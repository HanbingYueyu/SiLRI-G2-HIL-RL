"""Fail-closed source freshness evidence; never an authorization for motion.

Uncertainty is the monitor's empirical allowance, not a certified physical
bound. Every task threshold must be explicitly supplied by the caller.
"""
from dataclasses import asdict, dataclass, field
from fractions import Fraction
import math
import threading
import time
from types import MappingProxyType

from .clock_ipc import _snapshot_from_payload
from .clock_mapping import MAX_DRIFT_PPM, MAX_REPORT_INTEGER, MAX_WALL_JUMP_NS, SOURCES
from .live_clock import ClockSnapshot


def _finite(value, name, *, positive=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if (not finite or
            (value <= 0 if positive else value < 0)):
        raise ValueError(f'{name}: finite {"positive" if positive else "nonnegative"} number required')
    return value


def _decimal(value):
    return Fraction(str(value))


def _timestamp(value, name):
    # Same signed-int64 domain as clock snapshots. SDK uint64/Python integers
    # beyond it must not become trusted evidence through arithmetic coercion.
    if type(value) is not int or not 0 < value <= MAX_REPORT_INTEGER:
        raise ValueError(f'{name}: positive signed-int64 nanoseconds required')
    return value


def _pose(value, name):
    if (type(value) is not list or len(value) != 7 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in value) or
            abs(math.hypot(*value[3:])-1) > .01):
        raise ValueError(f'{name}: finite XYZ and unit xyzw quaternion required')


def _evidence(info, snapshot, now):
    if type(info) is not dict:
        raise ValueError('Observation metadata must be a dict')
    stamps = info['source_timestamp_ns']
    if type(stamps) is not dict or set(stamps) != set(SOURCES):
        raise ValueError('Exactly four source timestamps required')
    stamps = {key: _timestamp(stamps[key], key) for key in SOURCES}
    cameras = info['camera_timestamp_ns']
    if type(cameras) is not dict or set(cameras) != set(SOURCES[:2]):
        raise ValueError('Exactly two camera timestamps required')
    for key in SOURCES[:2]:
        if _timestamp(cameras[key], key) != stamps[key]:
            raise ValueError(f'Camera/source timestamp mismatch: {key}')
    queries = info['tf_queries']
    if type(queries) is not list or len(queries) != 2:
        raise ValueError('Both raw TF directions required')
    for query, direction in zip(queries, (('base_link', 'arm_l_end_link'),
                                           ('arm_l_end_link', 'base_link'))):
        if (type(query) is not dict or
                set(query) != {'target', 'source', 'timestamp_ns', 'pose'} or
                (query['target'], query['source']) != direction):
            raise ValueError('Invalid TF query direction or fields')
        _timestamp(query['timestamp_ns'], 'TF query')
        _pose(query['pose'], 'TF query pose')
    if queries[1]['timestamp_ns'] != stamps['tf']:
        raise ValueError('Selected TF/source timestamp mismatch')
    _pose(info['motion_pose'], 'motion_pose')
    for key in ('tf_position_error_m', 'tf_rotation_error_rad', 'read_duration_s'):
        if type(info[key]) is not float:
            raise ValueError(f'{key}: floating-point evidence required')
        _finite(info[key], key)
    if info['tf_rotation_error_rad'] > math.pi:
        raise ValueError('TF rotation difference exceeds pi')
    for key in ('read_start_monotonic_ns', 'read_end_monotonic_ns',
                'read_start_wall_ns', 'read_end_wall_ns', 'read_start_sdk_clock_ns',
                'read_end_sdk_clock_ns', 'sdk_clock_ns', 'received_monotonic_ns',
                'state_received_monotonic_ns'):
        _timestamp(info[key], key)
    start, end = info['read_start_monotonic_ns'], info['read_end_monotonic_ns']
    if not start <= info['state_received_monotonic_ns'] <= end <= now:
        raise ValueError('Read/receive monotonic evidence is out of order')
    if (info['received_monotonic_ns'] != end or
            info['sdk_clock_ns'] != info['read_end_sdk_clock_ns'] or
            info['read_start_sdk_clock_ns'] > info['read_end_sdk_clock_ns'] or
            info['read_start_wall_ns'] > info['read_end_wall_ns'] or
            info['read_duration_s'] != (end-start)/1e9):
        raise ValueError('Inconsistent read clock evidence')
    for side in ('start', 'end'):
        origin = info[f'read_{side}_wall_ns']-info[f'read_{side}_monotonic_ns']
        if abs(origin-snapshot.wall_minus_mono_ns) > MAX_WALL_JUMP_NS:
            raise ValueError('Read wall-minus-monotonic origin differs from mapping')
    return stamps


@dataclass(frozen=True)
class FreshnessLimits:
    camera_age_s: float
    state_age_s: float
    camera_skew_s: float
    tf_position_error_m: float
    tf_rotation_error_rad: float
    mapping_error_s: float

    def __post_init__(self):
        for name, value in vars(self).items():
            _finite(value, name, positive=True)


@dataclass(frozen=True)
class FreshnessDecision:
    code: str
    detail: str
    snapshot_sequence: int | None = None
    mapping_error_s: float | None = None
    source_intervals_ns: object = field(default_factory=lambda: MappingProxyType({}))
    age_intervals_s: object = field(default_factory=lambda: MappingProxyType({}))
    camera_skew_s: float | None = None


class _Rejected(Exception):
    def __init__(self, code, detail):
        self.code, self.detail = code, detail


class ObservationFreshnessGuard:
    """Serialize checks and commit source progress only on complete acceptance.

    ``client`` is a SnapshotClient (or an offline equivalent with ``read``).
    Boot/master/session authentication is owned by that client; snapshot shape,
    lease and successful-observation identity/sequence are rechecked here.
    Observation shape is owned by MotionBackend. This guard neither changes
    the observation nor owns/operates any command or stop interface.
    """

    def __init__(self, client, limits: FreshnessLimits, *, monotonic_ns=None):
        if not callable(getattr(client, 'read', None)):
            raise ValueError('A snapshot client is required')
        if type(limits) is not FreshnessLimits:
            raise ValueError('Explicit FreshnessLimits are required')
        self._client, self._limits = client, limits
        self._now = time.monotonic_ns if monotonic_ns is None else monotonic_ns
        if not callable(self._now):
            raise ValueError('A monotonic clock is required')
        self._lock = threading.RLock()
        self._previous_source_ns = {}
        self._snapshot = None
        self._last_decision = FreshnessDecision('not_checked', 'No observation checked')

    @property
    def previous_source_ns(self):
        with self._lock:
            return self._previous_source_ns.copy()

    @property
    def last_decision(self):
        with self._lock:
            return self._last_decision

    def _read_mapping(self):
        try:
            snapshot = self._client.read()
        except Exception as error:
            raise _Rejected('mapping_unavailable', f'{type(error).__name__}: {error}') from error
        now = self._now()
        if type(now) is not int or not 0 < now <= MAX_REPORT_INTEGER:
            raise _Rejected('mapping_invalid', 'Invalid local monotonic time')
        if type(snapshot) is not ClockSnapshot:
            raise _Rejected('mapping_invalid', 'Expected ClockSnapshot')
        if type(snapshot.valid_until_ns) is int and now > snapshot.valid_until_ns:
            raise _Rejected('mapping_expired', 'Snapshot lease expired before validation')
        previous = self._snapshot
        try:
            _snapshot_from_payload(
                asdict(snapshot),
                expected_master=(previous.expected_master if previous else snapshot.expected_master),
                previous_sequence=previous.sequence if previous else None,
                previous_session=previous.session_id if previous else None,
                received_mono_ns=now,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise _Rejected('mapping_invalid', str(error)) from error
        return snapshot, now

    def __call__(self, obs, info, after=None) -> bool:
        with self._lock:
            diagnostics = {}
            try:
                snapshot, now = self._read_mapping()
                diagnostics['snapshot_sequence'] = snapshot.sequence
                # Same 100 ppm forward extrapolation allowance as DiagnosticMapping.
                error = (_decimal(snapshot.empirical_error_ns) +
                         Fraction(max(0, now-snapshot.reference_mono_ns)*MAX_DRIFT_PPM, 1_000_000))
                diagnostics['mapping_error_s'] = float(error/1_000_000_000)
                if error > _decimal(self._limits.mapping_error_s)*1_000_000_000:
                    raise _Rejected('mapping_error', 'Extrapolated empirical error exceeds limit')
                stamps = _evidence(info, snapshot, now)
                sent_ns = None
                if after is not None:
                    _finite(after, 'after')
                    # Round upward so conversion cannot move the send earlier.
                    sent_ns = math.ceil(Fraction(after)*1_000_000_000)
                    if sent_ns > now:
                        raise ValueError('after is later than local monotonic time')
                denominator = 1-_decimal(snapshot.drift_ppm)/1_000_000
                intervals, ages = {}, {}
                for source in SOURCES:
                    # Snapshot offset is wall-minus-raw-PTP: ClockWindow already
                    # subtracts the 37-second correction. Invert its affine model
                    # source = origin + t - offset_ref - drift*(t-reference).
                    delta = (stamps[source]-snapshot.wall_minus_mono_ns-
                             snapshot.reference_mono_ns)
                    center = (snapshot.reference_mono_ns +
                              (delta+_decimal(snapshot.offset_at_reference_ns))/denominator)
                    radius = error/denominator
                    intervals[source] = (center-radius, center+radius)
                    ages[source] = (now-center-radius, now-center+radius)
                diagnostics['source_intervals_ns'] = MappingProxyType({
                    key: (math.floor(lo), math.ceil(hi)) for key, (lo, hi) in intervals.items()})
                diagnostics['age_intervals_s'] = MappingProxyType({
                    key: (float(lo/1_000_000_000), float(hi/1_000_000_000))
                    for key, (lo, hi) in ages.items()})
                for source, (earliest, _) in intervals.items():
                    camera = source in SOURCES[:2]
                    limit = self._limits.camera_age_s if camera else self._limits.state_age_s
                    if now-earliest > _decimal(limit)*1_000_000_000:
                        raise _Rejected(('camera_stale:' if camera else 'state_stale:')+source,
                                        f'{source} worst-case age exceeds {limit} seconds')
                    if earliest > now:
                        raise _Rejected('source_future:'+source,
                                        f'{source} earliest possible acquisition is in the future')
                left, right = intervals['left_wrist'], intervals['right_aux']
                skew = max(left[1]-right[0], right[1]-left[0])
                diagnostics['camera_skew_s'] = float(skew/1_000_000_000)
                if skew > _decimal(self._limits.camera_skew_s)*1_000_000_000:
                    raise _Rejected('camera_skew', 'Worst-case camera separation exceeds limit')
                for source in SOURCES:
                    previous = self._previous_source_ns.get(source)
                    if previous is not None and stamps[source] <= previous:
                        code = 'source_frozen:' if stamps[source] == previous else 'source_reversed:'
                        raise _Rejected(code+source, f'{source} timestamp did not strictly advance')
                if sent_ns is not None:
                    for source, (earliest, _) in intervals.items():
                        if earliest <= sent_ns:
                            raise _Rejected('not_after_command:'+source,
                                            f'{source} interval is not strictly after command send')
                for name, code in (('tf_position_error_m', 'tf_position_error'),
                                   ('tf_rotation_error_rad', 'tf_rotation_error')):
                    if info[name] > getattr(self._limits, name):
                        raise _Rejected(code, f'{name} exceeds explicit limit')
                decision = FreshnessDecision('ok', 'Freshness checks passed; motion is not authorized',
                                             **diagnostics)
            except _Rejected as error:
                self._last_decision = FreshnessDecision(error.code, error.detail, **diagnostics)
                return False
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                self._last_decision = FreshnessDecision('invalid_evidence', str(error), **diagnostics)
                return False
            self._previous_source_ns = stamps
            self._snapshot = snapshot
            self._last_decision = decision
            return True
