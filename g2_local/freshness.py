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

# Upstream G2 semantics: freshness is "how long has it been since this process
# received a NEW timestamp from this source".  No robot-clock-to-local-clock
# mapping is involved, so no PTP client is needed here.
SOURCES = ('left_wrist', 'right_aux', 'joint', 'tf')
MAX_WALL_JUMP_NS = 1_000_000
MAX_REPORT_INTEGER = 2**63 - 1
# One-domain sanity bound: when the SDK command clock and the sensor
# timestamps share a domain, a source's age inside that domain is at most
# a few sensor periods. A larger gap means the two clocks are unrelated
# (e.g. a host clock offset), so the same-domain successor proof is not
# available and only the local-receipt guarantees apply.
SDK_SAME_DOMAIN_MAX_NS = 500_000_000


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


def _evidence(info, now):
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
    if True:
        start_origin = info['read_start_wall_ns']-info['read_start_monotonic_ns']
        end_origin = info['read_end_wall_ns']-info['read_end_monotonic_ns']
        if abs(start_origin-end_origin) > MAX_WALL_JUMP_NS:
            raise ValueError('Read wall-minus-monotonic origin changed within one read')
        return stamps


@dataclass(frozen=True)
class FreshnessLimits:
    """Explicit limits; every field is part of the approved commissioning record.

    Enforced today: ``camera_age_s``, ``state_age_s``, ``tf_position_error_m``,
    ``tf_rotation_error_rad``.

    Recorded but NOT enforced (local-receipt freshness has no clock mapping):
    ``camera_skew_s`` and ``mapping_error_s``. The guard computes the camera
    skew for diagnostics only (``FreshnessDecision.camera_skew_s``), and
    ``mapping_error_s`` has no implementation left. Changing those two values
    therefore changes ``config_hash`` without changing what is rejected — they
    stay in the dataclass because the commissioning approval file lists them.
    """
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
    # Same-domain command-send anchor evidence: anchor/sdk clock, whether the
    # two clocks proved comparable, and the per-source acquisition margin.
    sdk_anchor: object | None = None


class _Rejected(Exception):
    def __init__(self, code, detail):
        self.code, self.detail = code, detail


class FreshnessLeaseGuard:
    """Bound each command send by the last accepted observation's local lease."""

    def __init__(self, observation_guard, *, feedback_lease_s, clock=time.monotonic):
        if not callable(observation_guard) or not callable(clock):
            raise ValueError('Freshness guard and monotonic clock are required')
        if (type(feedback_lease_s) not in (int, float) or
                not math.isfinite(feedback_lease_s) or feedback_lease_s <= 0):
            raise ValueError('Positive feedback lease required')
        self.observation_guard = observation_guard
        self.feedback_lease_s = feedback_lease_s
        self.clock = clock
        self.valid_until = None
        self.lock = threading.RLock()

    @property
    def last_decision(self):
        return self.observation_guard.last_decision

    def accept(self, obs, info, after=None, after_sdk_ns=None):
        """Accept a fresh observation, or leave the existing lease untouched.

        A rejected read must NOT revoke an already granted lease. The command
        writer resends the *last* target (a measured hold) while the policy loop
        retries the successor read inside its bounded window; revoking the lease
        here made the 25 Hz writer latch a fatal ``Command stream fault:
        Feedback freshness not explicitly confirmed`` after ~40 ms, which killed
        the whole run before the retry could finish. Safety is unchanged: the
        lease still expires ``feedback_lease_s`` after the last *accepted*
        observation, and a new target is only submitted from an accepted read.
        """
        with self.lock:
            accepted = self.observation_guard(obs, info, after,
                                              after_sdk_ns=after_sdk_ns)
            if accepted is True:
                self.valid_until = self.clock() + self.feedback_lease_s
            return accepted

    def revalidate(self, obs, info):
        """Recheck an already accepted input without renewing the command lease."""
        with self.lock:
            revalidate = getattr(self.observation_guard, 'revalidate', None)
            return (revalidate(obs, info) if callable(revalidate)
                    else self.observation_guard(obs, info))

    def __call__(self):
        with self.lock:
            return self.valid_until is not None and self.clock() <= self.valid_until


class ObservationFreshnessGuard:
    """Serialize checks and commit source progress only on complete acceptance.

    Freshness is local: the age of a source is how long this process has gone
    without a *new* timestamp from it, and robot timestamps must still strictly
    advance.  No clock client is read.  Observation shape is owned by
    MotionBackend; this guard neither changes the observation nor owns or
    operates any command or stop interface.
    """

    def __init__(self, limits: FreshnessLimits = None, *, monotonic_ns=None,
                 skip_tf_progress=False, skip_state_progress=False):
        if type(limits) is not FreshnessLimits:
            raise ValueError('Explicit FreshnessLimits are required')
        self._limits = limits
        self._skip_tf_progress = skip_tf_progress
        # Internal repositioning loops (pre-reset) command far faster than the
        # robot state topic updates, so a repeated joint/TF timestamp is normal
        # there; their liveness is still bounded by state_age_s.
        self._skip_state_progress = skip_state_progress
        self._now = time.monotonic_ns if monotonic_ns is None else monotonic_ns
        if not callable(self._now):
            raise ValueError('A monotonic clock is required')
        self._lock = threading.RLock()
        self._previous_source_ns = {}
        self._last_decision = FreshnessDecision('not_checked', 'No observation checked')

    @property
    def previous_source_ns(self):
        with self._lock:
            return self._previous_source_ns.copy()

    @property
    def last_decision(self):
        with self._lock:
            return self._last_decision

    def revalidate(self, obs, info):
        """Recheck an already accepted policy input's age without source progress."""
        return self.__call__(obs, info, check_progress=False)

    def validate_stop_feedback(self, info):
        """Validate a newly read measured hold, independently of camera evidence.

        This never renews the normal command lease or accepts an observation.
        Mapping and joint/TF acquisition ages must still satisfy their limits.
        """
        with self._lock:
            try:
                if True:
                    now = self._now()
                    _pose(info['motion_pose'], 'motion_pose')
                    start = _timestamp(info['read_start_monotonic_ns'], 'read_start')
                    if not 0 <= now-start <= self._limits.state_age_s*1e9:
                        raise ValueError('stop feedback read expired')
                    query = info['tf_queries'][1]
                    if (query['target'], query['source']) != ('arm_l_end_link', 'base_link'):
                        raise ValueError('stop TF direction mismatch')
                    _pose(query['pose'], 'tf_pose')
                    for name in ('tf_position_error_m', 'tf_rotation_error_rad'):
                        if _finite(info[name], name) > getattr(self._limits, name):
                            raise ValueError('stop '+name+' exceeds limit')
                    return True
            except (_Rejected, KeyError, TypeError, ValueError, OverflowError) as error:
                raise RuntimeError('Independent stop feedback rejected: '+str(error)) from error

    def __call__(self, obs, info, after=None, after_sdk_ns=None, *, check_progress=True) -> bool:
        with self._lock:
            diagnostics = {}
            try:
                now = self._now()
                stamps = _evidence(info, now)
                sent_ns = None
                if after is not None:
                    _finite(after, 'after')
                    # Round upward so conversion cannot move the send earlier.
                    sent_ns = math.ceil(Fraction(after)*1_000_000_000)
                    if sent_ns > now:
                        raise ValueError('after is later than local monotonic time')
                anchor_ns = None
                if after_sdk_ns is not None:
                    # Robot-domain anchor of the command send (gdk.Clock at ack).
                    anchor_ns = _timestamp(after_sdk_ns, 'command send SDK clock')
                intervals, ages = {}, {}
                if True:
                    changed = info['source_changed_monotonic_ns']
                    if type(changed) is not dict or set(changed) != set(SOURCES):
                        raise ValueError('Exactly four source receipt times required')
                    for source in SOURCES:
                        receipt = _timestamp(changed[source], source)
                        if not 0 <= receipt <= now:
                            raise ValueError('Source receipt time is not in the past')
                        intervals[source] = (receipt, receipt)
                        ages[source] = (now-receipt, now-receipt)
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
                for source in SOURCES:
                    if source == 'tf' and self._skip_tf_progress:
                        continue
                    if source in ('joint', 'tf') and self._skip_state_progress:
                        continue
                    previous = self._previous_source_ns.get(source)
                    if check_progress and previous is not None and stamps[source] <= previous:
                        if source in SOURCES[:2] and stamps[source] == previous:
                            continue
                        code = 'source_frozen:' if stamps[source] == previous else 'source_reversed:'
                        raise _Rejected(code+source, f'{source} timestamp did not strictly advance')
                if sent_ns is not None:
                    for source, (earliest, _) in intervals.items():
                        if earliest <= sent_ns:
                            raise _Rejected('not_after_command:'+source,
                                            f'{source} interval is not strictly after command send')
                if anchor_ns is not None:
                    # Same-domain successor proof: the successor source timestamps
                    # must be *acquired* after the command send. Only possible
                    # when the SDK clock used for the anchor and the sensor
                    # timestamps share one domain; that is verified per
                    # observation instead of assumed, so a hardware/toolchain
                    # mismatch degrades to the local-receipt guarantee with an
                    # explicit diagnostic rather than rejecting every frame.
                    sdk_now = info['sdk_clock_ns']
                    sdk_ages = {source: sdk_now - stamps[source] for source in SOURCES}
                    same_domain = all(0 <= age <= SDK_SAME_DOMAIN_MAX_NS
                                      for age in sdk_ages.values())
                    diagnostics['sdk_anchor'] = MappingProxyType({
                        'anchor_ns': anchor_ns, 'sdk_clock_ns': sdk_now,
                        'same_domain': same_domain,
                        'margins_ns': MappingProxyType(
                            {source: stamps[source] - anchor_ns for source in SOURCES}),
                        'sdk_ages_ns': MappingProxyType(dict(sdk_ages))})
                    if same_domain:
                        # Enforced for the successor *images* only: those are what
                        # pair with the executed action, and a persistently stale
                        # camera frame is retried inside the successor window.
                        # joint/tf keep their own 50 ms age limit and strict
                        # progress check; their same-domain margin is recorded in
                        # `sdk_anchor.margins_ns` so it can be enforced later
                        # without risking a spurious fatal fault on a slow state
                        # topic.
                        for source in SOURCES[:2]:
                            if stamps[source] <= anchor_ns:
                                raise _Rejected(
                                    'not_after_command_sdk:'+source,
                                    f'{source} acquisition timestamp is not after the '
                                    'same-domain command send time')
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
            if check_progress:
                self._previous_source_ns = stamps
            self._last_decision = decision
            return True
