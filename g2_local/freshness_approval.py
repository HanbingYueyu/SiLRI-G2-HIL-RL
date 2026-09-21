"""Offline evidence qualification and explicit limits; never motion permission.

Qualification is a recorded observation of process cleanup, not a signature or
proof against a malicious owner. Evidence remains untouched and is bound by
SHA-256; approval reopens and validates it. No thresholds have default values.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import subprocess

from .clock_ipc import _decode_json, _integer, _number, _text
from .clock_mapping import (MAX_DRIFT_PPM, MAX_PATH_DELAY_NS, MAX_REPORT_INTEGER,
                            MAX_RESIDUAL_NS, PTP_LEASE_NS, SOURCES)
from .freshness import FreshnessLimits, _decimal, _evidence, _finite, _timestamp
from .live_clock import ClockSnapshot


_MASTER = '044052.fffe.000010'
_FILES = ('evidence.jsonl', 'summary.json')
_LIMIT_NAMES = frozenset(field.name for field in fields(FreshnessLimits))
_FLAGS = dict(motion_authorized=False, thresholds_approved=False,
              source_clock_identity_proven=False, actions_discarded=True)
_PROCESS_COMMAND = ['/usr/bin/ps', '-eo', 'pid=,stat=,args=']


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def _identity(info):
    return info.st_dev, info.st_ino


@contextmanager
def _directory(path):
    """Walk every component without following symlinks or creating paths."""
    path = Path(os.path.abspath(path))
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                          dir_fd=fd)
            os.close(fd)
            fd = new
        yield fd
    finally:
        os.close(fd)


def _read(path, limit=64*1024*1024):
    path = Path(os.path.abspath(path))
    with _directory(path.parent) as directory:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=directory)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
                raise ValueError('Evidence must be a bounded independent regular file')
            raw = stream.read(limit+1)
            after = os.fstat(stream.fileno())
            named = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
            signature = lambda s: (_identity(s), s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if len(raw) > limit or signature(before) != signature(after) or signature(after) != signature(named):
                raise ValueError('Evidence path identity changed during read')
            with _directory(path.parent) as reopened:
                if _identity(os.fstat(directory)) != _identity(os.fstat(reopened)):
                    raise ValueError('Evidence directory identity changed during read')
    return raw, sha256(raw).hexdigest()


def _json(raw):
    value = _decode_json(raw)
    # Also catches finite-looking JSON exponents that overflow to infinity.
    def check(item):
        if type(item) is float and not math.isfinite(item):
            raise ValueError('nonfinite evidence')
        if isinstance(item, dict):
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
    check(value)
    return value


def _records(raw):
    rows = [_json(line) for line in raw.splitlines()]
    if not rows or any(type(row) is not dict for row in rows):
        raise ValueError('Invalid raw evidence records')
    return rows


def _exclusive(output, value):
    encoded = (_canonical(value)+'\n').encode()
    output = Path(os.path.abspath(output))
    with _directory(output.parent) as directory:
        fd = os.open(output.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o400, dir_fd=directory)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        with _directory(output.parent) as reopened:
            if _identity(os.fstat(directory)) != _identity(os.fstat(reopened)):
                raise ValueError('Output directory identity changed')
        os.fsync(directory)
    return value


def _process_listing():
    return subprocess.run(_PROCESS_COMMAND, check=True,
                          capture_output=True, text=True, timeout=10).stdout


def _residuals(listing):
    result = []
    for line in listing.splitlines():
        parts = line.strip().split(maxsplit=2)
        if (len(parts) != 3 or not parts[0].isascii() or not parts[0].isdigit() or
                not parts[1] or any(c not in 'RSDZTWXIKP<NLsl+' for c in parts[1])):
            raise ValueError('Invalid process listing')
        pid = _integer(int(parts[0]), minimum=1)
        # ps args is display text, not shell serialization. Splitting only
        # whitespace preserves literal unmatched quotes in unrelated argv.
        # No PID/ancestor exclusion is needed: our fixed ps command contains
        # none of the target tokens, including when it lists itself.
        argv = parts[2].split()
        modules = ('g2_local.freshness_audit', 'g2_local.clock_monitor')
        names = ('ptp4l', 'pmc', 'phc2sys', 'freshness_audit.py', 'clock_monitor.py',
                 'freshness_audit', 'clock_monitor', 'clock_monitor.p')
        if any(word.removeprefix('-m') in modules or Path(word.strip('[]')).name in names
               for word in argv):
            result.append(dict(pid=pid, command=parts[2]))
        elif parts[1].startswith('Z') and Path(argv[0].strip('[]')).name.startswith(('python', 'pypy')):
            # Zombie interpreters lose their module argv. Do not claim they
            # belong to this session, but cannot certify their cleanup either.
            result.append(dict(pid=pid, command=parts[2], reason='ambiguous_zombie_python'))
    return result


def _monitor(raw, session_id, initial_mono_ns, formal_end_mono_ns):
    rows = _records(raw)
    sessions = [r for r in rows if r.get('kind') == 'session']
    exits = [r for r in rows if r.get('kind') == 'exit']
    if (len(sessions) != 1 or sessions[0].get('session_id') != session_id or
            sessions[0].get('expected_master') != _MASTER or
            sessions[0].get('motion_authorized') is not False or
            len(exits) != 1 or type(exits[0].get('returncode')) is not int or
            exits[0]['returncode'] != 0 or rows[0] != sessions[0] or rows[-1] != exits[0] or
            exits[0].get('healthy') is not False or
            exits[0].get('reason') not in ('stop_requested', 'signal:2', 'signal:15') or
            any(r.get('kind') == 'cleanup_pending' for r in rows)):
        raise ValueError('Monitor identity or clean exit evidence missing')
    times = [_timestamp(row.get('mono_ns'), 'monitor timestamp') for row in rows]
    initial = _timestamp(initial_mono_ns, 'initial snapshot timestamp')
    formal_end = _timestamp(formal_end_mono_ns, 'formal end timestamp')
    if (any(a > b for a, b in zip(times, times[1:])) or
            not times[0] <= initial <= formal_end < times[-1]):
        raise ValueError('Monitor lifetime must cover the complete formal audit')
    for row, mono in zip(rows, times):
        if row.get('kind') == 'ptp':
            received = _timestamp(row.get('received_mono_ns'), 'PTP receipt timestamp')
            _timestamp(row.get('wall_ns'), 'PTP wall timestamp')
            if not times[0] <= received <= mono:
                raise ValueError('Monitor PTP timestamp ordering invalid')
        if row.get('kind') == 'properties':
            if _integer(row.get('returncode'), minimum=0, maximum=255) != 0:
                raise ValueError('Monitor properties query failed')


def record_qualification(audit_dir, monitor_evidence, output, *, process_listing=None):
    """After audit/monitor exit, record a fixed read-only cleanup process scan.

    ``process_listing`` is an offline test seam returning ps-format text.
    The CLI always uses the fixed system process listing, with no commands or
    process control accepted from the caller.
    """
    audit_dir = Path(os.path.abspath(audit_dir))
    if Path(os.path.abspath(output)) != audit_dir/'qualification.json':
        raise ValueError('Qualification output must be audit_dir/qualification.json')
    blobs = {name: _read(audit_dir/name) for name in _FILES}
    summary = _json(blobs['summary.json'][0])
    rows = _records(blobs['evidence.jsonl'][0])
    samples = [row for row in rows if row.get('event') == 'sample']
    if summary.get('status') != 'completed' or not samples:
        raise ValueError('Completed audit evidence required')
    monitor_path = str(Path(os.path.abspath(monitor_evidence)))
    monitor_raw, monitor_hash = _read(monitor_path)
    monitor_id = samples[0]['snapshot']['session_id']
    if rows[3].get('event') != 'formal_start' or rows[-1].get('event') != 'formal_end':
        raise ValueError('Formal timing evidence missing')
    _monitor(monitor_raw, monitor_id, rows[3]['initial_snapshot']['created_mono_ns'],
             rows[-1]['formal_end_mono_ns'])
    residuals = _residuals((process_listing or _process_listing)())
    if residuals:
        reasons = ', '.join(r.get('reason', 'audit_monitor_or_ptp') for r in residuals)
        raise ValueError('residual_process: '+reasons)
    # Bind a stable set: do not attest files that changed during the scan.
    for name, (_, digest) in blobs.items():
        if _read(audit_dir/name)[1] != digest:
            raise ValueError('Evidence hash changed during process check')
    if _read(monitor_path)[1] != monitor_hash:
        raise ValueError('Monitor hash changed during process check')
    return _exclusive(output, dict(
        schema=1, audit_session_id=_text(summary['audit_session_id']),
        monitor_session_id=monitor_id, checked_at_utc=datetime.now(timezone.utc).isoformat(),
        process_check=dict(command=_PROCESS_COMMAND, residuals=[], clean=True),
        audit_hashes={name: digest for name, (_, digest) in blobs.items()},
        monitor_evidence=monitor_path, monitor_sha256=monitor_hash,
        motion_authorized=False, thresholds_approved=False))


def _snapshot(payload, previous=None):
    if set(payload) != {f.name for f in fields(ClockSnapshot)}:
        raise ValueError('Unexpected snapshot fields')
    for name in ('schema', 'sequence', 'utc_offset_s', 'utc_offset_valid', 'leap61', 'leap59', 'ptp_timescale',
                 'reference_mono_ns', 'created_mono_ns', 'last_sample_mono_ns', 'valid_until_ns'):
        _integer(payload[name], minimum=0)
    _integer(payload['wall_minus_mono_ns'])
    _integer(payload['path_delay_ns'], minimum=0, maximum=MAX_PATH_DELAY_NS)
    for name, low, high in [('offset_at_reference_ns', -MAX_REPORT_INTEGER, MAX_REPORT_INTEGER),
                            ('drift_ppm', -MAX_DRIFT_PPM, MAX_DRIFT_PPM),
                            ('residual_ns', 0, MAX_RESIDUAL_NS), ('empirical_error_ns', 0, MAX_REPORT_INTEGER)]:
        _number(payload[name], minimum=low, maximum=high)
    for name in ('boot_id', 'session_id'):
        _text(payload[name])
    expected = dict(schema=1, healthy=True, reason='ok', expected_master=_MASTER,
                    actual_master=_MASTER, scale='raw_ptp', utc_offset_s=37, leap61=0, leap59=0, ptp_timescale=1)
    if any(type(payload[k]) is not type(v) or payload[k] != v for k, v in expected.items()):
        raise ValueError('Invalid snapshot identity/health/properties')
    snap = ClockSnapshot(**payload)
    if (snap.sequence < 1 or snap.utc_offset_valid not in (0, 1) or
            not 0 < snap.reference_mono_ns == snap.last_sample_mono_ns <= snap.created_mono_ns or
            snap.valid_until_ns != snap.last_sample_mono_ns+PTP_LEASE_NS):
        raise ValueError('Invalid snapshot lease')
    if previous is not None:
        if snap.sequence <= previous.sequence:
            raise ValueError('Snapshot sequence did not advance')
        if (snap.boot_id != previous.boot_id or snap.session_id != previous.session_id or
                snap.created_mono_ns < previous.created_mono_ns or
                snap.last_sample_mono_ns < previous.last_sample_mono_ns):
            raise ValueError('Snapshot identity/time changed')
        if snap.last_sample_mono_ns == previous.last_sample_mono_ns:
            mutable = {'sequence', 'created_mono_ns'}
            if any(payload[f.name] != getattr(previous, f.name) for f in fields(snap) if f.name not in mutable):
                raise ValueError('Same source sample changed mapping or lease')
    return snap


def _metrics(row, previous_row, previous_snapshot):
    info, raw = row['info'], row['inference']
    now = _timestamp(row['received_mono_ns'], 'received')
    snap = _snapshot(row['snapshot'], previous_snapshot)
    start = _timestamp(row['inference_start_mono_ns'], 'inference start')
    end = _timestamp(row['inference_end_mono_ns'], 'inference end')
    if not info['read_end_monotonic_ns'] <= start <= end <= now <= snap.valid_until_ns or snap.created_mono_ns > now:
        raise ValueError('Invalid inference/lease ordering')
    stamps = _evidence(info, snap, now)
    if previous_row is not None:
        old = previous_row['info']
        if info['read_start_monotonic_ns'] < previous_row['received_mono_ns']:
            raise ValueError('Read sequence reversed')
        if any(stamps[s] <= old['source_timestamp_ns'][s] for s in SOURCES):
            raise ValueError('Source sequence frozen or reversed')
    error = _decimal(snap.empirical_error_ns)+Fraction(max(0, now-snap.reference_mono_ns)*MAX_DRIFT_PPM, 1_000_000)
    denominator = 1-_decimal(snap.drift_ppm)/1_000_000
    intervals, ages, lower = {}, {}, {}
    for source in SOURCES:
        delta = stamps[source]-snap.wall_minus_mono_ns-snap.reference_mono_ns
        center = snap.reference_mono_ns+(delta+_decimal(snap.offset_at_reference_ns))/denominator
        lo, hi = center-error/denominator, center+error/denominator
        if lo > now:
            raise ValueError('Future source timestamp')
        intervals[source] = [math.floor(lo), math.ceil(hi)]
        ages[source], lower[source] = float((now-lo)/1_000_000), float((now-hi)/1_000_000)
    for interval in row['source_intervals_ns'].values():
        if type(interval) is not list or len(interval) != 2:
            raise ValueError('Raw source interval shape invalid')
        if _integer(interval[0]) > _integer(interval[1]):
            raise ValueError('Raw source interval ordering invalid')
    if intervals != row['source_intervals_ns']:
        raise ValueError('Raw source interval mismatch')
    left, right = intervals['left_wrist'], intervals['right_aux']
    metrics = dict(camera_age_ms={s: ages[s] for s in SOURCES[:2]},
                   camera_age_lower_ms={s: lower[s] for s in SOURCES[:2]},
                   state_age_ms={s: ages[s] for s in SOURCES[2:]},
                   state_age_lower_ms={s: lower[s] for s in SOURCES[2:]},
                   camera_skew_ms=max(left[1]-right[0], right[1]-left[0])/1e6,
                   source_camera_skew_ms=abs(stamps['left_wrist']-stamps['right_aux'])/1e6,
                   mapping_error_ms=float(error/1_000_000), mapping_residual_ms=snap.residual_ns/1e6,
                   mapping_drift_ppm=snap.drift_ppm, mapping_path_delay_ms=snap.path_delay_ns/1e6,
                   gdk_read_duration_ms=info['read_duration_s']*1000, inference_duration_ms=(end-start)/1e6,
                   snapshot_gap_ms=(snap.created_mono_ns-previous_snapshot.created_mono_ns)/1e6)
    if previous_row is not None:
        metrics['gdk_read_gap_ms'] = (info['read_start_monotonic_ns']-previous_row['info']['read_end_monotonic_ns'])/1e6
    candidate, pose = info['tf_queries'][1]['pose'], info['motion_pose']
    dot = abs(sum(a*b for a, b in zip(candidate[3:], pose[3:]))/(math.hypot(*candidate[3:])*math.hypot(*pose[3:])))
    tf = dict(tf_position_error_m=math.dist(candidate[:3], pose[:3]),
              tf_rotation_error_rad=2*math.acos(min(1., dot)))
    for key, value in tf.items():
        if not math.isclose(value, info[key], rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError('Raw TF pose error mismatch')
        metrics[key] = info[key]
    shape = raw['action_shape']
    if (raw['action_discarded'] is not True or type(shape) is not list or
            len(shape) != 2 or any(type(dimension) is not int for dimension in shape) or shape != [1, 6]):
        raise ValueError('Invalid discarded action')
    low, high = raw['action_min'], raw['action_max']
    if type(low) not in (int, float) or type(high) not in (int, float) or not -1+1e-6 <= low <= high <= 1-1e-6:
        raise ValueError('Invalid action range')
    for source, metric in [('cpu_prepare_ns', 'cpu_prepare_ms'), ('h2d_resize_ns', 'h2d_resize_ms'),
                           ('forward_ns', 'actor_forward_ms'), ('total_ns', 'actor_inference_ms')]:
        metrics[metric] = _integer(raw[source], minimum=0)/1e6
    for name in ('cuda_allocated', 'cuda_reserved', 'cuda_peak_allocated'):
        metrics[name+'_mib'] = _integer(raw[name+'_bytes'], minimum=0)/1024**2
    if raw['total_ns'] > end-start or any(raw[k] > raw['total_ns'] for k in ('cpu_prepare_ns', 'h2d_resize_ns', 'forward_ns')):
        raise ValueError('Inference timing inconsistent')
    if _canonical(metrics) != _canonical(row['metrics']):
        raise ValueError('Raw metric mismatch')
    # Use the more conservative reconstructed TF value, including roundoff.
    metrics.update({key: max(value, info[key]) for key, value in tf.items()})
    return metrics, snap


def _distribution(values):
    values = sorted(values)
    result = dict(min=values[0], max=values[-1])
    for key, q in [('p50', .5), ('p95', .95), ('p99', .99)]:
        index = (len(values)-1)*q
        lo, hi = math.floor(index), math.ceil(index)
        result[key] = values[lo]+(values[hi]-values[lo])*(index-lo)
    return result


def _summarize(rows):
    result = {}
    for key in sorted(set().union(*(row.keys() for row in rows))):
        values = [row[key] for row in rows if key in row]
        result[key] = ({s: _distribution([value[s] for value in values]) for s in values[0]}
                       if isinstance(values[0], dict) else _distribution(values))
    return result


def _validate(path):
    blobs = {name: _read(path/name) for name in (*_FILES, 'qualification.json')}
    summary, qualification = (_json(blobs[name][0]) for name in ('summary.json', 'qualification.json'))
    if any(qualification['audit_hashes'][name] != blobs[name][1] for name in _FILES):
        raise ValueError('Qualification hash mismatch')
    scan = qualification['process_check']
    if (_integer(qualification.get('schema'), minimum=1, maximum=1) != 1 or
            scan.get('clean') is not True or scan.get('residuals') != [] or
            scan.get('command') != _PROCESS_COMMAND or
            qualification.get('motion_authorized') is not False or qualification.get('thresholds_approved') is not False):
        raise ValueError('residual_process: clean qualification required')
    checked = datetime.fromisoformat(_text(qualification['checked_at_utc']))
    if checked.utcoffset() != timezone.utc.utcoffset(None) or checked.timestamp() <= 0:
        raise ValueError('Qualification timestamp must be positive UTC')
    monitor_raw, digest = _read(qualification['monitor_evidence'])
    if digest != qualification['monitor_sha256']:
        raise ValueError('Monitor hash mismatch')
    rows = _records(blobs['evidence.jsonl'][0])
    events = [row.get('event') for row in rows]
    if (len(rows) < 1005 or events[:4] != ['session', 'actor_metadata', 'warmup', 'formal_start'] or
            events[-1] != 'formal_end' or any(event != 'sample' for event in events[4:-1])):
        raise ValueError('samples: invalid complete audit event sequence')
    session, actor, warmup, begin, finish = rows[0], rows[1], rows[2], rows[3], rows[-1]
    samples = rows[4:-1]
    session_id = _text(session['audit_session_id'])
    if summary['audit_session_id'] != session_id or qualification['audit_session_id'] != session_id:
        raise ValueError('Audit session identity mismatch')
    for key, value in _FLAGS.items():
        if summary.get(key) is not value:
            raise ValueError('Authorization flags must remain fixed')
    if any(session.get(key) is not _FLAGS[key] for key in ('motion_authorized', 'thresholds_approved', 'actions_discarded')):
        raise ValueError('Raw authorization flags invalid')
    for record in (summary, session):
        _finite(record['inference_delay_s'], 'inference_delay_s')
    if (summary['status'] != 'completed' or summary['reason'] != '' or
            summary['actor_mode'] is not True or session['actor_mode'] is not True or
            summary['inference_delay_s'] != 0 or session['inference_delay_s'] != 0):
        raise ValueError('Completed real Actor session required')
    if _integer(summary['rejected_count'], minimum=0, maximum=36000) != 0:
        raise ValueError('rejected samples')
    for key in ('accepted_count', 'sample_count'):
        if _integer(summary[key], minimum=1000, maximum=36000) != len(samples):
            raise ValueError('samples count mismatch or too small')
    if (_integer(summary['duration_s'], minimum=120, maximum=1800) !=
            _integer(session['duration_s'], minimum=120, maximum=1800)):
        raise ValueError('duration too short or mismatched')
    start, end = _timestamp(begin['formal_start_mono_ns'], 'formal start'), _timestamp(finish['formal_end_mono_ns'], 'formal end')
    observed = (_timestamp(samples[-1]['received_mono_ns'], 'last receive')-
                _timestamp(samples[0]['info']['read_start_monotonic_ns'], 'first read'))/1e9
    elapsed = (end-start)/1e9
    for record in (summary, finish):
        _finite(record['formal_elapsed_s'], 'formal_elapsed_s', positive=True)
    _timestamp(summary['formal_start_mono_ns'], 'summary formal start')
    _timestamp(summary['formal_end_mono_ns'], 'summary formal end')
    if (observed < 120 or elapsed < 120 or
            not start <= samples[0]['info']['read_start_monotonic_ns'] <= samples[-1]['received_mono_ns'] <= end or
            summary['formal_start_mono_ns'] != start or summary['formal_end_mono_ns'] != end or
            summary['formal_elapsed_s'] != elapsed or finish['formal_elapsed_s'] != elapsed):
        raise ValueError('duration raw boundaries do not qualify')
    metadata = summary['actor_metadata']
    for record in (session, summary, warmup, metadata, actor['metadata'], warmup['metadata']):
        _integer(record['warmup_steps'], minimum=0)
        if record is not session:
            _integer(record['warmup_completed'], minimum=0)
    if actor['metadata']['warmup_completed'] != 0:
        raise ValueError('Initial warmup count must be zero')
    if _canonical(metadata) != _canonical(warmup['metadata']):
        raise ValueError('checkpoint/config/gpu metadata mismatch')
    initial = dict(actor['metadata'])
    initial['warmup_completed'] = metadata['warmup_completed']
    if _canonical(initial) != _canonical(metadata):
        raise ValueError('checkpoint/config/gpu changed after warmup')
    checkpoint = metadata['checkpoint_sha256']
    if len(checkpoint) != 64 or any(c not in '0123456789abcdef' for c in checkpoint):
        raise ValueError('Invalid checkpoint sha256')
    if metadata['gpu_name'] != 'NVIDIA GeForce RTX 3090' or metadata['device'] != 'cuda':
        raise ValueError('gpu identity mismatch')
    if (type(metadata['checkpoint_schema']) is not int or metadata['checkpoint_schema'] != 1 or
            type(metadata['checkpoint_version']) is not int or metadata['checkpoint_version'] < 0 or
            type(metadata['policy_config']) is not dict or not metadata['policy_config']):
        raise ValueError('checkpoint config missing')
    _integer(metadata['checkpoint_version'], minimum=0)
    requested = summary['warmup_steps']
    previous = _snapshot(begin['initial_snapshot'])
    warmup_start = _timestamp(warmup['start_mono_ns'], 'warmup start')
    warmup_end = _timestamp(warmup['end_mono_ns'], 'warmup end')
    if (type(requested) is not int or requested < 0 or
            any(value != requested for value in (session['warmup_steps'], warmup['warmup_steps'], warmup['warmup_completed'],
                summary['warmup_completed'], metadata['warmup_steps'], metadata['warmup_completed'])) or
            not previous.created_mono_ns <= warmup_start <= warmup_end <= start):
        raise ValueError('Warmup evidence mismatch')
    monitor_id = previous.session_id
    if qualification['monitor_session_id'] != monitor_id:
        raise ValueError('Monitor session mismatch')
    _monitor(monitor_raw, monitor_id, previous.created_mono_ns, end)
    metrics, old = [], None
    for row in samples:
        metric, previous = _metrics(row, old, previous)
        metrics.append(metric)
        old = row
    distributions = _summarize(metrics)
    for key, value in distributions.items():
        if key.startswith('tf_'):
            if any(not math.isclose(summary[key][q], v, abs_tol=1e-7) for q, v in value.items()):
                raise ValueError('TF summary mismatch')
        elif _canonical(summary[key]) != _canonical(value):
            raise ValueError('Recomputed summary mismatch: '+key)
    if (summary['action_min'] != min(r['inference']['action_min'] for r in samples) or
            summary['action_max'] != max(r['inference']['action_max'] for r in samples)):
        raise ValueError('Action summary mismatch')
    return dict(path=str(path), audit_session_id=session_id, monitor_session_id=monitor_id,
                hashes={name: digest for name, (_, digest) in blobs.items()},
                monitor_evidence=qualification['monitor_evidence'], monitor_sha256=qualification['monitor_sha256'],
                actor_metadata=metadata, observed_elapsed_s=observed, sample_count=len(samples), metrics=distributions)


@dataclass(frozen=True)
class ApprovalEvidence:
    paths: tuple[str, ...]
    evidence_json: str

    @property
    def sessions(self):
        return json.loads(self.evidence_json)['sessions']

    @property
    def worst_case(self):
        return json.loads(self.evidence_json)['worst_case']


def validate_sessions(paths) -> ApprovalEvidence:
    paths = tuple(Path(os.path.abspath(path)) for path in paths)
    if len(paths) != 3 or len(set(paths)) != 3:
        raise ValueError('Exactly three independent sessions required')
    try:
        sessions = [_validate(path) for path in paths]
        if (len({s['audit_session_id'] for s in sessions}) != 3 or
                len({s['monitor_session_id'] for s in sessions}) != 3):
            raise ValueError('Exactly three independent sessions required')
        for field in ('checkpoint_sha256', 'checkpoint_schema', 'checkpoint_version', 'policy_config', 'gpu_name', 'device'):
            if len({_canonical(s['actor_metadata'][field]) for s in sessions}) != 1:
                raise ValueError('checkpoint/config/gpu mismatch: '+field)
        worst = {}
        for limit, metric in [('camera_age_s', 'camera_age_ms'), ('state_age_s', 'state_age_ms'),
                              ('camera_skew_s', 'camera_skew_ms'), ('mapping_error_s', 'mapping_error_ms'),
                              ('tf_position_error_m', 'tf_position_error_m'), ('tf_rotation_error_rad', 'tf_rotation_error_rad')]:
            values = [s['metrics'][metric] for s in sessions]
            maxima = [v['max'] if 'max' in v else max(d['max'] for d in v.values()) for v in values]
            worst[limit] = max(maxima)/(1000 if metric.endswith('_ms') else 1)
        if worst['tf_position_error_m'] > .005 or worst['tf_rotation_error_rad'] > .02:
            raise ValueError('TF observed maxima exceed task ceilings')
        return ApprovalEvidence(tuple(map(str, paths)), _canonical(dict(sessions=sessions, worst_case=worst)))
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('Incomplete or invalid qualification evidence') from error


def approve_limits(evidence, limits, output):
    if type(evidence) is not ApprovalEvidence:
        raise ValueError('Validated ApprovalEvidence required')
    if type(limits) is not dict or set(limits) != _LIMIT_NAMES:
        raise ValueError('All six explicit limits are required, without defaults')
    for name, value in limits.items():
        _finite(value, name, positive=True)
    fresh = validate_sessions(evidence.paths)
    if fresh.evidence_json != evidence.evidence_json:
        raise ValueError('Evidence changed since validation')
    margins = {name: limits[name]-value for name, value in fresh.worst_case.items()}
    for name in ('camera_age_s', 'state_age_s', 'camera_skew_s', 'mapping_error_s'):
        if margins[name] <= 0:
            raise ValueError('Explicit positive worst-case margin required: '+name)
    if limits['tf_position_error_m'] != .005 or limits['tf_rotation_error_rad'] != .02:
        raise ValueError('TF task ceilings must remain 0.005 m and 0.02 rad')
    metadata = fresh.sessions[0]['actor_metadata']
    return _exclusive(output, dict(schema=1, limits=limits, margins=margins,
        worst_case=fresh.worst_case, sessions=fresh.sessions,
        checkpoint_sha256=metadata['checkpoint_sha256'], policy_config=metadata['policy_config'],
        gpu_name=metadata['gpu_name'], approved_at_utc=datetime.now(timezone.utc).isoformat(),
        thresholds_approved=True, motion_authorized=False, source_clock_identity_proven=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    qualification = commands.add_parser('qualify')
    qualification.add_argument('--audit-dir', required=True, type=Path)
    qualification.add_argument('--monitor-evidence', required=True, type=Path)
    qualification.add_argument('--output', required=True, type=Path)
    approve = commands.add_parser('approve')
    approve.add_argument('--sessions', required=True, nargs=3, type=Path)
    approve.add_argument('--output', required=True, type=Path)
    for name in sorted(_LIMIT_NAMES):
        approve.add_argument('--'+name.replace('_', '-'), required=True, type=float)
    args = parser.parse_args(argv)
    try:
        if args.operation == 'qualify':
            record_qualification(args.audit_dir, args.monitor_evidence, args.output)
        else:
            approve_limits(validate_sessions(args.sessions), {k: getattr(args, k) for k in _LIMIT_NAMES}, args.output)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'Qualification/approval rejected: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
