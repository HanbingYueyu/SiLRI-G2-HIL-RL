"""Finite read-only GDK load evidence; never approves thresholds or motion."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from .clock_ipc import SnapshotClient, _snapshot_from_payload
from .clock_mapping import MAX_DRIFT_PPM, SOURCES
from .clock_monitor import EXPECTED_MASTER, _create_session_directory
from .freshness import _decimal, _evidence, _finite, _timestamp
from .live_clock import ClockSnapshot


_MAX_LOG_BYTES = 64 * 1024 * 1024
_MAX_ROW_BYTES = 16384
_MAX_SAMPLES = 36000
_SAMPLE_PERIOD_S = .05


def validate_audit_duration(duration_s):
    if type(duration_s) is not int or not 30 <= duration_s <= 1800:
        raise ValueError('Audit duration must be explicit integer 30..1800 seconds')
    return duration_s


def _reader():
    # SDK initialization is deferred until output and monitor preflight succeed.
    from .gdk_backend import GdkReader
    return GdkReader()


def _mapping(client, now_fn, previous):
    snapshot = client.read()
    now = _timestamp(now_fn(), 'local monotonic time')
    if type(snapshot) is not ClockSnapshot:
        raise ValueError('Expected ClockSnapshot')
    _snapshot_from_payload(
        asdict(snapshot), expected_master=EXPECTED_MASTER,
        previous_sequence=None if previous is None else previous.sequence,
        previous_session=None if previous is None else previous.session_id,
        received_mono_ns=now)
    if previous is not None and snapshot.last_sample_mono_ns < previous.last_sample_mono_ns:
        raise ValueError('PTP sample timestamp reversed')
    return snapshot, now


def _measure(info, snapshot, now, previous_info, previous_snapshot, inference_ns):
    stamps = _evidence(info, snapshot, now)
    if previous_info is not None:
        if info['read_start_monotonic_ns'] < previous_info['read_end_monotonic_ns']:
            raise ValueError('GDK read time reversed')
        for source in SOURCES:
            old = previous_info['source_timestamp_ns'][source]
            if stamps[source] <= old:
                reason = 'frozen' if stamps[source] == old else 'reversed'
                raise ValueError(f'source_{reason}:{source}')
    error = (_decimal(snapshot.empirical_error_ns) +
             Fraction(max(0, now-snapshot.reference_mono_ns)*MAX_DRIFT_PPM, 1_000_000))
    denominator = 1-_decimal(snapshot.drift_ppm)/1_000_000
    intervals, ages = {}, {}
    for source in SOURCES:
        delta = stamps[source]-snapshot.wall_minus_mono_ns-snapshot.reference_mono_ns
        center = snapshot.reference_mono_ns+(delta+_decimal(snapshot.offset_at_reference_ns))/denominator
        radius = error/denominator
        lo, hi = center-radius, center+radius
        if lo > now:
            raise ValueError('source_future:'+source)
        intervals[source] = [math.floor(lo), math.ceil(hi)]
        ages[source] = [float((now-hi)/1_000_000), float((now-lo)/1_000_000)]
    left, right = intervals['left_wrist'], intervals['right_aux']
    metrics = dict(
        camera_age_ms={s: ages[s][1] for s in SOURCES[:2]},
        camera_age_lower_ms={s: ages[s][0] for s in SOURCES[:2]},
        state_age_ms={s: ages[s][1] for s in SOURCES[2:]},
        state_age_lower_ms={s: ages[s][0] for s in SOURCES[2:]},
        camera_skew_ms=max(left[1]-right[0], right[1]-left[0])/1e6,
        source_camera_skew_ms=abs(stamps['left_wrist']-stamps['right_aux'])/1e6,
        mapping_error_ms=float(error/1_000_000),
        mapping_residual_ms=snapshot.residual_ns/1e6,
        mapping_drift_ppm=snapshot.drift_ppm,
        mapping_path_delay_ms=snapshot.path_delay_ns/1e6,
        gdk_read_duration_ms=info['read_duration_s']*1000,
        inference_duration_ms=inference_ns/1e6,
        tf_position_error_m=info['tf_position_error_m'],
        tf_rotation_error_rad=info['tf_rotation_error_rad'])
    if previous_info is not None:
        metrics['gdk_read_gap_ms'] = (info['read_start_monotonic_ns']-previous_info['read_end_monotonic_ns'])/1e6
    if previous_snapshot is not None:
        metrics['snapshot_gap_ms'] = (snapshot.created_mono_ns-previous_snapshot.created_mono_ns)/1e6
    return metrics, intervals


def _distribution(values):
    values = sorted(values)
    def percentile(q):
        index = (len(values)-1)*q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo]+(values[hi]-values[lo])*(index-lo)
    return dict(min=values[0], p50=percentile(.5), p95=percentile(.95),
                p99=percentile(.99), max=values[-1])


def summarize_audit(rows):
    """Summarize validated metric rows using linear interpolated percentiles.

    Age metrics without ``lower`` use the worst-case upper age bound. Units
    are literal in every metric name; empty sessions have no distributions.
    """
    result = dict(motion_authorized=False, thresholds_approved=False,
                  source_clock_identity_proven=False, sample_count=len(rows),
                  percentile_method='linear interpolation', age_statistic='upper interval bound')
    for key in sorted({key for row in rows for key in row}):
        values = [row[key] for row in rows if key in row]
        if isinstance(values[0], dict):
            result[key] = {source: _distribution([v[source] for v in values]) for source in values[0]}
        else:
            result[key] = _distribution(values)
    return result


def run_audit(*, output, duration_s, socket_path=None, inference_delay_s=0.,
              client_factory=None, reader_factory=None, monotonic_ns=None, sleep=None):
    """Collect bounded evidence, closing owned read resources on every exit.

    Factories allow offline tests; returned resources belong to this run.
    Duration bounds the sampling loop, not an uninterruptible vendor call.
    No timeout thread releases SDK resources underneath a blocked SDK call.
    """
    validate_audit_duration(duration_s)
    _finite(inference_delay_s, 'inference_delay_s')
    if client_factory is None and socket_path is None:
        raise ValueError('An explicit monitor socket is required')
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(output)
    directory_fd = _create_session_directory(output)
    now_fn = time.monotonic_ns if monotonic_ns is None else monotonic_ns
    sleep_fn = time.sleep if sleep is None else sleep
    client = reader = stream = None
    rows, previous_info, previous_snapshot = [], None, None
    status, reason, failure = 'completed', '', None
    written = 0
    raw = {}

    def write_record(record):
        nonlocal written
        encoded = (json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n').encode()
        if len(encoded) > _MAX_ROW_BYTES or written+len(encoded) > _MAX_LOG_BYTES:
            raise ValueError('Audit evidence limit exceeded')
        stream.write(encoded)
        stream.flush()
        written += len(encoded)

    try:
        fd = os.open('evidence.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory_fd)
        stream = os.fdopen(fd, 'wb')
        write_record(dict(event='session', duration_s=duration_s,
                          inference_delay_s=inference_delay_s, motion_authorized=False,
                          thresholds_approved=False))
        client = (client_factory() if client_factory else SnapshotClient(
            Path(socket_path), timeout_s=.25, expected_master=EXPECTED_MASTER))
        previous_snapshot, _ = _mapping(client, now_fn, None)
        reader = (reader_factory or _reader)()
        start = _timestamp(now_fn(), 'audit start')
        deadline = start+duration_s*1_000_000_000
        while now_fn() < deadline:
            if len(rows) >= _MAX_SAMPLES:
                raise ValueError('Audit sample limit exceeded')
            raw = {}
            reader.observe()
            # Freeze metadata before another observe can replace it; no images.
            raw['info'] = deepcopy(reader.last_info)
            inference_start = now_fn()
            delay = min(inference_delay_s, max(0., (deadline-inference_start)/1e9))
            if delay:
                sleep_fn(delay)
            inference_end = now_fn()
            if inference_end < inference_start:
                raise ValueError('Inference monotonic time reversed')
            snapshot, now = _mapping(client, now_fn, previous_snapshot)
            raw['snapshot'] = asdict(snapshot)
            metrics, intervals = _measure(raw['info'], snapshot, now, previous_info,
                                          previous_snapshot, inference_end-inference_start)
            write_record(dict(event='sample', **raw, received_mono_ns=now,
                              inference_start_mono_ns=inference_start,
                              inference_end_mono_ns=inference_end,
                              source_intervals_ns=intervals, metrics=metrics))
            rows.append(metrics)
            previous_info, previous_snapshot = raw['info'], snapshot
            remaining = (deadline-now_fn())/1e9
            if remaining > 0:
                sleep_fn(min(_SAMPLE_PERIOD_S, remaining))
    except KeyboardInterrupt:
        status, reason = 'interrupted', 'operator_interrupt'
    except Exception as error:
        status, reason, failure = 'failed', f'{type(error).__name__}: {error}', error
        if stream is not None:
            try:
                write_record(dict(event='rejected', reason=reason[:1024], **raw))
            except (TypeError, ValueError, OSError):
                # Invalid/non-finite or oversized input cannot enter JSONL.
                pass
    finally:
        for resource in (reader, client):
            if resource is not None:
                try:
                    resource.close()
                except (Exception, KeyboardInterrupt) as error:
                    status, reason = 'failed', f'cleanup_failed: {type(error).__name__}: {error}'
                    if failure is None:
                        failure = RuntimeError(reason)
        try:
            result = summarize_audit(rows)
            result.update(status=status, reason=reason, duration_s=duration_s,
                          inference_delay_s=inference_delay_s)
            fd = os.open('summary.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory_fd)
            with os.fdopen(fd, 'w') as summary:
                json.dump(result, summary, allow_nan=False, indent=2)
                summary.write('\n')
        finally:
            if stream is not None:
                stream.close()
            os.close(directory_fd)
    if failure is not None:
        raise failure
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--seconds', type=int, required=True)
    parser.add_argument('--inference-delay-s', type=float, default=0.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path('runtime/freshness_audit')/uuid.uuid4().hex
    try:
        report = run_audit(output=output, duration_s=args.seconds, socket_path=args.socket,
                           inference_delay_s=args.inference_delay_s)
    except Exception as error:
        print(f'Audit failed: {error}; evidence directory: {output}', file=sys.stderr)
        return 1
    print(f'Audit {report["status"]}: {output}; motion_authorized=false; thresholds_approved=false')
    return 130 if report['status'] == 'interrupted' else 0


if __name__ == '__main__':
    raise SystemExit(main())
