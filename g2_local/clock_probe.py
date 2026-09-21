"""Concurrent PTP/GDK evidence, without adjusting clocks or commanding motion.

sudo is restricted to fixed linuxptp measurement commands, never Python/GDK.
The diagnostic mapping is retrospective and is NOT loaded by the motion code.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid
import numpy as np

from .clock_mapping import associate, fit_mapping, parse_ptp


def ptp_command(seconds, socket):
    if type(seconds) is not int or not 30 <= seconds <= 120:
        raise ValueError('PTP duration must be 30..120 seconds')
    if not re.fullmatch(r'/var/run/g2-clock-[a-zA-Z0-9-]+', socket):
        raise ValueError('Private measurement socket required')
    return ['sudo', '-n', '/usr/bin/timeout', '--signal=INT', '--kill-after=3s',
            f'{seconds}s', '/usr/bin/stdbuf', '-oL', '-eL', '/usr/sbin/ptp4l',
            '-i', 'enp3s0', '-2', '-E', '-S', '-s', '-m', '-q',
            '--free_running=1', '--utc_offset=37',
            f'--uds_address={socket}', f'--uds_ro_address={socket}-ro']


class PtpEvidence:
    def __init__(self, expected_master):
        self.expected = expected_master
        self.master = None
        self.samples = []
        self.errors = []

    def feed(self, line, received_ns):
        selected = re.search(r'selected best master clock (\S+)', line)
        if selected:
            self.master = selected[1]
            if self.master != self.expected:
                self.errors.append('PTP selected an unexpected master')
        if any(word in line for word in ('FAULTY', 'UNCALIBRATED to LISTENING',
                                          'SLAVE to LISTENING', 'clockcheck:', 'timed out')):
            self.errors.append('PTP fault/loss/jump: '+line)
        sample = parse_ptp(line, self.master)
        if 'master offset' in line and sample is None:
            self.errors.append('Malformed PTP offset report')
        if sample:
            if self.master != self.expected:
                self.errors.append('PTP measurement without expected master')
            elif not -1_000_000 <= received_ns-sample['mono_ns'] <= 500_000_000:
                self.errors.append('PTP log clock mismatch or delayed delivery')
            else:
                self.samples.append(sample)


def check_properties(raw):
    fields = {}
    for name in ('currentUtcOffset', 'currentUtcOffsetValid', 'leap61', 'leap59', 'ptpTimescale'):
        match = re.search(r'\b'+name+r'\s+(-?\d+)\b', raw)
        if not match:
            raise ValueError('Incomplete PTP TIME_PROPERTIES_DATA_SET')
        fields[name] = int(match[1])
    if (fields['currentUtcOffset'] != 37 or fields['ptpTimescale'] != 1 or
            fields['leap61'] != 0 or fields['leap59'] != 0 or
            fields['currentUtcOffsetValid'] not in (0, 1)):
        raise ValueError('PTP time scale/correction changed or leap announced')
    # Invalid UTC announcement is retained explicitly. linuxptp uses the
    # explicitly configured 37 s fallback; it is NOT UTC traceability proof.
    return fields


def transform_pose(transform):
    return [float(getattr(transform.translation, k)) for k in 'xyz'] + [
        float(getattr(transform.rotation, k)) for k in 'xyzw']


def query_tf_directions(tf):
    queries = []
    for target, source in (('base_link', 'arm_l_end_link'), ('arm_l_end_link', 'base_link')):
        transform, stamp = tf.lookup_transform_latest(target, source, True)
        queries.append(dict(target=target, source=source, timestamp_ns=int(stamp),
                            pose=transform_pose(transform)))
    return queries


def wait_for_tf(tf, *, timeout_s=10., retry_s=.05):
    """Boundedly wait for both TF directions before starting PTP evidence."""
    if not 0 < timeout_s <= 30 or not 0 < retry_s <= timeout_s:
        raise ValueError('Invalid TF preflight timeout')
    deadline = time.monotonic()+timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            return query_tf_directions(tf)
        except RuntimeError as exc:
            last_error = exc
            time.sleep(min(retry_s, max(0, deadline-time.monotonic())))
    raise TimeoutError('TF cache did not expose both arm_l_end_link directions') from last_error


def sample_gdk(reader, tf):
    start = time.monotonic_ns()
    start_wall = time.time_ns()
    stamps = {}
    for key, stream in reader.streams.items():
        stamps[key] = int(reader.camera.get_latest_image(stream, 100.).timestamp_ns)
    stamps['joint'] = int(reader.robot.get_joint_states()['timestamp'])
    queries = query_tf_directions(tf)
    # This installed SDK's reverse query matched motion status during the
    # preliminary check. Preserve BOTH; reject disagreement, never auto-swap.
    candidate = np.asarray(queries[1]['pose'])
    measured = reader.controller.read_end_effector_pose('arm_l_end_link')
    pose = np.asarray((*measured.position_m, *measured.orientation_xyzw), dtype=float)
    if (not np.isfinite(candidate).all() or not np.isfinite(pose).all() or
            abs(np.linalg.norm(candidate[3:])-1) > .01 or
            abs(np.linalg.norm(pose[3:])-1) > .01):
        raise ValueError('Invalid diagnostic TF/motion pose')
    position_error = float(np.linalg.norm(candidate[:3]-pose[:3]))
    dot = float(abs(np.dot(candidate[3:]/np.linalg.norm(candidate[3:]),
                           pose[3:]/np.linalg.norm(pose[3:]))))
    rotation_error = 2*math.acos(min(1., dot))
    stamps['tf'] = queries[1]['timestamp_ns']
    sdk_ns = int(reader.gdk.Clock.now_ns())
    wall = time.time_ns()
    mono = time.monotonic_ns()
    return dict(kind='gdk', start_mono_ns=start, start_wall_ns=start_wall,
                mono_ns=mono, wall_ns=wall, sdk_ns=sdk_ns, timestamps=stamps,
                tf_queries=queries, motion_pose=pose.tolist(),
                tf_position_error_m=position_error, tf_rotation_error_rad=rotation_error)


def summarize(events, *, master, session):
    ptp = PtpEvidence(master)
    rows, properties = [], []
    for event in events:
        if event['kind'] == 'ptp':
            ptp.feed(event['raw'], event['mono_ns'])
        elif event['kind'] == 'gdk':
            rows.append(event)
        elif event['kind'] == 'properties':
            if event['returncode'] != 0:
                raise ValueError('PTP properties query failed')
            properties.append((event['mono_ns'], check_properties(event['raw'])))
        elif event['kind'] == 'error':
            raise ValueError('Collection failed: '+event['error'])
    if ptp.errors:
        raise ValueError('; '.join(ptp.errors))
    if len(properties) < 2 or any(p != properties[0][1] for _, p in properties):
        raise ValueError('Need repeated, stable PTP time properties')
    mapping = fit_mapping(ptp.samples, master=master, utc_offset_s=37, session=session)
    if (properties[0][0]-mapping.start_ns > 12_000_000_000 or
            mapping.last_ns-properties[-1][0] > 12_000_000_000 or
            any(b[0]-a[0] > 12_000_000_000 for a, b in zip(properties, properties[1:]))):
        raise ValueError('PTP properties coverage gap')
    report = associate(mapping, rows)
    return dict(report, mapping=asdict(mapping), properties=properties[0][1],
                valid_for_live_use=False,
                note='Retrospective diagnostic only. Re-measure after expiry; no safety authorization.')


def collect(seconds, output, master):
    # No GDK import or resource allocation until sudo is available.
    subprocess.run(['sudo', '-n', 'true'], check=True, timeout=3)
    from .gdk_backend import GdkReader
    output.mkdir(parents=True, exist_ok=False)
    session = Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+uuid.uuid4().hex
    socket = '/var/run/g2-clock-'+uuid.uuid4().hex
    events, lock = [], threading.Lock()
    reader = tf = process = worker = None
    with (output/'evidence.jsonl').open('x') as log:
        def record(event):
            with lock:
                log.write(json.dumps(event, allow_nan=False)+'\n')
                log.flush()
                events.append(event)

        def pump():
            try:
                for line in process.stdout:
                    record(dict(kind='ptp', mono_ns=time.monotonic_ns(),
                                wall_ns=time.time_ns(), raw=line.rstrip()))
            except Exception as exc:
                record(dict(kind='error', error='PTP log reader: '+repr(exc)))

        record(dict(kind='session', session=session, expected_master=master,
                    command=ptp_command(seconds, socket), motion_authorized=False,
                    utc_correction_policy='explicit 37 s fallback; checked via pmc'))
        try:
            reader = GdkReader()
            tf = reader.gdk.TF()
            wait_for_tf(tf)
            process = subprocess.Popen(ptp_command(seconds, socket), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            worker = threading.Thread(target=pump, daemon=True)
            worker.start()
            start = time.monotonic()
            next_properties = start+8
            next_progress = start+5
            while time.monotonic()-start < seconds and process.poll() is None:
                record(sample_gdk(reader, tf))
                now = time.monotonic()
                if now >= next_properties:
                    reply = subprocess.run(['sudo', '-n', '/usr/sbin/pmc', '-u', '-b', '0',
                                            '-s', socket, 'GET TIME_PROPERTIES_DATA_SET'],
                                           capture_output=True, text=True, timeout=4)
                    record(dict(kind='properties', mono_ns=time.monotonic_ns(),
                                raw=reply.stdout+reply.stderr, returncode=reply.returncode))
                    next_properties = time.monotonic()+5
                if now >= next_progress:
                    print(f'只读采集 {int(now-start)}/{seconds}s：{output}', flush=True)
                    next_progress = now+5
                time.sleep(.2)
            code = process.wait(timeout=5)
            if code not in (0, 124):
                raise RuntimeError(f'PTP measurement exited {code}')
        except BaseException as exc:
            record(dict(kind='error', error=repr(exc)))
            print('采集中断；PTP 子进程由其固定 timeout 结束，不会校时。', flush=True)
            raise
        finally:
            # sudo-owned process has its own hard deadline. Do not use pkill
            # or touch unrelated linuxptp services. Drain its pipe before log close.
            if process is not None:
                process.wait(timeout=seconds+5)
            if worker is not None:
                worker.join(timeout=3)
            tf = None
            if reader is not None:
                reader.close()
    try:
        report = summarize(events, master=master, session=session)
    except (ValueError, KeyError, TypeError) as exc:
        report = dict(status='rejected', reason=str(exc), motion_authorized=False,
                      valid_for_live_use=False)
    with (output/'summary.json').open('x') as log:
        json.dump(report, log, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if report['status'] == 'diagnostic_consistent' else 2


def run_supervised(command, *, timeout_s, output):
    """Kill only our unprivileged worker if an SDK call hangs indefinitely."""
    if output.exists():
        raise FileExistsError(f'Clock evidence output already exists: {output}')
    def rejected(reason):
        output.mkdir(parents=True, exist_ok=True)
        with (output/'supervisor.json').open('x') as log:
            json.dump(dict(status='rejected', reason=reason,
                           motion_authorized=False, valid_for_live_use=False), log)
    try:
        code = subprocess.run(command, timeout=timeout_s).returncode
        if code:
            rejected(f'Acquisition worker exited with code {code}')
        return code
    except subprocess.TimeoutExpired:
        rejected('Acquisition worker timed out')
        print('采集进程超时，结果无效。PTP 测量子进程有独立的固定超时；不修改时钟。', flush=True)
        return 124


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=45)
    parser.add_argument('--master', required=True, help='Expected PTP grandmaster identity')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 30 <= args.seconds <= 120:
        parser.error('--seconds must be 30..120')
    if not re.fullmatch(r'[0-9a-f]{6}\.[0-9a-f]{4}\.[0-9a-f]{6}', args.master):
        parser.error('Use PTP identity format such as 044052.fffe.000010')
    output = args.output or Path('runtime/clock_probe')/uuid.uuid4().hex
    if args._worker:
        return collect(args.seconds, output, args.master)
    return run_supervised([sys.executable, '-m', 'g2_local.clock_probe', '--_worker',
                           '--seconds', str(args.seconds), '--master', args.master,
                           '--output', str(output)], timeout_s=args.seconds+20, output=output)


if __name__ == '__main__':
    raise SystemExit(main())
