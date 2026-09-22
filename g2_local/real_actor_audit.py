"""Single-command orchestration for the read-only RTX 3090 Actor audit.

This module only coordinates the existing ``clock_monitor`` and
``freshness_audit`` processes.  It never constructs a GDK command port, Gym
environment, or motion backend, and it never runs ``qualify`` automatically.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .clock_ipc import SnapshotClient
from .clock_monitor import EXPECTED_MASTER


def _validate_session(output, master, monitor_seconds, audit_seconds, checkpoint,
                      warmup_steps, startup_timeout_s):
    output = Path(output).absolute()
    if os.path.lexists(output):
        raise FileExistsError(f'Output session already exists: {output}')
    if master != EXPECTED_MASTER:
        raise ValueError('Expected master must be '+EXPECTED_MASTER)
    if type(monitor_seconds) is not int or monitor_seconds < 60:
        raise ValueError('monitor_seconds must be an integer >= 60')
    if type(audit_seconds) is not int or not 120 <= audit_seconds <= 1800:
        raise ValueError('audit_seconds must be an integer 120..1800')
    if monitor_seconds < audit_seconds + 30:
        raise ValueError('monitor_seconds must leave at least 30 seconds for clean shutdown')
    checkpoint = Path(checkpoint).absolute()
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise FileNotFoundError(f'Actor checkpoint is not a regular file: {checkpoint}')
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError('warmup_steps must be a nonnegative integer')
    if (type(startup_timeout_s) not in (int, float) or startup_timeout_s <= 0):
        raise ValueError('startup_timeout_s must be positive')
    return output, checkpoint


def _wait_for_healthy_monitor(process, socket_path, *, master, timeout_s,
                              client_factory, now_fn, sleep_fn):
    deadline = now_fn() + timeout_s
    client = None
    try:
        while now_fn() < deadline:
            code = process.poll()
            if code is not None:
                raise RuntimeError(f'clock monitor exited before healthy snapshot: {code}')
            if socket_path.exists():
                client = client_factory(path=socket_path, timeout_s=.25,
                                        expected_master=master)
                try:
                    snapshot = client.read()
                    if (getattr(snapshot, 'healthy', False) is True and
                            getattr(snapshot, 'reason', None) == 'ok'):
                        return
                except Exception:
                    pass
                finally:
                    client.close()
                    client = None
            sleep_fn(min(.25, max(0., deadline-now_fn())))
    finally:
        if client is not None:
            client.close()
    raise TimeoutError('clock monitor did not publish a healthy snapshot before timeout')


def _stop_monitor(process):
    if process is None or process.poll() is not None:
        return None if process is None else process.poll()
    process.send_signal(signal.SIGINT)
    try:
        return process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=5)


def _stop_audit(process):
    if process is None or process.poll() is not None:
        return None if process is None else process.poll()
    process.send_signal(signal.SIGINT)
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        return process.wait(timeout=5)


def run_session(*, output, master=EXPECTED_MASTER, monitor_seconds=300,
                audit_seconds=125, checkpoint, warmup_steps=10,
                startup_timeout_s=45., python=None, popen_factory=None,
                client_factory=None, now_fn=None, sleep_fn=None):
    """Run one safely ordered real-Actor read-only audit session.

    The returned status is the audit status when the monitor also exits cleanly;
    a nonzero monitor status wins only when the audit itself returned zero.
    """
    output, checkpoint = _validate_session(
        output, master, monitor_seconds, audit_seconds, checkpoint,
        warmup_steps, startup_timeout_s)
    python = sys.executable if python is None else python
    if type(python) is not str or not python:
        raise ValueError('python executable must be a nonempty string')
    popen = subprocess.Popen if popen_factory is None else popen_factory
    make_client = client_factory or (
        lambda **kwargs: SnapshotClient(kwargs['path'], timeout_s=kwargs['timeout_s'],
                                        expected_master=kwargs['expected_master']))
    now = time.monotonic if now_fn is None else now_fn
    sleep = time.sleep if sleep_fn is None else sleep_fn
    monitor_output = output/'monitor'
    audit_output = output/'audit'
    socket_path = monitor_output/'clock.sock'
    monitor_command = [python, '-m', 'g2_local.clock_monitor',
                       '--master', master, '--max-seconds', str(monitor_seconds),
                       '--output', str(monitor_output)]
    audit_command = [python, '-m', 'g2_local.freshness_audit',
                     '--socket', str(socket_path), '--seconds', str(audit_seconds),
                     '--warmup-steps', str(warmup_steps), '--device', 'cuda',
                     '--actor-checkpoint', str(checkpoint), '--output', str(audit_output)]
    monitor = audit = None
    audit_code = monitor_code = None
    try:
        monitor = popen(monitor_command, stdin=None, stdout=None, stderr=None, shell=False)
        _wait_for_healthy_monitor(
            monitor, socket_path, master=master, timeout_s=startup_timeout_s,
            client_factory=make_client, now_fn=now, sleep_fn=sleep)
        audit = popen(audit_command, stdin=None, stdout=None, stderr=None, shell=False)
        audit_code = audit.wait()
    except KeyboardInterrupt:
        audit_code = _stop_audit(audit)
        if audit_code in (None, 0):
            audit_code = 130
    finally:
        monitor_code = _stop_monitor(monitor)
    if audit_code is None:
        audit_code = 1
    if monitor_code not in (None, 0) and audit_code == 0:
        return 2
    return audit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', choices=[EXPECTED_MASTER], default=EXPECTED_MASTER)
    parser.add_argument('--monitor-seconds', type=int, default=300)
    parser.add_argument('--audit-seconds', type=int, default=125)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--actor-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--startup-timeout-s', type=float, default=45.)
    args = parser.parse_args(argv)
    try:
        code = run_session(
            output=args.output, master=args.master,
            monitor_seconds=args.monitor_seconds, audit_seconds=args.audit_seconds,
            checkpoint=args.actor_checkpoint, warmup_steps=args.warmup_steps,
            startup_timeout_s=args.startup_timeout_s)
    except (FileExistsError, FileNotFoundError, RuntimeError, TimeoutError, ValueError) as error:
        print(f'Read-only Actor audit launcher failed: {error}', file=sys.stderr)
        return 1
    print('Read-only Actor audit finished; qualify separately; '
          'motion_authorized=false; thresholds_approved=false')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
