"""External process boundaries are replaced; no sudo/PTP/GDK is executed."""
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest


MASTER = '044052.fffe.000010'
PYTHON = sys.executable
POPEN = subprocess.Popen


def monitor():
    from g2_local import clock_monitor
    return clock_monitor


def test_fixed_command_is_non_adjusting_get_only_and_finitely_bounded():
    cmd = monitor().ptp_monitor_command(3600, '/var/run/g2-live-a', '/var/run/g2-live-a-ro')
    assert cmd == ['/usr/bin/sudo', '-n', '/usr/bin/timeout', '--signal=INT',
                   '--kill-after=3s', '3600s', '/usr/bin/stdbuf', '-oL', '-eL',
                   '/usr/sbin/ptp4l', '-i', 'enp3s0', '-2', '-E', '-S', '-s', '-m', '-q',
                   '--free_running=1', '--utc_offset=37', '--uds_file_mode=0600',
                   '--uds_ro_file_mode=0666', '--uds_address=/var/run/g2-live-a',
                   '--uds_ro_address=/var/run/g2-live-a-ro']


@pytest.mark.parametrize('seconds', [True, 59, 43201, 60.0, '60'])
def test_invalid_duration_cannot_reach_process_launch(seconds):
    with pytest.raises(ValueError):
        monitor().ptp_monitor_command(seconds, '/var/run/g2-live-a', '/var/run/g2-live-a-ro')


@pytest.mark.parametrize('uds,ro', [('/tmp/a', '/tmp/a-ro'),
    ('/var/run/g2-live-a;id', '/var/run/g2-live-a;id-ro'),
    ('/var/run/g2-live-a', '/var/run/g2-live-b-ro'),
    ('/var/run/g2-live-'+'a'*100, '/var/run/g2-live-'+'a'*100+'-ro')])
def test_command_rejects_non_session_socket_paths(uds, ro):
    with pytest.raises(ValueError):
        monitor().ptp_monitor_command(60, uds, ro)


@pytest.fixture
def launch(monkeypatch):
    """Run harmless real pipes/processes instead of either external executable."""
    module = monitor()
    original = subprocess.Popen
    children, calls = [], []
    scripts = {'ptp': 'import time; time.sleep(10)',
               'pmc': 'print("TIME_PROPERTIES_DATA_SET\\ncurrentUtcOffset 37\\n'
                      'currentUtcOffsetValid 0\\nleap61 0\\nleap59 0\\nptpTimescale 1")'}

    def start(argv, **kwargs):
        calls.append((argv, kwargs))
        assert kwargs.get('shell', False) is False
        script = scripts['ptp' if argv[0] == '/usr/bin/sudo' else 'pmc']
        child = original([PYTHON, '-c', script, *argv], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(module.subprocess, 'Popen', start)
    yield scripts, calls, children
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3)
        if child.stdout:
            child.stdout.close()


def runtime(tmp_path):
    return monitor().MonitorRuntime(max_seconds=60, master=MASTER, output=tmp_path/'run')


def test_child_exit_publishes_unhealthy_and_never_restarts(tmp_path, launch):
    scripts, calls, _ = launch
    scripts['ptp'] = 'raise SystemExit(7)'
    item = runtime(tmp_path)
    assert item.run() == 7
    assert item.provider().healthy is False
    assert item.provider().reason == 'ptp_child_exit:7'
    assert len(calls) == 1
    assert calls[0][1]['start_new_session'] is True
    events = [json.loads(line) for line in (item.output/'evidence.jsonl').read_text().splitlines()]
    assert events[-1]['kind'] == 'exit'
    assert events[-1]['healthy'] is False


def test_stop_publishes_unhealthy_before_signalling_only_owned_group(tmp_path, launch, monkeypatch):
    item = runtime(tmp_path)
    seen = []
    killpg = os.killpg

    def signal_group(pgid, sig):
        seen.append((pgid, sig, item.provider().healthy, item.provider().reason))
        return killpg(pgid, sig)

    monkeypatch.setattr(monitor().os, 'killpg', signal_group)
    timer = threading.Timer(.15, item.request_stop)
    timer.start()
    try:
        assert item.run() == 0
    finally:
        timer.cancel()
        timer.join()
    assert seen == [(launch[2][0].pid, signal.SIGINT, False, 'stop_requested')]
    assert launch[2][0].poll() is not None
    assert not item.socket_path.exists()


@pytest.mark.parametrize('existing', ['output', 'socket', 'uds', 'uds_ro', 'symlink'])
def test_existing_resources_rejected_before_sudo(tmp_path, launch, monkeypatch, existing):
    item = runtime(tmp_path)
    if existing == 'output':
        item.output.mkdir()
    elif existing == 'socket':
        item.socket_path.parent.mkdir(parents=True)
        item.socket_path.touch()
    elif existing == 'symlink':
        item.output.symlink_to(tmp_path/'absent')
    else:
        target = getattr(item, existing)
        original = os.path.lexists
        monkeypatch.setattr(monitor().os.path, 'lexists', lambda p: str(p) == target or original(p))
    with pytest.raises(FileExistsError):
        item.run()
    assert launch[1] == []


@pytest.mark.parametrize('script,reason', [
    ('print("broken")', 'properties_invalid'),
    ('raise SystemExit(9)', 'properties_query_failed'),
    ('import time; time.sleep(10)', 'properties_query_timeout'),
    ('print("x"*20000)', 'properties_response_too_large'),
])
def test_properties_failures_stop_monitor_and_publish_unhealthy(tmp_path, launch, monkeypatch, script, reason):
    module = monitor()
    monkeypatch.setattr(module, '_FIRST_PROPERTIES_S', 0)
    monkeypatch.setattr(module, '_PMC_TIMEOUT_S', .15)
    launch[0]['pmc'] = script
    item = runtime(tmp_path)
    assert item.run() == 2
    assert item.provider().reason == reason
    assert item.provider().healthy is False
    argv = launch[1][1][0]
    assert argv == ['/usr/sbin/pmc', '-u', '-b', '0', '-s', item.uds_ro,
                    '-i', item._pmc_address, 'GET TIME_PROPERTIES_DATA_SET']
    assert item._pmc_address.startswith(f'/proc/{os.getpid()}/fd/')
    assert all(child.poll() is not None for child in launch[2])


@pytest.mark.parametrize('script,reason', [
    ('print("x"*5000)', 'ptp_line_too_large'),
    ('import sys; sys.stdout.buffer.write(b"\\xff\\n"); sys.stdout.flush(); import time; time.sleep(1)', 'monitor_error:UnicodeDecodeError'),
    ('print("ptp4l[1.000]: master offset nan s0 freq +0 path delay 10"); import time; time.sleep(1)', 'ptp_report_invalid'),
])
def test_corrupt_ptp_output_is_fail_closed(tmp_path, launch, script, reason):
    launch[0]['ptp'] = script
    item = runtime(tmp_path)
    assert item.run() == 2
    assert item.provider().reason == reason


def test_evidence_limit_ends_session_without_unbounded_log(tmp_path, launch, monkeypatch):
    monkeypatch.setattr(monitor(), '_MAX_LOG_BYTES', 2048)
    launch[0]['ptp'] = 'print("hello"*200); import time; time.sleep(1)'
    item = runtime(tmp_path)
    assert item.run() == 2
    assert item.provider().healthy is False
    assert (item.output/'evidence.jsonl').stat().st_size <= 2048


def test_cli_requires_explicit_bound_and_fixed_master(monkeypatch):
    for arguments in (['--master', MASTER], ['--master', '000000.0000.000000', '--max-seconds', '60']):
        monkeypatch.setattr(sys, 'argv', ['clock_monitor', *arguments])
        with pytest.raises(SystemExit) as exc:
            monitor().main()
        assert exc.value.code == 2


def test_probe_properties_uses_strict_shared_parser():
    from g2_local.clock_probe import check_properties
    raw = 'currentUtcOffset 37\ncurrentUtcOffsetValid 0\nleap61 0\nleap59 0\nptpTimescale 1\n'
    with pytest.raises(ValueError):
        check_properties(raw+'leap61 1\n')


def test_truncated_ptp_line_at_eof_is_rejected(tmp_path, launch):
    launch[0]['ptp'] = 'import sys; sys.stdout.write("ptp4l[1.0]: master off")'
    item = runtime(tmp_path)
    assert item.run() == 2
    assert item.provider().reason == 'ptp_output_truncated'


def test_log_failure_during_cleanup_still_reaps_pmc(tmp_path, launch, monkeypatch):
    item = runtime(tmp_path)
    # Simulate a privileged leader that cannot be reaped within the grace
    # period; PMC is still our ordinary child and must always be cleaned up.
    original_wait = POPEN.wait
    monkeypatch.setattr(monitor(), '_FIRST_PROPERTIES_S', 0)
    launch[0]['pmc'] = 'import time; time.sleep(10)'

    def wait(child, timeout=None):
        if child is item.process and timeout == 4:
            raise subprocess.TimeoutExpired('owned sudo', 4)
        return original_wait(child, timeout=timeout)

    monkeypatch.setattr(POPEN, 'wait', wait)
    record = item._record

    def record_failure(kind, **fields):
        if kind == 'cleanup_pending':
            raise OSError('disk full')
        return record(kind, **fields)

    monkeypatch.setattr(item, '_record', record_failure)
    timer = threading.Timer(.15, item.request_stop)
    timer.start()
    try:
        assert item.run() == 0
        assert launch[2][1].poll() is not None
    finally:
        timer.join()


def test_live_snapshot_and_successful_properties_then_sigterm(tmp_path, launch, monkeypatch):
    from g2_local.clock_ipc import SnapshotClient
    module = monitor()
    monkeypatch.setattr(module, '_FIRST_PROPERTIES_S', 0)
    item = runtime(tmp_path)
    now = time.monotonic_ns()
    origin = time.time_ns()-now
    item.window.feed_ptp(f'ptp4l[1.000]: selected best master clock {MASTER}', now, now+origin)
    raw = 'currentUtcOffset 37\ncurrentUtcOffsetValid 0\nleap61 0\nleap59 0\nptpTimescale 1\n'
    item.window.feed_properties(raw, now-14_000_000_000)
    for index in range(8):
        stamp = now-14_000_000_000+index*2_000_000_000
        seconds, fraction = divmod(stamp, 1_000_000_000)
        item.window.feed_ptp(f'ptp4l[{seconds}.{fraction:09d}]: master offset 55000000000 '
                            's0 freq +0 path delay 40000', stamp, stamp+origin)
    seen = []

    def inspect_and_stop():
        try:
            with_client = SnapshotClient(item.socket_path, timeout_s=.5, expected_master=MASTER)
            try:
                seen.append(with_client.read())
                seen.append(with_client.read())
            finally:
                with_client.close()
        except Exception as error:
            seen.append(error)
        finally:
            os.kill(os.getpid(), signal.SIGTERM)

    timer = threading.Timer(.2, inspect_and_stop)
    timer.start()
    try:
        assert item.run() == 0
    finally:
        timer.cancel()
        timer.join()
    assert len(seen) == 2 and all(s.healthy for s in seen)
    assert seen[1].sequence > seen[0].sequence
    assert seen[1].valid_until_ns == seen[0].valid_until_ns
    assert item.provider().reason == 'signal:15'
    events = [json.loads(line) for line in (item.output/'evidence.jsonl').read_text().splitlines()]
    assert any(e['kind'] == 'properties' and e['returncode'] == 0 for e in events)
    assert any(e['kind'] == 'mapping' and e['snapshot']['healthy'] for e in events)


@pytest.mark.parametrize('mutation', ['unlink', 'socket_permissions', 'parent_permissions'])
def test_invalid_ipc_socket_stops_the_session(tmp_path, launch, mutation):
    item = runtime(tmp_path)
    def invalidate():
        if mutation == 'unlink':
            item.socket_path.unlink()
        elif mutation == 'socket_permissions':
            item.socket_path.chmod(0o666)
        else:
            item.output.chmod(0o755)
    timer = threading.Timer(.15, invalidate)
    # A second timer is only a test deadline for an implementation that ignores
    # socket loss, keeping this failure bounded.
    deadline = threading.Timer(.4, item.request_stop)
    timer.start()
    deadline.start()
    try:
        assert item.run() == 2
        assert item.provider().reason == 'snapshot_socket_invalid'
    finally:
        timer.join()
        deadline.cancel()
        deadline.join()


def test_monitor_deadline_stops_owned_child_even_if_timeout_wrapper_misbehaves(tmp_path, launch, monkeypatch):
    item = runtime(tmp_path)
    ticks = iter([100., 164.])
    monkeypatch.setattr(monitor(), 'time', SimpleNamespace(
        monotonic=lambda: next(ticks), monotonic_ns=time.monotonic_ns, time_ns=time.time_ns))
    assert item.run() == 124
    assert item.provider().reason == 'monitor_deadline'
    assert launch[2][0].poll() is not None


def test_running_python_as_root_is_rejected_before_sudo(tmp_path, launch, monkeypatch, capsys):
    monkeypatch.setattr(monitor().os, 'geteuid', lambda: 0)
    item = runtime(tmp_path)
    assert item.run() == 2
    assert launch[1] == []
    assert not item.output.exists()
    assert 'ordinary operator' in capsys.readouterr().err


def test_exited_leader_revokes_snapshot_without_waiting_for_descendant_pipe(tmp_path, launch):
    launch[0]['ptp'] = ('import subprocess,sys; '
                        'subprocess.Popen([sys.executable,"-c","import time;time.sleep(1)"]); '
                        'raise SystemExit(7)')
    item = runtime(tmp_path)
    start = time.monotonic()
    assert item.run() == 7
    assert time.monotonic()-start < .5
    assert item.provider().reason == 'ptp_child_exit:7'


def test_signal_during_window_update_cannot_deadlock_with_ipc_provider(tmp_path):
    script = '''
import os, signal, sys, threading
from g2_local.clock_monitor import MonitorRuntime
item = MonitorRuntime(max_seconds=60, master='044052.fffe.000010', output=sys.argv[1])
acquired = threading.Event()
def ipc_read():
    with item._lock:
        acquired.set()
        item.provider()
item.window._lock.acquire()
worker = threading.Thread(target=ipc_read)
worker.start()
assert acquired.wait(1)
signal.signal(signal.SIGTERM, item.request_stop)
os.kill(os.getpid(), signal.SIGTERM)
assert item._failure is None, 'Signal handler must only defer shutdown'
item.window._lock.release()
worker.join(1)
assert not worker.is_alive()
print('signal returned without locks')
'''
    try:
        result = subprocess.run([PYTHON, '-c', script, str(tmp_path/'run')],
                                capture_output=True, text=True, timeout=2)
    except subprocess.TimeoutExpired:
        pytest.fail('Signal handler deadlocked while the IPC read waited for the window lock')
    assert result.returncode == 0, result.stderr
    assert 'signal returned without locks' in result.stdout


def test_listener_failure_with_intact_path_revokes_monitor(tmp_path, launch):
    item = runtime(tmp_path)
    seen = []
    def close_listener():
        item._server._listener.close()
        seen.append(item.socket_path.exists())
    timer = threading.Timer(.15, close_listener)
    deadline = threading.Timer(.4, item.request_stop)
    timer.start()
    deadline.start()
    try:
        assert item.run() == 2
        assert seen == [True]
        assert item.provider().healthy is False
        assert item.provider().reason == 'snapshot_server_unavailable'
        assert launch[2][0].poll() is not None
    finally:
        timer.cancel()
        timer.join()
        deadline.cancel()
        deadline.join()


def test_output_replaced_by_symlink_after_mkdir_cannot_modify_existing_target(tmp_path, launch, monkeypatch):
    item = runtime(tmp_path)
    victim = tmp_path/'existing'
    victim.mkdir(mode=0o755)
    original = os.mkdir
    path_mkdir = Path.mkdir

    def swap(actual):
        if actual == item.output and not (tmp_path/'detached').exists():
            actual.rename(tmp_path/'detached')
            actual.symlink_to(victim, target_is_directory=True)

    def replace_created(path, mode=0o777, *, dir_fd=None):
        original(path, mode, dir_fd=dir_fd)
        actual = Path(path) if dir_fd is None else Path(f'/proc/self/fd/{dir_fd}').resolve()/path
        swap(actual)

    def replace_created_path(path, *args, **kwargs):
        path_mkdir(path, *args, **kwargs)
        swap(path)

    monkeypatch.setattr(os, 'mkdir', replace_created)
    monkeypatch.setattr(Path, 'mkdir', replace_created_path)
    assert item.run() == 2
    assert stat.S_IMODE(victim.stat().st_mode) == 0o755
    assert list(victim.iterdir()) == []
    assert launch[1] == []


@pytest.mark.parametrize('ancestor', ['symlink', 'world_writable'])
def test_unsafe_output_ancestor_is_rejected_without_creating_external_session(tmp_path, launch, ancestor):
    launch[0]['ptp'] = 'raise SystemExit(7)'
    parent = tmp_path/'parent'
    if ancestor == 'symlink':
        victim = tmp_path/'existing'
        victim.mkdir(mode=0o755)
        parent.symlink_to(victim, target_is_directory=True)
    else:
        parent.mkdir()
        parent.chmod(0o777)
    item = monitor().MonitorRuntime(max_seconds=60, master=MASTER, output=parent/'run')
    assert item.run() == 2
    assert not item.output.exists()
    assert launch[1] == []


def test_pmc_anchored_datagram_address_can_receive_reply_from_another_process(tmp_path, launch, monkeypatch):
    item = runtime(tmp_path)
    remote = tmp_path/'remote.sock'
    raw = 'currentUtcOffset 37\ncurrentUtcOffsetValid 0\nleap61 0\nleap59 0\nptpTimescale 1\n'
    responder = POPEN([PYTHON, '-c',
        'import socket,sys; s=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM); '
        's.bind(sys.argv[1]); print("ready",flush=True); data,peer=s.recvfrom(256); '
        's.sendto(sys.argv[2].encode(),peer)', str(remote), raw],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert responder.stdout.readline() == 'ready\n'
        launch[0]['pmc'] = (
            'import socket,sys; s=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM); '
            's.settimeout(.2); s.bind(sys.argv[sys.argv.index("-i")+1]); '
            f's.sendto(b"GET TIME_PROPERTIES_DATA_SET",{str(remote)!r}); '
            'print(s.recv(8192).decode(),flush=True)')
        monkeypatch.setattr(monitor(), '_FIRST_PROPERTIES_S', 0)
        timer = threading.Timer(.4, item.request_stop)
        timer.start()
        try:
            assert item.run() == 0
        finally:
            timer.cancel()
            timer.join()
        assert responder.wait(timeout=.5) == 0
        events = [json.loads(line) for line in (item.output/'evidence.jsonl').read_text().splitlines()]
        assert any(e['kind'] == 'properties' and e['returncode'] == 0 for e in events)
    finally:
        if responder.poll() is None:
            responder.kill()
        responder.wait(timeout=1)
        responder.stdout.close()
        responder.stderr.close()
