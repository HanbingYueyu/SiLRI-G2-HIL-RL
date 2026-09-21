"""External process boundaries are replaced; no sudo/PTP/GDK is executed."""
import json
import os
from pathlib import Path
import pty
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


def timestamp_context():
    fields = Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()
    tty = int(fields[4])
    return ('tty', os.getsid(0), tty) if tty else ('ppid', os.getpid())


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
def credentials(monkeypatch):
    """Inject the sole interactive subprocess boundary without executing sudo."""
    state = {'calls': [], 'ticket_context': None, 'returncode': 0, 'inspect': None}

    def validate(argv, **kwargs):
        assert argv == ['/usr/bin/sudo', '-v']
        assert kwargs == dict(check=True, shell=False, stdin=None, stdout=None, stderr=None)
        state['calls'].append((argv, kwargs))
        if state['inspect'] is not None:
            state['inspect']()
        if state['returncode']:
            raise subprocess.CalledProcessError(state['returncode'], argv)
        state['ticket_context'] = timestamp_context()
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(monitor().subprocess, 'run', validate)
    return state


@pytest.fixture
def launch(monkeypatch, credentials):
    """Run harmless real pipes/processes instead of either external executable."""
    module = monitor()
    original = subprocess.Popen
    original_spawn = os.posix_spawn
    spawn_measurement = module._spawn_measurement
    children, calls = [], []
    scripts = {'ptp': 'import time; time.sleep(10)',
               'pmc': 'print("TIME_PROPERTIES_DATA_SET\\ncurrentUtcOffset 37\\n'
                      'currentUtcOffsetValid 0\\nleap61 0\\nleap59 0\\nptpTimescale 1")'}

    def start(argv, **kwargs):
        calls.append((argv, kwargs))
        assert kwargs.get('shell', False) is False
        assert argv[0] == '/usr/sbin/pmc'
        script = scripts['pmc']
        child = original([PYTHON, '-c', script, *argv], **kwargs)
        children.append(child)
        return child

    def spawn(executable, argv, env, **kwargs):
        assert executable == '/usr/bin/sudo'
        calls.append((argv, kwargs))
        context = ('ppid', os.getpid()) if kwargs.get('setsid') else timestamp_context()
        script = scripts['ptp'] if context == credentials['ticket_context'] else (
            'print("sudo: 需要密码"); raise SystemExit(1)')
        return original_spawn(PYTHON, [PYTHON, '-c', script, *argv], env, **kwargs)

    def measurement(argv):
        child = spawn_measurement(argv)
        children.append(child)
        return child

    monkeypatch.setattr(module.subprocess, 'Popen', start)
    monkeypatch.setattr(module.os, 'posix_spawn', spawn)
    monkeypatch.setattr(module, '_spawn_measurement', measurement)
    yield scripts, calls, children
    for child in children:
        if child.poll() is None:
            os.kill(child.pid, signal.SIGKILL)
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
    assert calls[0][1]['setpgroup'] == 0
    assert 'setsid' not in calls[0][1]
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
def test_existing_resources_rejected_before_sudo(tmp_path, launch, credentials, monkeypatch, existing):
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
    assert credentials['calls'] == []


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
    original_wait = monitor()._SpawnedProcess.wait
    monkeypatch.setattr(monitor(), '_FIRST_PROPERTIES_S', 0)
    launch[0]['pmc'] = 'import time; time.sleep(10)'

    def wait(child, timeout=None):
        if child is item.process and timeout == 4:
            raise subprocess.TimeoutExpired('owned sudo', 4)
        return original_wait(child, timeout=timeout)

    monkeypatch.setattr(monitor()._SpawnedProcess, 'wait', wait)
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
        monotonic=lambda: next(ticks, time.monotonic()), monotonic_ns=time.monotonic_ns,
        time_ns=time.time_ns, sleep=time.sleep))
    assert item.run() == 124
    assert item.provider().reason == 'monitor_deadline'
    assert launch[2][0].poll() is not None


def test_running_python_as_root_is_rejected_before_sudo(tmp_path, launch, credentials, monkeypatch, capsys):
    monkeypatch.setattr(monitor().os, 'geteuid', lambda: 0)
    item = runtime(tmp_path)
    assert item.run() == 2
    assert launch[1] == []
    assert credentials['calls'] == []
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
def test_unsafe_output_ancestor_is_rejected_without_creating_external_session(tmp_path, launch, credentials, ancestor):
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
    assert credentials['calls'] == []


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


def test_monitor_authenticates_itself_before_evidence_and_measurement_sudo(tmp_path, launch, credentials):
    item = runtime(tmp_path)
    launch[0]['ptp'] = 'raise SystemExit(0)'

    def before_authentication():
        assert not item.output.exists()
        assert launch[1] == []

    credentials['inspect'] = before_authentication
    # An external shell ticket intentionally does not set this Python
    # parent's timestamp context. Unauthenticated sudo -n rejects it in launch.
    assert item.run() == 0
    assert len(credentials['calls']) == 1
    assert launch[1][0][0][:2] == ['/usr/bin/sudo', '-n']
    assert launch[1][0][1]['setpgroup'] == 0
    assert (item.output/'evidence.jsonl').exists()


def test_failed_interactive_authentication_creates_no_output_or_ptp_child(tmp_path, launch, credentials):
    item = runtime(tmp_path)
    credentials['returncode'] = 1
    assert item.run() == 2
    assert len(credentials['calls']) == 1
    assert not item.output.exists()
    assert launch[1] == []


def test_path_created_during_authentication_is_not_overwritten(tmp_path, launch, credentials):
    item = runtime(tmp_path)
    credentials['inspect'] = lambda: item.output.mkdir()
    with pytest.raises(FileExistsError):
        item.run()
    assert len(credentials['calls']) == 1
    assert list(item.output.iterdir()) == []
    assert launch[1] == []


def test_measurement_preserves_authenticated_tty_but_owns_separate_process_group(tmp_path):
    # The harness becomes a session leader, acquires a real controlling PTY,
    # and substitutes only the sudo executable with harmless Python probes.
    # Kernel SID/tty_nr values decide ticket compatibility, not a boolean.
    script = r'''
import fcntl, json, os, subprocess, sys, termios
from pathlib import Path
from g2_local import clock_monitor as module
fcntl.ioctl(0, termios.TIOCSCTTY, 0)
original_popen, original_spawn = subprocess.Popen, os.posix_spawn
probe = "import json,os; f=open('/proc/self/stat').read().rsplit(') ',1)[1].split(); print(json.dumps(dict(pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgrp(),sid=os.getsid(0),tty=int(f[4]))),flush=True)"
authenticated = None
def validate(argv, **kwargs):
    global authenticated
    assert argv == ['/usr/bin/sudo','-v']
    child = original_popen([sys.executable,'-c',probe], stdout=subprocess.PIPE, text=True)
    authenticated = json.loads(child.communicate(timeout=1)[0])
    assert child.returncode == 0 and authenticated['tty'] != 0
    return subprocess.CompletedProcess(argv, 0)
def measurement_probe():
    return probe + "; " + "raise SystemExit(0 if (os.getsid(0),int(f[4]),os.getppid()) == " + repr((authenticated['sid'],authenticated['tty'],authenticated['ppid'])) + " else 1)"
def popen(argv, **kwargs):
    assert argv[0] == '/usr/bin/sudo'
    return original_popen([sys.executable,'-c',measurement_probe()], **kwargs)
def spawn(executable, argv, env, **kwargs):
    assert executable == '/usr/bin/sudo'
    return original_spawn(sys.executable,[sys.executable,'-c',measurement_probe()],env,**kwargs)
module.subprocess.run = validate
module.subprocess.Popen = popen
module.os.posix_spawn = spawn
item = module.MonitorRuntime(max_seconds=60, master='044052.fffe.000010', output=Path(sys.argv[1]))
code = item.run()
events = [json.loads(line) for line in (item.output/'evidence.jsonl').read_text().splitlines()]
measured = [json.loads(event['raw']) for event in events if event['kind']=='ptp'][0]
print(json.dumps(dict(code=code,authenticated=authenticated,measured=measured)),flush=True)
'''
    master_fd, slave_fd = pty.openpty()
    try:
        process = POPEN([PYTHON, '-c', script, str(tmp_path/'run')], stdin=slave_fd,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        start_new_session=True, text=True)
        try:
            stdout, stderr = process.communicate(timeout=3)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=1)
        assert process.returncode == 0, stderr
        evidence = json.loads(stdout.splitlines()[-1])
        auth, measured = evidence['authenticated'], evidence['measured']
        assert auth['tty'] != 0
        assert evidence['code'] == 0, evidence
        assert (measured['sid'], measured['tty'], measured['ppid']) == (auth['sid'], auth['tty'], auth['ppid'])
        assert measured['pgid'] == measured['pid']
        assert measured['pgid'] != auth['pgid']
    finally:
        os.close(slave_fd)
        os.close(master_fd)


def test_spawn_adapter_merges_output_and_reaps_nonzero_exit(monkeypatch):
    original = os.posix_spawn

    def harmless(executable, argv, env, **kwargs):
        assert executable == '/usr/bin/sudo'
        assert kwargs['setpgroup'] == 0
        assert 'setsid' not in kwargs
        return original(PYTHON, [PYTHON, '-c',
            'import sys; print("out"); print("err",file=sys.stderr); raise SystemExit(7)'],
            env, **kwargs)

    monkeypatch.setattr(monitor().os, 'posix_spawn', harmless)
    process = monitor()._spawn_measurement(['/usr/bin/sudo', '-n'])
    try:
        assert process.wait(timeout=1) == 7
        assert process.poll() == 7
        assert set(process.stdout.read().splitlines()) == {b'out', b'err'}
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)
    finally:
        if process.poll() is None:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=1)
        process.stdout.close()


def test_spawn_adapter_timeout_preserves_child_for_owned_group_cleanup(monkeypatch):
    original = os.posix_spawn
    monkeypatch.setattr(monitor().os, 'posix_spawn', lambda executable, argv, env, **kwargs:
        original(PYTHON, [PYTHON, '-c', 'import time; time.sleep(10)'], env, **kwargs))
    process = monitor()._spawn_measurement(['/usr/bin/sudo', '-n'])
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=.03)
        assert process.poll() is None
        assert os.getpgid(process.pid) == process.pid
        os.killpg(process.pid, signal.SIGINT)
        assert process.wait(timeout=1) == -signal.SIGINT
        assert process.poll() == -signal.SIGINT
    finally:
        if process.poll() is None:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=1)
        process.stdout.close()


def test_failed_spawn_closes_both_pipe_descriptors(monkeypatch):
    original_pipe = os.pipe2
    descriptors = []

    def pipe(flags):
        result = original_pipe(flags)
        descriptors.extend(result)
        return result

    def failed(*args, **kwargs):
        raise FileNotFoundError('measurement executable missing')

    monkeypatch.setattr(monitor().os, 'pipe2', pipe)
    monkeypatch.setattr(monitor().os, 'posix_spawn', failed)
    with pytest.raises(FileNotFoundError):
        monitor()._spawn_measurement(['/usr/bin/sudo', '-n'])
    assert len(descriptors) == 2
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_pipe_wrapper_failure_closes_descriptors_before_spawn(monkeypatch):
    original_pipe = os.pipe2
    descriptors = []

    def pipe(flags):
        result = original_pipe(flags)
        descriptors.extend(result)
        return result

    def failed(*args, **kwargs):
        raise OSError('cannot allocate pipe reader')

    monkeypatch.setattr(monitor().os, 'pipe2', pipe)
    monkeypatch.setattr(monitor().os, 'fdopen', failed)
    monkeypatch.setattr(monitor().os, 'posix_spawn',
                        lambda *args, **kwargs: pytest.fail('Spawn must not run without its pipe'))
    with pytest.raises(OSError, match='pipe reader'):
        monitor()._spawn_measurement(['/usr/bin/sudo', '-n'])
    try:
        for descriptor in descriptors:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
