"""Operator-owned foreground PTP measurement; never adjusts a clock or robot.

Sudo is limited to interactive credential validation and the fixed,
time-bounded ptp4l command. IPC and PMC run as the operator; neither accepts
a command from an Actor.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid

from .clock_ipc import SnapshotServer
from .live_clock import ClockWindow


EXPECTED_MASTER = '044052.fffe.000010'
_FIRST_PROPERTIES_S = 8
_PROPERTIES_INTERVAL_S = 5
_PMC_TIMEOUT_S = 3
_MAX_LINE_BYTES = 4096
_MAX_PROPERTIES_BYTES = 8192
_MAX_LOG_BYTES = 64 * 1024 * 1024


def _create_session_directory(path, *, create=True):
    """Validate existing ancestors read-only, or create under pinned parents."""
    if '..' in path.parts:
        raise ValueError('Session output cannot contain parent traversal')
    parts = path.absolute().parts
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    current = os.open(parts[0], flags)
    try:
        for part in parts[1:-1]:
            try:
                child = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    return None
                os.mkdir(part, 0o700, dir_fd=current)
                child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
            metadata = os.fstat(current)
            root_sticky = metadata.st_uid == 0 and metadata.st_mode & stat.S_ISVTX
            if (metadata.st_uid not in (0, os.getuid()) or
                    (metadata.st_mode & 0o002 and not root_sticky) or
                    (metadata.st_mode & 0o020 and metadata.st_gid != os.getgid() and
                     not root_sticky)):
                raise PermissionError('Session output ancestor is not trusted')
        if not create:
            return None
        os.mkdir(parts[-1], 0o700, dir_fd=current)
        created = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        if not stat.S_ISDIR(created.st_mode):
            raise PermissionError('New session directory was replaced')
        result = os.open(parts[-1], flags, dir_fd=current)
        opened = os.fstat(result)
        if (opened.st_uid != os.getuid() or stat.S_IMODE(opened.st_mode) != 0o700 or
                (created.st_dev, created.st_ino) != (opened.st_dev, opened.st_ino)):
            os.close(result)
            raise PermissionError('New session directory was replaced or is not private')
        return result
    finally:
        os.close(current)


def ptp_monitor_command(max_seconds, uds, uds_ro) -> list[str]:
    if type(max_seconds) is not int or not 60 <= max_seconds <= 43200:
        raise ValueError('PTP duration must be 60..43200 seconds')
    if (type(uds) is not str or
            not re.fullmatch(r'/var/run/g2-live-[a-zA-Z0-9-]{1,48}', uds) or
            uds_ro != uds+'-ro'):
        raise ValueError('Session-specific PTP UDS pair required')
    return ['/usr/bin/sudo', '-n', '/usr/bin/timeout', '--signal=INT',
            '--kill-after=3s', f'{max_seconds}s', '/usr/bin/stdbuf', '-oL', '-eL',
            '/usr/sbin/ptp4l', '-i', 'enp3s0', '-2', '-E', '-S', '-s', '-m', '-q',
            '--free_running=1', '--utc_offset=37', '--uds_file_mode=0600',
            '--uds_ro_file_mode=0666', f'--uds_address={uds}', f'--uds_ro_address={uds_ro}']


class _SpawnedProcess:
    """Own one posix_spawn child and reap only its exact PID."""

    def __init__(self, command, pid, stdout):
        self.args = command
        self.pid = pid
        self.stdout = stdout
        self.returncode = None

    def poll(self):
        if self.returncode is None:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode

    def wait(self, timeout=None):
        if timeout is None:
            if self.returncode is None:
                _, status = os.waitpid(self.pid, 0)
                self.returncode = os.waitstatus_to_exitcode(status)
            return self.returncode
        deadline = time.monotonic()+timeout
        while self.poll() is None:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(min(.01, remaining))
        return self.returncode


def _spawn_measurement(command):
    """Create an owned process group without losing the authenticated TTY.

    Python 3.10 Popen lacks process_group. posix_spawn sets the group in libc,
    avoiding a Python preexec_fn after the IPC thread has started. Unlike
    setsid, setpgroup=0 retains the session/controlling TTY timestamp context.
    """
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    stdout = None
    try:
        stdout = os.fdopen(read_fd, 'rb')
        actions = [
            (os.POSIX_SPAWN_OPEN, 0, '/dev/null', os.O_RDONLY, 0),
            (os.POSIX_SPAWN_DUP2, write_fd, 1),
            (os.POSIX_SPAWN_DUP2, write_fd, 2),
            (os.POSIX_SPAWN_CLOSE, read_fd),
            (os.POSIX_SPAWN_CLOSE, write_fd),
        ]
        pid = os.posix_spawn(command[0], command, os.environ,
                             file_actions=actions, setpgroup=0,
                             setsigdef=(signal.SIGPIPE, signal.SIGXFSZ))
    except BaseException:
        if stdout is None:
            os.close(read_fd)
        else:
            stdout.close()
        raise
    finally:
        os.close(write_fd)
    return _SpawnedProcess(command, pid, stdout)


class MonitorRuntime:
    """One finite measurement session, with no restart or motion capability."""

    def __init__(self, *, max_seconds, master, output):
        if master != EXPECTED_MASTER:
            raise ValueError('Expected master must be '+EXPECTED_MASTER)
        self.session_id = uuid.uuid4().hex
        self.uds = '/var/run/g2-live-'+self.session_id
        self.uds_ro = self.uds+'-ro'
        self.command = ptp_monitor_command(max_seconds, self.uds, self.uds_ro)
        self.max_seconds = max_seconds
        self.output = Path(output)
        self.socket_path = self.output/'clock.sock'
        self._pmc_path = self.output/'pmc.sock'
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.window = ClockWindow(master, boot, self.session_id)
        self._lock = threading.RLock()
        self._failure = None
        self._stop_signal = None
        self._used = False
        self.process = self._pmc = self._server = self._log = None
        self._log_bytes = 0
        self._last_state = None
        self._socket_identity = None
        self._directory_fd = None
        self._pmc_address = None

    def provider(self):
        with self._lock:
            snapshot = self.window.snapshot(time.monotonic_ns(), time.time_ns())
            if self._failure is not None:
                snapshot = replace(snapshot, healthy=False, reason=self._failure)
            return snapshot

    def _fail(self, reason):
        with self._lock:
            if self._failure is None:
                self._failure = reason
            return self.provider()

    def request_stop(self, signum=None, _frame=None):
        # Python may invoke this while the main thread holds the window lock
        # and the IPC thread holds the runtime lock. Even Event.set takes a
        # lock: the handler must only store a scalar for the main loop.
        self._stop_signal = 0 if signum is None else signum

    def _apply_stop(self):
        signum = self._stop_signal
        if signum is None:
            return False
        self._fail('stop_requested' if signum == 0 else f'signal:{signum}')
        return True

    def _record(self, kind, **fields):
        data = (json.dumps(dict(kind=kind, mono_ns=time.monotonic_ns(), **fields),
                           allow_nan=False, ensure_ascii=True)+'\n').encode()
        if self._log_bytes + len(data) > _MAX_LOG_BYTES:
            raise OSError('Clock evidence size limit reached')
        self._log.write(data)
        self._log.flush()
        self._log_bytes += len(data)

    def _state(self):
        snapshot = self.provider()
        state = (snapshot.healthy, snapshot.reason, snapshot.last_sample_mono_ns)
        if state != self._last_state:
            self._record('mapping', snapshot=asdict(snapshot))
            self._last_state = state
        # An expired lease is already unhealthy at the IPC boundary, so every
        # consumer rejects it immediately.  Keep the measurement process alive
        # to allow a later clean PTP report to publish a new lease; latching the
        # runtime here would close the socket before the next 2 s PTP report.
        # All structural faults remain terminal.
        if snapshot.reason not in ('ok', 'warming_up', 'lease_expired'):
            self._fail(snapshot.reason)
        return snapshot

    def _preflight(self):
        if os.geteuid() == 0:
            raise PermissionError('Run the Python monitor as the ordinary operator, not root')
        # Reject every pre-existing resource before any sudo invocation.
        for path in (self.output, self.socket_path, self._pmc_path, self.uds, self.uds_ro):
            if os.path.lexists(path):
                raise FileExistsError(f'Clock session resource already exists: {path}')
        _create_session_directory(self.output, create=False)

    def _prepare(self):
        # Authentication may wait for operator input. Check paths again before
        # creating evidence in case anything changed during that interval.
        self._preflight()
        self._directory_fd = _create_session_directory(self.output)
        if not self._directory_valid():
            raise PermissionError('Session output pathname was replaced')
        log_fd = os.open('evidence.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=self._directory_fd)
        self._log = os.fdopen(log_fd, 'wb')
        self._server = SnapshotServer(self.socket_path, self.provider,
                                      directory_fd=self._directory_fd)
        metadata = os.stat('clock.sock', dir_fd=self._directory_fd, follow_symlinks=False)
        self._socket_identity = (metadata.st_dev, metadata.st_ino)
        # PMC uses datagrams: the peer resolves the return address in its own
        # process, so /proc/self would refer to ptp4l instead of this owner.
        self._pmc_address = f'/proc/{os.getpid()}/fd/{self._directory_fd}/pmc.sock'

    def _directory_valid(self):
        try:
            if self._directory_fd is None:
                return False
            opened = os.fstat(self._directory_fd)
            named = self.output.lstat()
            return (stat.S_ISDIR(named.st_mode) and opened.st_uid == os.getuid() and
                    stat.S_IMODE(opened.st_mode) == 0o700 and
                    (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino))
        except OSError:
            return False

    def _socket_valid(self):
        try:
            if not self._directory_valid():
                return False
            metadata = os.stat('clock.sock', dir_fd=self._directory_fd, follow_symlinks=False)
            return (stat.S_ISSOCK(metadata.st_mode) and
                    (metadata.st_dev, metadata.st_ino) == self._socket_identity and
                    metadata.st_uid == os.getuid() and
                    stat.S_IMODE(metadata.st_mode) == 0o600)
        except OSError:
            return False

    def _start_pmc(self, selector):
        command = ['/usr/sbin/pmc', '-u', '-b', '0', '-s', self.uds_ro,
                   '-i', self._pmc_address, 'GET TIME_PROPERTIES_DATA_SET']
        self._pmc = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=True, shell=False)
        selector.register(self._pmc.stdout, selectors.EVENT_READ, 'pmc')

    def _loop(self, selector):
        started = time.monotonic()
        next_properties = started + _FIRST_PROPERTIES_S
        pmc_deadline = None
        ptp_buffer, pmc_buffer = b'', b''
        pmc_eof = False
        while True:
            if self._apply_stop():
                return 0
            if not self._server.is_serving():
                self._fail('snapshot_server_unavailable')
                return 2
            if not self._socket_valid():
                self._fail('snapshot_socket_invalid')
                return 2
            now = time.monotonic()
            if now >= started + self.max_seconds + 3:
                self._fail('monitor_deadline')
                return 124
            if self._pmc is None and now >= next_properties:
                self._start_pmc(selector)
                pmc_deadline = now + _PMC_TIMEOUT_S
                pmc_buffer, pmc_eof = b'', False
            for key, _ in selector.select(.05):
                chunk = os.read(key.fileobj.fileno(), _MAX_LINE_BYTES)
                if not chunk:
                    selector.unregister(key.fileobj)
                    if key.data == 'pmc':
                        pmc_eof = True
                    else:
                        if ptp_buffer:
                            self._fail('ptp_output_truncated')
                        try:
                            self.process.wait(timeout=.05)
                        except subprocess.TimeoutExpired:
                            self._fail('ptp_output_closed')
                    continue
                if key.data == 'pmc':
                    pmc_buffer += chunk
                    if len(pmc_buffer) > _MAX_PROPERTIES_BYTES:
                        self._fail('properties_response_too_large')
                else:
                    ptp_buffer += chunk
                    while b'\n' in ptp_buffer:
                        line, ptp_buffer = ptp_buffer.split(b'\n', 1)
                        if len(line) > _MAX_LINE_BYTES:
                            self._fail('ptp_line_too_large')
                            break
                        raw = line.decode('utf-8')
                        mono, wall = time.monotonic_ns(), time.time_ns()
                        self._record('ptp', raw=raw, received_mono_ns=mono, wall_ns=wall)
                        self.window.feed_ptp(raw, mono, wall)
                    if len(ptp_buffer) > _MAX_LINE_BYTES:
                        self._fail('ptp_line_too_large')
            if self._apply_stop():
                return 0
            if self._failure is not None:
                return 2
            # An inherited pipe must not hide the owned leader's exit.
            code = self.process.poll()
            if code is not None:
                if ptp_buffer:
                    self._fail('ptp_line_too_large' if len(ptp_buffer) >= _MAX_LINE_BYTES
                               else 'ptp_output_truncated')
                    return 2
                self._fail(f'ptp_child_exit:{code}')
                return code
            if self._pmc is not None:
                code = self._pmc.poll()
                if code is not None and pmc_eof:
                    raw = pmc_buffer.decode('utf-8')
                    self._record('properties', raw=raw, returncode=code)
                    if code:
                        self._fail('properties_query_failed')
                    else:
                        self.window.feed_properties(raw, time.monotonic_ns())
                    self._pmc.stdout.close()
                    self._pmc = None
                    next_properties = time.monotonic()+_PROPERTIES_INTERVAL_S
                elif time.monotonic() >= pmc_deadline:
                    self._fail('properties_query_timeout')
            self._state()
            if self._failure is not None:
                return 2

    def _stop_children(self):
        # Child leaders remain unreaped until poll/wait, so their PID cannot be
        # reused here. posix_spawn(setpgroup=0) makes sudo PID our owned PGID.
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                # The root-owned timeout is the final cleanup boundary. Do not
                # launch another privileged command or kill unrelated services.
                try:
                    self._record('cleanup_pending', pgid=self.process.pid,
                                 reason='Fixed privileged timeout remains the final boundary')
                except OSError:
                    pass
        if self._pmc is not None:
            if self._pmc.poll() is None:
                self._pmc.terminate()
                try:
                    self._pmc.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    self._pmc.kill()
                    self._pmc.wait(timeout=.5)
            self._pmc.stdout.close()
        if self.process is not None:
            self.process.stdout.close()

    def run(self) -> int:
        if self._used:
            raise RuntimeError('A clock monitor session cannot be restarted')
        self._used = True
        handlers = {}
        code = 2
        try:
            self._preflight()
            # Validation and measurement share this parent AND its controlling
            # TTY/session. A separate group must not turn a TTY ticket into a
            # no-TTY PPID ticket by using setsid/start_new_session.
            # Inherit the terminal; never read, pipe, capture, or save passwords.
            subprocess.run(['/usr/bin/sudo', '-v'], check=True, shell=False,
                           stdin=None, stdout=None, stderr=None)
            self._prepare()
            self._record('session', session_id=self.session_id,
                         expected_master=EXPECTED_MASTER, command=self.command,
                         motion_authorized=False, socket=str(self.socket_path))
            if threading.current_thread() is threading.main_thread():
                for sig in (signal.SIGINT, signal.SIGTERM):
                    handlers[sig] = signal.signal(sig, self.request_stop)
            if self._apply_stop():
                code = 0
                return code
            self.process = _spawn_measurement(self.command)
            print(f'Clock snapshot socket: {self.socket_path}', flush=True)
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ, 'ptp')
                code = self._loop(selector)
            return code
        except FileExistsError:
            self._fail('resource_exists')
            raise
        except Exception as error:
            self._fail('monitor_error:'+type(error).__name__)
            print(f'Clock monitor failed: {error}', file=sys.stderr, flush=True)
            return 2
        finally:
            # Publish failure before signalling any process or closing IPC.
            self._apply_stop()
            snapshot = self._fail('monitor_shutdown')
            try:
                if self._log is not None:
                    try:
                        self._record('exit', reason=snapshot.reason, healthy=False, returncode=code)
                    except OSError:
                        pass
                self._stop_children()
            finally:
                if self._server is not None:
                    self._server.close()
                if self._log is not None:
                    self._log.close()
                if self._directory_fd is not None:
                    os.close(self._directory_fd)
                    self._directory_fd = None
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', required=True, choices=[EXPECTED_MASTER])
    parser.add_argument('--max-seconds', required=True, type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 60 <= args.max_seconds <= 43200:
        parser.error('--max-seconds must be 60..43200')
    output = args.output or Path('runtime/clock_monitor')/uuid.uuid4().hex
    return MonitorRuntime(max_seconds=args.max_seconds, master=args.master, output=output).run()


if __name__ == '__main__':
    raise SystemExit(main())
