"""Bounded, read-only transport for live clock snapshots.

This module only transports evidence.  It does not adjust clocks, inspect or
change GDK state, or authorize motion.
"""
from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import threading
import time

from .clock_mapping import (
    MAX_DRIFT_PPM,
    MAX_PATH_DELAY_NS,
    MAX_REPORT_INTEGER,
    MAX_RESIDUAL_NS,
    MIN_REPORT_INTEGER,
    PTP_LEASE_NS,
    UTC_OFFSET_S,
)
from .live_clock import ClockSnapshot


_REQUEST = {'op': 'snapshot', 'schema': 1}
_RESPONSE_LIMIT = 4096
_CONNECTION_TIMEOUT_S = 0.25
_SNAPSHOT_KEYS = frozenset(field.name for field in fields(ClockSnapshot))
_REASON_PATTERN = re.compile(r'[a-z][a-z0-9_]{0,127}\Z')


def _reject_constant(_token):
    raise ValueError('Non-finite JSON number')


def _object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _decode_json(raw):
    try:
        text = raw.decode('utf-8')
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError('Invalid JSON message') from error


def _encode_json(value):
    return (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Snapshot IPC deadline expired')
    return remaining


def _receive_bounded(connection, limit, deadline):
    message = bytearray()
    over_limit = False
    while True:
        connection.settimeout(_remaining(deadline))
        chunk = connection.recv(1024)
        if not chunk:
            if over_limit:
                raise ValueError('Snapshot IPC message exceeds its limit')
            return bytes(message)
        if over_limit or len(message) + len(chunk) > limit:
            over_limit = True
        else:
            message.extend(chunk)


def _send_bounded(connection, message, deadline):
    remaining = memoryview(message)
    while remaining:
        connection.settimeout(_remaining(deadline))
        sent = connection.send(remaining)
        if sent == 0:
            raise ConnectionError('Snapshot IPC connection closed during send')
        remaining = remaining[sent:]


def _read_boot_id():
    try:
        boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError as error:
        raise ValueError('Cannot read local boot identity') from error
    if not boot_id:
        raise ValueError('Local boot identity is empty')
    return boot_id


def _integer(value, *, minimum=MIN_REPORT_INTEGER, maximum=MAX_REPORT_INTEGER):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('Invalid snapshot integer')
    return value


def _number(value, *, minimum, maximum):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError('Invalid snapshot number')
    if not minimum <= value <= maximum:
        raise ValueError('Snapshot number is out of range')
    return value


def _text(value):
    if type(value) is not str or not value:
        raise ValueError('Invalid snapshot identity')
    return value


def _reason(value):
    if type(value) is not str or _REASON_PATTERN.fullmatch(value) is None:
        raise ValueError('Invalid snapshot reason')
    return value


def _snapshot_from_payload(payload, *, expected_master, previous_sequence,
                           previous_session, received_mono_ns):
    if type(payload) is not dict or frozenset(payload) != _SNAPSHOT_KEYS:
        raise ValueError('Unexpected snapshot fields')

    if _integer(payload['schema'], minimum=1, maximum=1) != 1:
        raise ValueError('Unsupported snapshot schema')
    sequence = _integer(payload['sequence'], minimum=1)
    if previous_sequence is not None and sequence <= previous_sequence:
        raise ValueError('Snapshot sequence did not advance')
    if type(payload['healthy']) is not bool:
        raise ValueError('Invalid snapshot health')
    reason = _reason(payload['reason'])
    if not payload['healthy']:
        raise ValueError(f'Clock snapshot is unhealthy: {reason}')
    if reason != 'ok':
        raise ValueError('Healthy snapshot has an invalid reason')

    boot_id = _text(payload['boot_id'])
    if boot_id != _read_boot_id():
        raise ValueError('Clock snapshot belongs to another boot')
    session_id = _text(payload['session_id'])
    if previous_session is not None and session_id != previous_session:
        raise ValueError('Clock snapshot session changed')
    if (_text(payload['expected_master']) != expected_master or
            _text(payload['actual_master']) != expected_master):
        raise ValueError('Clock master identity mismatch')
    if _text(payload['scale']) != 'raw_ptp':
        raise ValueError('Unsupported clock scale')

    if _integer(payload['utc_offset_s'], minimum=0, maximum=100) != UTC_OFFSET_S:
        raise ValueError('UTC correction mismatch')
    _integer(payload['utc_offset_valid'], minimum=0, maximum=1)
    if (_integer(payload['leap61'], minimum=0, maximum=1) != 0 or
            _integer(payload['leap59'], minimum=0, maximum=1) != 0 or
            _integer(payload['ptp_timescale'], minimum=0, maximum=1) != 1):
        raise ValueError('Unsupported PTP time properties')

    reference_ns = _integer(payload['reference_mono_ns'], minimum=1)
    _number(payload['offset_at_reference_ns'],
            minimum=-MAX_REPORT_INTEGER, maximum=MAX_REPORT_INTEGER)
    _number(payload['drift_ppm'], minimum=-MAX_DRIFT_PPM,
            maximum=MAX_DRIFT_PPM)
    _number(payload['residual_ns'], minimum=0, maximum=MAX_RESIDUAL_NS)
    _integer(payload['path_delay_ns'], minimum=0, maximum=MAX_PATH_DELAY_NS)
    _number(payload['empirical_error_ns'], minimum=0,
            maximum=MAX_REPORT_INTEGER)
    _integer(payload['wall_minus_mono_ns'])

    created_ns = _integer(payload['created_mono_ns'], minimum=1)
    last_sample_ns = _integer(payload['last_sample_mono_ns'], minimum=1)
    valid_until_ns = _integer(payload['valid_until_ns'], minimum=1)
    if (reference_ns != last_sample_ns or
            not last_sample_ns <= created_ns <= received_mono_ns or
            last_sample_ns > MAX_REPORT_INTEGER - PTP_LEASE_NS or
            valid_until_ns != last_sample_ns + PTP_LEASE_NS or
            received_mono_ns > valid_until_ns):
        raise ValueError('Clock snapshot lease is invalid or expired')

    try:
        return ClockSnapshot(**payload)
    except TypeError as error:
        raise ValueError('Invalid clock snapshot') from error


def _secure_parent(path):
    parent = path.parent
    try:
        parent.mkdir(mode=0o700)
    except FileExistsError:
        pass
    try:
        return os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as error:
        raise PermissionError('Snapshot socket parent must be a real private directory') from error


def _validate_directory_fd(fd, parent):
    metadata = os.fstat(fd)
    named = parent.lstat()
    if (not stat.S_ISDIR(metadata.st_mode) or not stat.S_ISDIR(named.st_mode) or
            metadata.st_uid != os.getuid() or
            stat.S_IMODE(metadata.st_mode) != 0o700 or
            (metadata.st_dev, metadata.st_ino) != (named.st_dev, named.st_ino)):
        raise PermissionError('Snapshot socket parent must be the same private directory')


class SnapshotServer:
    """Serve exactly one read-only snapshot request per Unix connection."""

    def __init__(self, path: Path, provider, *, request_limit=256, directory_fd=None):
        """Borrow a private directory descriptor to anchor filesystem writes.

        When supplied, directory_fd must name path.parent. The server owns a
        duplicate until close, so caller lifetime and pathname replacement
        cannot redirect bind/chmod/unlink to a different directory.
        """
        self.path = Path(path)
        if not callable(provider):
            raise TypeError('Snapshot provider must be callable')
        if (type(request_limit) is not int or
                not 1 <= request_limit <= _RESPONSE_LIMIT):
            raise ValueError('Invalid request limit')
        self._provider = provider
        self._request_limit = request_limit
        self._closed = threading.Event()
        self._close_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active_connection = None
        self._close_complete = False
        self._socket_identity = None
        self._listener = None
        self._thread = None
        self._directory_fd = None
        listener = None
        try:
            self._directory_fd = (_secure_parent(self.path) if directory_fd is None
                                  else os.dup(directory_fd))
            _validate_directory_fd(self._directory_fd, self.path.parent)
            try:
                os.stat(self.path.name, dir_fd=self._directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(self.path)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(f'/proc/self/fd/{self._directory_fd}/{self.path.name}')
            # Pin the socket inode before chmod. O_PATH opens a socket without
            # connecting, and O_NOFOLLOW prevents a substituted symlink.
            socket_fd = os.open(self.path.name, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=self._directory_fd)
            try:
                metadata = os.fstat(socket_fd)
                if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.getuid():
                    raise PermissionError('Snapshot socket was replaced')
                self._socket_identity = (metadata.st_dev, metadata.st_ino)
                _validate_directory_fd(self._directory_fd, self.path.parent)
                os.chmod(f'/proc/self/fd/{socket_fd}', 0o600)
            finally:
                os.close(socket_fd)
            listener.listen()
            listener.settimeout(0.05)
            self._listener = listener
            self._thread = threading.Thread(
                target=self._serve,
                name=f'clock-snapshot:{self.path.name}',
                daemon=True,
            )
            self._thread.start()
        except Exception:
            if listener is not None:
                listener.close()
            self._unlink_owned_socket()
            if self._directory_fd is not None:
                os.close(self._directory_fd)
                self._directory_fd = None
            raise

    def _unlink_owned_socket(self):
        if self._socket_identity is None or self._directory_fd is None:
            return
        try:
            metadata = os.stat(self.path.name, dir_fd=self._directory_fd, follow_symlinks=False)
            if ((metadata.st_dev, metadata.st_ino) == self._socket_identity and
                    stat.S_ISSOCK(metadata.st_mode)):
                os.unlink(self.path.name, dir_fd=self._directory_fd)
        except FileNotFoundError:
            pass

    def _read_request(self, connection):
        deadline = time.monotonic() + _CONNECTION_TIMEOUT_S
        message = _receive_bounded(connection, self._request_limit, deadline)
        if (len(message) > self._request_limit or not message.endswith(b'\n') or
                message.count(b'\n') != 1):
            raise ValueError('Invalid request framing')
        request = _decode_json(message[:-1])
        if (type(request) is not dict or set(request) != {'op', 'schema'} or
                type(request['op']) is not str or request['op'] != 'snapshot' or
                type(request['schema']) is not int or request['schema'] != 1):
            raise ValueError('Invalid request')

    def _reply(self, connection, response):
        encoded = _encode_json(response)
        if len(encoded) > _RESPONSE_LIMIT:
            encoded = _encode_json({
                'schema': 1,
                'ok': False,
                'error': 'snapshot_unavailable',
            })
        deadline = time.monotonic() + _CONNECTION_TIMEOUT_S
        _send_bounded(connection, encoded, deadline)

    def _handle(self, connection):
        try:
            self._read_request(connection)
        except (OSError, ValueError):
            try:
                self._reply(connection, {
                    'schema': 1,
                    'ok': False,
                    'error': 'invalid_request',
                })
            except OSError:
                pass
            return
        try:
            snapshot = self._provider()
            if type(snapshot) is not ClockSnapshot:
                raise ValueError('Provider returned a non-snapshot')
            response = asdict(snapshot)
            self._reply(connection, response)
        except Exception:
            try:
                self._reply(connection, {
                    'schema': 1,
                    'ok': False,
                    'error': 'snapshot_unavailable',
                })
            except OSError:
                pass

    def _serve(self):
        while not self._closed.is_set():
            try:
                connection, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self._active_lock:
                if self._closed.is_set():
                    connection.close()
                    break
                self._active_connection = connection
            try:
                with connection:
                    self._handle(connection)
            finally:
                with self._active_lock:
                    if self._active_connection is connection:
                        self._active_connection = None

    def is_serving(self):
        """Whether the worker and listening socket are still live.

        This checks service resources, not just the persistent socket inode.
        Owners must treat False as terminal for this server instance.
        """
        if (self._closed.is_set() or self._thread is None or
                not self._thread.is_alive() or self._listener is None):
            return False
        try:
            return self._listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
        except OSError:
            return False

    def close(self):
        with self._close_lock:
            if self._close_complete:
                return
            thread = self._thread
            if thread is threading.current_thread():
                raise RuntimeError('Snapshot server cannot close from its serving thread')
            self._closed.set()
            listener = self._listener
            if listener is not None:
                listener.close()
            with self._active_lock:
                connection = self._active_connection
                if connection is not None:
                    try:
                        connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    connection.close()
            if thread is not None:
                thread.join()
            self._unlink_owned_socket()
            if self._directory_fd is not None:
                os.close(self._directory_fd)
                self._directory_fd = None
            self._close_complete = True


class SnapshotClient:
    """Read and validate live snapshots without retaining stale fallback data."""

    def __init__(self, path: Path, *, timeout_s: float, expected_master: str):
        self.path = Path(path)
        if (type(timeout_s) not in (int, float) or isinstance(timeout_s, bool) or
                not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError('A finite positive socket timeout is required')
        if type(expected_master) is not str or not expected_master:
            raise ValueError('Expected master identity is required')
        self._timeout_s = float(timeout_s)
        self._expected_master = expected_master
        self._sequence = None
        self._session = None
        self._closed = False
        self._lock = threading.Lock()

    def _receive(self, connection, deadline):
        response = _receive_bounded(connection, _RESPONSE_LIMIT, deadline)
        if (len(response) > _RESPONSE_LIMIT or not response.endswith(b'\n') or
                response.count(b'\n') != 1):
            raise ValueError('Invalid snapshot response framing')
        return _decode_json(response[:-1])

    def read(self) -> ClockSnapshot:
        with self._lock:
            if self._closed:
                raise RuntimeError('Snapshot client is closed')
            deadline = time.monotonic() + self._timeout_s
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(_remaining(deadline))
                connection.connect(str(self.path))
                _send_bounded(connection, _encode_json(_REQUEST), deadline)
                connection.shutdown(socket.SHUT_WR)
                payload = self._receive(connection, deadline)
            received_mono_ns = time.monotonic_ns()
            if type(payload) is dict and payload.get('ok') is False:
                raise ValueError('Snapshot server rejected the request')
            snapshot = _snapshot_from_payload(
                payload,
                expected_master=self._expected_master,
                previous_sequence=self._sequence,
                previous_session=self._session,
                received_mono_ns=received_mono_ns,
            )
            self._sequence = snapshot.sequence
            self._session = snapshot.session_id
            return snapshot

    def close(self):
        with self._lock:
            self._closed = True
