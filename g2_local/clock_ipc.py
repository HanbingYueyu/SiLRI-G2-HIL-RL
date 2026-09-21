"""Bounded, read-only transport for live clock snapshots.

This module only transports evidence.  It does not adjust clocks, inspect or
change GDK state, or authorize motion.
"""
from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path
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


def _snapshot_from_payload(payload, *, expected_master, previous_sequence,
                           previous_session, received_mono_ns):
    if type(payload) is not dict or frozenset(payload) != _SNAPSHOT_KEYS:
        raise ValueError('Unexpected snapshot fields')

    if _integer(payload['schema'], minimum=1, maximum=1) != 1:
        raise ValueError('Unsupported snapshot schema')
    sequence = _integer(payload['sequence'], minimum=1)
    if previous_sequence is not None and sequence <= previous_sequence:
        raise ValueError('Snapshot sequence did not advance')
    if type(payload['healthy']) is not bool or not payload['healthy']:
        raise ValueError('Clock snapshot is unhealthy')
    if _text(payload['reason']) != 'ok':
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
        parent.chmod(0o700)
    except FileExistsError:
        metadata = parent.lstat()
        if (not stat.S_ISDIR(metadata.st_mode) or
                metadata.st_uid != os.getuid() or
                stat.S_IMODE(metadata.st_mode) != 0o700):
            raise PermissionError('Snapshot socket parent must be private')
    metadata = parent.lstat()
    if (not stat.S_ISDIR(metadata.st_mode) or
            metadata.st_uid != os.getuid() or
            stat.S_IMODE(metadata.st_mode) != 0o700):
        raise PermissionError('Snapshot socket parent must be private')


class SnapshotServer:
    """Serve exactly one read-only snapshot request per Unix connection."""

    def __init__(self, path: Path, provider, *, request_limit=256):
        self.path = Path(path)
        if not callable(provider):
            raise TypeError('Snapshot provider must be callable')
        if (type(request_limit) is not int or
                not 1 <= request_limit <= _RESPONSE_LIMIT):
            raise ValueError('Invalid request limit')
        self._provider = provider
        self._request_limit = request_limit
        self._closed = threading.Event()
        self._socket_identity = None
        self._listener = None
        self._thread = None

        _secure_parent(self.path)
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(self.path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.path))
            self.path.chmod(0o600)
            metadata = self.path.lstat()
            self._socket_identity = (metadata.st_dev, metadata.st_ino)
            listener.listen()
            listener.settimeout(0.05)
        except Exception:
            listener.close()
            self._unlink_owned_socket()
            raise
        self._listener = listener
        self._thread = threading.Thread(
            target=self._serve,
            name=f'clock-snapshot:{self.path.name}',
            daemon=True,
        )
        self._thread.start()

    def _unlink_owned_socket(self):
        if self._socket_identity is None:
            return
        try:
            metadata = self.path.lstat()
            if ((metadata.st_dev, metadata.st_ino) == self._socket_identity and
                    stat.S_ISSOCK(metadata.st_mode)):
                self.path.unlink()
        except FileNotFoundError:
            pass

    def _read_request(self, connection):
        message = bytearray()
        while len(message) <= self._request_limit:
            chunk = connection.recv(min(256, self._request_limit + 1 - len(message)))
            if not chunk:
                break
            message.extend(chunk)
            if b'\n' in chunk:
                break
        if (len(message) > self._request_limit or not message.endswith(b'\n') or
                message.count(b'\n') != 1):
            raise ValueError('Invalid request framing')
        request = _decode_json(bytes(message[:-1]))
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
        connection.sendall(encoded)

    def _handle(self, connection):
        connection.settimeout(_CONNECTION_TIMEOUT_S)
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
            with connection:
                self._handle(connection)

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        listener = self._listener
        if listener is not None:
            listener.close()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)
        self._unlink_owned_socket()


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

    def _receive(self, connection):
        response = bytearray()
        while len(response) <= _RESPONSE_LIMIT:
            chunk = connection.recv(min(1024, _RESPONSE_LIMIT + 1 - len(response)))
            if not chunk:
                break
            response.extend(chunk)
        if (len(response) > _RESPONSE_LIMIT or not response.endswith(b'\n') or
                response.count(b'\n') != 1):
            raise ValueError('Invalid snapshot response framing')
        return _decode_json(bytes(response[:-1]))

    def read(self) -> ClockSnapshot:
        with self._lock:
            if self._closed:
                raise RuntimeError('Snapshot client is closed')
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self._timeout_s)
                connection.connect(str(self.path))
                connection.sendall(_encode_json(_REQUEST))
                connection.shutdown(socket.SHUT_WR)
                payload = self._receive(connection)
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
