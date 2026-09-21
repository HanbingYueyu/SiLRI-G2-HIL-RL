from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import socket
import stat
import threading
import time

import pytest

from g2_local.clock_ipc import SnapshotClient, SnapshotServer
from g2_local.live_clock import ClockSnapshot


MASTER = '044052.fffe.000010'


def local_boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def healthy_snapshot(*, sequence=1, **changes):
    now_ns = time.monotonic_ns()
    last_sample_ns = now_ns - 100_000_000
    values = dict(
        schema=1,
        sequence=sequence,
        healthy=True,
        reason='ok',
        boot_id=local_boot_id(),
        session_id='test-session',
        expected_master=MASTER,
        actual_master=MASTER,
        scale='raw_ptp',
        utc_offset_s=37,
        utc_offset_valid=1,
        leap61=0,
        leap59=0,
        ptp_timescale=1,
        reference_mono_ns=last_sample_ns,
        offset_at_reference_ns=18_000_000_000.0,
        drift_ppm=10.0,
        residual_ns=100.0,
        path_delay_ns=40_000,
        empirical_error_ns=2_040_100.0,
        wall_minus_mono_ns=1_700_000_000_000_000_000,
        created_mono_ns=now_ns,
        last_sample_mono_ns=last_sample_ns,
        valid_until_ns=last_sample_ns + 2_500_000_000,
    )
    values.update(changes)
    return ClockSnapshot(**values)


class SnapshotSource:
    def __init__(self, snapshot):
        self._snapshot = snapshot
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self._snapshot

    def set(self, snapshot):
        with self._lock:
            self._snapshot = snapshot


@pytest.fixture
def server_factory():
    servers = []

    def start(path, snapshot=None):
        source = SnapshotSource(snapshot or healthy_snapshot())
        server = SnapshotServer(path, source)
        server.source = source
        servers.append(server)
        return server

    yield start
    for server in reversed(servers):
        server.close()


def exchange(path, request):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(.5)
        connection.connect(str(path))
        connection.sendall(request)
        connection.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = connection.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    return b''.join(chunks)


@contextmanager
def raw_reply_server(path, reply, *, delay_s=0):
    path.parent.mkdir(mode=0o700)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)

    def serve():
        try:
            connection, _ = listener.accept()
            with connection:
                connection.recv(4096)
                if delay_s:
                    time.sleep(delay_s)
                if reply:
                    connection.sendall(reply)
        except (BrokenPipeError, OSError):
            pass
        finally:
            listener.close()

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        yield path
    finally:
        thread.join(timeout=1)
        assert not thread.is_alive()


def encoded_snapshot(snapshot):
    return json.dumps(asdict(snapshot), separators=(',', ':')).encode() + b'\n'


def test_client_reads_valid_snapshot_and_checks_boot_master_sequence(tmp_path,
                                                                    server_factory):
    server = server_factory(tmp_path / 'clock.sock', healthy_snapshot(sequence=3))
    client = SnapshotClient(server.path, timeout_s=.05, expected_master=MASTER)

    assert client.read().sequence == 3
    server.source.set(healthy_snapshot(sequence=4))
    assert client.read().sequence == 4


@pytest.mark.parametrize('wire_request', [
    b'{}\n',
    b'{"op":"stop","schema":1}\n',
    b'{"op":"snapshot","schema":1,"extra":true}\n',
    b'{"op":"snapshot","schema":1,"schema":1}\n',
    b'{"op":"snapshot","schema":1}\n{}\n',
    b'\xff\n',
    b'x' * 257,
])
def test_server_rejects_every_non_snapshot_request(tmp_path, server_factory,
                                                   wire_request):
    server = server_factory(tmp_path / 'clock.sock')

    reply = exchange(server.path, wire_request)

    assert json.loads(reply) == {
        'schema': 1,
        'ok': False,
        'error': 'invalid_request',
    }


@pytest.mark.parametrize('change', [
    {'boot_id': 'wrong-boot'},
    {'expected_master': 'wrong-master', 'actual_master': 'wrong-master'},
    {'actual_master': 'wrong-master'},
    {'valid_until_ns': 1},
])
def test_client_rejects_wrong_identity_or_expired_lease(tmp_path, server_factory, change):
    server = server_factory(tmp_path / 'clock.sock', healthy_snapshot(**change))
    client = SnapshotClient(server.path, timeout_s=.05, expected_master=MASTER)

    with pytest.raises(ValueError):
        client.read()


def test_client_rejects_sequence_rollback_and_session_change(tmp_path, server_factory):
    first = healthy_snapshot(sequence=3)
    server = server_factory(tmp_path / 'clock.sock', first)
    client = SnapshotClient(server.path, timeout_s=.05, expected_master=MASTER)
    assert client.read().sequence == 3

    server.source.set(replace(first, sequence=3))
    with pytest.raises(ValueError):
        client.read()

    server.source.set(replace(first, sequence=4, session_id='replacement-session'))
    with pytest.raises(ValueError):
        client.read()


def invalid_payloads():
    valid = asdict(healthy_snapshot(sequence=7))
    extra = dict(valid, extra='field')
    missing = dict(valid)
    missing.pop('reason')
    boolean_sequence = dict(valid, sequence=True)
    excessive_drift = dict(valid, drift_ppm=101.0)
    negative_delay = dict(valid, path_delay_ns=-1)
    extended_lease = dict(valid, valid_until_ns=valid['valid_until_ns'] + 1)
    unhealthy = dict(valid, healthy=False, reason='ptp_fault')
    for payload in (extra, missing, boolean_sequence, excessive_drift,
                    negative_delay, extended_lease, unhealthy):
        yield json.dumps(payload, separators=(',', ':')).encode() + b'\n'
    yield encoded_snapshot(healthy_snapshot(sequence=7)).replace(b'10.0', b'NaN', 1)


@pytest.mark.parametrize('reply', list(invalid_payloads()))
def test_client_strictly_rejects_malformed_types_ranges_and_keys(tmp_path, reply):
    with raw_reply_server(tmp_path / 'raw' / 'clock.sock', reply) as path:
        client = SnapshotClient(path, timeout_s=.05, expected_master=MASTER)
        with pytest.raises(ValueError):
            client.read()


def test_client_rejects_oversize_response_and_timeout(tmp_path):
    with raw_reply_server(tmp_path / 'large' / 'clock.sock', b'x' * 4097) as path:
        client = SnapshotClient(path, timeout_s=.05, expected_master=MASTER)
        with pytest.raises(ValueError):
            client.read()

    with raw_reply_server(tmp_path / 'slow' / 'clock.sock', b'', delay_s=.1) as path:
        client = SnapshotClient(path, timeout_s=.01, expected_master=MASTER)
        with pytest.raises(TimeoutError):
            client.read()


def test_socket_and_parent_are_current_user_only(tmp_path, server_factory):
    server = server_factory(tmp_path / 'private' / 'clock.sock')

    assert stat.S_IMODE(server.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(server.path.stat().st_mode) == 0o600


def test_server_rejects_preexisting_insecure_parent(tmp_path):
    parent = tmp_path / 'shared'
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)

    with pytest.raises(PermissionError):
        SnapshotServer(parent / 'clock.sock', healthy_snapshot)


def test_provider_failure_is_fail_closed_without_stopping_server(tmp_path):
    snapshot = healthy_snapshot()
    calls = 0

    def flaky_provider():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError('provider failed')
        return snapshot

    server = SnapshotServer(tmp_path / 'clock.sock', flaky_provider)
    try:
        reply = exchange(server.path, b'{"op":"snapshot","schema":1}\n')
        assert json.loads(reply) == {
            'schema': 1,
            'ok': False,
            'error': 'snapshot_unavailable',
        }
        client = SnapshotClient(server.path, timeout_s=.05, expected_master=MASTER)
        assert client.read() == snapshot
    finally:
        server.close()


def test_close_is_idempotent_and_disables_future_use(tmp_path, server_factory):
    server = server_factory(tmp_path / 'clock.sock')
    client = SnapshotClient(server.path, timeout_s=.05, expected_master=MASTER)
    assert client.read().healthy is True

    client.close()
    client.close()
    with pytest.raises(RuntimeError):
        client.read()

    server.close()
    server.close()
    assert not server.path.exists()
