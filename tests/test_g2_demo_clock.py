import json
from types import SimpleNamespace
import pytest
from g2_local import demo_clock


@pytest.mark.parametrize('healthy', [True, False])
def test_existing_monitor_revalidated_without_starting_or_stopping_it(tmp_path, monkeypatch, healthy):
    directory = tmp_path / 'runtime/shared-demo-clock'
    directory.mkdir(parents=True, mode=0o700)
    socket = tmp_path / 'clock.sock'
    socket.touch()
    (directory / 'current.json').write_text(json.dumps({'socket': str(socket)}))
    closed = []
    class Client:
        def __init__(self, path, **kwargs):
            assert path == socket and kwargs['expected_master'] == 'master'
        def read(self):
            if not healthy:
                raise ValueError('lease_expired')
        def close(self):
            closed.append(True)
    monkeypatch.setattr(demo_clock, 'SnapshotClient', Client)
    def forbidden(*args, **kwargs):
        raise AssertionError('Must not launch or stop existing monitor')
    monkeypatch.setattr(demo_clock.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(demo_clock.subprocess, 'run', forbidden)
    monkeypatch.setattr(demo_clock, '_stop_monitor', forbidden)
    if healthy:
        assert demo_clock.ensure_clock(tmp_path, 'master') == socket
    else:
        with pytest.raises(RuntimeError, match='lease_expired'):
            demo_clock.ensure_clock(tmp_path, 'master')
    assert closed == [True]


@pytest.mark.parametrize('failure', [False, True])
def test_new_background_monitor_survives_only_successful_start(tmp_path, monkeypatch, failure):
    (tmp_path / 'runtime').mkdir()
    child = SimpleNamespace()
    stopped = []
    monkeypatch.setattr(demo_clock.subprocess, 'run', lambda *a, **k: None)
    monkeypatch.setattr(demo_clock.subprocess, 'Popen', lambda *a, **k: child)
    def wait(*args, **kwargs):
        if failure:
            raise RuntimeError('unhealthy')
    monkeypatch.setattr(demo_clock, '_wait_for_healthy_monitor', wait)
    monkeypatch.setattr(demo_clock, '_stop_monitor', stopped.append)
    if failure:
        with pytest.raises(RuntimeError, match='unhealthy'):
            demo_clock.ensure_clock(tmp_path, 'master')
        assert stopped == [child]
    else:
        path = demo_clock.ensure_clock(tmp_path, 'master')
        saved = json.loads((tmp_path / 'runtime/shared-demo-clock/current.json').read_text())
        assert saved['socket'] == str(path)
        assert not stopped
