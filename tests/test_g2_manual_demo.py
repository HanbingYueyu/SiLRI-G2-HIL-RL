import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('failure', [None, 'clock', 'demo', 'interrupt', 'retry', 'exhausted'])
def test_session_owns_unique_clock_and_cleans_it_on_all_exits(tmp_path, monkeypatch, failure):
    from g2_local import manual_demo as launch
    root = tmp_path
    (root / 'runtime').mkdir()
    source = root / 'runtime/site-demo.json'
    original = {'commissioning': {'clock_socket': '/old/clock.sock'}}
    source.write_text(json.dumps(original))
    monkeypatch.setattr(launch, 'load_training_config', lambda *a, **k: SimpleNamespace(
        motion_permitted=True, motion=SimpleNamespace(auto_reset=SimpleNamespace(enabled=False)),
        commissioning=SimpleNamespace(expected_master='044052.fffe.000010')))
    monitors, stopped, configs = [], [], []
    def popen(argv, **kwargs):
        assert argv[2] == 'g2_local.clock_monitor'
        assert not kwargs.get('start_new_session', False)
        process = SimpleNamespace(argv=argv)
        monitors.append(process)
        return process
    def wait(process, path, **kwargs):
        assert path.parent == launch.Path(process.argv[-1])
        if failure == 'clock':
            raise TimeoutError('not healthy')
        if failure == 'exhausted' or (failure == 'retry' and '-retry' not in str(path)):
            raise RuntimeError('clock exited: 2')
    if failure in ('retry', 'exhausted'):
        monkeypatch.setattr(launch, '_startup_clock_failure', lambda output: ('ptp_fault', True))
    def demo(argv):
        path = launch.Path(argv[argv.index('--config') + 1])
        config = json.loads(path.read_text())
        assert config['commissioning']['clock_socket'] == str(
            launch.Path(monitors[-1].argv[-1]) / 'clock.sock')
        context = launch.Path(argv[argv.index('--context') + 1])
        assert json.loads(context.read_text())['visual_reset_monotonic_ns'] > 0
        configs.append(path)
        if failure == 'demo':
            raise RuntimeError('demo failed')
        if failure == 'interrupt':
            raise KeyboardInterrupt
        return 0
    monkeypatch.setattr(launch.subprocess, 'Popen', popen)
    monkeypatch.setattr(launch, '_wait_for_healthy_monitor', wait)
    monkeypatch.setattr(launch, '_stop_monitor', lambda p: stopped.append(p))
    def unexpected_input(*args):
        raise AssertionError('Launcher must not require an Enter confirmation')
    monkeypatch.setattr('builtins.input', unexpected_input)
    for _ in range(2):
        if failure in ('clock', 'demo', 'exhausted'):
            with pytest.raises((TimeoutError, RuntimeError)):
                launch.run(root=root, demo_main=demo)
        else:
            assert launch.run(root=root, demo_main=demo) == (130 if failure == 'interrupt' else 0)
    assert stopped == monitors
    assert monitors[0].argv[-1] != monitors[1].argv[-1]
    assert json.loads(source.read_text()) == original
    assert len(configs) == (0 if failure in ('clock', 'exhausted') else 2)
    assert len(monitors) == (6 if failure == 'exhausted' else 4 if failure == 'retry' else 2)


@pytest.mark.parametrize('reason,raw,expected', [
    ('ptp_fault', 'UNCALIBRATED to LISTENING on ANNOUNCE_RECEIPT_TIMEOUT_EXPIRES', True),
    ('master_mismatch', 'UNCALIBRATED to LISTENING on ANNOUNCE_RECEIPT_TIMEOUT_EXPIRES', False),
    ('ptp_fault', 'FAULTY', False),
])
def test_startup_retry_requires_recorded_announce_timeout(tmp_path, reason, raw, expected):
    from g2_local.manual_demo import _startup_clock_failure
    (tmp_path / 'evidence.jsonl').write_text(
        json.dumps(dict(kind='ptp', raw=raw)) + '\n' +
        json.dumps(dict(kind='exit', reason=reason)) + '\n')
    assert _startup_clock_failure(tmp_path) == (reason, expected)
