import json
from types import SimpleNamespace

import pytest


SECTION_SIX = '''## 6. 上游起始位置

```bash
cd /home/flyfuture/g2_hinge_assembly && \\
./scripts/run_gdk_sam3_python.sh g2_adapter/run_bilateral_flow_gdk.py \\
  --from-stage right_to_lr1 --through-stage left_to_teach_start \\
  --allow-motion
```
'''


@pytest.mark.parametrize('failure', [None, 'demo', 'interrupt'])
def test_demo_consumes_the_operator_clock_and_never_spawns_one(
        tmp_path, monkeypatch, failure):
    """Time synchronisation runs in its own terminal; collection only consumes it."""
    from g2_local import manual_demo as launch
    root = tmp_path
    (root / 'runtime').mkdir()
    source = root / 'runtime/train-fixed-fridge-20260928-camera-relaxed.json'
    original = {'task': {'control_hz': 30, 'max_episode_steps': 900},
                'commissioning': {'clock_socket': '/old/clock.sock'}}
    source.write_text(json.dumps(original))
    (root / '常用命令.md').write_text(SECTION_SIX)
    monkeypatch.setattr(launch, 'load_training_config', lambda *a, **k: SimpleNamespace(
        motion_permitted=True, motion=SimpleNamespace(auto_reset=SimpleNamespace(enabled=False)),
        commissioning=SimpleNamespace(expected_master='044052.fffe.000010')))
    clock = tmp_path / 'clock.sock'
    clock.touch()
    configs, commands = [], []

    def forbidden(*args, **kwargs):
        raise AssertionError('Collection must never start a clock monitor')

    monkeypatch.setattr(launch.subprocess, 'Popen', forbidden)

    def demo(argv):
        path = launch.Path(argv[argv.index('--config') + 1])
        config = json.loads(path.read_text())
        assert config['task']['control_hz'] == 30
        context = launch.Path(argv[argv.index('--context') + 1])
        assert json.loads(context.read_text())['visual_reset_monotonic_ns'] > 0
        assert '--exit-after-labeled-demo' in argv
        configs.append(path)
        if failure == 'demo':
            raise RuntimeError('demo failed')
        if failure == 'interrupt':
            raise KeyboardInterrupt
        return 0

    def command_runner(argv, **kwargs):
        commands.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    if failure == 'demo':
        with pytest.raises(RuntimeError):
            launch.run(root=root, demo_main=demo,
                       command_runner=command_runner)
    else:
        assert launch.run(root=root, demo_main=demo,
                          command_runner=command_runner) == (
            130 if failure == 'interrupt' else 0)
    # The session copy carries the operator's socket; the base config is untouched.
    assert json.loads(source.read_text()) == original
    assert len(configs) == 1
    if failure is None:
        pre_reset, visual_reset = commands
        assert pre_reset[0][2:5] == ['-m', 'g2_local.pre_reset', '--config']
        assert pre_reset[0][-1] == '--allow-motion'
        assert pre_reset[1]['cwd'] == root
        assert visual_reset[0] == [
            './scripts/run_gdk_sam3_python.sh',
            'g2_adapter/run_bilateral_flow_gdk.py',
            '--from-stage', 'right_to_lr1', '--through-stage',
            'left_to_teach_start', '--allow-motion']
        assert visual_reset[1]['cwd'] == launch.Path('/home/flyfuture/g2_hinge_assembly')
    else:
        assert commands == []


@pytest.mark.parametrize('pre_reset_code,visual_reset_code,expected_calls,expected_result', [
    (1, 0, 1, 1), (0, 2, 2, 2),
])
def test_post_demo_reset_stops_if_either_reset_stage_fails(
        tmp_path, pre_reset_code, visual_reset_code, expected_calls, expected_result):
    from g2_local.manual_demo import _run_post_demo_reset
    (tmp_path / '常用命令.md').write_text('''## 6. 上游起始位置
```bash
cd /home/flyfuture/g2_hinge_assembly && \\
./scripts/run_gdk_sam3_python.sh g2_adapter/run_bilateral_flow_gdk.py \\
--relief-extra-lift-mm 4 \\
--vlm-enable false --tts-enable false \\
--teach-max-acceleration-rad-s2 20.0 \\
--from-stage left_to_hole_offset --through-stage right_to_lr1 \\
--allow-motion
```
''')
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        code = pre_reset_code if len(calls) == 1 else visual_reset_code
        return SimpleNamespace(returncode=code)

    result = _run_post_demo_reset(tmp_path, tmp_path / 'config.json', runner)

    assert result == expected_result
    assert len(calls) == expected_calls
    if expected_calls == 2:
        assert calls[1][0] == [
            './scripts/run_gdk_sam3_python.sh',
            'g2_adapter/run_bilateral_flow_gdk.py',
            '--relief-extra-lift-mm', '4',
            '--vlm-enable', 'false', '--tts-enable', 'false',
            '--teach-max-acceleration-rad-s2', '20.0',
            '--from-stage', 'left_to_hole_offset',
            '--through-stage', 'right_to_lr1', '--allow-motion']
