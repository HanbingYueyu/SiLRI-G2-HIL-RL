"""Operator-launched human demonstration; no clock process is involved."""
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

from .training_config import load_training_config


def _upstream_reset_command(root):
    """Read Section 6 verbatim and return its cwd and argv for shell-free call."""
    candidates = (Path(root).parent / '常用命令.md', Path(root) / '常用命令.md')
    document = next((path for path in candidates if path.is_file()), None)
    if document is None:
        raise FileNotFoundError('常用命令.md with Section 6 is required for visual reset')
    text = document.read_text()
    match = re.search(r'^## 6\. 上游起始位置\s*$([\s\S]*?)(?=^## |\Z)', text,
                      flags=re.MULTILINE)
    if match is None:
        raise ValueError('Section 6 上游起始位置 was not found')
    block = re.search(r'```bash\s*\n([\s\S]*?)\n```', match.group(1))
    if block is None:
        raise ValueError('Section 6 bash command block was not found')
    command_text = block.group(1).replace(chr(92) + '\n', ' ')
    tokens = shlex.split(command_text, comments=False, posix=True)
    expected = ['cd', '/home/flyfuture/g2_hinge_assembly', '&&',
                './scripts/run_gdk_sam3_python.sh',
                'g2_adapter/run_bilateral_flow_gdk.py']
    if (tokens[:len(expected)] != expected or '--allow-motion' not in tokens or
            any(token in (';', '|', '>', '<', '&&', '||') for token in tokens[3:])):
        raise ValueError('Section 6 must contain the unchanged G2 reset command')
    return Path(tokens[1]), tokens[3:]


def _run_post_demo_reset(root, config_path, command_runner):
    # Read Section 6 for every labeled episode so operator edits take effect
    # without copying stage names or numeric flags into SiLRI.
    visual_cwd, visual_argv = _upstream_reset_command(root)
    pre_reset = ['bash', str(Path(root) / 'run_g2_python.sh'),
                 '-m', 'g2_local.pre_reset',
                 '--config', str(config_path), '--allow-motion']
    print('Y/F 回合已保存；先执行 +Z 5 cm、再 +Y 10 cm 的 SiLRI 受限位移。',
          flush=True)
    result = command_runner(pre_reset, cwd=root, check=False)
    if result.returncode != 0:
        print(f'预复位失败（退出码 {result.returncode}）；未调用第 6 节视觉复位命令。',
              file=sys.stderr, flush=True)
        return result.returncode or 1
    print('位移完成；调用《常用命令.md》第 6 节原样的上游视觉复位命令。', flush=True)
    result = command_runner(visual_argv, cwd=visual_cwd, check=False)
    if result.returncode != 0:
        print(f'第 6 节视觉复位命令失败（退出码 {result.returncode}）；停止流程。',
              file=sys.stderr, flush=True)
        return result.returncode or 1
    print('视觉复位完成。退出上游程序后，重新运行人工采集命令采下一条。', flush=True)
    return 0


def run(*, root=None, demo_main=None, command_runner=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    command_runner = subprocess.run if command_runner is None else command_runner
    source = root / 'runtime/train-fixed-fridge-20260928-camera-relaxed.json'
    config = load_training_config(source, cli_allow_motion=True)
    if not config.motion_permitted or config.motion.auto_reset.enabled:
        raise PermissionError('Demo permission missing or automatic reset enabled')
    run_id = 'demo-' + uuid.uuid4().hex[:12]
    session = root / 'runtime' / (run_id + '-launch')
    session.mkdir(mode=0o700)
    raw = json.loads(source.read_text())
    config_path = session / 'config.json'
    with config_path.open('x') as stream:
        json.dump(raw, stream, ensure_ascii=False, indent=2)
    load_training_config(config_path, cli_allow_motion=True)
    try:
        if demo_main is None:
            from .real_train import main as demo_main
        print('等待双键开始；请保持夹持起点，并确保上游运动程序已退出。'
              '（SpaceMouse 独占锁：采集与上游复位程序不能同时运行。）', flush=True)
        context_path = session / 'context.json'
        context = dict(episode_id=run_id+'-ep1', target_offset_m=[0., 0., 0.],
                       ee_reset_offset=[0.]*6,
                       approach_source='operator_visual_confirmation_current_fixture_baseline',
                       grasp_description='hinge held; manual reset; gripper stays closed',
                       visual_reset_monotonic_ns=time.monotonic_ns(),
                       visual_confidence=None, upstream_frame_id=None)
        with context_path.open('x') as stream:
            json.dump(context, stream)
        print('采集目录：', root/'runtime'/run_id, flush=True)
        print('30 秒内两键都按住再全部松开开始；按 Y/F 结束并保存，随后自动执行机械位移和第 6 节视觉复位。', flush=True)
        # Keep the demo's own stop/cleanup path in this process.
        result = demo_main(['demo', '--run-id', run_id, '--config', str(config_path),
                            '--output', str(root/'runtime'/run_id),
                            '--context', str(context_path),
                            '--hid-device', '/dev/spacemouse-compact', '--allow-motion',
                            '--exit-after-labeled-demo'])
        if result != 0:
            return result
        return _run_post_demo_reset(root, config_path, command_runner)
    except KeyboardInterrupt:
        return 130


def main():
    import argparse
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.parse_args()
        root = Path(__file__).resolve().parents[1]
        config = load_training_config(
            root / 'runtime/train-fixed-fridge-20260928-camera-relaxed.json',
            cli_allow_motion=True)
        if not config.motion_permitted or config.motion.auto_reset.enabled:
            raise PermissionError('Demo permission missing or automatic reset enabled')
        # Freshness is local (see freshness.ObservationFreshnessGuard local mode):
        # no clock process is started, read or required.
        return run(root=root)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f'人工采集启动/运行失败：{type(error).__name__}: {error}',
              file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
