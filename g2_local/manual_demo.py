"""Operator-launched human demo with a session-owned read-only clock monitor."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .clock_ipc import SnapshotClient
from .real_actor_audit import _stop_monitor, _wait_for_healthy_monitor
from .training_config import load_training_config


def _startup_clock_failure(output):
    """Only a recorded announce timeout permits a fresh startup session."""
    try:
        rows = [json.loads(line) for line in
                (output / 'evidence.jsonl').read_text().splitlines()]
    except (OSError, ValueError):
        return 'unknown', False
    exits = [row for row in rows if row.get('kind') == 'exit']
    reason = exits[-1].get('reason', 'unknown') if exits else 'unknown'
    timeout = any('UNCALIBRATED to LISTENING on ANNOUNCE_RECEIPT_TIMEOUT_EXPIRES'
                  in row.get('raw', '') for row in rows if row.get('kind') == 'ptp')
    return reason, reason == 'ptp_fault' and timeout


def run(*, root=None, demo_main=None, shared_clock=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    source = root / 'runtime/site-demo.json'
    config = load_training_config(source, cli_allow_motion=True)
    if not config.motion_permitted or config.motion.auto_reset.enabled:
        raise PermissionError('Demo permission missing or automatic reset enabled')
    run_id = 'demo-' + uuid.uuid4().hex[:12]
    session = root / 'runtime' / (run_id + '-launch')
    session.mkdir(mode=0o700)
    monitor_output = Path('/tmp') / ('g2-' + run_id)
    socket_path = Path(shared_clock) if shared_clock is not None else monitor_output / 'clock.sock'
    raw = json.loads(source.read_text())
    raw['commissioning']['clock_socket'] = str(socket_path)
    config_path = session / 'config.json'
    with config_path.open('x') as stream:
        json.dump(raw, stream, ensure_ascii=False, indent=2)
    load_training_config(config_path, cli_allow_motion=True)
    master = config.commissioning.expected_master
    monitor = None
    try:
        if shared_clock is None:
            print('自动启动只读时钟监控；如提示请输入 sudo 密码。', flush=True)
        deadline = time.monotonic() + 120.
        for attempt in range(0 if shared_clock is not None else 3):
            if attempt:
                monitor_output = Path('/tmp') / ('g2-' + run_id + f'-retry{attempt}')
                socket_path = monitor_output / 'clock.sock'
            # No demo environment or command port exists during these retries.
            monitor = subprocess.Popen([
                sys.executable, '-m', 'g2_local.clock_monitor', '--master', master,
                '--max-seconds', '1800', '--output', str(monitor_output)],
                stdin=None, stdout=None, stderr=None, shell=False)
            print(f'等待时钟就绪（总预算 120 秒，第 {attempt+1}/3 次）...', flush=True)
            try:
                _wait_for_healthy_monitor(
                    monitor, socket_path, master=master,
                    timeout_s=max(.001, deadline-time.monotonic()),
                    client_factory=lambda **kw: SnapshotClient(
                        kw['path'], timeout_s=kw['timeout_s'], expected_master=kw['expected_master']),
                    now_fn=time.monotonic, sleep_fn=time.sleep)
                break
            except (RuntimeError, TimeoutError) as error:
                reason, retryable = _startup_clock_failure(monitor_output)
                if not retryable or attempt == 2 or time.monotonic() >= deadline:
                    raise RuntimeError(f'{error}; monitor_reason={reason}; evidence={monitor_output}') from error
                _stop_monitor(monitor)
                monitor = None
                print('启动时主时钟公告超时，自动新建只读时钟会话重连；尚未进入采集。', flush=True)
        raw['commissioning']['clock_socket'] = str(socket_path)
        with config_path.open('w') as stream:
            json.dump(raw, stream, ensure_ascii=False, indent=2)
        load_training_config(config_path, cli_allow_motion=True)
        if demo_main is None:
            from .real_train import main as demo_main
        print('时钟已就绪，直接等待双键开始；请保持夹持起点，并确保上游运动程序已退出。', flush=True)
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
        print('30 秒内两键都按住再全部松开开始；Y/F 结束，保存后 Ctrl+C 退出。', flush=True)
        # Keep the demo's own stop/cleanup path in this process.
        return demo_main(['demo', '--run-id', run_id, '--config', str(config_path),
                          '--output', str(root/'runtime'/run_id),
                          '--context', str(context_path),
                          '--hid-device', '/dev/spacemouse-compact', '--allow-motion'])
    except KeyboardInterrupt:
        return 130
    finally:
        # Only stop our child; never pkill or remove earlier evidence.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            _stop_monitor(monitor)
        finally:
            signal.signal(signal.SIGINT, previous)


def main():
    try:
        from .demo_clock import ensure_clock
        root = Path(__file__).resolve().parents[1]
        config = load_training_config(root / 'runtime/site-demo.json', cli_allow_motion=True)
        if not config.motion_permitted or config.motion.auto_reset.enabled:
            raise PermissionError('Demo permission missing or automatic reset enabled')
        return run(root=root, shared_clock=ensure_clock(root, config.commissioning.expected_master))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f'人工采集启动/运行失败：{type(error).__name__}: {error}',
              file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
