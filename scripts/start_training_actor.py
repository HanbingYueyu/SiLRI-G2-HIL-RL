"""Operator launcher for real-robot training episodes.

It wraps ``g2_local.real_train actor`` and can drive the whole per-episode loop,
including the Section 6 upstream reset between episodes:

  one episode (write context + start the Actor here):
      bash run_g2_python.sh scripts/start_training_actor.py --allow-motion
  whole loop (Section 6 runs automatically after each episode):
      bash run_g2_python.sh scripts/start_training_actor.py --allow-motion --loop 20
  write the context file only:
      bash run_g2_python.sh scripts/start_training_actor.py --write-context

Why the loop exists: the SpaceMouse is an exclusive lock, so the Actor must be
gone before the upstream reset can take the mouse. Each episode therefore runs
in its own Actor process (also keeps every GDK initialization in a fresh
process). In ``--loop`` mode this script does, per episode:

  1. publish a fresh context;
  2. start one Actor subprocess and hand the terminal to it;
  3. when the Actor exits (Y/F then ``Ctrl+C``), run the same guarded retraction
     demonstration collection uses (``g2_local.pre_reset``: +Z 50 mm, then
     +Y 100 mm) in its own process;
  4. run Section 6 verbatim, in the foreground, so the operator can still drive
     its SpaceMouse gates;
  5. confirm Section 6 exited cleanly and the mouse is free, then next episode.

Section 6 runs before the *next* episode, not after the last one, and is skipped
entirely for the first episode when ``--skip-first-reset`` is given (that is also
the right flag for a single episode whose Section 6 you already ran by hand).

The Learner is not touched: it stays up across episodes.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


DEFAULT_RUN_ID = 'offline-pretrain-20260928-camera30-02'
DEFAULT_CONFIG = 'runtime/train-fixed-fridge-20260928-camera-relaxed.json'
DEFAULT_CONTEXT = 'runtime/episode-context.json'


def parser():
    cli = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--run-id', default=DEFAULT_RUN_ID,
                     help='Must equal the running Learner --run-id')
    cli.add_argument('--config', default=DEFAULT_CONFIG,
                     help='Must equal the running Learner --config')
    cli.add_argument('--context', default=DEFAULT_CONTEXT,
                     help='Same path the Actor polls for the current episode')
    cli.add_argument('--hid-device', default='/dev/spacemouse-compact')
    cli.add_argument('--allow-motion', action='store_true',
                     help='Start the Actor after writing the context file')
    cli.add_argument('--write-context', action='store_true',
                     help='Only publish a fresh context file, then exit')
    cli.add_argument('--auto-reset', action='store_true',
                     help='Pass --auto-reset to the Actor (lift/return inside the '
                          'Actor process; still experimental, never changes the seed)')
    cli.add_argument('--loop', type=int, default=0, metavar='N',
                     help='Run N episodes back to back, auto-running Section 6 after '
                          'each one. 0 = single episode.')
    cli.add_argument('--skip-first-reset', action='store_true',
                     help='The scene is already reset for the first episode, so start '
                          'the Actor without running Section 6 first. Use it in single '
                          'mode too when you just ran Section 6 by hand, otherwise '
                          'Section 6 runs twice.')
    cli.add_argument('--skip-pre-reset', action='store_true',
                     help='Do not run the guarded +Z 5 cm / +Y 10 cm retraction between '
                          'an episode and Section 6 (the Actor then hands the arm over '
                          'from wherever the episode ended)')
    return cli


def spacemouse_holder(path):
    """Return a reason string when another process holds the HID lock."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as error:
        return f'无法打开：{error}'
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return '另一个进程持有独占 fcntl 锁'
    finally:
        os.close(fd)
    return None


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def upstream_reset_command(root):
    """Section 6 verbatim, from 常用命令.md, validated exactly like demo mode."""
    from g2_local.manual_demo import _upstream_reset_command
    return _upstream_reset_command(root)


def write_context(root, path, *, run_id):
    target = resolve(root, path)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    context = dict(
        episode_id=f'{run_id}-{time.strftime("%Y%m%d-%H%M%S")}-{time.monotonic_ns()}',
        target_offset_m=[0., 0., 0.], ee_reset_offset=[0.] * 6,
        approach_source='operator_visual_confirmation_current_fixture_baseline',
        grasp_description='hinge held; upstream reset; gripper stays closed',
        # Local monotonic receipt time of the completed upstream visual reset.
        visual_reset_monotonic_ns=time.monotonic_ns(),
        visual_confidence=None, upstream_frame_id=None)
    temporary = target.parent / f'.{target.name}.tmp'
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(context, stream)
    os.replace(temporary, target)
    print(f'已写入 context：{target}', flush=True)
    print(f'  episode_id={context["episode_id"]}', flush=True)
    print(f'  visual_reset_monotonic_ns={context["visual_reset_monotonic_ns"]}', flush=True)
    return target, context


def actor_command(args, root, config_path, output, context_path):
    return [sys.executable, '-m', 'g2_local.real_train', 'actor',
            '--run-id', args.run_id, '--config', str(config_path),
            '--output', str(output), '--context', str(context_path),
            '--hid-device', args.hid_device, '--allow-motion',
            *(['--auto-reset'] if args.auto_reset else []),
            # Loop mode: let the Actor exit by itself after Y/F so this script can
            # run the upstream reset without an extra Ctrl+C.
            *(['--exit-after-labeled-demo'] if args.loop else [])]


def run_child(argv, *, cwd):
    """Run a child in the foreground; its Ctrl+C must not kill this loop.

    The terminal delivers SIGINT to the whole foreground process group, so the
    loop ignores it while a child owns the terminal and restores the default
    handler in between.
    """
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return subprocess.run(argv, cwd=cwd, check=False)
    finally:
        signal.signal(signal.SIGINT, previous)


def run_one_episode(args, root, config_path, *, index):
    busy = spacemouse_holder(args.hid_device)
    if busy is not None:
        print(f'⚠️  {args.hid_device} 已被占用（{busy}）。'
              'SpaceMouse 是独占锁：先退出上一个 Actor，再重跑本命令。', flush=True)
        return 2
    context_path, context = write_context(root, args.context, run_id=args.run_id)
    output = root / 'runtime' / f'actor-live-{time.strftime("%Y%m%d-%H%M%S")}'
    print(f'—— 第 {index} 回合 ——', flush=True)
    print(f'Actor 日志目录：{output}', flush=True)
    print('保持夹持起点，确保上游运动程序已退出、现场急停可用。', flush=True)
    print('① 30 秒内双键按住再全部松开：策略动作；SpaceMouse 接管，松手交还；Y/F 结束。',
          flush=True)
    print('   （若 Actor 打印“等待 SpaceMouse 就绪…”，它只是在等设备自己上报，'
          '不需要你做任何操作，也不会发动作。）', flush=True)
    if args.loop:
        print('本回合按 Y/F 后 Actor 会自动干净退出，脚本随即自动跑第 6 节复位、'
              '自动写新 context 并起下一个 Actor；你只需要按双键。'
              '（想中途停整个循环：在本终端按 Ctrl+C。）', flush=True)
    result = run_child(actor_command(args, root, config_path, output, context_path),
                       cwd=root)
    print(f'上下文 episode_id={context["episode_id"]} 结束，退出码 {result.returncode}',
          flush=True)
    return result.returncode


def run_section_six(root):
    cwd, argv = upstream_reset_command(root)
    print(f'▶ 自动执行第 6 节上游复位：{" ".join(argv)}', flush=True)
    print('（现在由你操作 SpaceMouse 走完它的阶段，脚本会等它退出。）', flush=True)
    result = run_child(argv, cwd=cwd)
    print(f'第 6 节退出码 {result.returncode}', flush=True)
    return result.returncode


def run_pre_reset(root, config_path):
    """The same guarded retraction the demonstration path runs after Y/F.

    Demonstration collection always retracts (+Z 50 mm, then +Y 100 mm) before
    handing over to Section 6, because the arm is still holding the hinge at the
    fixture. Training episodes end in the same pose, so the loop reproduces that
    sequence instead of dropping the arm straight into the upstream program.
    The module is called in its own process (fresh GDK initialisation) and its
    reset-only ceilings come from the shared code, never from this script.
    """
    argv = ['bash', str(Path(root) / 'run_g2_python.sh'),
            '-m', 'g2_local.pre_reset', '--config', str(config_path), '--allow-motion']
    print('▶ 先执行 Y/F 后的受限位移：+Z 5 cm，再 +Y 10 cm（退出码非 0 就不跑第 6 节）。',
          flush=True)
    result = run_child(argv, cwd=root)
    print(f'预复位退出码 {result.returncode}', flush=True)
    return result.returncode


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    args = parser().parse_args(argv)
    if args.write_context:
        if args.allow_motion or args.loop:
            parser().error('--write-context 不能与 --allow-motion/--loop 同时使用')
        write_context(root, args.context, run_id=args.run_id)
        print('请立刻按住双键再全部松开（超过配置的 context 时效后必须重新复位并重写）。',
              flush=True)
        return 0
    if not args.allow_motion:
        parser().error('现场复位、退出上游运动程序后，使用 --allow-motion 启动')
    if args.loop < 0 or args.loop > 200:
        parser().error('--loop 取 0..200')

    config_path = resolve(root, args.config)
    episodes = args.loop if args.loop else 1
    for index in range(1, episodes + 1):
        # Section 6 also owns the SpaceMouse, so refuse before touching anything
        # when another Actor is still alive.
        busy = spacemouse_holder(args.hid_device)
        if busy is not None:
            print(f'⚠️  {args.hid_device} 已被占用（{busy}）。'
                  'SpaceMouse 是独占锁：先退出上一个 Actor，再重跑本命令。', flush=True)
            return 2
        if not (index == 1 and args.skip_first_reset):
            if index > 1 and not args.skip_pre_reset:
                # Episode N-1 ended holding the hinge at the fixture; retract the
                # same way demonstration collection does before Section 6.
                code = run_pre_reset(root, config_path)
                if code != 0:
                    print('预复位没有正常退出：停止循环，先处理现场，然后重跑本命令。',
                          flush=True)
                    return code or 1
            # Section 6 owns the SpaceMouse, so no Actor may be alive here.
            code = run_section_six(root)
            if code != 0:
                print('第 6 节没有正常退出：停止循环，先处理现场，然后重跑本命令。',
                      flush=True)
                return code or 1
        code = run_one_episode(args, root, config_path, index=index)
        if code not in (0, 130):  # 0 = Y/F clean exit (loop mode); 130 = operator Ctrl+C
            print(f'Actor 以 {code} 退出（不是 Y/F 也不是 Ctrl+C）：停止循环，先看上面的报错。',
                  flush=True)
            return code
        if not args.loop:
            return 0
    print(f'已跑完 {episodes} 个回合，循环结束。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
