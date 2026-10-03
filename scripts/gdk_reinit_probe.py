"""Read-only bench probe: can GDK re-acquire TF after init/release in one process?

`_automatic_reset` created a second motion environment inside the Actor process,
right after the episode's environment was closed. That failed on hardware with
``TimeoutError: TF cache did not expose both arm_l_end_link directions``. This
probe separates the two candidate causes without moving the robot:

  * "slow": TF needs longer than the 2 s reader budget after a re-init;
  * "never": the second ``gdk_init()`` in the same process never sees the frame.

It only calls ``observe()`` (read-only observation); ``allow_motion=False`` and
no command is ever submitted. Run it with no Actor/upstream program running:

    bash run_g2_python.sh scripts/gdk_reinit_probe.py --attempts 3 --timeout 20
"""
import argparse
import time
from pathlib import Path
import sys

DEFAULT_ADAPTER = '/home/flyfuture/g2_hinge_assembly'
# Both GdkReader and wait_for_tf bound their own budget to 30 s.
MAX_BUDGET_S = 30.


def probe(attempt, *, adapter_root, timeout_s, settle_s):
    from g2_local.gdk_backend import GdkReader, wait_for_tf
    started = time.monotonic()
    reader = None
    try:
        reader = GdkReader(adapter_root=adapter_root, timeout_s=timeout_s,
                           allow_motion=False)
        ready = time.monotonic() - started
        first = wait_for_tf(reader.tf, timeout_s=timeout_s)
        observation = reader.observe()
        time.sleep(settle_s)
        moved = wait_for_tf(reader.tf, timeout_s=timeout_s)
        advancing = any(moved[i]['timestamp_ns'] != first[i]['timestamp_ns']
                        for i in range(len(first)))
        print(f'attempt {attempt}: TF ready {ready:.2f}s after gdk_init; '
              f'state_ok={observation.get("state") is not None}; '
              f'frames_advancing={advancing}; '
              f'tf_stamps={[q["timestamp_ns"] for q in first]}', flush=True)
        return True, ready, advancing
    except BaseException as error:
        elapsed = time.monotonic() - started
        print(f'attempt {attempt}: FAILED after {elapsed:.2f}s '
              f'({type(error).__name__}: {error})', flush=True)
        return False, elapsed, False
    finally:
        if reader is not None:
            try:
                reader.close()
            except BaseException as error:
                print(f'attempt {attempt}: close failed: {error!r}', flush=True)
        # Give the library a moment to release its threads before re-initializing.
        time.sleep(2.)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=20.,
                        help='TF readiness budget per attempt, 1..30 s')
    parser.add_argument('--settle', type=float, default=2.)
    parser.add_argument('--adapter-root', default=DEFAULT_ADAPTER)
    args = parser.parse_args(argv)
    budget = min(args.timeout, MAX_BUDGET_S)
    if not 1 <= args.attempts <= 10 or not 1. <= budget or not 0. <= args.settle <= 10.:
        parser.error('1..10 attempts, 1..30 s timeout, 0..10 s settle')

    results = [probe(index + 1, adapter_root=args.adapter_root,
                     timeout_s=budget, settle_s=args.settle)
               for index in range(args.attempts)]
    first, rest = results[0], results[1:]
    if not first[0]:
        print('结论：连第一次 gdk_init 都没拿到 TF —— 先检查机器人/GDK/上游占用，再重跑。',
              flush=True)
        return 1
    if all(ok for ok, _, _ in rest):
        print(f'结论：同进程重复 gdk_init **能**拿到 TF（第一次 {first[1]:.2f}s，'
              f'后续 {[round(r[1], 2) for r in rest]}s）→ 自动复位只需放宽等待/重试。',
              flush=True)
        return 0
    print('结论：**同进程第二次 gdk_init 拿不到 TF** → 自动复位、以及"一个 Actor 进程'
          '连跑多回合"都必须改成"一个进程只做一次 GDK 初始化"'
          '（每回合一个子进程，或每回合重启 Actor）。', flush=True)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
