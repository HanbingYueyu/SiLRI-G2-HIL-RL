"""Read-only probe: are the SDK command clock and the sensor timestamps one domain?

The successor proof added to ``ObservationFreshnessGuard`` compares a command's
send time — read as ``gdk.Clock.now_ns()`` right after the acknowledgement —
with the successor observation's own timestamps:

    successor_timestamp_ns > send_sdk_clock_ns

That is a same-domain comparison and needs no PTP, but it is only valid when the
SDK clock and the camera/joint/TF timestamps really share one domain. The guard
verifies this per observation (``sdk_clock_ns - stamp`` must be within
``SDK_SAME_DOMAIN_MAX_NS``) and, if it is not, falls back to the local-receipt
guarantee with an explicit ``sdk_anchor.same_domain = False`` record.

This probe measures it on the bench without moving the robot:

    bash run_g2_python.sh scripts/gdk_clock_probe.py --reads 20

Expected when the domains match: every ``sdk_now - stamp`` is a small positive
number (milliseconds), and the verdict is SAME DOMAIN. Run it with no Actor and
no upstream motion program active.
"""
import argparse
from pathlib import Path
import statistics
import sys
import time

DEFAULT_ADAPTER = '/home/flyfuture/g2_hinge_assembly'
SOURCES = ('left_wrist', 'right_aux', 'joint', 'tf')
SAME_DOMAIN_MAX_NS = 500_000_000


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reads', type=int, default=20)
    parser.add_argument('--timeout', type=float, default=2.)
    parser.add_argument('--adapter-root', default=DEFAULT_ADAPTER)
    args = parser.parse_args(argv)
    if not 1 <= args.reads <= 200:
        parser.error('reads must be 1..200')

    from g2_local.gdk_backend import GdkReader
    reader = GdkReader(adapter_root=args.adapter_root, timeout_s=args.timeout,
                       allow_motion=False)
    rows = []
    try:
        for index in range(args.reads):
            observation = reader.observe()
            anchors = reader.read_sdk_clock_ns()
            info = reader.last_info
            stamps = info['source_timestamp_ns']
            ages = {source: anchors - stamps[source] for source in SOURCES}
            rows.append((index, anchors, ages, info))
            time.sleep(.05)
    finally:
        reader.close()

    print(f'{"#":>3}  ' + '  '.join(f'{s:>12}' for s in SOURCES) + '   same_domain')
    for index, _, ages, _ in rows:
        ok = all(0 <= age <= SAME_DOMAIN_MAX_NS for age in ages.values())
        print(f'{index:>3}  ' + '  '.join(f'{ages[s]/1e6:>10.1f}ms' for s in SOURCES)
              + f'   {ok}')
    all_ages = [age for _, _, ages, _ in rows for age in ages.values()]
    same = [all(0 <= age <= SAME_DOMAIN_MAX_NS for age in ages.values())
            for _, _, ages, _ in rows]
    print()
    print(f'样本 {len(rows)} 次；sdk_clock - stamp：最小 {min(all_ages)/1e6:.2f}ms，'
          f'中位 {statistics.median(all_ages)/1e6:.2f}ms，最大 {max(all_ages)/1e6:.2f}ms')
    if all(same):
        print('结论：SAME DOMAIN —— 同域发送锚有效，successor 的"动作后采集"会被强制检查。')
        return 0
    print('结论：**NOT the same domain**（至少一次采样超出 0..500ms）—— guard 会自动退回到'
          '"本地接收 + 时间戳严格前进"的保证，并在证据里记 '
          '`sdk_anchor.same_domain=false`；这种情况下不要期待同域证明，需要单独处理。')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
