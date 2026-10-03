#!/usr/bin/env python3
"""Read-only summary of a tcpdump PTP capture taken on the G2 link.

Usage:
    python3 scripts/analyze_ptp_capture.py runtime/ptp-link-YYYYmmdd-HHMMSS.log

The capture must be plain tcpdump text output, e.g.

    sudo timeout 180 tcpdump -i enp3s0 -s 256 -nn 'ether proto 0x88f7' -tttt \\
      2>&1 | tee runtime/ptp-link-$(date +%Y%m%d-%H%M%S).log

Nothing here touches the robot, the clock, or the network; it only reads a file
that was captured separately.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_TIMESTAMP = re.compile(r"^(\d+\.\d+)")
_MESSAGE_TYPE = re.compile(r"msg type : ([a-z_ ]+?) msg")

# linuxptp defaults observed on this link: announce every 2 s, sync every 1 s.
# 3 x announce interval is the receipt timeout that makes ptp4l drop the master.
_ANNOUNCE_SILENCE_S = 3.0
_SYNC_SILENCE_S = 2.0


def parse(path: Path) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for line in path.read_text(errors="replace").splitlines():
        stamp = _TIMESTAMP.match(line)
        if stamp is None:
            continue
        kind = _MESSAGE_TYPE.search(line)
        rows.append((float(stamp.group(1)),
                     kind.group(1).strip().replace(" ", "_") if kind else "other"))
    return rows


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip())
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"capture file not found: {path}", file=sys.stderr)
        return 1
    rows = parse(path)
    if not rows:
        print("no timestamped PTP packets parsed; is this tcpdump text output?")
        return 1

    start, end = rows[0][0], rows[-1][0]
    counts: dict[str, int] = {}
    for _, kind in rows:
        counts[kind] = counts.get(kind, 0) + 1
    print(f"file        : {path}")
    print(f"span        : {end - start:.1f} s")
    print(f"packets     : {len(rows)}")
    print("by type     : " + ", ".join(
        f"{kind}={count}" for kind, count in
        sorted(counts.items(), key=lambda item: (-item[1], item[0]))))

    buckets = [0] * max(int(end - start) + 1, 1)
    for stamp, _ in rows:
        buckets[min(int(stamp - start), len(buckets) - 1)] += 1
    print("\nper-second packet count (bar length capped at 60):")
    for index, count in enumerate(buckets):
        print(f"  +{index:4d}s {count:4d}  {'#' * min(count, 60)}")

    for target, limit in (("announce", _ANNOUNCE_SILENCE_S), ("sync", _SYNC_SILENCE_S)):
        stamps = [stamp for stamp, kind in rows if kind == target]
        if len(stamps) < 2:
            print(f"\n{target}: {len(stamps)} packet(s); no gap analysis")
            continue
        gaps = sorted(stamps[index + 1] - stamps[index]
                      for index in range(len(stamps) - 1))
        print(f"\n{target}: {len(stamps)} packets, median gap {gaps[len(gaps) // 2]:.2f} s, "
              f"max gap {gaps[-1]:.2f} s")
        for index in range(len(stamps) - 1):
            gap = stamps[index + 1] - stamps[index]
            if gap > limit:
                print(f"   SILENCE {gap:6.2f} s  {stamps[index]:.3f} -> {stamps[index + 1]:.3f}")
        print(f"   last {target} at +{stamps[-1] - start:.2f} s")

    if "announce" in counts and "sync" in counts:
        last_announce = max(stamp for stamp, kind in rows if kind == "announce")
        last_sync = max(stamp for stamp, kind in rows if kind == "sync")
        print(f"\ntail of capture: last announce +{last_announce - start:.2f} s, "
              f"last sync +{last_sync - start:.2f} s "
              "(if these are well before the span end, the master went silent "
              "and stayed silent; if the capture simply ended, re-capture longer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
