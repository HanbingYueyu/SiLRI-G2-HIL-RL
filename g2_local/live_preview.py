"""Bounded HID-only preview. No GDK imports, motion, or automatic intervention."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
from .spacemouse import LiveInputGate


def sample(reader, gate, emit):
    try:
        frame = reader.poll()
        now = time.monotonic()
        proposal = gate.update(frame, now=now)
        emit(dict(asdict(proposal), preview_only=True, raw_axes=frame.axes,
                  buttons=frame.buttons, axis_times=frame.axis_times,
                  ready=frame.ready, received_monotonic=now))
    except Exception as exc:
        proposal = gate.invalidate()
        emit(dict(asdict(proposal), preview_only=True, error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis-map', required=True,
                        help='Explicit signed raw axes for forward,left,up; not auto-calibrated')
    parser.add_argument('--left-button', type=int, required=True, choices=(0, 1))
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--max-age', type=float, default=.25)
    parser.add_argument('--deadzone', type=float, default=.1)
    parser.add_argument('--device')
    parser.add_argument('--adapter-root', type=Path, default=Path('/home/flyfuture/g2_hinge_assembly'))
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 300:
        parser.error('--seconds must be finite in (0,300]')
    gate = LiveInputGate(axis_map=tuple(int(i) for i in args.axis_map.split(',')),
                         left_button=args.left_button, max_age=args.max_age,
                         deadzone=args.deadzone)
    sys.path.insert(0, str(args.adapter_root.resolve()))
    from g2_adapter.spacemouse_input import CompactHID
    def emit(row):
        print(json.dumps(row), flush=True)
    with CompactHID(args.device) as reader:
        deadline = time.monotonic() + args.seconds
        try:
            while time.monotonic() < deadline:
                sample(reader, gate, emit)
                time.sleep(.02)
        finally:
            emit(dict(asdict(gate.invalidate()), preview_only=True, event='closed'))


if __name__ == '__main__':
    main()
