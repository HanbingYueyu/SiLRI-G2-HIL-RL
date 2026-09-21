"""Normalized proposals only. No hardware control or implicit intervention."""
from dataclasses import dataclass
import math
import time
from .contract import vector


@dataclass(frozen=True)
class Proposal:
    action: tuple[float, ...]
    mode: str
    blocked: bool


class ProposalMapper:
    """axis_map: signed raw indices for forward, left, up, respectively.

    Indices are 1-based, restricted to the three translation channels.
    Signs must be calibrated; no default axis mapping is supplied.
    valid is supplied by the reader's freshness/connection checks, not a
    hardware enable. Rotations are base-frame rotation-vector components,
    not absolute Euler angles. Left button never selects intervention.
    """
    def __init__(self, *, axis_map, deadzone=0.1):
        self.axis_map = tuple(axis_map)
        if (len(self.axis_map) != 3 or
                any(type(i) is not int for i in self.axis_map) or
                set(map(abs, self.axis_map)) != {1, 2, 3}):
            raise ValueError('Expected signed permutation of 1,2,3')
        if not math.isfinite(deadzone) or not 0 <= deadzone < 1:
            raise ValueError('Deadzone must be finite in [0,1)')
        self.deadzone = deadzone
        self.mode = None
        self.armed = False

    def update(self, axes, *, left_pressed, valid):
        try:
            raw = vector(axes, 6)
            if any(abs(v) > 1 for v in raw):
                raise ValueError('Expected normalized HID axes')
            if type(left_pressed) is not bool or type(valid) is not bool:
                raise ValueError('Boolean mode and validity required')
        except Exception:
            self.armed = False
            raise
        mode = 'rotation' if left_pressed else 'translation'
        if mode != self.mode or not valid:
            self.armed = False
        self.mode = mode
        xyz = tuple(raw[abs(i) - 1] * (1 if i > 0 else -1) for i in self.axis_map)
        if valid and all(abs(v) <= self.deadzone for v in xyz):
            self.armed = True
        if not self.armed:
            return Proposal((0.,) * 6, mode, True)
        values = tuple(math.copysign(max(0., abs(v) - self.deadzone) /
                                     (1 - self.deadzone), v) for v in xyz)
        forward, left, up = values
        action = (0., 0., 0., left, forward, up) if left_pressed else (*values, 0., 0., 0.)
        return Proposal(action, mode, False)


class LiveInputGate:
    """Fail-closed preview gate. Silent HID is not a verified connection heartbeat.

    A previously observed neutral report may remain a zero proposal while idle.
    It cannot arm a new source or survive mode changes/read failures. Nonzero
    commands always require fresh reports; silent neutral is NOT a heartbeat.
    Button timestamps are unavailable in the reference reader, so this is NOT
    a complete robot enable/deadman mechanism.
    """
    def __init__(self, *, axis_map, left_button, max_age=.25, deadzone=.1):
        if type(left_button) is not int or left_button not in (0, 1):
            raise ValueError('Explicit left button index 0 or 1 required')
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError('Positive finite maximum age required')
        self.mapper = ProposalMapper(axis_map=axis_map, deadzone=deadzone)
        self.left_button = left_button
        self.max_age = max_age
        self.neutral_evidence = None
        self.input_valid = False
        self.fresh = False

    def invalidate(self):
        self.neutral_evidence = None
        self.input_valid = False
        self.fresh = False
        return self.mapper.update((0.,)*6, left_pressed=False, valid=False)

    def update(self, frame, *, now):
        try:
            raw = vector(frame.axes, 6)
            stamps = tuple(frame.axis_times)
            button = frame.buttons[self.left_button]
            neutral = all(abs(v) <= self.mapper.deadzone for v in raw[:3])
            self.fresh = bool(math.isfinite(now) and frame.ready and len(stamps) == 2
                     and all(t is not None and math.isfinite(t) and
                             0 <= now-t <= self.max_age for t in stamps))
            evidence = (stamps, button)
            idle = (math.isfinite(now) and frame.ready and neutral and
                    self.neutral_evidence == evidence and
                    all(t is not None and math.isfinite(t) and t <= now for t in stamps))
            self.input_valid = bool(self.fresh or idle)
            proposal = self.mapper.update(raw, left_pressed=button, valid=self.input_valid)
            if self.fresh and neutral and not proposal.blocked:
                self.neutral_evidence = evidence
            elif not idle:
                self.neutral_evidence = None
            return proposal
        except Exception:
            self.invalidate()
            raise


class HumanInput:
    """Gym intervention callback; explicit activation, never bound to a button.

    Reader lifecycle belongs to the caller (use CompactHID as a context manager).
    Any read/validation error latches this instance; reconnect explicitly with
    a new reader/source and reset the episode. This is not a hardware deadman.
    Previously observed neutral silence keeps zero; stale nonzero input aborts.
    """
    def __init__(self, reader, *, axis_map, left_button, clock=time.monotonic):
        self.reader = reader
        self.gate = LiveInputGate(axis_map=axis_map, left_button=left_button)
        self.clock = clock
        self.active = False
        self.fault = None

    def set_active(self, active):
        if type(active) is not bool:
            raise ValueError('Explicit boolean activation required')
        if self.fault is not None:
            raise RuntimeError('Input fault latched; replace source before recovery')
        if active != self.active:
            self.gate.invalidate()
        self.active = active

    def __call__(self):
        if self.fault is not None:
            raise RuntimeError('Input fault latched; replace source before recovery')
        try:
            frame = self.reader.poll()
            now = self.clock()
            proposal = self.gate.update(frame, now=now)
            if self.active:
                if not self.gate.input_valid:
                    raise RuntimeError('Active human input missing or stale')
                return True, proposal.action
            return False, None
        except Exception as exc:
            self.fault = str(exc)
            self.active = False
            self.gate.invalidate()
            raise


class RotationCheck:
    """Event-driven read-only check, not a calibration of physical robot motion.

    Default mapping comes from the recorded 2026-09-20 device orientation.
    Each stage has its own timeout. Existing freshness gate remains unchanged.
    """
    stages = ('press', 'neutral', 'positive', 'return_positive', 'negative',
              'return_negative', 'release', 'passed')

    def __init__(self, *, axis, started_at, timeout=60., axis_map=(-2, -1, -3),
                 left_button=0):
        if axis not in ('roll', 'pitch', 'yaw'):
            raise ValueError('Unknown rotation axis')
        if not math.isfinite(started_at) or not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError('Finite start and timeout in (0,300] required')
        self.gate = LiveInputGate(axis_map=axis_map, left_button=left_button)
        self.index = {'roll': 3, 'pitch': 4, 'yaw': 5}[axis]
        self.axis = axis
        self.stage = 'press'
        self.reason = 'waiting_for_left_button'
        self.deadline = started_at + timeout
        self.timeout = timeout
        self.last_now = started_at

    def status(self):
        return dict(stage=self.stage, reason=self.reason, axis=self.axis,
                    preview_only=True)

    def fail(self, reason):
        self.gate.invalidate()
        self.stage, self.reason = 'failed', reason
        return self.status()

    def update(self, frame, *, now):
        if self.stage in ('passed', 'failed'):
            return self.status()
        if not math.isfinite(now) or now < self.last_now:
            return self.fail('invalid_clock')
        self.last_now = now
        if now >= self.deadline:
            return self.fail('timeout_at_' + self.stage)
        try:
            proposal = self.gate.update(frame, now=now)
            left = frame.buttons[self.gate.left_button]
            neutral = all(abs(v) <= self.gate.mapper.deadzone for v in frame.axes[:3])
            advance = False
            if self.stage == 'press':
                advance = frame.ready and left
                self.reason = 'waiting_for_left_button_and_axis_reports'
            elif self.stage == 'release':
                advance = not left and neutral and not proposal.blocked
                self.reason = 'release_left_button_and_return_to_center'
            elif not left:
                return self.fail('left_button_released_early')
            elif proposal.blocked or not self.gate.fresh:
                self.reason = 'fresh_neutral_report_required'
            elif self.stage in ('neutral', 'return_positive', 'return_negative'):
                advance = neutral
                self.reason = 'return_to_center'
            else:
                value = proposal.action[self.index]
                others = [abs(v) for i, v in enumerate(proposal.action) if i != self.index]
                if max(others) > .15:
                    self.reason = 'cross_axis_input'
                else:
                    advance = (value >= .2 if self.stage == 'positive' else value <= -.2)
                    self.reason = 'waiting_for_' + self.stage + '_input'
            if advance:
                self.stage = self.stages[self.stages.index(self.stage) + 1]
                self.deadline = now + self.timeout
                self.reason = 'stage_entered'
                if self.stage == 'passed':
                    self.gate.invalidate()
            return dict(self.status(), action=proposal.action, blocked=proposal.blocked)
        except Exception:
            self.fail('invalid_input')
            raise


def main():
    """Preview JSONL input only; does not open HID or import robot interfaces."""
    import argparse
    from dataclasses import asdict
    import json
    import sys
    parser = argparse.ArgumentParser(description='Offline normalized action preview, no motion')
    parser.add_argument('--axis-map', required=True,
                        help='Signed raw axes for forward,left,up; e.g. --axis-map=2,-1,3')
    parser.add_argument('--deadzone', type=float, default=.1)
    args = parser.parse_args()
    mapper = ProposalMapper(axis_map=tuple(int(x) for x in args.axis_map.split(',')),
                            deadzone=args.deadzone)
    for line in sys.stdin:
        row = json.loads(line)
        proposal = mapper.update(row['axes'], left_pressed=row['left_pressed'], valid=row['valid'])
        print(json.dumps(dict(asdict(proposal), preview_only=True)), flush=True)


if __name__ == '__main__':
    main()
