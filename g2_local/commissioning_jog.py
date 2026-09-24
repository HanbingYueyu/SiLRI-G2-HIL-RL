"""Supervised, no-Gym SpaceMouse micro-jog for position-mode commissioning.

Preview is the default and never constructs a command port.  Motion is only
available with --allow-motion and remains limited to a small neighborhood of
the measured startup pose.  This is not a certified speed limiter or E-stop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np

from .command_stream import CommandStream
from .gdk_backend import GdkCommandPort, validate_pose
from .spacemouse import Proposal


MAX_LINEAR_SPEED_M_S = 0.01
MAX_ANGULAR_SPEED_RAD_S = 0.2
MAX_LOCAL_TRANSLATION_M = 0.15
MAX_LOCAL_ROTATION_RAD = math.radians(90.0)
LOOP_PERIOD_S = 0.02
MAX_LOOP_GAP_S = 0.06
FEEDBACK_LEASE_S = 0.08


@dataclass(frozen=True)
class MicroTarget:
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


def _quat_multiply(left, right):
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )


def _rotation_distance(first, second):
    q0 = np.asarray(first, dtype=float)
    q1 = np.asarray(second, dtype=float)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(q0, q1)))))


def _pose_diagnostics(measured, target):
    measured = np.asarray(validate_pose(measured), dtype=float)
    target_position = np.asarray(target.position_m, dtype=float)
    target_orientation = np.asarray(validate_pose((*target.position_m, *target.orientation_xyzw)), dtype=float)[3:]
    return {
        'measured_orientation_xyzw': measured[3:].tolist(),
        'target_orientation_xyzw': target_orientation.tolist(),
        'measured_target_position_error_mm': float(np.linalg.norm(measured[:3] - target_position) * 1000.0),
        'measured_target_rotation_error_deg': math.degrees(_rotation_distance(measured[3:], target_orientation)),
    }


def make_micro_target(baseline, measured, action, *, dt_s, reference_pose=None):
    """Integrate a bounded step, optionally from the previous commanded target."""
    baseline = np.asarray(validate_pose(baseline), dtype=float)
    measured = np.asarray(validate_pose(measured), dtype=float)
    reference = np.asarray(validate_pose(measured if reference_pose is None else reference_pose), dtype=float)
    action = np.asarray(tuple(float(value) for value in action), dtype=float)
    if action.shape != (6,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.0):
        raise ValueError('Action must contain six finite normalized values')
    if not math.isfinite(dt_s) or not 0.0 < dt_s <= MAX_LOOP_GAP_S:
        raise ValueError('Control-loop interval exceeded its bound')

    position_radius = float(np.linalg.norm(measured[:3] - baseline[:3]))
    rotation_radius = _rotation_distance(measured[3:], baseline[3:])
    if position_radius > MAX_LOCAL_TRANSLATION_M + 1e-12 or rotation_radius > MAX_LOCAL_ROTATION_RAD + 1e-12:
        raise ValueError('Measured pose exceeded the micro-jog local envelope')
    reference_position_radius = float(np.linalg.norm(reference[:3] - baseline[:3]))
    reference_rotation_radius = _rotation_distance(reference[3:], baseline[3:])
    if (reference_position_radius > MAX_LOCAL_TRANSLATION_M + 1e-12 or
            reference_rotation_radius > MAX_LOCAL_ROTATION_RAD + 1e-12):
        raise ValueError('Previous target exceeded the micro-jog local envelope')

    translation = action[:3].copy()
    rotation = action[3:].copy()
    translation_norm = float(np.linalg.norm(translation))
    rotation_norm = float(np.linalg.norm(rotation))
    if translation_norm > 1.0:
        translation /= translation_norm
    if rotation_norm > 1.0:
        rotation /= rotation_norm

    target_position = reference[:3].copy() + translation * MAX_LINEAR_SPEED_M_S * dt_s
    target_orientation = reference[3:].copy()
    rotvec = rotation * MAX_ANGULAR_SPEED_RAD_S * dt_s
    angle = float(np.linalg.norm(rotvec))
    if angle > 0.0:
        delta_q = (*tuple(rotvec * (math.sin(angle / 2.0) / angle)), math.cos(angle / 2.0))
        target_orientation = np.asarray(_quat_multiply(delta_q, target_orientation))
        target_orientation /= np.linalg.norm(target_orientation)

    if (float(np.linalg.norm(target_position - baseline[:3])) > MAX_LOCAL_TRANSLATION_M + 1e-12 or
            _rotation_distance(target_orientation, baseline[3:]) > MAX_LOCAL_ROTATION_RAD + 1e-12):
        raise ValueError('Requested micro-jog would exceed the local envelope')
    return MicroTarget(tuple(float(v) for v in target_position),
                       tuple(float(v) for v in target_orientation))


def select_micro_target(baseline, measured, proposal, *, dt_s, held_target, active_target=None):
    """Integrate active motion from the last target; latch measured pose at neutral."""
    if not proposal.blocked and any(proposal.action):
        reference_pose = measured if active_target is None else _pose_values(active_target)
        target = make_micro_target(
            baseline, measured, proposal.action, dt_s=dt_s,
            reference_pose=reference_pose,
        )
        return target, None, target

    # Keep the live feedback and envelope validation even while holding; only
    # the target is latched, rather than chasing each new measured pose.
    measured_target = make_micro_target(
        baseline, measured, (0.0,) * 6, dt_s=dt_s,
    )
    if held_target is None:
        held_target = measured_target
    return held_target, held_target, None


def _full_axis_map(values):
    values = tuple(values)
    if len(values) == 3:
        values += (-5, -4, -6)
    if (len(values) != 6 or set(map(abs, values[:3])) != {1, 2, 3} or
            set(map(abs, values[3:])) != {4, 5, 6}):
        raise ValueError('axis map must contain signed permutations of 1,2,3 and 4,5,6')
    return values


def parse_axis_map(text):
    try:
        values = tuple(int(part.strip()) for part in text.split(','))
        return _full_axis_map(values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def map_jog_input(frame, *, now, axis_map, left_button, deadzone=0.1, max_age_s=0.1):
    """Use native XYZ; holding the left button also enables native RXYZ."""
    mode = 'six_dof' if frame.buttons[left_button] else 'translation'
    blocked = Proposal((0.0,) * 6, mode, True)
    raw = np.asarray(tuple(float(value) for value in frame.axes), dtype=float)
    stamps = tuple(frame.axis_times)
    if (raw.shape != (6,) or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1.0) or
            len(stamps) != 2):
        raise ValueError('Malformed SpaceMouse input frame')
    if type(frame.buttons[left_button]) is not bool:
        raise ValueError('SpaceMouse button state must be boolean')
    if stamps[0] is None and (not frame.buttons[left_button] or stamps[1] is None):
        return blocked

    mapped = np.asarray([raw[abs(index) - 1] * (1 if index > 0 else -1)
                         for index in _full_axis_map(axis_map)], dtype=float)
    values = np.sign(mapped) * np.maximum(0.0, np.abs(mapped) - deadzone) / (1 - deadzone)
    if not frame.buttons[left_button]:
        values[3:] = 0.0
    for group in range(2):
        selected = values[group * 3:group * 3 + 3]
        if np.any(selected):
            stamp = stamps[group]
            if (stamp is None or not math.isfinite(stamp) or stamp > now or
                    now - stamp > max_age_s):
                raise RuntimeError('SpaceMouse active-axis report is stale; stopping micro-jog')
        magnitude = float(np.linalg.norm(selected))
        if magnitude > 1.0:
            selected /= magnitude
    return Proposal(tuple(float(value) for value in values), mode, False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Supervised SpaceMouse micro-jog; no Gym, preview by default.')
    parser.add_argument('--adapter-root', type=Path, default=Path('/home/flyfuture/g2_hinge_assembly'))
    parser.add_argument('--device', help='SpaceMouse hidraw path; auto-detect if exactly one is connected')
    parser.add_argument('--list-devices', action='store_true')
    parser.add_argument('--axis-map', type=parse_axis_map, default=(-2, -1, -3, -5, -4, -6))
    parser.add_argument('--left-button', type=int, choices=(0, 1), default=0)
    parser.add_argument('--deadzone', type=float, default=0.1)
    parser.add_argument('--duration-s', type=float, default=15.0)
    parser.add_argument('--allow-motion', action='store_true',
                        help='Explicitly enable the supervised micro-jog command stream')
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_s) or args.duration_s < 0.1:
        parser.error('--duration-s must be finite and at least 0.1 seconds')
    if not math.isfinite(args.deadzone) or not 0.0 <= args.deadzone < 0.5:
        parser.error('--deadzone must be in [0,0.5)')
    return args


def _adapter_modules(root):
    root = root.resolve()
    if not (root / 'g2_adapter/control.py').is_file():
        raise FileNotFoundError(f'G2 adapter not found under {root}')
    sys.path.insert(0, str(root))
    from g2_adapter.control import G2Controller
    from g2_adapter.spacemouse_input import CompactHID, discover_devices
    return G2Controller, CompactHID, discover_devices


def _pose_values(pose):
    return tuple((*pose.position_m, *pose.orientation_xyzw))


def next_loop_deadline(previous_deadline, finished_at):
    """Keep cadence without accumulating missed ticks after a slow cycle."""
    return max(previous_deadline + LOOP_PERIOD_S, finished_at)


def _require_supported_motion_mode(controller, *, expected_control_mode=None):
    controller.checked_arm_state()
    status = controller.motion_status_summary()
    if (status['mode'] != 1 or status['control_mode'] not in (1, 3) or
            status['error_code'] != 0 or
            (expected_control_mode is not None and
             status['control_mode'] != expected_control_mode)):
        raise RuntimeError(
            'Refusing micro-jog: expected healthy mode=1 and unchanged '
            f'control_mode in {{1,3}}, got {status}'
        )
    return status


def run(args):
    G2Controller, CompactHID, discover_devices = _adapter_modules(args.adapter_root)
    if args.list_devices:
        print(json.dumps({'devices': discover_devices()}, ensure_ascii=False), flush=True)
        return 0

    import agibot_gdk as gdk

    gdk_initialized = False
    stream = None
    stream_stopped = True
    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    try:
        with CompactHID(args.device) as source:
            # Even a failed initialization may have allocated SDK resources.
            gdk_initialized = True
            if gdk.gdk_init() != gdk.GDKRes.kSuccess:
                raise RuntimeError('GDK initialization failed')
            robot = gdk.Robot()
            time.sleep(0.5)
            controller = G2Controller(gdk, robot, allow_motion=args.allow_motion)
            # This selects validation semantics only; the commissioning jog
            # does not call set_control_mode or change the robot's mode.
            controller.set_position_control_phase(True)
            status = _require_supported_motion_mode(controller)
            command_control_mode = status['control_mode']
            baseline_pose = controller.read_end_effector_pose('arm_l_end_link')
            baseline = _pose_values(baseline_pose)
            validate_pose(baseline)
            print(json.dumps({
                'event': 'ready', 'operation': 'no_gym_spacemouse_micro_jog',
                'permission': 'motion_enabled' if args.allow_motion else 'preview_no_commands',
                'device': source.path, 'control_status': status,
                'control_mode_policy': 'use_existing_mode_1_or_3; never_switch_mode',
                'input_policy': 'XYZ_without_left_button; XYZ_plus_native_rotation_with_left_button',
                'effective_axis_map': args.axis_map,
                'baseline_xyz_m': baseline[:3],
                'limits': {'linear_setpoint_m_s': MAX_LINEAR_SPEED_M_S,
                           'angular_setpoint_deg_s': math.degrees(MAX_ANGULAR_SPEED_RAD_S),
                           'translation_radius_mm': 150.0,
                           'rotation_radius_deg': 90.0,
                           'combined_axes_allowed': True,
                           'translation_speed_is_vector_capped': True},
                'note': 'Envelope is measured from startup pose; software setpoint limits are not certified physical speed or stopping-distance guarantees.'
            }, ensure_ascii=False, allow_nan=False), flush=True)

            last_loop = time.monotonic()
            last_report = -math.inf
            held_target = None
            active_target = None
            feedback = {'received_at': None, 'status': None}

            if args.allow_motion:
                def feedback_fresh():
                    received_at = feedback['received_at']
                    observed_status = feedback['status']
                    return bool(received_at is not None and
                                time.monotonic() - received_at <= FEEDBACK_LEASE_S and
                                observed_status is not None and
                                observed_status.get('mode') == 1 and
                                observed_status.get('control_mode') == command_control_mode and
                                observed_status.get('error_code') == 0)

                port = GdkCommandPort(controller, expected_mode=command_control_mode, allow_motion=True,
                                      freshness_guard=feedback_fresh, life_time_s=0.08)
                stream = CommandStream(port, command_timeout=0.10, send_timeout=0.07,
                                       stop_timeout=1.0, rate_hz=50.0)
                stream_stopped = False

            started = time.monotonic()
            next_tick = started
            while time.monotonic() - started < args.duration_s:
                now = time.monotonic()
                if now - next_tick > MAX_LOOP_GAP_S:
                    raise RuntimeError('Control loop overrun; stopping micro-jog')
                frame = source.poll()
                proposal = map_jog_input(frame, now=time.monotonic(), axis_map=args.axis_map,
                                          left_button=args.left_button, deadzone=args.deadzone)
                current_status = _require_supported_motion_mode(
                    controller, expected_control_mode=command_control_mode,
                )
                measured_pose = controller.read_end_effector_pose('arm_l_end_link')
                measured = _pose_values(measured_pose)
                feedback['received_at'] = time.monotonic()
                feedback['status'] = current_status
                dt_s = max(0.001, min(MAX_LOOP_GAP_S, feedback['received_at'] - last_loop))
                target, held_target, active_target = select_micro_target(
                    baseline, measured, proposal, dt_s=dt_s,
                    held_target=held_target, active_target=active_target,
                )

                if stream is not None:
                    sequence = stream.submit(target)
                    stream.wait_sent(sequence, timeout=0.07)
                if feedback['received_at'] - last_report >= 0.2:
                    print(json.dumps({
                        'event': 'jog' if any(proposal.action) and not proposal.blocked else 'hold_or_armed',
                        'input_mode': proposal.mode, 'blocked': proposal.blocked,
                        'raw_axes': frame.axes, 'buttons': frame.buttons,
                        'action': proposal.action,
                        'measured_xyz_m': measured[:3], 'target_xyz_m': target.position_m,
                        **_pose_diagnostics(measured, target),
                        'command_stream_enabled': stream is not None,
                        'gdk_send_acknowledged': stream is not None,
                    }, ensure_ascii=False, allow_nan=False), flush=True)
                    last_report = feedback['received_at']

                last_loop = feedback['received_at']
                next_tick = next_loop_deadline(next_tick, time.monotonic())
                time.sleep(max(0.0, next_tick - time.monotonic()))
            return 0
    finally:
        try:
            if stream is not None:
                stream.stop()
                stream_stopped = True
        finally:
            if gdk_initialized and stream_stopped:
                gdk.gdk_release()
            elif gdk_initialized:
                print('CRITICAL: GDK SDK call did not stop cleanly; release withheld. Use hardware E-stop if motion persists.',
                      file=sys.stderr, flush=True)
            signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv=None):
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print('Operator interrupted; command stream stopping at measured pose.', flush=True)
        return 130
    except Exception as exc:
        print(f'Micro-jog refused/stopped: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
