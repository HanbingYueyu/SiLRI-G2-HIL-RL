import math
from types import SimpleNamespace

import pytest


def test_micro_jog_preserves_simultaneous_axes_within_the_fresh_hid_group():
    from g2_local.commissioning_jog import map_jog_input

    frame = SimpleNamespace(
        axes=(0.2, -0.55, 0.0, 0.25, -0.5, 0.75),
        buttons=(False, False),
        axis_times=(9.98, 8.0),
        ready=True,
    )
    proposal = map_jog_input(frame, now=10.0, axis_map=(-2, -1, -3),
                             left_button=0, deadzone=0.1)

    assert proposal.blocked is False
    assert proposal.action == pytest.approx((0.5, -1 / 9, 0.0, 0.0, 0.0, 0.0))


def test_left_button_enables_native_rotation_without_disabling_translation():
    from g2_local.commissioning_jog import map_jog_input

    frame = SimpleNamespace(
        axes=(0.2, -0.55, 0.0, 0.25, -0.5, 0.75),
        buttons=(True, False),
        axis_times=(9.98, 9.99),
        ready=True,
    )
    proposal = map_jog_input(frame, now=10.0, axis_map=(-2, -1, -3, -5, -4, -6),
                             left_button=0, deadzone=0.1)

    assert proposal.blocked is False
    assert proposal.action == pytest.approx((0.5, -1 / 9, 0.0, 4 / 9, -1 / 6, -13 / 18))


def test_legacy_three_axis_cli_map_expands_to_native_six_axes():
    from g2_local.commissioning_jog import parse_args

    args = parse_args(['--axis-map=-2,-1,-3'])

    assert args.axis_map == (-2, -1, -3, -5, -4, -6)


def test_micro_target_integrates_translation_and_rotation_together():
    from g2_local.commissioning_jog import make_micro_target

    baseline = (0.3, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0)
    target = make_micro_target(
        baseline, baseline, (1.0, 0.0, 0.0, 0.0, 0.0, 1.0), dt_s=0.02,
    )

    assert target.position_m == pytest.approx((0.3002, 0.2, 0.8))
    assert target.orientation_xyzw == pytest.approx((0.0, 0.0, math.sin(0.002), math.cos(0.002)))


def test_fresh_motion_axes_control_immediately_without_startup_button_report():
    from g2_local.commissioning_jog import map_jog_input

    frame = SimpleNamespace(
        axes=(0.0, 0.0, -0.55, 0.0, 0.0, 0.0),
        buttons=(False, False),
        axis_times=(9.99, None),
        ready=False,
    )
    proposal = map_jog_input(frame, now=10.0, axis_map=(-2, -1, -3),
                             left_button=0, deadzone=0.1)

    assert proposal.blocked is False
    assert proposal.action == pytest.approx((0.0, 0.0, 0.5, 0.0, 0.0, 0.0))


def test_micro_target_limits_translation_and_base_frame_rotation():
    from g2_local.commissioning_jog import make_micro_target

    baseline = (0.3, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0)
    target = make_micro_target(
        baseline,
        baseline,
        (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        dt_s=0.02,
    )
    assert target.position_m == pytest.approx((0.3, 0.2, 0.8))
    angle = 2 * math.acos(min(1.0, abs(target.orientation_xyzw[3])))
    assert angle == pytest.approx(0.2 * 0.02)

    translated = make_micro_target(
        baseline,
        baseline,
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        dt_s=0.02,
    )
    assert translated.position_m == pytest.approx((0.3002, 0.2, 0.8))
    next_target = make_micro_target(
        baseline,
        (*translated.position_m, *baseline[3:]),
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        dt_s=0.02,
    )
    assert next_target.position_m == pytest.approx((0.3004, 0.2, 0.8))

    lag_angle = math.radians(0.12)
    lagged_feedback = (*translated.position_m, 0.0, 0.0,
                       math.sin(lag_angle / 2), math.cos(lag_angle / 2))
    rebased = make_micro_target(
        baseline,
        lagged_feedback,
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        dt_s=0.02,
    )
    assert rebased.position_m == pytest.approx(translated.position_m)
    assert _quaternion_distance_deg(rebased.orientation_xyzw, lagged_feedback[3:]) == pytest.approx(math.degrees(0.2 * 0.02))

    diagonal = make_micro_target(
        baseline, baseline, (1.0, 1.0, 0.0, 0.0, 0.0, 0.0), dt_s=0.02,
    )
    expected_component = 0.01 * 0.02 / math.sqrt(2)
    assert diagonal.position_m == pytest.approx((0.3 + expected_component, 0.2 + expected_component, 0.8))


def test_micro_target_uses_the_commissioning_test_envelope():
    from g2_local.commissioning_jog import make_micro_target

    baseline = (0.3, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0)
    inside = (0.449, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0)
    target = make_micro_target(baseline, inside, (0, 0, 0, 0, 0, 0), dt_s=0.02)
    assert target.position_m == pytest.approx(inside[:3])

    outside = (0.451, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match='envelope'):
        make_micro_target(baseline, outside, (0, 0, 0, 0, 0, 0), dt_s=0.02)

    almost_ninety_deg = math.radians(89.9)
    inside_rotation = (*baseline[:3], math.sin(almost_ninety_deg / 2), 0.0, 0.0,
                       math.cos(almost_ninety_deg / 2))
    target = make_micro_target(baseline, inside_rotation, (0, 0, 0, 0, 0, 0), dt_s=0.02)
    assert target.orientation_xyzw == pytest.approx(inside_rotation[3:])

    over_ninety_deg = math.radians(90.1)
    outside_rotation = (*baseline[:3], math.sin(over_ninety_deg / 2), 0.0, 0.0,
                        math.cos(over_ninety_deg / 2))
    with pytest.raises(ValueError, match='envelope'):
        make_micro_target(baseline, outside_rotation, (0, 0, 0, 0, 0, 0), dt_s=0.02)


def test_micro_jog_diagnostics_expose_measured_and_target_orientation():
    from g2_local.commissioning_jog import MicroTarget, _pose_diagnostics

    measured = (0.3, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0)
    target = MicroTarget((0.3, 0.2, 0.8), (0.0, 0.0, math.sin(0.005), math.cos(0.005)))

    diagnostics = _pose_diagnostics(measured, target)

    assert diagnostics['measured_orientation_xyzw'] == pytest.approx(measured[3:])
    assert diagnostics['target_orientation_xyzw'] == pytest.approx(target.orientation_xyzw)
    assert diagnostics['measured_target_rotation_error_deg'] == pytest.approx(math.degrees(0.01))


def test_neutral_micro_jog_latches_hold_target_until_next_active_input():
    from g2_local.commissioning_jog import MicroTarget, select_micro_target
    from g2_local.spacemouse import Proposal

    baseline = (0.3, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0)
    first_feedback = (0.301, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0)
    later_feedback = (0.302, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0)
    neutral = Proposal((0.0,) * 6, 'translation', False)

    target, held, active_target = select_micro_target(
        baseline, first_feedback, neutral, dt_s=0.02, held_target=None, active_target=None,
    )
    assert target == MicroTarget(first_feedback[:3], first_feedback[3:])
    assert held == target

    target, held, active_target = select_micro_target(
        baseline, later_feedback, neutral, dt_s=0.02, held_target=held,
        active_target=active_target,
    )
    assert target == MicroTarget(first_feedback[:3], first_feedback[3:])
    assert held == target

    active = Proposal((1.0, 0.0, 0.0, 0.0, 0.0, 0.0), 'translation', False)
    moving_target, held, active_target = select_micro_target(
        baseline, later_feedback, active, dt_s=0.02, held_target=held,
        active_target=active_target,
    )
    assert moving_target.position_m == pytest.approx((0.3022, 0.2, 0.8))
    assert held is None

    post_motion_feedback = (0.30205, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0)
    target, held, active_target = select_micro_target(
        baseline, post_motion_feedback, neutral, dt_s=0.02, held_target=held,
        active_target=active_target,
    )
    assert target == MicroTarget(post_motion_feedback[:3], post_motion_feedback[3:])
    assert held == target


def test_active_micro_jog_accumulates_from_last_command_and_neutral_rebases_to_feedback():
    from g2_local.commissioning_jog import MicroTarget, select_micro_target
    from g2_local.spacemouse import Proposal

    baseline = (0.3, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0)
    measured = baseline
    active = Proposal((1.0, 0.0, 0.0, 0.0, 0.0, 0.0), 'translation', False)
    neutral = Proposal((0.0,) * 6, 'translation', False)

    first, held, active_target = select_micro_target(
        baseline, measured, active, dt_s=0.02, held_target=None, active_target=None,
    )
    second, held, active_target = select_micro_target(
        baseline, measured, active, dt_s=0.02, held_target=held,
        active_target=active_target,
    )
    assert first.position_m == pytest.approx((0.3002, 0.2, 0.8))
    assert second.position_m == pytest.approx((0.3004, 0.2, 0.8))

    stopped_feedback = (0.30005, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0)
    hold, held, active_target = select_micro_target(
        baseline, stopped_feedback, neutral, dt_s=0.02, held_target=held,
        active_target=active_target,
    )
    assert hold == MicroTarget(stopped_feedback[:3], stopped_feedback[3:])
    assert held == hold
    assert active_target is None


def test_commissioning_jog_accepts_existing_joint_impedance_mode_without_switching():
    from g2_local.commissioning_jog import _require_supported_motion_mode

    class Controller:
        def __init__(self):
            self.control_mode = 3
            self.mode_changes = []

        def checked_arm_state(self):
            return ()

        def motion_status_summary(self):
            return {
                'mode': 1, 'control_mode': self.control_mode,
                'error_code': 0, 'error_msg': '',
            }

        def set_control_mode(self, value):
            self.mode_changes.append(value)

    controller = Controller()

    status = _require_supported_motion_mode(controller)

    assert status['control_mode'] == 3
    assert controller.mode_changes == []


def test_commissioning_jog_accepts_150_second_supervised_run():
    from g2_local.commissioning_jog import parse_args

    assert parse_args(['--duration-s', '150']).duration_s == 150.0


def test_loop_schedule_rebases_after_a_slow_cycle_instead_of_accumulating_lag():
    from g2_local.commissioning_jog import next_loop_deadline

    assert next_loop_deadline(10.02, 10.055) == pytest.approx(10.055)


def _quaternion_distance_deg(first, second):
    import numpy as np

    q1 = np.asarray(first, dtype=float)
    q2 = np.asarray(second, dtype=float)
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q1, q2))))))
