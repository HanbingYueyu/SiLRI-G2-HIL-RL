"""Guarded, transition-free +Z/+Y reposition before upstream visual reset."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .contract import vector
from .motion_env import create_motion_env
from .training_config import load_training_config


LIFT_M = .05
SHIFT_Y_M = .10
# Tolerance so a start pose exactly on the workspace edge can still complete the
# full retraction (floating-point and controller settling).
RESET_MARGIN_M = .005
RESET_LINEAR_SPEED_M_S = .04


def reset_only_high(motion):
    """Reset-only ceilings: the configured workspace plus exactly this retraction.

    The +Z 50 mm / +Y 100 mm motion is fixed, slow (40 mm/s) and scoped to the
    reset; deriving its ceiling from the *approved* workspace means any start
    pose that is valid for the demonstration/training workspace can finish it.
    Fixed 0.28/1.00 ceilings used to refuse any start within 65 mm (Y) or 15 mm
    (Z) of the workspace edge, which blocked domain-randomized start poses:

        y=0.183 + 0.100 = 0.283 > 0.280  ->  ValueError, no visual reset

    Derived values for the current workspace (high = 0.6665, 0.2451, 0.965):
    Y = 0.2451 + 0.105 = 0.3501 m, Z = 0.965 + 0.055 = 1.020 m. The normal
    command path still uses the unchanged workspace limits; only this
    reset-scoped motion sees the wider ceiling.
    """
    high = motion.limits.workspace_high
    return (high[0], high[1] + SHIFT_Y_M + RESET_MARGIN_M,
            high[2] + LIFT_M + RESET_MARGIN_M)


def reset_only_config(config):
    """Use the post-demo reset's Y/Z ceiling and faster translation speed."""
    motion = config.motion
    original_high = motion.limits.workspace_high
    reset_high = reset_only_high(motion)
    if (any(reset < configured for reset, configured in
            zip(reset_high, original_high))):
        raise ValueError('Reset-only ceilings must not narrow the configured workspace')
    limits = replace(motion.limits, workspace_high=reset_high)
    limits.validate_motion()
    reset = replace(motion.auto_reset, linear_speed_m_s=RESET_LINEAR_SPEED_M_S)
    return replace(config, motion=replace(motion, limits=limits, auto_reset=reset))


def pre_reset_waypoints(current, motion):
    """Return +Z 50 mm, then +base_link-Y 100 mm, prechecking the full path."""
    current = vector(current, 7)
    motion.limits.validate_motion()
    if abs(np.linalg.norm(current[3:]) - 1.) > .01:
        raise ValueError('Pre-reset requires a unit measured quaternion')
    points = (
        (*current[:2], current[2] + LIFT_M, *current[3:]),
        (current[0], current[1] + SHIFT_Y_M,
         current[2] + LIFT_M, *current[3:]),
    )
    reference = current
    for index, pose in enumerate((current, *points)):
        if any(value < low or value > high for value, low, high in zip(
                pose[:3], motion.limits.workspace_low, motion.limits.workspace_high)):
            stage = ('current' if index == 0 else
                     'lift_z_50mm' if index == 1 else 'translate_y_100mm')
            raise ValueError(
                f'Pre-reset {stage} target {tuple(pose[:3])} exceeds absolute '
                f'workspace low={motion.limits.workspace_low} '
                f'high={motion.limits.workspace_high}')
        if motion.local_envelope is not None:
            motion.local_envelope.check(pose, reference)
    return points


def run_pre_reset(env, motion, *, start_limits=None, emit=print, monotonic=time.monotonic):
    """Execute the fixed path through the commissioned backend, with no RL steps."""
    backend = env.backend
    reset = motion.auto_reset
    scale = np.asarray(motion.limits.action_scale, dtype=np.float64)
    try:
        observed = backend.observe()
        initial = vector(observed['state'], 7)
        if start_limits is not None and any(
                value < low or value > high for value, low, high in zip(
                    initial[:3], start_limits.workspace_low, start_limits.workspace_high)):
            raise ValueError('Pre-reset start lies outside the normal demo workspace')
        points = pre_reset_waypoints(initial, motion)
        # Latch the live start pose so the existing local workspace guard covers
        # both targets and every intermediate commanded pose.
        backend.begin_episode({'state': initial})
        deadline = monotonic() + reset.timeout_s
        for phase, target in zip(('lift_z_50mm', 'translate_y_100mm'), points):
            emit(f'自动复位：开始 {phase}，目标 base_link XYZ={target[:3]} m')
            while True:
                if monotonic() >= deadline:
                    raise TimeoutError('Pre-reset movement timed out')
                position = np.asarray(observed['state'][:3], dtype=np.float64)
                delta = np.asarray(target[:3], dtype=np.float64) - position
                distance = float(np.linalg.norm(delta))
                if distance <= reset.position_tolerance_m:
                    break
                max_delta = reset.linear_speed_m_s * backend.step_period
                delta *= min(1., max_delta / max(distance, 1e-12))
                action = np.zeros(6, dtype=np.float64)
                action[:3] = delta / scale[:3]
                orientation_delta = (Rotation.from_quat(initial[3:]) *
                                     Rotation.from_quat(observed['state'][3:]).inv())
                action[3:] = np.clip(orientation_delta.as_rotvec() / scale[3:], -1., 1.)
                if np.any(np.abs(action[:3]) > 1. + 1e-9):
                    raise RuntimeError('Pre-reset speed exceeds normalized action limit')
                result = backend.execute(action)
                observed = result.observation
        emit('自动复位：+Z 50 mm、再 +Y 100 mm 已完成。')
        # The retraction is allowed past the workspace ceiling, so the arm now
        # commonly rests OUTSIDE the demo/training workspace. Say so explicitly:
        # starting an episode from there is refused by `plan_target`
        # ("Measured pose already outside configured workspace"), and the fix is
        # to let the Section 6 upstream reset bring the arm back to the start
        # pose first.
        position = np.asarray(observed['state'][:3], dtype=np.float64)
        low = np.asarray(motion.limits.workspace_low, dtype=np.float64)
        high = np.asarray(motion.limits.workspace_high, dtype=np.float64)
        outside = [axis for axis in range(3) if position[axis] < low[axis]
                   or position[axis] > high[axis]]
        if outside:
            detail = ', '.join(
                f'{"xyz"[axis]}={position[axis]:.4f} '
                f'{"<" if position[axis] < low[axis] else ">"} '
                f'{low[axis] if position[axis] < low[axis] else high[axis]:.4f}'
                for axis in outside)
            emit(f'注意：机械臂现在停在**采集工作空间之外**（{detail}）。'
                 '请先跑《常用命令.md》第 6 节把臂带回起始位姿，再按双键开始下一回合；'
                 '否则会报 Measured pose already outside configured workspace。')
        else:
            emit(f'机械臂当前位姿 {np.round(position, 4).tolist()} 仍在采集工作空间内。')
        return observed
    finally:
        env.close()


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config', required=True, type=Path)
    cli.add_argument('--allow-motion', action='store_true')
    return cli


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        config = load_training_config(args.config, cli_allow_motion=args.allow_motion)
        if not args.allow_motion or not config.motion_permitted:
            raise PermissionError('Pre-reset requires commissioned --allow-motion permission')
        reset_config = reset_only_config(config)
        coordinator = SimpleNamespace(outcome=lambda observation: (0., False),
                                      intervention=None)
        env = create_motion_env(reset_config, coordinator,
                                cli_allow_motion=args.allow_motion,
                                skip_tf_progress=True, skip_state_progress=True)
        reset_high = reset_only_high(config.motion)
        print(f'仅本次预复位使用 Y≤{reset_high[1]:.3f} m、Z≤{reset_high[2]:.3f} m'
              f'（= 工作空间上限 + 本次位移 + {RESET_MARGIN_M*1000:.0f} mm 裕度）；'
              '采集/训练边界不变。', flush=True)
        run_pre_reset(env, reset_config.motion, start_limits=config.motion.limits)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f'自动预复位失败：{type(error).__name__}: {error}',
              file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
