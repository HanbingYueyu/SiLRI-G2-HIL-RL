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
RESET_Y_HIGH_M = .28
RESET_Z_HIGH_M = 1.00
RESET_LINEAR_SPEED_M_S = .04


def reset_only_config(config):
    """Use the post-demo reset's Y/Z ceiling and faster translation speed."""
    motion = config.motion
    original_high = motion.limits.workspace_high
    if original_high[1] > RESET_Y_HIGH_M or original_high[2] > RESET_Z_HIGH_M:
        raise ValueError('Reset-only ceilings must not narrow the configured workspace')
    limits = replace(motion.limits,
                     workspace_high=(original_high[0], RESET_Y_HIGH_M, RESET_Z_HIGH_M))
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
                                skip_tf_progress=True)
        print(f'仅本次预复位使用 Y≤{RESET_Y_HIGH_M:.2f} m、'
              f'Z≤{RESET_Z_HIGH_M:.2f} m；采集/训练边界不变。', flush=True)
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
