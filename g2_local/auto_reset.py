"""Keep-grasp lift/return outside Gym transitions. Never opens hardware itself."""
from dataclasses import dataclass
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .contract import vector


@dataclass(frozen=True)
class AutoResetConfig:
    enabled: bool = False
    lift_m: float = .05
    linear_speed_m_s: float = .01
    angular_speed_rad_s: float = .2
    position_tolerance_m: float = .001
    rotation_tolerance_rad: float = .01
    timeout_s: float = 30.

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('auto_reset.enabled must be boolean')
        for key in ('lift_m', 'linear_speed_m_s', 'angular_speed_rad_s',
                    'position_tolerance_m', 'rotation_tolerance_rad', 'timeout_s'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'auto_reset.{key} must be positive and finite')
        if self.position_tolerance_m >= self.lift_m:
            raise ValueError('Reset tolerance must be smaller than lift')


def reset_waypoints(current, reference, motion):
    current, reference = vector(current, 7), vector(reference, 7)
    lifted = (*current[:2], current[2]+motion.auto_reset.lift_m, *current[3:])
    for pose in (current, lifted, reference):
        if abs(np.linalg.norm(pose[3:])-1.) > .01:
            raise ValueError('Reset requires unit quaternions')
        if any(not lo <= value <= hi for value, lo, hi in
               zip(pose[:3], motion.limits.workspace_low, motion.limits.workspace_high)):
            raise ValueError('Automatic reset path exceeds absolute workspace')
        if motion.local_envelope is not None:
            motion.local_envelope.check(pose, reference)
    return lifted, reference


def run_reset(env, reference, motion, *, poll, emit=lambda *a, **kw: None):
    """Own/close a fresh commissioned environment; execute no Gym steps or rewards.

    Fixed intermediate lift, then straight Cartesian return with shortest rotation.
    Bounds are not collision checking. Any fault aborts; no retry/re-arm here.
    """
    backend = env.backend
    cfg = motion.auto_reset
    try:
        if not cfg.enabled:
            raise PermissionError('Automatic reset disabled')
        poll()
        observed = backend.observe()
        points = reset_waypoints(observed['state'], reference, motion)
        backend.begin_episode({'state': reference})
        deadline = time.monotonic()+cfg.timeout_s
        scale = np.asarray(motion.limits.action_scale)
        for phase, goal in zip(('lift', 'return'), points):
            emit('reset_phase', phase=phase, target=list(goal))
            while True:
                poll()
                if time.monotonic() >= deadline:
                    raise TimeoutError('Automatic reset timed out')
                pose = np.asarray(observed['state'])
                delta = np.asarray(goal[:3])-pose[:3]
                rotation = (Rotation.from_quat(goal[3:]) *
                            Rotation.from_quat(pose[3:]).inv()).as_rotvec()
                if (np.linalg.norm(delta) <= cfg.position_tolerance_m and
                        np.linalg.norm(rotation) <= cfg.rotation_tolerance_rad):
                    break
                for value, maximum in ((delta, cfg.linear_speed_m_s*backend.step_period),
                                       (rotation, cfg.angular_speed_rad_s*backend.step_period)):
                    value *= min(1., maximum/max(np.linalg.norm(value), 1e-12))
                action = np.concatenate((delta, rotation))/scale
                action /= max(1., float(np.max(np.abs(action))))
                observed = backend.execute(action).observation
        return observed
    finally:
        env.close()
