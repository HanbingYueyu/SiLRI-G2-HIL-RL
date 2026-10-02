"""Offline base-frame target planning. This module never connects to hardware."""
import numpy as np
from scipy.spatial.transform import Rotation
from .contract import vector


def plan_target(measured_pose, action, config):
    """Return target XYZ/XYZW and effective normalized candidate after clipping.

    The second value is NOT acknowledged/executed until a driver accepts it.
    Rotations are left-multiplied base-frame increments. Collision, orientation
    envelope, velocity, and timestamp checks remain the driver's responsibility.
    """
    config.validate_motion()
    pose = np.asarray(vector(measured_pose, 7))
    proposal = np.clip(vector(action, 6), -1., 1.)
    low, high = np.asarray(config.workspace_low), np.asarray(config.workspace_high)
    if np.any(pose[:3] < low) or np.any(pose[:3] > high):
        # Name the offending axis and the overage: this is what an operator sees
        # when the arm was left parked outside the workspace (for example by the
        # post-episode pre-reset, which is allowed past the workspace ceiling but
        # must be followed by the Section 6 reset before a new episode starts).
        detail = ', '.join(
            f'{"xyz"[axis]}={pose[axis]:.4f} {"<" if pose[axis] < low[axis] else ">"} '
            f'{"limit"}= {low[axis] if pose[axis] < low[axis] else high[axis]:.4f} '
            f'(over {abs(pose[axis] - (low[axis] if pose[axis] < low[axis] else high[axis])):.4f} m)'
            for axis in range(3)
            if pose[axis] < low[axis] or pose[axis] > high[axis])
        raise ValueError(
            'Measured pose already outside configured workspace: ' + detail
            + f'; pose_xyz={np.round(pose[:3], 4).tolist()} '
            + f'workspace_low={low.tolist()} workspace_high={high.tolist()}')
    if abs(np.linalg.norm(pose[3:]) - 1.) > .01:
        raise ValueError('Measured quaternion must be unit length')
    scale = np.asarray(config.action_scale)
    position = np.clip(pose[:3] + proposal[:3]*scale[:3], low, high)
    quaternion = (Rotation.from_rotvec(proposal[3:]*scale[3:]) *
                  Rotation.from_quat(pose[3:])).as_quat()
    # Subtracting nearby positions can turn a valid full-scale input into
    # 1.0000000000000009. Bound only numerical roundoff at this planning exit;
    # transition validation remains strict for arbitrary driver outputs.
    translation = (position-pose[:3])/scale[:3]
    if np.any(np.abs(translation) > 1. + 1e-10):
        raise ValueError('Effective translation exceeds normalized bounds')
    effective = np.concatenate((np.clip(translation, -1., 1.), proposal[3:]))
    return tuple(np.concatenate((position, quaternion))), tuple(effective)
