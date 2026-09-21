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
        raise ValueError('Measured pose already outside configured workspace')
    if abs(np.linalg.norm(pose[3:]) - 1.) > .01:
        raise ValueError('Measured quaternion must be unit length')
    scale = np.asarray(config.action_scale)
    position = np.clip(pose[:3] + proposal[:3]*scale[:3], low, high)
    quaternion = (Rotation.from_rotvec(proposal[3:]*scale[3:]) *
                  Rotation.from_quat(pose[3:])).as_quat()
    effective = np.concatenate(((position-pose[:3])/scale[:3], proposal[3:]))
    return tuple(np.concatenate((position, quaternion))), tuple(effective)
