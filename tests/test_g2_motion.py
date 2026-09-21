import math
import numpy as np
import pytest
from g2_local.config import LocalTaskConfig


def config():
    return LocalTaskConfig(action_scale=(.01,.01,.01,.1,.1,.1),
                           workspace_low=(-1,-1,-1), workspace_high=(1,1,1))


def test_workspace_clipping_records_effective_action():
    from g2_local.motion import plan_target
    target, executed = plan_target((.995,0,0,0,0,0,1), (1,0,0,0,0,0), config())
    assert target[:3] == pytest.approx((1,0,0))
    assert executed == pytest.approx((.5,0,0,0,0,0))


def test_rotation_increment_is_in_base_frame():
    from g2_local.motion import plan_target
    from scipy.spatial.transform import Rotation
    initial = Rotation.from_euler('z', math.pi/2)
    target, _ = plan_target((0,0,0,*initial.as_quat()), (0,0,0,1,0,0), config())
    actual = Rotation.from_quat(target[3:]).apply([0,0,1])
    assert actual == pytest.approx([0,-math.sin(.1),math.cos(.1)])


@pytest.mark.parametrize('pose', [(2,0,0,0,0,0,1), (0,0,0,0,0,0,0)])
def test_invalid_measured_pose_is_rejected(pose):
    from g2_local.motion import plan_target
    with pytest.raises(ValueError):
        plan_target(pose, (0,)*6, config())


def test_no_implicit_physical_limits():
    from g2_local.motion import plan_target
    with pytest.raises(ValueError):
        plan_target((0,0,0,0,0,0,1), (0,)*6, LocalTaskConfig())
