import math
import pytest
from scipy.spatial.transform import Rotation
from g2_local.local_envelope import LocalEnvelope


def test_relative_translation_rotation_and_quaternion_sign():
    limits = LocalEnvelope((-.02, -.035, -.09), (.07, .02, .02), math.radians(25))
    start = (.55, .17, .92, 0, 0, 0, 1)
    limits.check((.595, .159, .855, 0, 0, 0, -1), start)
    with pytest.raises(ValueError, match='translation'):
        limits.check((.621, .17, .92, 0, 0, 0, 1), start)
    with pytest.raises(ValueError, match='rotation'):
        limits.check((*start[:3], *Rotation.from_euler('z', 26, degrees=True).as_quat()), start)
    shifted = (.65, .27, .92, 0, 0, 0, 1)
    limits.check((.695, .259, .855, 0, 0, 0, 1), shifted)


def test_invalid_limits_rejected():
    with pytest.raises(ValueError):
        LocalEnvelope((.01, -.02, -.02), (.03, .03, .03), .4)
    with pytest.raises(ValueError):
        LocalEnvelope((-.02,)*3, (.03,)*3, float('nan'))
