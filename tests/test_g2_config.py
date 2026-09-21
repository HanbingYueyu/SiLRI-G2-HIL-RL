import unittest
from g2_local.config import LocalTaskConfig


class ConfigTests(unittest.TestCase):
    def test_default_cannot_enable_motion(self):
        with self.assertRaises(ValueError):
            LocalTaskConfig().validate_motion()

    def test_explicit_limits_required(self):
        config = LocalTaskConfig(action_scale=(.001,) * 6,
                                 workspace_low=(0, 0, 0),
                                 workspace_high=(1, 1, 1))
        config.validate_motion()

    def test_bad_limits_rejected(self):
        for scale in ((0,) * 6, (float('nan'),) * 6):
            with self.assertRaises(ValueError):
                LocalTaskConfig(action_scale=scale,
                                workspace_low=(0, 0, 0),
                                workspace_high=(1, 1, 1)).validate_motion()

    def test_duplicate_cameras_rejected(self):
        with self.assertRaises(ValueError):
            LocalTaskConfig(camera_keys=('left_wrist', 'left_wrist'))
