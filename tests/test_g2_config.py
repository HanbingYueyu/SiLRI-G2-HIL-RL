import unittest
from g2_local.config import HingeInsertTaskConfig, LocalTaskConfig


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

    def test_hinge_mvp_defaults_are_explicit_and_motion_limits_stay_required(self):
        config = HingeInsertTaskConfig()
        self.assertEqual(config.control_hz, 10.)
        self.assertEqual(config.max_episode_steps, 80)
        self.assertEqual(config.action_scale[:3], (.0015,) * 3)
        self.assertEqual(config.action_scale[3:], (.026,) * 3)
        self.assertEqual(config.reward_source, 'human')
        self.assertEqual(config.motion_config().action_scale, config.action_scale)
        with self.assertRaises(ValueError):
            config.motion_config().validate_motion()

    def test_hinge_workspace_is_still_supplied_by_operator(self):
        config = HingeInsertTaskConfig()
        motion = config.motion_config(workspace_low=(0, 0, 0),
                                      workspace_high=(1, 1, 1))
        motion.validate_motion()

    def test_hinge_config_rejects_non_mvp_values(self):
        for kwargs in ({'control_hz': 0}, {'max_episode_steps': 0},
                       {'reward_source': 'policy'}, {'action_scale': (0.,) * 6}):
            with self.assertRaises(ValueError):
                HingeInsertTaskConfig(**kwargs)
