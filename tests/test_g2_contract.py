import unittest

from g2_local.contract import EpisodeContext, select_action, transition


class ContractTests(unittest.TestCase):
    def test_zero_human_action_retains_control(self):
        decision = select_action([.5] * 6, human_active=True, human=[0] * 6)
        self.assertEqual(decision.selected_action, (0.0,) * 6)
        self.assertTrue(decision.is_intervention)

    def test_clip_preserves_proposal(self):
        decision = select_action([2] * 6)
        self.assertEqual(decision.policy_proposal, (2.0,) * 6)
        self.assertEqual(decision.selected_action, (1.0,) * 6)

    def test_invalid_action_rejected(self):
        for action in ([0] * 7, [float('nan')] * 6):
            with self.assertRaises(ValueError):
                select_action(action)

    def test_transition_uses_acknowledged_execution(self):
        decision = select_action([.8] * 6)
        row = transition('before', 'after', decision, [0.2] * 6,
                         reward=0, terminated=False, truncated=True,
                         reward_source='human', success_label=False)
        self.assertEqual(row['action'], (.2,) * 6)
        self.assertFalse(row['done'])
        self.assertTrue(row['truncated'])
        metadata = row['complementary_info']
        self.assertEqual(metadata['policy_action'], (.8,) * 6)
        self.assertIsNone(metadata['human_action'])
        self.assertEqual(metadata['executed_action'], (.2,) * 6)
        self.assertEqual(metadata['reward_source'], 'human')
        self.assertFalse(metadata['success_label'])

    def test_invalid_successor_never_becomes_training_data(self):
        with self.assertRaises(ValueError):
            transition('before', None, select_action([0] * 6), [0] * 6,
                       reward=0, terminated=False, truncated=False)

    def test_episode_randomization_is_metadata(self):
        context = EpisodeContext('episode-1', (.05, 0, 0), 'visual_reapproach', 'unknown')
        self.assertEqual(context.target_offset_m, (.05, 0, 0))
        self.assertEqual(context.ee_reset_offset, (0.,) * 6)
        with self.assertRaises(ValueError):
            EpisodeContext('', (0, 0, 0), 'visual_reapproach', 'unknown')

    def test_episode_randomization_keeps_pose_offset_separate(self):
        context = EpisodeContext('episode-2', (.005, -.002, 0), 'visual', 'unknown',
                                (.001, -.001, .002, 0., 0., .01))
        self.assertEqual(context.ee_reset_offset[2], .002)
        with self.assertRaises(ValueError):
            EpisodeContext('episode-3', (0, 0, 0), 'visual', 'unknown', (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
