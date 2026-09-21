import unittest
from g2_local.contract import EpisodeContext
from g2_local.episode import EpisodeRunner, StepResult


class Backend:
    def __init__(self):
        self.stops = 0
        self.actions = []
        self.fail = False

    def observe(self):
        return {'frame': 0}

    def execute(self, action):
        self.actions.append(action)
        if self.fail:
            raise RuntimeError('command rejected')
        return StepResult({'frame': len(self.actions)}, tuple(x / 2 for x in action), 0, False)

    def stop(self):
        self.stops += 1


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.runner = EpisodeRunner(self.backend, max_steps=2)
        self.context = EpisodeContext('e1', (.05, 0, 0), 'visual', 'perturbed')

    def test_reset_required(self):
        with self.assertRaises(RuntimeError):
            self.runner.step([0] * 6)
        self.assertEqual(self.backend.actions, [])

    def test_actual_action_and_episode_metadata(self):
        self.runner.reset(self.context)
        row = self.runner.step([1] * 6)
        self.assertEqual(row['action'], (.5,) * 6)
        self.assertEqual(row['complementary_info']['target_offset_m'], (.05, 0, 0))
        self.assertEqual(row['complementary_info']['step_id'], 0)

    def test_intervention_is_cumulative_and_timeout_keeps_final_obs(self):
        self.runner.reset(self.context)
        self.runner.step([1] * 6, human_active=True, human=[0] * 6)
        row = self.runner.step([0] * 6)
        self.assertTrue(row['truncated'])
        self.assertFalse(row['done'])
        self.assertTrue(row['complementary_info']['episode_assisted'])
        self.assertEqual(row['next_state'], {'frame': 2})
        with self.assertRaises(RuntimeError):
            self.runner.step([0] * 6)

    def test_failure_stops_and_requires_reset(self):
        self.runner.reset(self.context)
        self.backend.fail = True
        with self.assertRaisesRegex(RuntimeError, 'command rejected'):
            self.runner.step([0] * 6)
        self.assertEqual(self.backend.stops, 1)
        with self.assertRaises(RuntimeError):
            self.runner.step([0] * 6)
        self.assertEqual(len(self.backend.actions), 1)

    def test_reset_clears_assistance_and_does_not_move(self):
        self.runner.reset(self.context)
        self.runner.step([0] * 6, human_active=True, human=[0] * 6)
        self.runner.reset(EpisodeContext('e2', (0, 0, 0), 'visual', 'unknown'))
        row = self.runner.step([0] * 6)
        self.assertFalse(row['complementary_info']['episode_assisted'])
        self.assertEqual(row['state'], {'frame': 0})

    def test_invalid_budget(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                EpisodeRunner(self.backend, max_steps=value)
