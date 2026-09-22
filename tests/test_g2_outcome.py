import numpy as np
import pytest

from g2_local.config import HingeInsertTaskConfig, LocalTaskConfig
from g2_local.contract import EpisodeContext
from g2_local.env import G2LocalEnv
from g2_local.motion_backend import MotionBackend
from g2_local.outcome import HumanBinaryOutcome, OutcomeDecision


class Reader:
    def __init__(self):
        self.state = np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float32)
        self.closed = False

    def observe(self):
        self.last_info = {'captured': 1.}
        return {
            'state': self.state.copy(),
            'left_wrist': np.zeros((8, 8, 3), dtype=np.uint8),
            'right_aux': np.zeros((8, 8, 3), dtype=np.uint8),
        }

    def close(self):
        self.closed = True


class Port:
    def __init__(self):
        self.sent = []
        self.stopped = False

    def send(self, target):
        self.sent.append(target)

    def stop(self):
        self.stopped = True


def _backend(reader, port, outcome):
    return MotionBackend(
        reader, port,
        config=LocalTaskConfig(action_scale=(.01,) * 6,
                                workspace_low=(-1.,) * 3,
                                workspace_high=(1.,) * 3),
        observation_guard=lambda obs, info, after: True,
        outcome=outcome,
        command_timeout=.3, send_timeout=.1, step_period=.02,
        allow_motion=True)


def test_human_binary_outcome_maps_pending_success_and_failure_explicitly():
    labels = iter((None, True, False))
    outcome = HumanBinaryOutcome(
        HingeInsertTaskConfig(success_reward=10., step_reward=-.05),
        lambda observation: next(labels))

    assert outcome(None) == OutcomeDecision(-.05, False, 'human', None)
    assert outcome(None) == OutcomeDecision(10., True, 'human', True)
    assert outcome(None) == OutcomeDecision(-.05, True, 'human', False)


@pytest.mark.parametrize('label', (1, np.bool_(True), 'success'))
def test_human_binary_outcome_rejects_non_boolean_operator_labels(label):
    outcome = HumanBinaryOutcome(HingeInsertTaskConfig(), lambda observation: label)
    with pytest.raises(ValueError, match='boolean or None'):
        outcome(None)


def test_motion_backend_and_gym_preserve_outcome_provenance():
    reader, port = Reader(), Port()
    outcome = lambda observation: OutcomeDecision(10., True, 'human', True)
    driver = _backend(reader, port, outcome)
    env = G2LocalEnv(driver, max_steps=2)
    try:
        env.reset(options={'context': EpisodeContext('e', (0, 0, 0), 'vision', 'grasp')})
        _, reward, done, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert (reward, done, truncated) == (10., True, False)
        assert info['reward_source'] == 'human'
        assert info['success_label'] is True
    finally:
        env.close()


def test_gym_success_uses_explicit_failure_label_not_reward_sign():
    reader, port = Reader(), Port()
    driver = _backend(reader, port,
                      lambda observation: OutcomeDecision(.5, True, 'human', False))
    env = G2LocalEnv(driver, max_steps=2)
    try:
        env.reset(options={'context': EpisodeContext('e', (0, 0, 0), 'vision', 'grasp')})
        _, reward, done, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert (reward, done) == (.5, True)
        assert info['success_label'] is False
        assert info['succeed'] is False
    finally:
        env.close()
