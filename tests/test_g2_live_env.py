import numpy as np
import pytest

from g2_local.config import HingeInsertTaskConfig
from g2_local.contract import EpisodeContext
from g2_local.freshness import FreshnessDecision, FreshnessLimits
from g2_local.live_env import create_g2_env


def _limits():
    return FreshnessLimits(
        camera_age_s=.1, state_age_s=.05, camera_skew_s=.05,
        tf_position_error_m=.005, tf_rotation_error_rad=.02,
        mapping_error_s=.005)


class Reader:
    def __init__(self):
        self.closed = False
        self.last_info = {'source': 'fixture'}
        self.observes = 0

    def observe(self):
        self.observes += 1
        return {
            'state': np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float32),
            'left_wrist': np.zeros((8, 8, 3), dtype=np.uint8),
            'right_aux': np.zeros((8, 8, 3), dtype=np.uint8),
        }

    def close(self):
        self.closed = True


class Client:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class Guard:
    def __init__(self):
        self.calls = []
        self.last_decision = FreshnessDecision('not_checked', 'fixture')

    def __call__(self, obs, info, after=None):
        self.calls.append(after)
        self.last_decision = FreshnessDecision('ok', 'fixture')
        return True


def test_create_g2_env_is_read_only_and_does_not_construct_command_port():
    reader = Reader()
    client = Client()
    guard = Guard()
    env = create_g2_env(
        clock_socket='/tmp/fixture-clock.sock', expected_master='fixture-master',
        limits=_limits(), task=HingeInsertTaskConfig(),
        reader_factory=lambda **kwargs: reader,
        client_factory=lambda **kwargs: client,
        guard_factory=lambda client, limits: guard)
    context = EpisodeContext('episode', (0., 0., 0.), 'vision', 'fixed_gripper')
    try:
        observation, info = env.reset(options={'context': context})
        assert observation['state'].shape == (7,)
        assert info['backend'] == 'gdk_read_only'
        assert guard.calls == [None]
        assert not hasattr(env.backend, 'port')
        assert not hasattr(env.backend, 'stream')
        with pytest.raises(PermissionError, match='read-only'):
            env.step(np.zeros(6, dtype=np.float32))
        assert env.runner.active is False
    finally:
        env.close()
    assert reader.closed is True
    assert client.closed is True


def test_create_g2_env_rejects_guard_failure_before_reset_completes():
    reader = Reader()
    client = Client()

    class RejectingGuard(Guard):
        def __call__(self, obs, info, after=None):
            self.calls.append(after)
            self.last_decision = FreshnessDecision('mapping_expired', 'expired')
            return False

    guard = RejectingGuard()
    env = create_g2_env(
        clock_socket='/tmp/fixture-clock.sock', expected_master='fixture-master',
        limits=_limits(), reader_factory=lambda **kwargs: reader,
        client_factory=lambda **kwargs: client,
        guard_factory=lambda client, limits: guard)
    context = EpisodeContext('episode', (0., 0., 0.), 'vision', 'fixed_gripper')
    try:
        with pytest.raises(RuntimeError, match='mapping_expired'):
            env.reset(options={'context': context})
    finally:
        env.close()
