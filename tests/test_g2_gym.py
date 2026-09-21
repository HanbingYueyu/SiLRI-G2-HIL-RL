import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from g2_local.config import LocalTaskConfig
from g2_local.contract import EpisodeContext
from g2_local.env import G2LocalEnv, SyntheticBackend
from g2_local.freshness import FreshnessDecision
from g2_local.motion_backend import MotionBackend


def test_gym_contract():
    env = G2LocalEnv(SyntheticBackend(), max_steps=3)
    check_env(env, skip_render_check=True)


def test_timeout_and_executed_action():
    env = G2LocalEnv(SyntheticBackend(), max_steps=1)
    obs, info = env.reset(seed=10)
    assert obs['left_wrist'].dtype == np.uint8
    obs, reward, done, truncated, info = env.step(np.ones(6))
    assert not done and truncated
    assert info['backend'] == 'synthetic'
    assert np.array_equal(info['executed_action'], np.ones(6))
    with pytest.raises(RuntimeError):
        env.step(np.zeros(6))


def test_missing_camera_rejected():
    backend = SyntheticBackend()
    original = backend.observe
    def incomplete():
        obs = original()
        del obs['right_aux']
        return obs
    backend.observe = incomplete
    env = G2LocalEnv(backend)
    with pytest.raises(ValueError):
        env.reset()
    with pytest.raises(RuntimeError, match='Reset required'):
        env.step(np.ones(6))
    assert np.array_equal(backend.state[:3], np.zeros(3))


def test_intervention_failure_stops_and_requires_reset():
    class Backend(SyntheticBackend):
        def __init__(self):
            super().__init__()
            self.stopped = False
        def stop(self):
            self.stopped = True
    def disconnected():
        raise OSError('HID disconnected')
    backend = Backend()
    env = G2LocalEnv(backend, intervention=disconnected)
    env.reset()
    with pytest.raises(OSError, match='HID disconnected'):
        env.step(np.ones(6))
    assert backend.stopped
    env.intervention = None
    with pytest.raises(RuntimeError, match='Reset required'):
        env.step(np.ones(6))
    assert np.array_equal(backend.state[:3], np.zeros(3))


def test_guard_reason_reaches_gym_and_reset_does_not_reconstruct_backend():
    class Reader:
        def __init__(self):
            self.closed = False
            self.last_info = {}

        def observe(self):
            return dict(state=np.array([0., 0., 0., 0., 0., 0., 1.]),
                        left_wrist=np.zeros((8, 8, 3), dtype=np.uint8),
                        right_aux=np.zeros((8, 8, 3), dtype=np.uint8))

        def close(self):
            self.closed = True

    class Port:
        def __init__(self):
            self.sent = []
            self.stop_calls = 0

        def send(self, target):
            self.sent.append(target)

        def stop(self):
            self.stop_calls += 1

    class Guard:
        def __init__(self):
            self.calls = 0
            self.healthy = True
            self.last_decision = FreshnessDecision('not_checked', 'fixture')

        def __call__(self, obs, info, after):
            self.calls += 1
            reject = not self.healthy and after is not None
            self.last_decision = FreshnessDecision(
                'not_after_command:tf' if reject else 'ok', 'fixture')
            return not reject

    reader, port, guard = Reader(), Port(), Guard()
    config = LocalTaskConfig(action_scale=(.01,) * 6,
                             workspace_low=(-1,) * 3,
                             workspace_high=(1,) * 3)
    backend = MotionBackend(
        reader, port, config=config, observation_guard=guard,
        outcome=lambda obs: (0., False), command_timeout=.3,
        send_timeout=.1, step_period=.02, allow_motion=True)
    env = G2LocalEnv(backend, max_steps=2)
    context = EpisodeContext('test', (0, 0, 0), 'fixture', 'fixture')
    try:
        env.reset(options={'context': context})
        guard.healthy = False
        with pytest.raises(RuntimeError, match='not_after_command:tf'):
            env.step(np.zeros(6))
        assert env.runner.active is False
        assert port.stop_calls == 1
        guard.healthy = True
        with pytest.raises(RuntimeError, match='reconstruct'):
            env.reset(options={'context': context})
        assert env.backend is backend
        assert port.stop_calls == 1
    finally:
        env.close()
    assert reader.closed is True
