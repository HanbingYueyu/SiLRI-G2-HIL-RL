import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from g2_local.env import G2LocalEnv, SyntheticBackend


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
