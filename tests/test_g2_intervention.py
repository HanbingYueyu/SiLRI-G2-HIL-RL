from types import SimpleNamespace
import numpy as np
import pytest
from g2_local.env import G2LocalEnv, SyntheticBackend
from g2_local import spacemouse


class Reader:
    def __init__(self):
        self.frame = SimpleNamespace(axes=(0,)*6, buttons=(False,False),
                                     axis_times=(1.,1.), ready=True)
    def poll(self):
        return self.frame


def test_explicit_takeover_and_executed_action_labels():
    reader = Reader()
    source = spacemouse.HumanInput(reader, axis_map=(-2,-1,-3), left_button=0,
                                   clock=lambda: 1.)
    env = G2LocalEnv(SyntheticBackend(), intervention=source)
    env.reset()
    reader.frame.buttons = (True,False)
    _, _, _, _, info = env.step(np.ones(6))
    assert not info['is_intervention']  # Left button never activates takeover.
    source.set_active(True)
    _, _, _, _, info = env.step(np.ones(6))
    assert info['is_intervention']
    assert np.array_equal(info['executed_action'], np.zeros(6))
    reader.frame.axes = (0,0,-.55,0,0,0)
    _, _, _, _, info = env.step(np.ones(6))
    assert np.allclose(info['executed_action'], [0,0,0,0,0,.5])


def test_active_stale_input_fault_is_latched_and_ends_episode():
    reader = Reader()
    source = spacemouse.HumanInput(reader, axis_map=(-2,-1,-3), left_button=0,
                                   clock=lambda: 2.)
    source.set_active(True)
    env = G2LocalEnv(SyntheticBackend(), intervention=source)
    env.reset()
    with pytest.raises(RuntimeError, match='stale'):
        env.step(np.ones(6))
    reader.frame.axis_times = (2.,2.)
    env.reset()
    with pytest.raises(RuntimeError, match='latched'):
        env.step(np.ones(6))
    with pytest.raises(RuntimeError):
        source.set_active(True)
    assert np.array_equal(env.backend.state[:3], np.zeros(3))


def test_reader_disconnect_does_not_fall_back_to_policy():
    class BrokenReader:
        def poll(self):
            raise OSError('device unplugged')
    source = spacemouse.HumanInput(BrokenReader(), axis_map=(-2,-1,-3), left_button=0)
    env = G2LocalEnv(SyntheticBackend(), intervention=source)
    env.reset()
    with pytest.raises(OSError, match='unplugged'):
        env.step(np.ones(6))
    assert not env.runner.active
    assert np.array_equal(env.backend.state[:3], np.zeros(3))
    with pytest.raises(RuntimeError, match='latched'):
        source()


def test_active_neutral_idle_does_not_abort():
    reader = Reader()
    now = [1.]
    source = spacemouse.HumanInput(reader, axis_map=(-2,-1,-3), left_button=0,
                                   clock=lambda: now[0])
    source.set_active(True)
    assert source()[1] == (0.,)*6
    now[0] = 20.
    assert source() == (True, (0.,)*6)
