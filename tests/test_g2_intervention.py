from types import SimpleNamespace
import numpy as np
import pytest
from g2_local.env import G2LocalEnv, SyntheticBackend
from g2_local import spacemouse
from g2_local.training_config import InterventionConfig


class Reader:
    def __init__(self):
        self.frame = SimpleNamespace(axes=(0,)*6, buttons=(False,False),
                                     axis_times=(1.,1.), ready=True)
    def poll(self):
        return self.frame


def frame(axes=(0,)*6, stamps=(1.,1.), buttons=(False,False), ready=True):
    return SimpleNamespace(axes=axes, axis_times=stamps, buttons=buttons, ready=ready)


def explicit_config():
    return InterventionConfig(axis_map=(1,2,3), left_button=0, right_button=1,
                              engage_deadzone=.12, release_deadzone=.08,
                              release_hold_s=.25, report_max_age_s=.2)


def automatic_source():
    reader = Reader()
    now = [1.]
    source = spacemouse.AutomaticIntervention(reader, explicit_config(), clock=lambda: now[0])
    assert source() == (False, None)
    return source, reader, now


def engaged_source():
    source, reader, now = automatic_source()
    reader.frame = frame(axes=(.6,0,0,0,0,0))
    assert source()[0] is True
    return source, reader, now


def test_motion_engages_and_only_fresh_held_neutral_releases():
    source, reader, now = automatic_source()
    reader.frame = frame(axes=(.6,0,0,0,0,0), stamps=(1.,1.))
    assert source()[0] is True
    reader.frame = frame(stamps=(1.1,1.1)); now[0] = 1.1
    assert source()[0] is True
    reader.frame = frame(stamps=(1.4,1.4)); now[0] = 1.4
    assert source() == (False, None)


def test_silent_neutral_never_releases_active_intervention():
    source, reader, now = engaged_source()
    reader.frame = frame(stamps=(1.,1.)); now[0] = 5.
    assert source()[0] is True


def test_unplug_latches_fault_instead_of_policy_fallback():
    class BrokenReader:
        def poll(self):
            raise OSError('device unplugged')
    source = spacemouse.AutomaticIntervention(BrokenReader(), explicit_config())
    with pytest.raises(OSError): source()
    with pytest.raises(RuntimeError, match='latched'): source()


def test_stale_nonzero_aborts_and_latches():
    source, reader, now = engaged_source()
    now[0] = 2.
    with pytest.raises(RuntimeError, match='stale nonzero'):
        source()
    with pytest.raises(RuntimeError, match='latched'):
        source()


def test_stale_input_inside_release_deadzone_aborts():
    source, reader, now = engaged_source()
    reader.frame = frame(axes=(.05,0,0,0,0,0), stamps=(1.,1.))
    now[0] = 2.
    with pytest.raises(RuntimeError, match='stale nonzero'):
        source()
    with pytest.raises(RuntimeError, match='latched'):
        source()


def test_repeated_neutral_snapshot_cannot_accumulate_release_hold():
    source, reader, now = engaged_source()
    reader.frame = frame(stamps=(1.1,1.1)); now[0] = 1.1
    assert source()[0] is True
    reader.frame = frame(stamps=(1.2,1.2)); now[0] = 1.2
    assert source()[0] is True
    now[0] = 1.39  # Still age-fresh, but no new axis-channel packets.
    assert source()[0] is True
    reader.frame = frame(stamps=(1.5,1.2)); now[0] = 1.5
    assert source()[0] is True
    reader.frame = frame(stamps=(1.6,1.6)); now[0] = 1.6
    assert source()[0] is True
    reader.frame = frame(stamps=(1.9,1.9)); now[0] = 1.9
    assert source() == (False, None)


def test_malformed_report_latches_and_invalidates_last_frame():
    source, reader, now = automatic_source()
    assert source.last_frame is reader.frame
    reader.frame = frame(axes=(float('nan'),0,0,0,0,0))
    with pytest.raises(ValueError):
        source()
    assert source.last_frame is None
    with pytest.raises(RuntimeError, match='latched'):
        source()


def test_left_button_selects_rotation_and_neutral_hold_resets_on_motion():
    source, reader, now = automatic_source()
    reader.frame = frame(stamps=(1.1,1.1), buttons=(True,False)); now[0] = 1.1
    assert source() == (False, None)
    reader.frame = frame(axes=(.6,0,0,0,0,0), stamps=(1.2,1.2), buttons=(True,False)); now[0] = 1.2
    assert source()[1] == pytest.approx((0,0,0,0,.5652173913,0))
    reader.frame = frame(stamps=(1.3,1.3), buttons=(True,False)); now[0] = 1.3
    assert source()[0] is True
    reader.frame = frame(axes=(.6,0,0,0,0,0), stamps=(1.4,1.4), buttons=(True,False)); now[0] = 1.4
    assert source()[0] is True
    reader.frame = frame(stamps=(1.5,1.5), buttons=(True,False)); now[0] = 1.5
    assert source()[0] is True
    reader.frame = frame(stamps=(1.8,1.8), buttons=(True,False)); now[0] = 1.8
    assert source() == (False, None)


@pytest.mark.parametrize('bad_frame', [
    frame(ready=False),
    frame(stamps=(None,1.)),
    frame(stamps=(float('nan'),1.)),
    frame(stamps=(2.,1.)),
])
def test_not_ready_or_malformed_time_aborts(bad_frame):
    source, reader, now = automatic_source()
    reader.frame = bad_frame
    with pytest.raises(ValueError):
        source()
    with pytest.raises(RuntimeError, match='latched'):
        source()


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
