from types import SimpleNamespace
import pytest
from g2_local import spacemouse


def frame(axes=(0,)*6, stamps=(1., 1.), buttons=(False, False), ready=True):
    return SimpleNamespace(axes=axes, axis_times=stamps, buttons=buttons, ready=ready)


def test_freshness_and_recovery():
    gate = spacemouse.LiveInputGate(axis_map=(1, 2, 3), left_button=1, max_age=.2)
    assert not gate.update(frame(), now=1.1).blocked
    assert gate.update(frame(axes=(.55,0,0,0,0,0)), now=1.1).action[0] == pytest.approx(.5)
    assert gate.update(frame(axes=(.55,0,0,0,0,0)), now=1.3).blocked
    assert gate.update(frame(axes=(.55,0,0,0,0,0), stamps=(1.4,1.4)), now=1.4).blocked
    assert not gate.update(frame(stamps=(1.4,1.4)), now=1.4).blocked
    gate.update(frame(stamps=(1.4,1.4), buttons=(False,True)), now=1.4)
    assert gate.update(frame(axes=(.55,0,0,0,0,0), stamps=(1.4,1.4), buttons=(False,True)), now=1.4).action == pytest.approx((0,0,0,0,.5,0))


@pytest.mark.parametrize('stamps', [(None,None), (2.,2.), (float('nan'),1.)])
def test_missing_future_nonfinite_stamps_block(stamps):
    gate = spacemouse.LiveInputGate(axis_map=(1,2,3), left_button=0)
    assert gate.update(frame(stamps=stamps), now=1.).blocked


def test_disconnect_latches_until_neutral():
    gate = spacemouse.LiveInputGate(axis_map=(1,2,3), left_button=0)
    gate.update(frame(), now=1.)
    assert gate.invalidate().blocked
    assert gate.update(frame(axes=(1,0,0,0,0,0)), now=1.).blocked


def test_reader_error_emits_blocked_record_and_propagates():
    class Disconnected:
        def poll(self):
            raise OSError('disconnected')
    from g2_local import live_preview
    output = []
    gate = spacemouse.LiveInputGate(axis_map=(1,2,3), left_button=0)
    gate.update(frame(), now=1.)
    with pytest.raises(OSError):
        live_preview.sample(Disconnected(), gate, output.append)
    assert output[-1]['blocked'] is True
    assert output[-1]['action'] == (0.,)*6
    assert gate.update(frame(axes=(1,0,0,0,0,0)), now=1.).blocked


def test_observed_neutral_survives_idle_but_stale_motion_does_not():
    gate = spacemouse.LiveInputGate(axis_map=(1,2,3), left_button=0)
    assert not gate.update(frame(), now=1.).blocked
    assert gate.fresh
    assert not gate.update(frame(), now=20.).blocked
    assert not gate.fresh  # A zero proposal from silence is not fresh release evidence.
    assert gate.update(frame(axes=(.55,0,0,0,0,0), stamps=(20.,20.)), now=20.).action[0] == pytest.approx(.5)
    assert gate.update(frame(axes=(.55,0,0,0,0,0), stamps=(20.,20.)), now=21.).blocked


def test_idle_neutral_cannot_arm_new_source_or_changed_mode():
    gate = spacemouse.LiveInputGate(axis_map=(1,2,3), left_button=0)
    assert gate.update(frame(), now=20.).blocked
    gate.update(frame(stamps=(20.,20.)), now=20.)
    assert gate.update(frame(stamps=(20.,20.), buttons=(True,False)), now=21.).blocked
