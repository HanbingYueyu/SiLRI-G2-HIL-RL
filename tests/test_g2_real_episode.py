from types import SimpleNamespace

import numpy as np
import pytest

from g2_local.contract import EpisodeContext
from g2_local.config import HingeInsertTaskConfig
from g2_local.outcome import OutcomeDecision
from g2_local.real_episode import RealEpisodeCoordinator


class Keys:
    def __init__(self, keys=()): self.keys = list(keys)
    def read_available(self, *, limit):
        result, self.keys = self.keys[:limit], self.keys[limit:]
        return result


class Intervention:
    def __init__(self):
        self.last_frame = SimpleNamespace(buttons=(False, False), pressed=(), ready=True)
        self.gate = SimpleNamespace(fresh=True)
        self.fault = None
    def __call__(self): return False, None


def context(*, reset_monotonic_ns=39_000_000_000, target_offset_m=(0., 0., 0.),
            ee_reset_offset=(0.,) * 6):
    return EpisodeContext('episode-1', target_offset_m, 'visual', 'fixed',
                          ee_reset_offset, visual_reset_monotonic_ns=reset_monotonic_ns)


def waiting_episode(*, now_ns=40_000_000_000, context_max_age_s=5.0,
                    target_xy_range_m=.05, keys=()):
    task = HingeInsertTaskConfig(target_xy_range_m=target_xy_range_m)
    return RealEpisodeCoordinator(Intervention(), Keys(keys), task,
                                  context_max_age_s=context_max_age_s,
                                  clock_ns=lambda: now_ns)


def running_episode():
    machine = waiting_episode()
    machine.offer_context(context())
    machine.intervention.last_frame = SimpleNamespace(buttons=(True, True),
                                                       pressed=(0, 1), ready=True)
    assert machine.observe_start_frame() is False
    machine.intervention.last_frame = SimpleNamespace(buttons=(False, False),
                                                       pressed=(), ready=True)
    assert machine.observe_start_frame() is True
    return machine


def valid_successor():
    return dict(state=np.array((0., 0., 0., 0., 0., 0., 1.), dtype=np.float32),
                left_wrist=np.zeros((2, 2, 3), dtype=np.uint8),
                right_aux=np.zeros((2, 2, 3), dtype=np.uint8))


def test_y_during_step_is_consumed_once_after_successor():
    machine = running_episode()
    token = machine.begin_step()
    machine.request_terminal('success')
    assert machine.outcome(valid_successor()) == OutcomeDecision(10., True, 'human', True)
    with pytest.raises(RuntimeError, match='not running'): machine.begin_step()


def test_stale_or_out_of_range_reset_context_cannot_start():
    machine = waiting_episode(now_ns=40_000_000_000,
                              context_max_age_s=5.0,
                              target_xy_range_m=.05)
    with pytest.raises(ValueError, match='stale reset context'):
        machine.offer_context(context(reset_monotonic_ns=1_000_000_000))
    with pytest.raises(ValueError, match='target offset outside configured range'):
        machine.offer_context(context(reset_monotonic_ns=39_000_000_000,
                                      target_offset_m=(.06, 0., 0.)))


def test_terminal_key_buffer_is_flushed_before_episode_start():
    machine = waiting_episode(keys=('y',))
    machine.offer_context(context())
    machine.intervention.last_frame = SimpleNamespace(buttons=(True, True),
                                                       pressed=(0, 1), ready=True)
    assert machine.observe_start_frame() is False
    machine.intervention.last_frame = SimpleNamespace(buttons=(False, False),
                                                       pressed=(), ready=True)
    assert machine.observe_start_frame() is True
    assert machine.begin_step().step_id == 0
    assert machine.outcome(valid_successor()) == OutcomeDecision(-.05, False,
                                                                  'human', None)


def test_abort_invalidates_step_and_clears_pending_state():
    machine = running_episode()
    token = machine.begin_step()
    machine.request_terminal('failure')
    machine.abort_step(token)
    with pytest.raises(RuntimeError): machine.outcome(valid_successor())
    assert machine.state == 'ABORTED'
    assert machine.context is None


def test_outcome_requires_successor_and_input_fault_aborts():
    machine = running_episode()
    token = machine.begin_step()
    with pytest.raises(ValueError, match='successor'): machine.outcome(None)
    machine.intervention.fault = 'disconnected'
    machine.intervention.last_frame = None
    with pytest.raises(RuntimeError, match='Input fault'): machine.outcome(valid_successor())
    assert machine.state == 'ABORTED'


def test_context_metadata_defaults_and_bounds():
    legacy = EpisodeContext('e', (0, 0, 0), 'visual', 'fixed')
    assert legacy.visual_reset_monotonic_ns is None
    assert legacy.visual_confidence is None
    assert legacy.upstream_frame_id is None
    with pytest.raises(ValueError): EpisodeContext('e', (0, 0, 0), 'visual', 'fixed',
                                                   visual_confidence=float('nan'))


def test_legacy_context_parses_but_cannot_authorize_real_start():
    machine = waiting_episode()
    machine.offer_context(EpisodeContext('legacy', (0, 0, 0), 'visual', 'fixed'))
    machine.intervention.last_frame = SimpleNamespace(buttons=(True, True),
                                                       pressed=(0, 1), ready=True)
    assert machine.observe_start_frame() is False
    machine.intervention.last_frame = SimpleNamespace(buttons=(False, False),
                                                       pressed=(), ready=True)
    with pytest.raises(ValueError, match='stale reset context'):
        machine.observe_start_frame()


def test_ee_reset_offset_must_fit_configured_range():
    machine = waiting_episode()
    with pytest.raises(ValueError, match='EE reset offset outside configured range'):
        machine.offer_context(context(ee_reset_offset=(.004, 0., 0., 0., 0., 0.)))


def test_conflicting_terminal_keys_abort_without_transition():
    machine = running_episode()
    machine.begin_step()
    machine.keys.source.keys.extend(['y', 'f'])
    with pytest.raises(RuntimeError, match='Conflicting'):
        machine.outcome(valid_successor())
    assert machine.state == 'ABORTED'
    assert machine.context is None


def test_stale_hid_report_cannot_start_chord():
    machine = waiting_episode()
    machine.offer_context(context())
    machine.intervention.gate = SimpleNamespace(fresh=False)
    machine.intervention.last_frame = SimpleNamespace(buttons=(True, True),
                                                       pressed=(0, 1), ready=True)
    assert machine.observe_start_frame() is False
    machine.intervention.gate.fresh = True
    machine.intervention.last_frame = SimpleNamespace(buttons=(False, False),
                                                       pressed=(), ready=True)
    assert machine.observe_start_frame() is False


def test_direct_start_bypass_is_unavailable():
    machine = waiting_episode()
    machine.offer_context(context())
    assert not hasattr(machine, 'confirm_start_chord')
    with pytest.raises(RuntimeError, match='not running'): machine.begin_step()


def test_missing_gate_cannot_start_chord():
    machine = waiting_episode()
    machine.offer_context(context())
    del machine.intervention.gate
    machine.intervention.last_frame = SimpleNamespace(buttons=(True, True),
                                                       pressed=(0, 1), ready=True)
    assert machine.observe_start_frame() is False
    machine.intervention.last_frame = SimpleNamespace(buttons=(False, False),
                                                       pressed=(), ready=True)
    assert machine.observe_start_frame() is False


def test_malformed_successor_cannot_seal_pending_terminal():
    machine = running_episode()
    machine.begin_step()
    machine.request_terminal('success')
    assert not hasattr(machine, 'finish_step')
    for malformed in ({'valid': True},
                      dict(valid_successor(), left_wrist=np.zeros((2, 2), dtype=np.uint8)),
                      dict(valid_successor(), state=np.array((0.,) * 7))):
        with pytest.raises(ValueError, match='successor'):
            machine.outcome(malformed)
        assert machine.running
    assert machine.outcome(valid_successor()) == OutcomeDecision(10., True, 'human', True)
