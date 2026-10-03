"""Freshness checks with local receipt times (upstream G2 semantics).

The guard never converts a robot timestamp into local time.  The age of a
source is how long this process has gone without a *new* timestamp from it. A
stamp that moves backwards is an anomaly; a repeated stamp is tolerated while the
source keeps changing inside its age limit (two guarded reads can fall inside one
publication cycle of a source).
"""
import numpy as np
import pytest

from g2_local.freshness import (FreshnessLeaseGuard, FreshnessLimits,
                               ObservationFreshnessGuard)

ORIGIN = 1_700_000_000_000_000_000
NOW = 20_000_000_000
SOURCE_NOW = ORIGIN + 2_000_000_000
SOURCES = ('left_wrist', 'right_aux', 'joint', 'tf')


def limits(**changes):
    values = dict(camera_age_s=.100, state_age_s=.050, camera_skew_s=.050,
                  tf_position_error_m=.005, tf_rotation_error_rad=.020,
                  mapping_error_s=.005)
    values.update(changes)
    return FreshnessLimits(**values)


def observation():
    return dict(state=np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float32),
                left_wrist=np.zeros((2, 3, 3), dtype=np.uint8),
                right_aux=np.zeros((2, 3, 3), dtype=np.uint8))


def evidence(**source_changes):
    stamps = dict.fromkeys(SOURCES, SOURCE_NOW - 10_000_000)
    stamps.update(source_changes)
    pose = [0., 0., 0., 0., 0., 0., 1.]
    info = dict(source_timestamp_ns=stamps,
                camera_timestamp_ns={k: stamps[k] for k in SOURCES[:2]},
                tf_queries=[dict(target='base_link', source='arm_l_end_link',
                                 timestamp_ns=stamps['tf'], pose=pose.copy()),
                            dict(target='arm_l_end_link', source='base_link',
                                 timestamp_ns=stamps['tf'], pose=pose.copy())],
                tf_position_error_m=0., tf_rotation_error_rad=0., motion_pose=pose,
                read_start_monotonic_ns=NOW-5_000_000, read_end_monotonic_ns=NOW,
                read_start_wall_ns=ORIGIN+NOW-5_000_000, read_end_wall_ns=ORIGIN+NOW,
                read_start_sdk_clock_ns=SOURCE_NOW-5_000_000,
                read_end_sdk_clock_ns=SOURCE_NOW, sdk_clock_ns=SOURCE_NOW,
                received_monotonic_ns=NOW, state_received_monotonic_ns=NOW-4_000_000,
                read_duration_s=.005)
    info['source_changed_monotonic_ns'] = dict.fromkeys(SOURCES, NOW-5_000_000)
    return info


def guard(monotonic=None, **kwargs):
    return ObservationFreshnessGuard(limits(), monotonic_ns=monotonic or (lambda: NOW),
                                     **kwargs)


def test_no_limit_has_a_production_default():
    with pytest.raises(TypeError):
        FreshnessLimits()


def test_a_freshly_received_observation_is_accepted():
    g = guard()
    assert g(observation(), evidence()) is True
    assert g.last_decision.code == 'ok'


def test_a_repeated_frame_ages_out_instead_of_being_accepted():
    now = [NOW]
    g = guard(lambda: now[0])
    info = evidence()
    assert g(observation(), info) is True
    now[0] = NOW + int(limits().camera_age_s*1e9) + 1
    assert g(observation(), info) is False
    assert g.last_decision.code == 'camera_stale:left_wrist'


def test_a_delayed_receipt_is_rejected_immediately():
    g = guard()
    late = NOW - int(limits().camera_age_s*1e9) - 1
    info = evidence()
    info['source_changed_monotonic_ns'] = dict.fromkeys(SOURCES, late)
    assert g(observation(), info) is False
    assert g.last_decision.code == 'camera_stale:left_wrist'


def test_a_repeated_state_stamp_is_tolerated_but_a_frozen_source_is_not():
    """Hardware regression: the TF cache repeated one stamp while its age was still
    0 ms, and the old strict-advance rule aborted a healthy episode 100 steps in."""
    g = guard()
    assert g(observation(), evidence()) is True
    # Same stamp, but the source changed 5 ms ago: a benign duplicate read.
    assert g(observation(), evidence()) is True
    # A source that stopped changing ages past its limit and is rejected.
    stale = evidence()
    stale['source_changed_monotonic_ns'] = dict.fromkeys(
        SOURCES, NOW - int(limits().state_age_s*1e9) - 1)
    assert g(observation(), stale) is False
    assert g.last_decision.code == 'state_stale:joint'


def test_a_state_stamp_that_moves_backwards_is_still_rejected():
    g = guard()
    assert g(observation(), evidence()) is True
    assert g(observation(), evidence(joint=SOURCE_NOW - 20_000_000)) is False
    assert g.last_decision.code == 'source_reversed:joint'


def test_a_recent_camera_frame_may_be_reused_but_the_age_still_bounds_it():
    now = [NOW]
    g = guard(lambda: now[0])
    first = evidence()
    assert g(observation(), first) is True
    reused = evidence(left_wrist=first['source_timestamp_ns']['left_wrist'],
                      right_aux=first['source_timestamp_ns']['right_aux'],
                      joint=SOURCE_NOW-9_000_000, tf=SOURCE_NOW-9_000_000)
    reused['source_changed_monotonic_ns'].update(joint=NOW-4_000_000, tf=NOW-4_000_000)
    now[0] = NOW + 1_000_000
    assert g(observation(), reused) is True
    now[0] = NOW + int(limits().camera_age_s*1e9) + 1
    tail = evidence(joint=SOURCE_NOW-8_000_000, tf=SOURCE_NOW-8_000_000)
    tail['source_changed_monotonic_ns'].update(joint=now[0]-1_000_000, tf=now[0]-1_000_000)
    assert g(observation(), tail) is False
    assert g.last_decision.code == 'camera_stale:left_wrist'


def test_a_successor_must_be_received_after_the_command_was_sent():
    g = guard()
    assert g(observation(), evidence(), after=NOW/1e9) is False
    assert g.last_decision.code.startswith('not_after_command')


def test_source_intervals_are_the_local_receipt_times():
    g = guard()
    assert g(observation(), evidence()) is True
    intervals = g.last_decision.source_intervals_ns
    assert intervals['left_wrist'] == (NOW-5_000_000, NOW-5_000_000)
    assert intervals['tf'] == (NOW-5_000_000, NOW-5_000_000)


def test_receipt_time_from_the_future_is_invalid_evidence():
    g = guard()
    info = evidence()
    info['source_changed_monotonic_ns'] = dict.fromkeys(SOURCES, NOW+1)
    assert g(observation(), info) is False
    assert g.last_decision.code == 'invalid_evidence'


def test_missing_receipt_evidence_is_rejected():
    g = guard()
    info = evidence()
    del info['source_changed_monotonic_ns']
    assert g(observation(), info) is False
    assert g.last_decision.code == 'invalid_evidence'


def test_skip_tf_progress_still_requires_the_other_sources_to_advance():
    g = guard(skip_tf_progress=True)
    assert g(observation(), evidence()) is True
    reused_tf = evidence(tf=SOURCE_NOW-10_000_000, joint=SOURCE_NOW-9_000_000)
    reused_tf['source_changed_monotonic_ns'].update(joint=NOW-4_000_000, tf=NOW-4_000_000)
    assert g(observation(), reused_tf) is True


def test_stop_feedback_uses_the_local_read_time():
    g = guard()
    info = evidence()
    info['joint_timestamp_ns'] = info['source_timestamp_ns']['joint']
    assert g.validate_stop_feedback(info) is True
    stale = evidence()
    stale['joint_timestamp_ns'] = stale['source_timestamp_ns']['joint']
    stale['read_start_monotonic_ns'] = NOW - int(limits().state_age_s*1e9) - 1
    with pytest.raises(RuntimeError, match='stop feedback read expired'):
        guard().validate_stop_feedback(stale)


def test_stop_feedback_rejects_a_bad_tf_direction():
    g = guard()
    info = evidence()
    info['joint_timestamp_ns'] = info['source_timestamp_ns']['joint']
    info['tf_queries'][1] = dict(info['tf_queries'][1],
                                 target='base_link', source='arm_l_end_link')
    with pytest.raises(RuntimeError, match='stop TF direction mismatch'):
        g.validate_stop_feedback(info)


def test_same_domain_successor_must_be_acquired_after_the_command():
    """A successor older than the same-domain send anchor must be refused."""
    fresh = guard()
    info = evidence()
    # stamps are 10 ms old inside the SDK domain; the command was sent 20 ms ago.
    assert fresh(observation(), info, after=(NOW - 10_000_000)/1e9,
                 after_sdk_ns=SOURCE_NOW - 20_000_000) is True
    assert fresh.last_decision.sdk_anchor['same_domain'] is True
    assert fresh.last_decision.sdk_anchor['margins_ns']['left_wrist'] == 10_000_000

    stale = guard()
    assert stale(observation(), info, after=(NOW - 10_000_000)/1e9,
                 after_sdk_ns=SOURCE_NOW - 5_000_000) is False
    assert stale.last_decision.code == 'not_after_command_sdk:left_wrist'


def test_unrelated_clocks_degrade_to_the_local_receipt_guarantee():
    """An unrelated SDK clock must not reject every frame, but must be recorded."""
    info = evidence()
    shifted = SOURCE_NOW + 20_000_000_000
    info['sdk_clock_ns'] = info['read_end_sdk_clock_ns'] = shifted
    g = guard()
    assert g(observation(), info, after=(NOW - 10_000_000)/1e9, after_sdk_ns=SOURCE_NOW) is True
    anchor = g.last_decision.sdk_anchor
    assert anchor['same_domain'] is False
    assert anchor['sdk_ages_ns']['joint'] == 20_000_000_000 + 10_000_000


def test_rejected_read_keeps_the_previous_lease_until_it_expires():
    """A rejected read must not revoke an existing lease (successor retries)."""
    now = [10.0]
    guard = FreshnessLeaseGuard(FakeGuard([True, False]), feedback_lease_s=1.5,
                                clock=lambda: now[0])
    assert guard.accept(object(), {}, after=1.0) is True
    assert guard() is True
    assert guard.accept(object(), {}, after=1.0) is False
    # The previous lease still holds: the writer keeps resending the last target
    # while the policy loop retries its successor read.
    assert guard() is True
    now[0] += 1.6
    assert guard() is False


class FakeGuard:
    """Minimal observation guard: returns the queued verdicts in order."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.last_decision = None

    def __call__(self, obs, info, after=None, after_sdk_ns=None):
        return self.verdicts.pop(0) if self.verdicts else False
