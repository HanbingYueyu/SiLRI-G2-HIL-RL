"""Offline freshness checks with literal raw-PTP/monotonic clock evidence."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from g2_local.freshness import FreshnessLimits, ObservationFreshnessGuard
from g2_local.live_clock import ClockSnapshot, ClockWindow


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


def snapshot(**changes):
    # A 55-second software PTP offset has ALREADY had 37 seconds removed.
    values = dict(schema=1, sequence=1, healthy=True, reason='ok',
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  session_id='test-session', expected_master='044052.fffe.000010',
                  actual_master='044052.fffe.000010', scale='raw_ptp',
                  utc_offset_s=37, utc_offset_valid=0, leap61=0, leap59=0,
                  ptp_timescale=1, reference_mono_ns=NOW,
                  offset_at_reference_ns=18_000_000_000., drift_ppm=0.,
                  residual_ns=0., path_delay_ns=0, empirical_error_ns=2_000_000.,
                  wall_minus_mono_ns=ORIGIN, created_mono_ns=NOW,
                  last_sample_mono_ns=NOW, valid_until_ns=NOW+2_500_000_000)
    values.update(changes)
    return ClockSnapshot(**values)


def observation():
    return dict(state=np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float32),
                left_wrist=np.zeros((2, 3, 3), dtype=np.uint8),
                right_aux=np.zeros((2, 3, 3), dtype=np.uint8))


def evidence(**source_changes):
    stamps = dict.fromkeys(SOURCES, SOURCE_NOW - 10_000_000)
    stamps.update(source_changes)
    pose = [0., 0., 0., 0., 0., 0., 1.]
    return dict(source_timestamp_ns=stamps,
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
                read_duration_s=.005, motion_status={'control_mode': 1, 'error_code': 0},
                backend='gdk_read_only')


@pytest.fixture
def rig():
    state = SimpleNamespace(snapshot=snapshot(), now=NOW)
    class Client:
        def read(self):
            result = state.snapshot
            state.snapshot = replace(result, sequence=result.sequence+1)
            return result
    state.client = Client()
    state.guard = ObservationFreshnessGuard(state.client, limits(),
                                          monotonic_ns=lambda: state.now)
    return state


def test_no_limit_has_a_production_default():
    with pytest.raises(TypeError):
        FreshnessLimits()


def test_policy_revalidation_checks_age_without_resetting_source_progress(rig):
    obs, info = observation(), evidence()
    assert rig.guard(obs, info)
    assert rig.guard.revalidate(obs, info)
    assert not rig.guard(obs, info)  # A new observation still cannot repeat stamps.
    rig.now += 60_000_000
    assert not rig.guard.revalidate(obs, info)


@pytest.mark.parametrize('field', list(FreshnessLimits.__annotations__))
@pytest.mark.parametrize('value', [True, False, 0, -1., float('nan'), float('inf'),
                                   -float('inf'), '0.1', np.float64(.1),
                                   pytest.param(10**1000, id='overflow')])
def test_limits_reject_non_exact_or_non_positive_finite_numbers(field, value):
    with pytest.raises(ValueError):
        limits(**{field: value})


def test_fresh_sources_commit_and_protect_diagnostic_state(rig):
    obs, info = observation(), evidence()
    original = deepcopy(info)
    assert rig.guard(obs, info, after=None) is True
    assert rig.guard.previous_source_ns == original['source_timestamp_ns']
    assert info == original
    decision = rig.guard.last_decision
    assert decision.code == 'ok'
    assert decision.source_intervals_ns['tf'] == (NOW-12_000_000, NOW-8_000_000)
    assert decision.age_intervals_s['tf'] == (.008, .012)
    with pytest.raises(FrozenInstanceError):
        decision.code = 'corrupted'
    with pytest.raises(TypeError):
        decision.source_intervals_ns['tf'] = (0, 0)
    rig.guard.previous_source_ns['tf'] = 0
    assert rig.guard.previous_source_ns == original['source_timestamp_ns']
    assert rig.guard(obs, evidence(**dict.fromkeys(SOURCES, SOURCE_NOW-9_000_000)), None)
    assert decision.source_intervals_ns['tf'] == (NOW-12_000_000, NOW-8_000_000)


@pytest.mark.parametrize('source,age_ns,code', [
    ('left_wrist', 98_000_000, 'camera_stale:left_wrist'),
    ('right_aux', 98_000_000, 'camera_stale:right_aux'),
    ('joint', 48_000_000, 'state_stale:joint'),
    ('tf', 48_000_000, 'state_stale:tf'),
])
def test_worst_case_age_boundary_and_one_nanosecond_outside(rig, source, age_ns, code):
    # Keep the other camera nearby but inside its own age limit.
    stamps = {source: SOURCE_NOW-age_ns}
    if source in SOURCES[:2]:
        other = 'right_aux' if source == 'left_wrist' else 'left_wrist'
        stamps[other] = SOURCE_NOW-age_ns+1_000_000
    assert rig.guard(observation(), evidence(**stamps), None)
    before = rig.guard.previous_source_ns
    rig.now += 1
    assert rig.guard(observation(), evidence(**stamps), None) is False
    assert rig.guard.last_decision.code == code
    assert rig.guard.previous_source_ns == before


@pytest.mark.parametrize('source', SOURCES)
def test_future_uses_earliest_interval_boundary(rig, source):
    assert rig.guard(observation(), evidence(**{source: SOURCE_NOW+2_000_000}), None)
    before = rig.guard.previous_source_ns
    assert not rig.guard(observation(), evidence(**{source: SOURCE_NOW+2_000_001}), None)
    assert rig.guard.last_decision.code == 'source_future:'+source
    assert rig.guard.previous_source_ns == before


def test_camera_skew_uses_worst_interval_separation(rig):
    assert rig.guard(observation(), evidence(left_wrist=SOURCE_NOW-48_000_000,
                                            right_aux=SOURCE_NOW-2_000_000), None)
    assert not rig.guard(observation(), evidence(left_wrist=SOURCE_NOW-48_000_000,
                                                right_aux=SOURCE_NOW-1_999_999), None)
    assert rig.guard.last_decision.code == 'camera_skew'


@pytest.mark.parametrize('source', SOURCES)
@pytest.mark.parametrize('delta,code', [(0, 'source_frozen:'), (-1, 'source_reversed:')])
def test_each_source_must_advance_and_failure_is_transactional(rig, source, delta, code):
    assert rig.guard(observation(), evidence(), None)
    before = rig.guard.previous_source_ns
    stamps = dict.fromkeys(SOURCES, SOURCE_NOW-9_000_000)
    stamps[source] = SOURCE_NOW-10_000_000+delta
    assert not rig.guard(observation(), evidence(**stamps), None)
    assert rig.guard.last_decision.code == code+source
    assert rig.guard.previous_source_ns == before
    assert rig.guard(observation(), evidence(**dict.fromkeys(SOURCES, SOURCE_NOW-9_500_000)), None)


@pytest.mark.parametrize('source', SOURCES)
@pytest.mark.parametrize('delta,accepted', [(0, False), (1, True), (-1, False)])
def test_every_source_earliest_time_must_be_strictly_after_command(rig, source, delta, accepted):
    stamps = dict.fromkeys(SOURCES, SOURCE_NOW-9_000_000)
    stamps[source] = SOURCE_NOW-10_000_000+delta
    assert rig.guard(observation(), evidence(**stamps), 19.988) is accepted
    if not accepted:
        assert rig.guard.last_decision.code == 'not_after_command:'+source
        assert rig.guard.previous_source_ns == {}


@pytest.mark.parametrize('field,boundary,code', [
    ('tf_position_error_m', .005, 'tf_position_error'),
    ('tf_rotation_error_rad', .020, 'tf_rotation_error'),
])
def test_pose_error_bounds_and_failed_observation_do_not_commit(rig, field, boundary, code):
    info = evidence()
    info[field] = boundary + 1e-9
    assert not rig.guard(observation(), info, None)
    assert rig.guard.last_decision.code == code
    assert rig.guard.previous_source_ns == {}
    info[field] = boundary
    assert rig.guard(observation(), info, None)


@pytest.mark.parametrize('drift,stamp,interval', [
    (100., SOURCE_NOW-9_999_000, (19_987_999_799, 19_992_000_201)),
    (-100., SOURCE_NOW-10_001_000, (19_988_000_199, 19_991_999_801)),
])
def test_drift_is_inverted_and_uncertainty_propagated_outward(rig, drift, stamp, interval):
    rig.snapshot = replace(rig.snapshot, drift_ppm=drift)
    assert rig.guard(observation(), evidence(**dict.fromkeys(SOURCES, stamp)), None)
    assert rig.guard.last_decision.source_intervals_ns['joint'] == interval


def test_mapping_error_grows_with_extrapolation_before_any_source_check(rig):
    rig.snapshot = replace(rig.snapshot, reference_mono_ns=NOW-1_000_000_000,
                           last_sample_mono_ns=NOW-1_000_000_000,
                           valid_until_ns=NOW+1_500_000_000,
                           empirical_error_ns=4_900_000.)
    assert rig.guard(observation(), evidence(), None)
    assert rig.guard.last_decision.mapping_error_s == .005
    rig.now += 1
    assert not rig.guard(observation(), {}, None)
    assert rig.guard.last_decision.code == 'mapping_error'


def test_expired_mapping_rejects_before_missing_source_metadata(rig):
    rig.now += 2_500_000_001
    assert not rig.guard(observation(), {}, None)
    assert rig.guard.last_decision.code == 'mapping_expired'
    assert rig.guard.previous_source_ns == {}


@pytest.mark.parametrize('source', SOURCES)
@pytest.mark.parametrize('value', [True, 1., '1', 0, -1, float('nan'),
                                   1 << 63, (1 << 64)-1, np.int64(1)])
def test_source_timestamp_type_and_signed_int64_bound_are_enforced(rig, source, value):
    assert not rig.guard(observation(), evidence(**{source: value}), None)
    assert rig.guard.last_decision.code == 'invalid_evidence'
    assert rig.guard.previous_source_ns == {}


@pytest.mark.parametrize('field', [
    'read_start_monotonic_ns', 'read_end_monotonic_ns', 'read_start_wall_ns',
    'read_end_wall_ns', 'read_start_sdk_clock_ns', 'read_end_sdk_clock_ns',
    'sdk_clock_ns', 'received_monotonic_ns', 'state_received_monotonic_ns',
])
@pytest.mark.parametrize('value', [True, 1., 0, -1, 1 << 63])
def test_local_and_sdk_clock_evidence_requires_positive_int64(rig, field, value):
    info = evidence()
    info[field] = value
    assert not rig.guard(observation(), info, None)
    assert rig.guard.last_decision.code == 'invalid_evidence'


@pytest.mark.parametrize('field', ['tf_position_error_m', 'tf_rotation_error_rad', 'read_duration_s'])
@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), -1., '0'])
def test_nonfinite_or_invalid_error_and_duration_evidence_is_rejected(rig, field, value):
    info = evidence()
    info[field] = value
    assert not rig.guard(observation(), info, None)
    assert rig.guard.last_decision.code == 'invalid_evidence'


@pytest.mark.parametrize('field', [
    'source_timestamp_ns', 'camera_timestamp_ns', 'tf_queries', 'motion_pose',
    'read_start_monotonic_ns', 'read_end_monotonic_ns', 'read_start_wall_ns',
    'read_end_wall_ns', 'read_start_sdk_clock_ns', 'read_end_sdk_clock_ns',
    'sdk_clock_ns', 'received_monotonic_ns', 'state_received_monotonic_ns',
    'read_duration_s', 'tf_position_error_m', 'tf_rotation_error_rad',
])
def test_missing_evidence_fields_fail_closed(rig, field):
    info = evidence()
    del info[field]
    assert not rig.guard(observation(), info, None)
    assert rig.guard.last_decision.code == 'invalid_evidence'


@pytest.mark.parametrize('fault', [
    'source_extra', 'source_missing', 'camera_mismatch', 'source_not_dict',
    'tf_direction', 'tf_stamp', 'tf_stamp_overflow', 'tf_pose_nan',
    'tf_quaternion', 'motion_pose_bool', 'motion_quaternion', 'tf_query_count',
    'reversed_read', 'future_receive', 'reversed_sdk', 'sdk_alias',
    'received_alias', 'state_outside_read', 'duration_mismatch', 'wall_jump',
])
def test_inconsistent_evidence_is_rejected_transactionally(rig, fault):
    assert rig.guard(observation(), evidence(), None)
    before = rig.guard.previous_source_ns
    info = evidence(**dict.fromkeys(SOURCES, SOURCE_NOW-9_000_000))
    if fault == 'source_extra':
        info['source_timestamp_ns']['extra'] = SOURCE_NOW
    elif fault == 'source_missing':
        del info['source_timestamp_ns']['joint']
    elif fault == 'source_not_dict':
        info['source_timestamp_ns'] = list(info['source_timestamp_ns'].items())
    elif fault == 'camera_mismatch':
        info['camera_timestamp_ns']['right_aux'] += 1
    elif fault == 'tf_direction':
        info['tf_queries'].reverse()
    elif fault == 'tf_stamp':
        info['tf_queries'][1]['timestamp_ns'] += 1
    elif fault == 'tf_stamp_overflow':
        info['tf_queries'][0]['timestamp_ns'] = 1 << 63
    elif fault == 'tf_pose_nan':
        info['tf_queries'][0]['pose'][0] = float('nan')
    elif fault == 'tf_quaternion':
        info['tf_queries'][1]['pose'][6] = 0.
    elif fault == 'motion_pose_bool':
        info['motion_pose'][0] = True
    elif fault == 'motion_quaternion':
        info['motion_pose'][6] = 0.
    elif fault == 'tf_query_count':
        info['tf_queries'].pop()
    elif fault == 'reversed_read':
        info['read_start_monotonic_ns'] = NOW+1
    elif fault == 'future_receive':
        info['read_end_monotonic_ns'] = NOW+1
    elif fault == 'reversed_sdk':
        info['read_start_sdk_clock_ns'] = SOURCE_NOW+1
    elif fault == 'sdk_alias':
        info['sdk_clock_ns'] += 1
    elif fault == 'received_alias':
        info['received_monotonic_ns'] += 1
    elif fault == 'state_outside_read':
        info['state_received_monotonic_ns'] = NOW+1
    elif fault == 'duration_mismatch':
        info['read_duration_s'] = .004
    elif fault == 'wall_jump':
        info['read_start_wall_ns'] += 1_000_001
    assert not rig.guard(observation(), info, None)
    assert rig.guard.last_decision.code == 'invalid_evidence'
    assert rig.guard.previous_source_ns == before


@pytest.mark.parametrize('after', [True, -1., float('nan'), float('inf'), '20', 1e20])
def test_invalid_after_is_diagnostic_and_does_not_commit(rig, after):
    assert not rig.guard(observation(), evidence(), after)
    assert rig.guard.last_decision.code == 'invalid_evidence'
    assert rig.guard.previous_source_ns == {}


@pytest.mark.parametrize('changes', [
    {'schema': True}, {'sequence': 0}, {'healthy': 1}, {'healthy': False},
    {'reason': 'fault'}, {'boot_id': 'different-boot'}, {'session_id': ''},
    {'actual_master': 'another-master'}, {'scale': 'utc'}, {'utc_offset_s': 36},
    {'utc_offset_valid': True}, {'leap61': 1}, {'leap59': 1}, {'ptp_timescale': 0},
    {'offset_at_reference_ns': float('nan')}, {'offset_at_reference_ns': True},
    {'drift_ppm': 100.000001}, {'drift_ppm': float('inf')}, {'residual_ns': -1.},
    {'path_delay_ns': 1_000_001}, {'empirical_error_ns': float('nan')},
    {'wall_minus_mono_ns': 1 << 63}, {'created_mono_ns': NOW+1},
    {'reference_mono_ns': NOW-1}, {'valid_until_ns': NOW+2_500_000_001},
])
def test_snapshot_types_ranges_identity_and_lease_are_rechecked(rig, changes):
    rig.snapshot = replace(rig.snapshot, **changes)
    assert not rig.guard(observation(), {}, None)
    assert rig.guard.last_decision.code == 'mapping_invalid'
    assert rig.guard.previous_source_ns == {}


@pytest.mark.parametrize('changes', [{'sequence': 1}, {'session_id': 'new-session'},
    {'expected_master': 'other', 'actual_master': 'other'}])
def test_snapshot_identity_and_sequence_follow_successful_observations(rig, changes):
    assert rig.guard(observation(), evidence(), None)
    before = rig.guard.previous_source_ns
    rig.snapshot = replace(rig.snapshot, **changes)
    assert not rig.guard(observation(), evidence(**dict.fromkeys(SOURCES, SOURCE_NOW-9_000_000)), None)
    assert rig.guard.last_decision.code == 'mapping_invalid'
    assert rig.guard.previous_source_ns == before


def test_failed_observation_does_not_commit_snapshot_sequence(rig):
    assert not rig.guard(observation(), {}, None)
    rig.snapshot = replace(rig.snapshot, sequence=1)
    assert rig.guard(observation(), evidence(), None)


def test_mapping_failure_has_a_stable_diagnostic_without_stale_fallback(rig):
    assert rig.guard(observation(), evidence(), None)
    before = rig.guard.previous_source_ns
    def disconnected():
        raise TimeoutError()
    rig.client.read = disconnected
    assert not rig.guard(observation(), evidence(), None)
    assert rig.guard.last_decision.code == 'mapping_unavailable'
    assert 'TimeoutError' in rig.guard.last_decision.detail
    assert rig.guard.previous_source_ns == before


def test_lease_exact_boundary_is_accepted_and_one_nanosecond_later_rejected(rig):
    rig.snapshot = replace(rig.snapshot, reference_mono_ns=NOW-2_500_000_000,
                           last_sample_mono_ns=NOW-2_500_000_000, valid_until_ns=NOW)
    assert rig.guard(observation(), evidence(), None)
    rig.now += 1
    assert not rig.guard(observation(), evidence(), None)
    assert rig.guard.last_decision.code == 'mapping_expired'


def test_clock_window_raw_ptp_offset_is_consumed_without_second_utc_correction(rig):
    snap = rig.snapshot
    window = ClockWindow(snap.expected_master, snap.boot_id, snap.session_id)
    window.feed_properties('currentUtcOffset 37\ncurrentUtcOffsetValid 0\n'
                           'leap61 0\nleap59 0\nptpTimescale 1', 6_000_000_000)
    window.feed_ptp('selected best master clock '+snap.expected_master,
                    6_000_000_000, ORIGIN+6_000_000_000)
    for seconds in range(6, 21, 2):
        window.feed_ptp(f'ptp4l[{seconds}.000000000]: master offset 55000000000 '
                        's2 freq 0 path delay 0', seconds*1_000_000_000,
                        ORIGIN+seconds*1_000_000_000)
    rig.snapshot = window.snapshot(NOW, ORIGIN+NOW)
    assert rig.guard(observation(), evidence(), None)
    assert rig.guard.last_decision.source_intervals_ns['joint'] == (19_988_000_000, 19_992_000_000)


def test_nonzero_drift_at_shifted_reference_changes_mapped_source_age(rig):
    # offset(ref=19s)=18s, +100ppm -> offset(19.99s)=18.000099s.
    rig.snapshot = replace(rig.snapshot, drift_ppm=100., reference_mono_ns=19_000_000_000,
                           last_sample_mono_ns=19_000_000_000, valid_until_ns=21_500_000_000)
    assert rig.guard(observation(), evidence(**dict.fromkeys(SOURCES, SOURCE_NOW-10_099_000)), None)
    assert rig.guard.last_decision.source_intervals_ns['joint'] == (19_987_899_789, 19_992_100_211)


def test_concurrent_same_observation_can_be_committed_only_once(rig):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: rig.guard(observation(), evidence(), None), range(2)))
    assert sorted(results) == [False, True]
    assert rig.guard.previous_source_ns == dict.fromkeys(SOURCES, SOURCE_NOW-10_000_000)
    assert rig.guard.last_decision.code == 'source_frozen:left_wrist'


def test_large_explicit_limits_are_legal_but_cannot_bypass_runtime_checks(rig):
    guard = ObservationFreshnessGuard(rig.client, limits(mapping_error_s=1.),
                                     monotonic_ns=lambda: rig.now)
    rig.snapshot = replace(rig.snapshot, empirical_error_ns=60_000_000.)
    assert not guard(observation(), evidence(), None)
    assert guard.last_decision.code == 'state_stale:joint'


def test_source_int64_max_boundary_is_valid_when_mapping_and_read_evidence_agree(rig):
    maximum = (1 << 63)-1
    origin = maximum-100_000_000_000
    rig.snapshot = replace(rig.snapshot, wall_minus_mono_ns=origin,
                           offset_at_reference_ns=-80_010_000_000.)
    info = evidence(**dict.fromkeys(SOURCES, maximum))
    info.update(read_start_wall_ns=origin+NOW-5_000_000, read_end_wall_ns=origin+NOW,
                read_start_sdk_clock_ns=maximum-5_000_000,
                read_end_sdk_clock_ns=maximum, sdk_clock_ns=maximum)
    assert rig.guard(observation(), info, None)
    assert rig.guard.last_decision.source_intervals_ns['joint'] == (19_988_000_000, 19_992_000_000)
