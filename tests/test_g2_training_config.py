"""Training configuration is exact, auditable, and never grants motion alone."""

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from g2_local.config import HingeInsertTaskConfig
from g2_local.training_config import load_training_config


def valid_payload():
    return {
        'schema': 1, 'mode': 'train', 'requested_motion': False,
        'task': {'control_hz': 10.0, 'max_episode_steps': 80,
                 'fix_gripper': True, 'action_scale': [.0015, .0015, .0015,
                                                       .026, .026, .026],
                 'success_reward': 10.0, 'failure_reward': -1.0,
                 'step_reward': -.05, 'reward_source': 'human',
                 'target_xy_range_m': .05, 'ee_xyz_range_m': .003,
                 'ee_rpy_range_rad': .008726646259971648},
        'motion': {'workspace_low': [.20, .20, .70],
                   'workspace_high': [.40, .50, 1.00], 'control_mode': 1,
                   'command_timeout_s': .25, 'send_timeout_s': .05,
                   'stop_timeout_s': 1.0, 'reader_timeout_s': 2.0,
                   'command_lifetime_s': .1, 'send_rate_hz': 50.0,
                   'adapter_root': '/home/flyfuture/g2_hinge_assembly'},
        'freshness': {'camera_age_s': .1, 'state_age_s': .05,
                      'camera_skew_s': .05, 'tf_position_error_m': .005,
                      'tf_rotation_error_rad': .02, 'mapping_error_s': .005},
        'observation': {'camera_keys': ['left_wrist', 'right_aux'],
                        'image_size': 128,
                        'camera_rois': {'left_wrist': [0, 0, 1280, 1056],
                                        'right_aux': [0, 0, 1280, 1056]},
                        'raw_rgb_logging': False},
        'intervention': {'axis_map': [-2, -1, -3, -5, -4, -6], 'left_button': 0,
                         'right_button': 1, 'engage_deadzone': .12,
                         'release_deadzone': .08, 'release_hold_s': .25,
                         'report_max_age_s': .25},
        'optimization': {'online_capacity': 100000, 'human_capacity': 50000,
                         'min_online_transitions': 256, 'online_batch_size': 128,
                         'human_batch_size': 128, 'utd_ratio': 1,
                         'actor_lr': 3e-4, 'critic_lr': 3e-4,
                         'expert_lr': 3e-4, 'lagrange_lr': 3e-4,
                         'target_update_interval': 1, 'publish_interval': 1,
                         'checkpoint_interval': 1000},
        'runtime': {'seed': 1234, 'device': 'cuda', 'learner_host': '127.0.0.1',
                    'learner_port': 50175, 'queue_capacity': 64,
                    'queue_put_timeout_s': 1.0, 'transport_timeout_s': 5.0,
                    'context_max_age_s': 30.0, 'parameter_heartbeat_s': 1.0,
                    'learner_silence_timeout_s': 5.0,
                    'operator_poll_interval_s': .01},
        'commissioning': {'profile': 'unapproved', 'clock_socket': '/run/g2/clock.sock',
                          'expected_master': '044052.fffe.000010',
                          'evidence': []}}


def write_config(tmp_path, payload=None):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(valid_payload() if payload is None else payload))
    return path


def test_config_needs_cli_and_approved_evidence_before_motion(tmp_path):
    payload = valid_payload(); payload['requested_motion'] = True
    loaded = load_training_config(write_config(tmp_path, payload), cli_allow_motion=False)
    assert loaded.motion_permitted is False
    assert loaded.task.action_scale == (.0015, .0015, .0015, .026, .026, .026)
    assert loaded.task.failure_reward == -1.0
    assert load_training_config(write_config(tmp_path, payload), cli_allow_motion=True).motion_permitted is False


def test_config_rejects_image_size_not_supported_by_policy(tmp_path):
    payload = valid_payload()
    payload['observation']['image_size'] = 160
    with pytest.raises(ValueError, match='image_size.*128'):
        load_training_config(write_config(tmp_path, payload), cli_allow_motion=False)


def test_manifest_is_canonical_and_refuses_existing_output(tmp_path):
    loaded = load_training_config(write_config(tmp_path), cli_allow_motion=False)
    manifest = loaded.write_manifest(tmp_path / 'run', run_id='run-1', role='learner')
    data = json.loads(manifest.read_text())
    assert data['config_sha256'] == loaded.config_hash
    assert data['config'] == valid_payload()
    assert manifest.name == 'run_manifest.json'
    assert manifest.stat().st_mode & 0o777 == 0o400
    assert manifest.parent.stat().st_mode & 0o777 == 0o700
    assert json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n' == manifest.read_text()
    with pytest.raises(FileExistsError):
        loaded.write_manifest(tmp_path / 'run', run_id='run-1', role='learner')


def bad_workspace(payload):
    payload['motion']['workspace_high'] = [.1, .1, .1]
    return payload


def missing_freshness(payload):
    del payload['freshness']
    return payload


def reversed_deadzone_hysteresis(payload):
    payload['intervention']['release_deadzone'] = .2
    return payload


def unknown_key(payload):
    payload['unknown'] = 1
    return payload


@pytest.mark.parametrize('mutation', [bad_workspace, missing_freshness,
                                      reversed_deadzone_hysteresis, unknown_key])
def test_invalid_or_ambiguous_config_is_rejected(tmp_path, mutation):
    with pytest.raises(ValueError):
        load_training_config(write_config(tmp_path, mutation(valid_payload())), cli_allow_motion=False)


@pytest.mark.parametrize('path,value', [
    (('task', 'failure_reward'), True), (('task', 'success_reward'), float('nan')),
    (('freshness', 'camera_age_s'), True), (('motion', 'send_rate_hz'), float('inf')),
    (('runtime', 'queue_capacity'), True), (('observation', 'image_size'), True),
    (('intervention', 'axis_map'), [True, -1, -3]),
    (('optimization', 'actor_lr'), float('nan')),
])
def test_bool_numbers_and_nonfinite_numbers_are_rejected(tmp_path, path, value):
    payload = valid_payload()
    payload[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        load_training_config(write_config(tmp_path, payload), cli_allow_motion=False)


@pytest.mark.parametrize('section,key', [
    ('task', 'failure_reward'), ('motion', 'workspace_low'),
    ('observation', 'camera_rois'), ('runtime', 'learner_port'),
    ('commissioning', 'evidence'),
])
def test_missing_nested_key_is_rejected(tmp_path, section, key):
    payload = valid_payload()
    del payload[section][key]
    with pytest.raises(ValueError):
        load_training_config(write_config(tmp_path, payload), cli_allow_motion=False)


@pytest.mark.parametrize('section,key,value', [
    ('observation', 'camera_rois', {'left_wrist': [0, 0, 1280, 1056]}),
    ('observation', 'camera_rois', {'left_wrist': [0, 0, 0, 1056], 'right_aux': [0, 0, 1280, 1056]}),
    ('motion', 'command_lifetime_s', .3),
    ('runtime', 'learner_silence_timeout_s', .5),
    ('runtime', 'operator_poll_interval_s', .3),
    ('intervention', 'axis_map', [-2, -2, -3]),
])
def test_invalid_relationships_are_rejected(tmp_path, section, key, value):
    payload = valid_payload()
    payload[section][key] = value
    with pytest.raises(ValueError):
        load_training_config(write_config(tmp_path, payload), cli_allow_motion=False)


def test_duplicate_json_key_and_symlinked_config_are_rejected(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text('{"schema":1,"schema":1}')
    with pytest.raises(ValueError):
        load_training_config(path, cli_allow_motion=False)
    path.write_text(json.dumps(valid_payload()))
    link = tmp_path / 'link.json'
    link.symlink_to(path)
    with pytest.raises(ValueError):
        load_training_config(link, cli_allow_motion=False)


def test_approved_profile_requires_all_verified_evidence_and_both_flags(tmp_path):
    payload = valid_payload()
    payload['requested_motion'] = True
    payload['commissioning']['profile'] = 'approved'
    names = ('freshness_approval', 'xyz_rpy_direction_scale',
             'software_stop_lease_expiry', 'hardware_estop')
    evidence = []
    for name in names:
        path = tmp_path / f'{name}.json'
        content = json.dumps({'schema': 1, 'limits': payload['freshness'],
                              'thresholds_approved': True, 'motion_authorized': False}) if name == 'freshness_approval' else name
        path.write_text(content)
        evidence.append({'kind': name, 'path': str(path),
                         'sha256': hashlib.sha256(content.encode()).hexdigest()})
    payload['commissioning']['evidence'] = evidence
    config_path = write_config(tmp_path, payload)
    assert load_training_config(config_path, cli_allow_motion=False).motion_permitted is False
    assert load_training_config(config_path, cli_allow_motion=True).motion_permitted is True
    payload['freshness']['camera_age_s'] = .2
    with pytest.raises(ValueError, match='approved freshness'):
        load_training_config(write_config(tmp_path, payload), cli_allow_motion=True)
    payload['freshness']['camera_age_s'] = .1
    payload['requested_motion'] = False
    assert load_training_config(write_config(tmp_path, payload), cli_allow_motion=True).motion_permitted is False
    payload['requested_motion'] = True
    payload['commissioning']['evidence'] = evidence[:-1]
    assert load_training_config(write_config(tmp_path, payload), cli_allow_motion=True).motion_permitted is False
    payload['commissioning']['evidence'] = evidence
    evidence[0]['sha256'] = '0' * 64
    assert load_training_config(write_config(tmp_path, payload), cli_allow_motion=True).motion_permitted is False


def test_symlinked_evidence_is_rejected(tmp_path):
    payload = valid_payload()
    path = tmp_path / 'evidence.txt'
    path.write_text('freshness')
    link = tmp_path / 'evidence-link.txt'
    link.symlink_to(path)
    payload['commissioning']['evidence'] = [{'kind': 'freshness_approval', 'path': str(link),
                                               'sha256': hashlib.sha256(b'freshness').hexdigest()}]
    with pytest.raises(ValueError):
        load_training_config(write_config(tmp_path, payload), cli_allow_motion=False)


def test_symlinked_parent_of_config_is_rejected(tmp_path):
    actual = tmp_path / 'actual'
    actual.mkdir()
    config = write_config(actual)
    alias = tmp_path / 'alias'
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError):
        load_training_config(alias / config.name, cli_allow_motion=False)


def test_symlinked_parent_of_evidence_is_rejected(tmp_path):
    actual = tmp_path / 'actual'
    actual.mkdir()
    evidence = actual / 'approval.txt'
    evidence.write_text('freshness')
    alias = tmp_path / 'alias'
    alias.symlink_to(actual, target_is_directory=True)
    payload = valid_payload()
    payload['commissioning']['evidence'] = [{
        'kind': 'freshness_approval', 'path': str(alias / evidence.name),
        'sha256': hashlib.sha256(b'freshness').hexdigest(),
    }]
    with pytest.raises(ValueError):
        load_training_config(write_config(tmp_path, payload), cli_allow_motion=False)


def test_manifest_directory_swap_cannot_write_into_existing_directory(tmp_path, monkeypatch):
    loaded = load_training_config(write_config(tmp_path), cli_allow_motion=False)
    output = tmp_path / 'run'
    moved = tmp_path / 'moved-new-run'
    rival = tmp_path / 'existing-run'
    rival.mkdir()
    (rival / 'sentinel').write_text('existing')
    original_open = os.open
    swapped = False

    def swap_before_manifest_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and Path(path).name == 'run_manifest.json' and flags & os.O_EXCL:
            if output.exists():
                output.rename(moved)
            rival.rename(output)
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, 'open', swap_before_manifest_open)
    with pytest.raises(FileExistsError):
        loaded.write_manifest(output, run_id='run-1', role='learner')
    assert swapped
    assert (output / 'sentinel').read_text() == 'existing'
    assert not (output / 'run_manifest.json').exists()


def test_manifest_stage_swap_before_open_rejects_existing_directory(tmp_path, monkeypatch):
    loaded = load_training_config(write_config(tmp_path), cli_allow_motion=False)
    substitute = tmp_path / 'existing-stage'
    substitute.mkdir()
    (substitute / 'sentinel').write_text('existing')
    original_mkdir = os.mkdir
    stage_name = None

    def replace_new_stage(name, mode=0o777, *, dir_fd=None):
        nonlocal stage_name
        original_mkdir(name, mode, dir_fd=dir_fd)
        if name.startswith('.run.manifest-'):
            stage_name = name
            os.rename(name, 'moved-new-stage', src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            os.rename(substitute.name, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)

    monkeypatch.setattr(os, 'mkdir', replace_new_stage)
    with pytest.raises(ValueError):
        loaded.write_manifest(tmp_path / 'run', run_id='run-1', role='learner')
    assert stage_name is not None
    assert (tmp_path / stage_name / 'sentinel').read_text() == 'existing'
    assert not (tmp_path / stage_name / 'run_manifest.json').exists()
    assert not (tmp_path / 'run').exists()


def test_failed_manifest_cleanup_does_not_remove_substituted_stage(tmp_path, monkeypatch):
    loaded = load_training_config(write_config(tmp_path), cli_allow_motion=False)
    substitute = tmp_path / 'existing-empty-stage'
    substitute.mkdir()
    output = tmp_path / 'run'
    original_open = os.open
    stage_name = None

    # The stage name is resolved from the pinned descriptor; the rename is
    # performed in the test's known parent rather than through production code.
    def swap_with_parent(path, flags, *args, **kwargs):
        nonlocal stage_name
        if stage_name is None and Path(path).name == 'run_manifest.json' and flags & os.O_EXCL:
            stage_name = Path(os.readlink(f"/proc/self/fd/{kwargs['dir_fd']}")).name
            (tmp_path / stage_name).rename(tmp_path / 'moved-new-stage')
            substitute.rename(tmp_path / stage_name)
            output.mkdir()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, 'open', swap_with_parent)
    with pytest.raises(FileExistsError):
        loaded.write_manifest(output, run_id='run-1', role='learner')
    assert stage_name is not None
    assert (tmp_path / stage_name).is_dir()
    assert not (tmp_path / stage_name / 'run_manifest.json').exists()


def test_manifest_parent_must_be_owned_and_private(tmp_path):
    loaded = load_training_config(write_config(tmp_path), cli_allow_motion=False)
    shared = tmp_path / 'shared'
    shared.mkdir()
    shared.chmod(0o777)
    with pytest.raises(ValueError):
        loaded.write_manifest(shared / 'run', run_id='run-1', role='learner')
    assert not (shared / 'run').exists()


def test_loaded_payload_cannot_be_mutated_and_hash_is_canonical(tmp_path):
    payload = valid_payload()
    loaded = load_training_config(write_config(tmp_path, payload), cli_allow_motion=False)
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=False).encode()).hexdigest()
    assert loaded.config_hash == expected
    with pytest.raises(TypeError):
        loaded.canonical_payload['schema'] = 2
    with pytest.raises(TypeError):
        loaded.canonical_payload['task']['failure_reward'] = 0
    assert load_training_config(write_config(tmp_path, deepcopy(payload)), cli_allow_motion=False).config_hash == expected


def test_hinge_failure_reward_requires_finite_number():
    for bad in (True, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            HingeInsertTaskConfig(failure_reward=bad)


def test_checked_in_example_stays_read_only_and_unapproved():
    path = Path(__file__).resolve().parents[1] / 'configs/g2_real_training_readonly.json'
    loaded = load_training_config(path, cli_allow_motion=True)
    assert loaded.requested_motion is False
    assert loaded.motion_permitted is False
    assert loaded.commissioning.profile == 'unapproved'
    assert loaded.commissioning.evidence == ()
    assert loaded.intervention.axis_map == (-2, -1, -3, -5, -4, -6)
