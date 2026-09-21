"""Offline qualification exercises actual recorder output and hostile evidence."""
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import shutil
import stat

import pytest

from g2_local import freshness_audit
from test_g2_actor_audit import ActorRig


@pytest.fixture
def approval():
    assert importlib.util.find_spec('g2_local.freshness_approval') is not None
    return importlib.import_module('g2_local.freshness_approval')


@pytest.fixture(scope='module')
def recorded(tmp_path_factory):
    root = tmp_path_factory.mktemp('approval-source')
    for index in range(3):
        rig = ActorRig()
        rig.snapshot_fault = lambda snap, i=index: replace(snap, session_id=f'monitor-{i}')
        freshness_audit.run_audit(
            output=root/str(index), duration_s=121, actor_checkpoint=Path('trusted.pt'),
            device='cuda', warmup_steps=2, probe_factory=rig.probe,
            client_factory=rig.client, reader_factory=rig.reader,
            monotonic_ns=lambda: rig.now, sleep=rig.sleep)
        (root/str(index)/'monitor.jsonl').write_text(
            json.dumps(dict(kind='session', mono_ns=1, session_id=f'monitor-{index}',
                            motion_authorized=False, expected_master='044052.fffe.000010'))+'\n'+
            json.dumps(dict(kind='exit', mono_ns=2, returncode=0, healthy=False, reason='monitor_shutdown'))+'\n')
    return root


@pytest.fixture
def sessions(recorded, tmp_path):
    paths = []
    for index in range(3):
        target = tmp_path/str(index)
        shutil.copytree(recorded/str(index), target)
        paths.append(target)
    return paths


def qualify(approval, paths):
    for path in paths:
        approval.record_qualification(path, path/'monitor.jsonl', path/'qualification.json',
                                      process_listing=lambda: '')
    return paths


def mutate(path, fault):
    summary = json.loads((path/'summary.json').read_text())
    rows = [json.loads(line) for line in (path/'evidence.jsonl').read_text().splitlines()]
    samples = [r for r in rows if r['event'] == 'sample']
    fault(summary, rows, samples)
    (path/'summary.json').write_text(json.dumps(summary))
    (path/'evidence.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def proposed():
    return dict(camera_age_s=.050, state_age_s=.050, camera_skew_s=.010,
                mapping_error_s=.005, tf_position_error_m=.005, tf_rotation_error_rad=.02)


def test_qualifies_real_recorder_and_recomputes_raw_upper_bounds(approval, sessions):
    evidence = approval.validate_sessions(qualify(approval, sessions))
    assert evidence.worst_case['camera_age_s'] == .032
    assert evidence.worst_case['state_age_s'] == .032
    assert evidence.worst_case['camera_skew_s'] == .004
    assert evidence.worst_case['mapping_error_s'] == .002
    assert evidence.sessions[0]['metrics']['camera_age_ms']['left_wrist']['p99'] == 32.
    assert evidence.sessions[0]['observed_elapsed_s'] >= 120.


@pytest.mark.parametrize('count', [0, 1, 2, 4])
def test_requires_exactly_three(approval, sessions, count):
    with pytest.raises(ValueError, match='three independent'):
        approval.validate_sessions((sessions*2)[:count])


def test_duplicate_directory_is_not_independent(approval, sessions):
    qualify(approval, sessions)
    with pytest.raises(ValueError, match='three independent'):
        approval.validate_sessions([sessions[0]]*3)


@pytest.mark.parametrize('fault', [
    lambda s, r, a: s.update(status='failed'),
    lambda s, r, a: s.update(duration_s=119),
    lambda s, r, a: s.update(accepted_count=999),
    lambda s, r, a: s.update(rejected_count=1),
    lambda s, r, a: s['actor_metadata'].update(checkpoint_sha256='b'*64),
    lambda s, r, a: s['actor_metadata'].update(gpu_name='other GPU'),
    lambda s, r, a: s['actor_metadata']['policy_config'].update(extra=True),
    lambda s, r, a: a[1]['snapshot'].update(sequence=a[0]['snapshot']['sequence']),
    lambda s, r, a: a[1]['snapshot'].update(valid_until_ns=a[1]['snapshot']['valid_until_ns']+1),
    lambda s, r, a: a[1]['info']['source_timestamp_ns'].update(joint=a[0]['info']['source_timestamp_ns']['joint']),
    lambda s, r, a: s.update(motion_authorized=True),
    lambda s, r, a: s.update(thresholds_approved=True),
    lambda s, r, a: s.update(source_clock_identity_proven=True),
    lambda s, r, a: s.update(actions_discarded=False),
    lambda s, r, a: s['camera_skew_ms'].update(max=float('nan')),
    lambda s, r, a: s['camera_age_ms']['left_wrist'].update(max=1.),
    lambda s, r, a: a[0]['metrics']['camera_age_ms'].update(left_wrist=1.),
    lambda s, r, a: a[0]['source_intervals_ns']['joint'].__setitem__(0, 1),
    lambda s, r, a: a[0]['inference'].update(action_discarded=False),
    lambda s, r, a: a[0]['inference'].update(action_max=2.),
    lambda s, r, a: a[0]['info']['motion_pose'].__setitem__(0, .006),
    lambda s, r, a: a[0].update(inference_start_mono_ns=a[0]['received_mono_ns']+1),
    lambda s, r, a: s.update(formal_elapsed_s=999.),
])
def test_rejects_faults_even_when_qualification_hashes_match(approval, sessions, fault):
    mutate(sessions[0], fault)
    with pytest.raises(ValueError):
        approval.validate_sessions(qualify(approval, sessions))


def test_changed_raw_evidence_breaks_qualification_hash(approval, sessions):
    qualify(approval, sessions)
    with (sessions[0]/'evidence.jsonl').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='hash'):
        approval.validate_sessions(sessions)


@pytest.mark.parametrize('listing', ['123 python -m g2_local.freshness_audit',
                                    '123 python -m g2_local.clock_monitor',
                                    '123 /usr/sbin/ptp4l -i enp3s0',
                                    '123 /usr/sbin/pmc -u'])
def test_residual_process_prevents_attestation(approval, sessions, listing):
    path = sessions[0]
    with pytest.raises(ValueError, match='residual_process'):
        approval.record_qualification(path, path/'monitor.jsonl', path/'qualification.json',
                                      process_listing=lambda: listing)
    assert not (path/'qualification.json').exists()


def test_requires_recorded_clean_process_check(approval, sessions):
    with pytest.raises((ValueError, FileNotFoundError)):
        approval.validate_sessions(sessions)


@pytest.mark.parametrize('target', ['directory', 'ancestor', 'summary.json', 'evidence.jsonl', 'qualification.json'])
def test_symlink_paths_are_rejected(approval, sessions, tmp_path, target):
    qualify(approval, sessions)
    if target in ('directory', 'ancestor'):
        link = tmp_path/'link'
        link.symlink_to(sessions[0] if target == 'directory' else tmp_path, target_is_directory=True)
        sessions[0] = link if target == 'directory' else link/'0'
    else:
        path = sessions[0]/target
        saved = path.with_suffix('.saved')
        path.rename(saved)
        path.symlink_to(saved)
    with pytest.raises((ValueError, OSError)):
        approval.validate_sessions(sessions)


def test_approval_is_exclusive_readonly_and_never_authorizes_motion(approval, sessions, tmp_path):
    evidence = approval.validate_sessions(qualify(approval, sessions))
    output = tmp_path/'limits.json'
    artifact = approval.approve_limits(evidence, proposed(), output)
    assert json.loads(output.read_text()) == artifact
    assert artifact['limits'] == proposed()
    assert artifact['thresholds_approved'] is True
    assert artifact['motion_authorized'] is False
    assert artifact['margins']['camera_age_s'] == pytest.approx(.018)
    assert len(artifact['sessions']) == 3
    assert artifact['checkpoint_sha256'] == 'a'*64
    assert not stat.S_IMODE(output.stat().st_mode) & 0o222
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        approval.approve_limits(evidence, proposed(), output)
    assert output.read_bytes() == original


@pytest.mark.parametrize('name', list(proposed()))
@pytest.mark.parametrize('value', [None, True, 0., -1., float('nan'), float('inf')])
def test_all_six_values_are_explicit_finite_positive(approval, sessions, tmp_path, name, value):
    evidence = approval.validate_sessions(qualify(approval, sessions))
    limits = proposed()
    if value is None:
        del limits[name]
    else:
        limits[name] = value
    with pytest.raises(ValueError):
        approval.approve_limits(evidence, limits, tmp_path/'limits.json')
    assert not (tmp_path/'limits.json').exists()


@pytest.mark.parametrize('name,value', [('camera_age_s', .032), ('state_age_s', .031),
                                      ('camera_skew_s', .004), ('mapping_error_s', .002),
                                      ('tf_position_error_m', .004), ('tf_rotation_error_rad', .03)])
def test_worst_case_margins_and_task_geometry_are_required(approval, sessions, tmp_path, name, value):
    evidence = approval.validate_sessions(qualify(approval, sessions))
    limits = proposed()
    limits[name] = value
    with pytest.raises(ValueError):
        approval.approve_limits(evidence, limits, tmp_path/'limits.json')


def test_changed_evidence_after_validation_cannot_be_approved(approval, sessions, tmp_path):
    evidence = approval.validate_sessions(qualify(approval, sessions))
    mutate(sessions[0], lambda s, r, a: s.update(rejected_count=1))
    with pytest.raises(ValueError):
        approval.approve_limits(evidence, proposed(), tmp_path/'limits.json')


def test_qualification_cli_consumes_actual_monitor_kind_records(approval, sessions, monkeypatch):
    monkeypatch.setattr(approval, '_process_listing', lambda: '')
    path = sessions[0]
    assert approval.main(['qualify', '--audit-dir', str(path), '--monitor-evidence',
                          str(path/'monitor.jsonl'), '--output', str(path/'qualification.json')]) == 0
    result = json.loads((path/'qualification.json').read_text())
    assert result['process_check']['clean'] is True
    assert result['motion_authorized'] is False


def test_approval_cli_has_no_implicit_limits(approval, tmp_path):
    with pytest.raises(SystemExit) as error:
        approval.main(['approve', '--sessions', 'a', 'b', 'c', '--output', str(tmp_path/'limits.json')])
    assert error.value.code == 2
    assert not (tmp_path/'limits.json').exists()


def test_historical_boot_identity_needs_no_current_boot_match(approval, sessions):
    for index, path in enumerate(sessions):
        def change(summary, rows, samples):
            rows[3]['initial_snapshot']['boot_id'] = f'historical-boot-{index}'
            for sample in samples:
                sample['snapshot']['boot_id'] = f'historical-boot-{index}'
        mutate(path, change)
    assert len(approval.validate_sessions(qualify(approval, sessions)).sessions) == 3


def test_import_does_not_load_hardware_or_motion(approval, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name not in ('torch', 'agibot_gdk', 'g2_local.gdk_backend', 'g2_local.actor_inference',
                            'g2_local.motion_backend', 'g2_local.clock_monitor', 'g2_local.freshness_audit')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    importlib.reload(approval)


@pytest.mark.parametrize('field,value', [('checkpoint_schema', True), ('checkpoint_version', -1),
                                        ('checkpoint_version', True)])
def test_invalid_checkpoint_metadata_in_every_record_is_rejected(approval, sessions, field, value):
    for path in sessions:
        def change(summary, rows, samples):
            summary['actor_metadata'][field] = value
            rows[1]['metadata'][field] = value
            rows[2]['metadata'][field] = value
        mutate(path, change)
    with pytest.raises(ValueError, match='checkpoint'):
        approval.validate_sessions(qualify(approval, sessions))


def test_repeated_audit_identity_in_copied_directory_is_rejected(approval, sessions):
    identity = json.loads((sessions[0]/'summary.json').read_text())['audit_session_id']
    def change(summary, rows, samples):
        summary['audit_session_id'] = rows[0]['audit_session_id'] = identity
    mutate(sessions[1], change)
    with pytest.raises(ValueError, match='three independent'):
        approval.validate_sessions(qualify(approval, sessions))


def test_qualification_rejects_failed_monitor_exit(approval, sessions):
    path = sessions[0]
    records = [json.loads(line) for line in (path/'monitor.jsonl').read_text().splitlines()]
    records[-1]['returncode'] = 2
    (path/'monitor.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    with pytest.raises(ValueError, match='exit'):
        qualify(approval, sessions)
    assert not (path/'qualification.json').exists()


def test_hardlinked_evidence_cannot_hide_shared_identity(approval, sessions, tmp_path):
    qualify(approval, sessions)
    os.link(sessions[0]/'evidence.jsonl', tmp_path/'alias')
    with pytest.raises(ValueError, match='independent'):
        approval.validate_sessions(sessions)


def test_qualification_is_exclusive(approval, sessions):
    qualify(approval, sessions)
    before = (sessions[0]/'qualification.json').read_bytes()
    with pytest.raises(FileExistsError):
        qualify(approval, sessions)
    assert (sessions[0]/'qualification.json').read_bytes() == before


def test_read_only_validation_preserves_all_input_bytes_and_modes(approval, sessions):
    qualify(approval, sessions)
    files = [path/name for path in sessions for name in ('summary.json', 'evidence.jsonl', 'monitor.jsonl', 'qualification.json')]
    before = [(file.read_bytes(), file.stat().st_mode, file.stat().st_mtime_ns) for file in files]
    approval.validate_sessions(sessions)
    assert [(file.read_bytes(), file.stat().st_mode, file.stat().st_mtime_ns) for file in files] == before


def test_120_second_request_cannot_qualify_short_observed_span(approval, sessions):
    def shorten(summary, rows, samples):
        del rows[1004:-1]
        summary['accepted_count'] = summary['sample_count'] = 1000
    mutate(sessions[0], shorten)
    with pytest.raises(ValueError, match='duration'):
        approval.validate_sessions(qualify(approval, sessions))
