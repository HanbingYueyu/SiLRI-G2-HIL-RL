"""Fake-only CLI and lifecycle tests for real training orchestration."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from g2_local import real_train


TEMPLATE = Path(__file__).resolve().parents[1] / 'configs/g2_real_training_readonly.json'


@pytest.mark.parametrize('key,error', [('Y', None), ('F', RuntimeError), ('Y', KeyboardInterrupt)])
def test_terminal_single_byte_and_restoration(monkeypatch, key, error):
    import os
    import pty
    import select
    import termios
    master, slave = pty.openpty()
    try:
        with os.fdopen(os.dup(slave), 'r') as stream:
            monkeypatch.setattr(real_train.sys, 'stdin', stream)
            original = termios.tcgetattr(slave)
            def run():
                with real_train._TerminalInput() as terminal:
                    os.write(master, key.encode())
                    assert select.select([slave], [], [], .2)[0]
                    assert terminal.read_available(limit=1) == [key]
                    if error:
                        raise error()
            if error:
                with pytest.raises(error):
                    run()
            else:
                run()
            assert termios.tcgetattr(slave) == original
    finally:
        os.close(master)
        os.close(slave)


def _config(tmp_path, *, mode='train'):
    data = json.loads(TEMPLATE.read_text())
    data['mode'] = mode
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(data))
    path.chmod(0o400)
    return path


def test_preflight_writes_manifest_before_importing_gdk(tmp_path, monkeypatch):
    imported = []
    monkeypatch.setattr(real_train, 'import_gdk_runtime', lambda: imported.append(True))
    code = real_train.main(['actor', '--run-id', 'run-1', '--config',
                            str(_config(tmp_path)), '--output', str(tmp_path / 'run')])
    assert code == 2
    assert imported == []
    manifest = json.loads((tmp_path / 'run/run_manifest.json').read_text())
    assert manifest['run_id'] == 'run-1' and manifest['role'] == 'actor'
    assert json.loads((tmp_path / 'run/run_result.json').read_text())['status'] == 'motion_not_permitted'


def test_learner_cannot_request_motion_or_use_eval_mode(tmp_path):
    config = _config(tmp_path, mode='eval')
    assert real_train.main(['learner', '--run-id', 'r', '--config', str(config),
                            '--output', str(tmp_path / 'one')]) == 2
    assert real_train.main(['learner', '--run-id', 'r', '--config', str(config),
                            '--output', str(tmp_path / 'two'), '--allow-motion']) == 2
    assert not (tmp_path / 'one').exists()
    assert not (tmp_path / 'two').exists()


def test_eval_accepts_the_same_train_profile_as_learner(tmp_path):
    from g2_local.training_config import load_training_config
    loaded = load_training_config(_config(tmp_path), cli_allow_motion=False)
    args = SimpleNamespace(role='eval', allow_motion=False,
                           checkpoint=tmp_path / 'checkpoint.pt',
                           context=None, hid_device=None)
    real_train._validate_cli(args, loaded)


def test_rejects_invalid_run_id_before_creating_output(tmp_path):
    assert real_train.main(['actor', '--run-id', 'bad/id', '--config',
                            str(_config(tmp_path)), '--output', str(tmp_path / 'run')]) == 2
    assert not (tmp_path / 'run').exists()


def test_eval_summary_excludes_assisted_success_from_unassisted_denominator(tmp_path):
    output = tmp_path / 'run'
    output.mkdir(mode=0o700)
    evidence = real_train.RunEvidenceWriter(output, output / 'run_manifest.json', role='eval')
    sink = real_train.EvalTransitionSink('run-1', 'hash-1', 3, {'weight': 1}, evidence)
    sink.send_transition_batch([{'done': True, 'truncated': False,
                                 'complementary_info': {
                                     'episode_id': 'ep-1', 'step_id': 0,
                                     'is_intervention': True, 'success_label': True,
                                     'selected_action': (2., 0., 0., 0., 0., 0.),
                                     'executed_action': (1., 0., 0., 0., 0., 0.),
                                     'target_offset_m': (0., 0., 0.),
                                     'ee_reset_offset': (0.,) * 6,
                                     'actor_version': 3}}])
    summary = sink.summary()
    assert summary['intervention_free_success_rate_denominator'] == 0
    assert summary['intervention_free_success_rate_numerator'] == 0
    assert summary['episodes'] == 1
    assert summary['successes'] == 1
    assert summary['action_clipping_count'] == 1


def test_eval_frozen_parameters_heartbeat_without_timeout(tmp_path):
    output = tmp_path / 'run'
    output.mkdir(mode=0o700)
    evidence = real_train.RunEvidenceWriter(output, output / 'run_manifest.json', role='eval')
    sink = real_train.EvalTransitionSink('run-1', 'hash-1', 3, {'weight': 1}, evidence)
    first = sink.receive_latest_parameters()
    second = sink.receive_latest_parameters()
    assert first.version == second.version == 3
    assert second.message_sequence > first.message_sequence


def test_boolean_failure_gets_failure_stop_reason(tmp_path):
    output = tmp_path / 'run'
    output.mkdir(mode=0o700)
    evidence = real_train.RunEvidenceWriter(output, output / 'run_manifest.json', role='eval')
    sink = real_train.EvalTransitionSink('run-1', 'hash-1', 3, {}, evidence)
    sink.send_transition_batch([{'done': True, 'truncated': False,
                                 'complementary_info': {'episode_id': 'ep-1', 'step_id': 0,
                                                        'is_intervention': False,
                                                        'success_label': False,
                                                        'selected_action': (0.,) * 6,
                                                        'executed_action': (0.,) * 6,
                                                        'actor_version': 3}}])
    row = json.loads((output / 'episode_summaries.jsonl').read_text())
    assert row['success'] is False
    assert row['stop_reason'] == 'failure'


def test_eval_checkpoint_checks_identity_without_constructing_optimizer(tmp_path):
    import torch
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'schema': 1, 'run_id': 'run-1', 'config_hash': 'hash-1',
                'manifest_digest': 'hash-1', 'camera_keys': ('left_wrist', 'right_aux'),
                'image_size': 128, 'action_size': 6, 'version': 4,
                'published_version': 3,
                'published_actor_state': {'weight': torch.tensor([3.])}}, checkpoint)
    checkpoint.chmod(0o600)
    frozen = real_train.load_eval_checkpoint(checkpoint, run_id='run-1',
                                             config_hash='hash-1')
    assert frozen.version == 3
    assert frozen.actor_state['weight'].tolist() == [3.]
    with pytest.raises(ValueError, match='identity'):
        real_train.load_eval_checkpoint(checkpoint, run_id='other', config_hash='hash-1')


def test_train_records_episode_only_after_successful_upload(tmp_path):
    output = tmp_path / 'run'
    output.mkdir(mode=0o700)
    evidence = real_train.RunEvidenceWriter(output, output / 'run_manifest.json', role='actor')
    sent = []

    class Transport:
        def send_transition_batch(self, rows):
            sent.extend(rows)

    transport = real_train.EvidenceActorTransport(Transport(), evidence)
    row = {'done': True, 'truncated': False,
           'complementary_info': {'episode_id': 'episode-1', 'step_id': 0,
                                  'is_intervention': False, 'success_label': True,
                                  'selected_action': (0.,) * 6,
                                  'executed_action': (0.,) * 6,
                                  'actor_version': 2}}
    transport.send_transition_batch([row])
    assert sent == [row]
    summary = json.loads((output / 'episode_summaries.jsonl').read_text())
    assert summary['episode_id'] == 'episode-1'
    assert summary['success'] is True
    assert summary['intervention_ratio'] == 0


def test_freshness_rejection_is_recorded_without_transition(tmp_path):
    output = tmp_path / 'run'
    output.mkdir(mode=0o700)
    evidence = real_train.RunEvidenceWriter(output, output / 'run_manifest.json', role='actor')
    sink = real_train.EvalTransitionSink('run-1', 'hash-1', 1, {}, evidence)
    sink.telemetry('freshness_reject', episode_id='ep-1', code='camera_stale')
    sink.finalize_unfinished('actor_failure')
    row = json.loads((output / 'episode_summaries.jsonl').read_text())
    assert row['freshness_rejects'] == 1
    assert row['length'] == 0
    assert sink.summary()['intervention_free_success_rate_denominator'] == 0


def test_cli_records_actual_unconfirmed_actor_stop(tmp_path, monkeypatch):
    config = _config(tmp_path)
    class Loaded:
        mode = 'train'
        motion_permitted = True
        def write_manifest(self, output, *, run_id, role):
            output.mkdir(mode=0o700)
            manifest = output / 'run_manifest.json'
            manifest.write_text('{}')
            return manifest
    monkeypatch.setattr(real_train, 'load_training_config', lambda *a, **k: Loaded())
    monkeypatch.setattr(real_train, '_validate_cli', lambda *a: None)
    def fail_role(args, loaded, evidence):
        error = RuntimeError('learner disconnected')
        error.stop_unconfirmed = True
        raise error
    monkeypatch.setattr(real_train, 'run_role', fail_role)
    code = real_train.main(['actor', '--run-id', 'r', '--config', str(config),
                            '--output', str(tmp_path / 'run'), '--allow-motion'])
    assert code == 1
    assert json.loads((tmp_path / 'run/run_result.json').read_text())['status'] == 'stop_unconfirmed'


def test_actor_failure_evidence_error_keeps_unconfirmed_outcome():
    error = RuntimeError('Source observation freshness not confirmed: camera_stale')
    def failed_write(*args, **kwargs):
        raise OSError('evidence write failed')
    transport = SimpleNamespace(tracker=SimpleNamespace(finalize_unfinished=failed_write))
    runtime = SimpleNamespace(stop_confirmed=False, stop_reason='actor_failure',
                              freshness_rejects=1)
    evidence = SimpleNamespace(event=failed_write)
    real_train.record_actor_failure(runtime, transport, evidence, 'actor', error)
    assert error.stop_unconfirmed is True


def test_result_file_records_stop_outcome_when_event_append_fails(tmp_path):
    output = tmp_path / 'run'
    output.mkdir(mode=0o700)
    evidence = real_train.RunEvidenceWriter(output, output / 'run_manifest.json', role='actor')
    def fail_event(*args, **kwargs):
        raise OSError('events file unavailable')
    evidence.event = fail_event
    evidence.finish('stop_unconfirmed', error=RuntimeError('freshness failed'))
    assert json.loads((output / 'run_result.json').read_text())['status'] == 'stop_unconfirmed'


def test_interrupt_with_unconfirmed_stop_is_a_failure(tmp_path, monkeypatch):
    class Loaded:
        mode = 'train'
        motion_permitted = True
        def write_manifest(self, output, *, run_id, role):
            output.mkdir(mode=0o700)
            manifest = output / 'run_manifest.json'
            manifest.write_text('{}')
            return manifest
    monkeypatch.setattr(real_train, 'load_training_config', lambda *a, **k: Loaded())
    monkeypatch.setattr(real_train, '_validate_cli', lambda *a: None)
    def interrupt(args, loaded, evidence):
        error = KeyboardInterrupt()
        error.stop_unconfirmed = True
        raise error
    monkeypatch.setattr(real_train, 'run_role', interrupt)
    output = tmp_path / 'run'
    code = real_train.main(['actor', '--run-id', 'r', '--config', str(tmp_path / 'config'),
                            '--output', str(output), '--allow-motion'])
    assert code == 1
    assert json.loads((output / 'run_result.json').read_text())['status'] == 'stop_unconfirmed'


def test_summary_reads_boolean_label_from_validated_actor_transition(tmp_path):
    from test_g2_real_actor import actor_rig
    rig = actor_rig()
    step = rig.env.step
    def terminal(action):
        observation, reward, _, _, info = step(action)
        info['success_label'] = True
        return observation, reward, True, False, info
    rig.env.step = terminal
    rig.runtime.run(max_completed_steps=1)
    row = rig.transport.sent[0]
    assert row['complementary_info']['success_label'] is True
    output = tmp_path / 'run'
    output.mkdir(mode=0o700)
    evidence = real_train.RunEvidenceWriter(output, output / 'run_manifest.json', role='eval')
    sink = real_train.EvalTransitionSink('run-1', 'hash-1', 3, {}, evidence)
    sink.send_transition_batch([row])
    assert sink.summary()['successes'] == 1


def test_learner_passes_owned_output_checkpoint_path_to_runtime(tmp_path, monkeypatch):
    import grpc
    from g2_local import real_learner
    from lerobot.transport import services_pb2_grpc as rpc
    created = []
    class Learner:
        version = 0
        stopped = SimpleNamespace(wait=lambda timeout: True)
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.checkpoint_path = kwargs['checkpoint_path']
        def publish_parameters(self):
            return object()
        def snapshot_counts(self):
            return {}
    class Service:
        failure = None
        def __init__(self, learner):
            pass
        def publish(self, envelope):
            pass
        def close(self):
            pass
    class Server:
        def add_insecure_port(self, address):
            assert address == '127.0.0.1:50175'
            return 50175
        def start(self):
            pass
        def stop(self, grace):
            return SimpleNamespace(wait=lambda: None)
    monkeypatch.setattr(real_learner, 'RealLearnerRuntime', Learner)
    monkeypatch.setattr(real_learner, 'GrpcLearnerService', Service)
    monkeypatch.setattr(grpc, 'server', lambda pool: Server())
    monkeypatch.setattr(rpc, 'add_LearnerServiceServicer_to_server', lambda *a: None)
    output = tmp_path / 'run'
    output.mkdir(mode=0o700)
    evidence = real_train.RunEvidenceWriter(output, output / 'run_manifest.json', role='learner')
    loaded = SimpleNamespace(config_hash='hash-1', runtime=SimpleNamespace(
        learner_port=50175, transport_timeout_s=1.))
    args = SimpleNamespace(checkpoint=None, run_id='run-1')
    assert real_train._run_learner(args, loaded, evidence) == 0
    assert created[0]['checkpoint_path'] == output / 'checkpoint.pt'
