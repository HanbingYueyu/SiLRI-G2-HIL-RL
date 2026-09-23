"""Fake-only CLI and lifecycle tests for real training orchestration."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from g2_local import real_train


TEMPLATE = Path(__file__).resolve().parents[1] / 'configs/g2_real_training_readonly.json'


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
                                     'is_intervention': True, 'success_label': 'success',
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


def test_actor_disconnect_stops_before_transport_close(tmp_path):
    events = []
    output = tmp_path / 'run'
    output.mkdir(mode=0o700)
    evidence = real_train.RunEvidenceWriter(output, output / 'run_manifest.json', role='actor')

    class Runtime:
        def stop(self, reason):
            events.append('command_stop')

        def close_environment(self):
            events.append('env_close')

    class Transport:
        def close(self):
            events.append('transport_close')

    real_train.stop_actor(Runtime(), Transport(), evidence, 'learner_disconnect')
    assert events == ['command_stop', 'env_close', 'transport_close']
    assert json.loads((output / 'run_result.json').read_text())['status'] == 'learner_disconnect'


def test_stop_failure_is_recorded_and_transport_still_closes(tmp_path):
    events = []
    output = tmp_path / 'run'
    output.mkdir(mode=0o700)
    evidence = real_train.RunEvidenceWriter(output, output / 'run_manifest.json', role='actor')

    class Runtime:
        def stop(self, reason):
            events.append('command_stop')
            raise RuntimeError('stop not confirmed')

        def close_environment(self):
            events.append('env_close')

    class Transport:
        def close(self):
            events.append('transport_close')

    with pytest.raises(RuntimeError, match='stop not confirmed'):
        real_train.stop_actor(Runtime(), Transport(), evidence, 'learner_disconnect')
    assert events == ['command_stop', 'env_close', 'transport_close']
    assert json.loads((output / 'run_result.json').read_text())['status'] == 'stop_unconfirmed'


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
                                  'is_intervention': False, 'success_label': 'success',
                                  'selected_action': (0.,) * 6,
                                  'executed_action': (0.,) * 6,
                                  'actor_version': 2}}
    transport.send_transition_batch([row])
    assert sent == [row]
    summary = json.loads((output / 'episode_summaries.jsonl').read_text())
    assert summary['episode_id'] == 'episode-1'
    assert summary['success'] is True
    assert summary['intervention_ratio'] == 0
