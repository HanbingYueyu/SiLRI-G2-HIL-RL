"""Focused real Learner contract tests."""

import pytest
import torch
import numpy as np
import random

from g2_local.real_learner import GrpcLearnerService, RealLearnerRuntime, load_checkpoint
from g2_local.runtime import train_batch
from g2_local.training_config import OptimizationConfig, RuntimeConfig
from lerobot.transport import services_pb2 as pb
from lerobot.transport.utils import send_bytes_in_chunks, transitions_to_bytes


def _config():
    optimization = OptimizationConfig(8, 4, 1, 1, 1, 2, 1e-4, 1e-4,
                                      1e-4, 1e-4, 2, 2, 3)
    runtime = RuntimeConfig(1, 'cpu', '127.0.0.1', 50175, 4, 1., 2.,
                            1., .1, 1., .05)
    observation = type('Observation', (), {'camera_keys': ('left_wrist', 'right_aux'),
                                            'image_size': 128})()
    return type('Config', (), {'optimization': optimization, 'runtime': runtime,
                                'observation': observation, 'config_hash': 'hash-1'})()


def _row(step=0, *, human=False, executed=0.):
    obs = {'observation.state': torch.tensor([[0., 0., 0., 0., 0., 0., 1.]])}
    obs.update({f'observation.images.{key}': torch.zeros(1, 3, 128, 128)
                for key in ('left_wrist', 'right_aux')})
    action = (executed, 0., 0., 0., 0., 0.)
    info = dict(run_id='run-1', config_hash='hash-1', transition_id=f'run-1/ep-1/{step}',
                episode_id='ep-1', step_id=step, actor_version=0, synthetic=False,
                policy_action=(1., 0., 0., 0., 0., 0.),
                human_action=(0., 0., 0., 0., 0., 0.) if human else None,
                selected_action=(0., 0., 0., 0., 0., 0.) if human else
                                (1., 0., 0., 0., 0., 0.),
                executed_action=action, is_intervention=human,
                reward_source='environment', success_label=None,
                gate_summary={'fresh': True}, target_offset_m=(0., 0., 0.),
                ee_reset_offset=(0.,) * 6, approach_source='test',
                grasp_description='test', visual_reset_monotonic_ns=None,
                visual_confidence=None, upstream_frame_id=None)
    return dict(state=obs, next_state=obs, action=torch.tensor(action), reward=0.,
                done=False, truncated=False, complementary_info=info)


def _learner():
    torch.set_num_threads(2)
    return RealLearnerRuntime(config=_config(), run_id='run-1')


def test_duplicate_does_not_mutate_replay_or_update_budget():
    learner = _learner()
    row = _row(human=True)
    assert learner.ingest([row]).accepted == 1
    before = learner.snapshot_counts()
    assert learner.ingest([row]).duplicates == 1
    assert learner.snapshot_counts() == before


def test_invalid_row_in_batch_leaves_all_replay_untouched():
    learner = _learner()
    bad = _row(1)
    bad['complementary_info']['config_hash'] = 'wrong'
    with pytest.raises(ValueError):
        learner.ingest([_row(), bad])
    assert learner.snapshot_counts()['online'] == 0


def test_replay_stores_executed_action_and_only_interventions_in_human_pool():
    learner = _learner()
    learner.ingest([_row(human=True), _row(1)])
    assert len(learner.online_replay) == 2
    assert len(learner.human_replay) == 1
    assert torch.equal(learner.human_replay.actions[0], torch.zeros(6))
    assert torch.equal(learner.online_replay.actions[0], torch.zeros(6))


def test_resume_preserves_replay_position_and_requires_identity(tmp_path):
    learner = _learner()
    learner.ingest([_row(human=True)])
    checkpoint = learner.save_checkpoint(tmp_path / 'checkpoint.pt')
    restored = load_checkpoint(checkpoint, expected_run_id='run-1',
                               expected_config_hash='hash-1')
    assert restored.software_state_restored
    assert restored.physical_episode_state == 'WAITING_FOR_RESET'
    assert restored.online_replay.position == learner.online_replay.position
    assert restored.seen_transition_ids == learner.seen_transition_ids
    with pytest.raises(ValueError):
        load_checkpoint(checkpoint, expected_run_id='other',
                        expected_config_hash='hash-1')


def test_online_update_skips_human_optimizers_and_uses_configured_utd():
    learner = _learner()
    learner.ingest([_row()])
    assert len(learner.update_for_interactions()) == 2
    assert learner.version == 2
    assert len(learner.optimizers['expert'].state) == 0
    assert len(learner.optimizers['actor'].state) > 0
    assert learner.message_sequence == 0
    assert learner.update_for_interactions() == []


def test_human_update_advances_expert_and_actor_optimizer():
    learner = _learner()
    learner.ingest([_row(human=True)])
    metrics = learner.update_once()
    assert 'expert' in metrics and 'actor_bc' in metrics
    assert len(learner.optimizers['expert'].state) > 0


def test_grpc_ingress_rejects_entire_bad_batch_before_mutation():
    learner = _learner()
    service = GrpcLearnerService(learner)
    bad = _row(1)
    bad['complementary_info']['config_hash'] = 'wrong'
    packet = transitions_to_bytes([_row(), bad])
    with pytest.raises(ValueError):
        service.SendTransitions(send_bytes_in_chunks(packet, pb.Transition), None)
    assert len(learner.online_replay) == 0
    assert learner.stopped.is_set()


def test_heartbeat_keeps_version_and_state_with_new_sequence():
    learner = _learner()
    first = learner.publish_parameters()
    second = learner.publish_parameters()
    assert second.version == first.version == 0
    assert second.message_sequence == first.message_sequence + 1
    assert all(torch.equal(first.actor_state[key], second.actor_state[key])
               for key in first.actor_state)


def test_training_does_not_mutate_sampled_replay_batch():
    learner = _learner()
    learner.ingest([_row()])
    batch = learner.online_replay.sample(1)
    assert 'is_intervention' not in batch
    train_batch(learner.policy, learner.optimizers, batch, ('critic',))
    assert 'is_intervention' not in batch


def test_grpc_valid_batch_updates_once_per_configured_utd():
    learner = _learner()
    service = GrpcLearnerService(learner)
    packet = transitions_to_bytes([_row()])
    service.SendTransitions(send_bytes_in_chunks(packet, pb.Transition), None)
    assert learner.version == 2
    assert learner.snapshot_counts()['online'] == 1
    service.SendTransitions(send_bytes_in_chunks(packet, pb.Transition), None)
    assert learner.version == 2
    assert learner.snapshot_counts()['online'] == 1


def test_resume_rejects_symlink_and_corrupt_provenance(tmp_path):
    learner = _learner()
    learner.ingest([_row()])
    checkpoint = learner.save_checkpoint(tmp_path / 'checkpoint.pt')
    link = tmp_path / 'link.pt'
    link.symlink_to(checkpoint)
    with pytest.raises(ValueError):
        load_checkpoint(link, expected_run_id='run-1', expected_config_hash='hash-1')
    payload = torch.load(checkpoint, weights_only=False)
    payload['records'][0]['complementary_info']['transition_id'] = 'wrong'
    corrupt = tmp_path / 'corrupt.pt'
    torch.save(payload, corrupt)
    with pytest.raises(ValueError):
        load_checkpoint(corrupt, expected_run_id='run-1',
                        expected_config_hash='hash-1')


def test_resume_rejects_dedup_ids_that_do_not_match_records(tmp_path):
    learner = _learner()
    learner.ingest([_row()])
    checkpoint = learner.save_checkpoint(tmp_path / 'checkpoint.pt')
    payload = torch.load(checkpoint, weights_only=False)
    payload['seen_transition_ids'] = ['run-1/ep-1/999']
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError):
        load_checkpoint(checkpoint, expected_run_id='run-1',
                        expected_config_hash='hash-1')


def test_resume_rejects_group_writable_checkpoint_before_deserialization(tmp_path):
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'not a checkpoint')
    checkpoint.chmod(0o660)
    with pytest.raises(ValueError, match='trusted'):
        load_checkpoint(checkpoint, expected_run_id='run-1',
                        expected_config_hash='hash-1')


def test_resume_rejects_replay_action_mismatch(tmp_path):
    learner = _learner()
    learner.ingest([_row()])
    checkpoint = learner.save_checkpoint(tmp_path / 'checkpoint.pt')
    payload = torch.load(checkpoint, weights_only=False)
    payload['online_replay']['actions'][0, 0] = .75
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError):
        load_checkpoint(checkpoint, expected_run_id='run-1',
                        expected_config_hash='hash-1')


def test_checkpoint_restores_random_generators(tmp_path):
    learner = _learner()
    checkpoint = learner.save_checkpoint(tmp_path / 'checkpoint.pt')
    saved_torch = torch.get_rng_state().clone()
    saved_numpy = np.random.get_state()
    saved_python = random.getstate()
    torch.rand(3)
    np.random.rand(3)
    random.random()
    load_checkpoint(checkpoint, expected_run_id='run-1',
                    expected_config_hash='hash-1')
    assert torch.equal(torch.get_rng_state(), saved_torch)
    assert np.array_equal(np.random.get_state()[1], saved_numpy[1])
    assert random.getstate() == saved_python


def test_checkpoint_cadence_and_failed_publication_stop_updates(tmp_path):
    learner = RealLearnerRuntime(config=_config(), run_id='run-1',
                                 checkpoint_path=tmp_path / 'checkpoint.pt')
    learner.ingest([_row()])
    learner.update_once()
    assert not (tmp_path / 'checkpoint.pt').exists()
    learner.update_once()
    assert not (tmp_path / 'checkpoint.pt').exists()
    learner.update_once()
    assert (tmp_path / 'checkpoint.pt').exists()
    failing = _learner()
    failing.ingest([_row()])
    failing.publish = lambda _: (_ for _ in ()).throw(ConnectionError('stream failed'))
    failing.update_once()
    with pytest.raises(ConnectionError):
        failing.update_once()
    assert failing.stopped.is_set()
    before = failing.snapshot_counts()
    with pytest.raises(RuntimeError):
        failing.update_once()
    assert failing.snapshot_counts() == before
