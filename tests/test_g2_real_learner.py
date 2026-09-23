"""Focused real Learner contract tests."""

from dataclasses import replace
import pytest
import torch
import numpy as np
import random
import threading

from g2_local.real_learner import GrpcLearnerService, RealLearnerRuntime, load_checkpoint
from g2_local.runtime import train_batch
from g2_local.training_config import OptimizationConfig, RuntimeConfig
from lerobot.transport import services_pb2 as pb
from lerobot.transport.utils import (bytes_to_state_dict, send_bytes_in_chunks,
                                     transitions_to_bytes)


def _config(*, optimization=None, runtime=None):
    optimization = optimization or OptimizationConfig(8, 4, 1, 1, 1, 2, 1e-4, 1e-4,
                                                      1e-4, 1e-4, 2, 2, 3)
    runtime = runtime or RuntimeConfig(1, 'cpu', '127.0.0.1', 50175, 4, 1., 2.,
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


def test_checkpoint_mid_utd_retains_remaining_updates_on_resume(tmp_path):
    optimization = replace(_config().optimization, utd_ratio=3,
                           publish_interval=1, checkpoint_interval=1)
    learner = RealLearnerRuntime(config=_config(optimization=optimization), run_id='run-1',
                                 checkpoint_path=tmp_path / 'checkpoint.pt')
    calls = 0

    def publish(_envelope):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError('stop after first checkpoint')

    learner.publish = publish
    learner.ingest([_row()])
    with pytest.raises(ConnectionError):
        learner.update_for_interactions()
    restored = load_checkpoint(tmp_path / 'checkpoint.pt', expected_run_id='run-1',
                               expected_config_hash='hash-1')
    assert restored.version == 1
    assert restored.snapshot_counts()['budget'] == 2
    assert len(restored.update_for_interactions()) == 2
    assert restored.runtime.version == 3


def test_warmup_preserves_pending_utd_credits():
    optimization = replace(_config().optimization, min_online_transitions=2)
    learner = RealLearnerRuntime(config=_config(optimization=optimization), run_id='run-1')
    learner.ingest([_row()])
    assert learner.update_for_interactions() == []
    assert learner.snapshot_counts()['budget'] == 2
    learner.ingest([_row(1)])
    assert len(learner.update_for_interactions()) == 4
    assert learner.snapshot_counts()['budget'] == 0


def test_heartbeat_uses_last_published_policy_below_publish_interval():
    learner = _learner()
    initial = learner.publish_parameters()
    learner.ingest([_row()])
    learner.update_once()
    heartbeat = learner.heartbeat_parameters()
    assert learner.version == 1
    assert heartbeat.version == initial.version == 0
    assert heartbeat.message_sequence > initial.message_sequence
    assert all(torch.equal(heartbeat.actor_state[key], initial.actor_state[key])
               for key in initial.actor_state)


def test_parameter_stream_serializes_last_published_state_below_interval():
    learner = _learner()
    initial = {key: value.detach().cpu().clone()
               for key, value in learner.policy.actor.state_dict().items()}
    learner.ingest([_row()])
    learner.update_once()
    service = GrpcLearnerService(learner)
    context = _StreamContext()
    stream = service.StreamParameters(pb.Empty(), context)
    chunks = []
    for chunk in stream:
        chunks.append(chunk.data)
        if chunk.transfer_state == pb.TransferState.TRANSFER_END:
            break
    payload = bytes_to_state_dict(b''.join(chunks))
    context.cancel()
    stream.close()
    assert payload['version'] == 0
    assert payload['message_sequence'] == 0
    assert all(torch.equal(payload['actor_state'][key], initial[key]) for key in initial)


@pytest.mark.parametrize('corrupt', ('next_state', 'reward'))
def test_resume_rejects_corrupted_replay_fields(tmp_path, corrupt):
    learner = _learner()
    learner.ingest([_row()])
    checkpoint = learner.save_checkpoint(tmp_path / 'checkpoint.pt')
    payload = torch.load(checkpoint, weights_only=False)
    if corrupt == 'next_state':
        payload['online_replay']['next_states']['observation.state'][0, 0] = .25
    else:
        payload['online_replay']['rewards'][0] = .25
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError):
        load_checkpoint(checkpoint, expected_run_id='run-1',
                        expected_config_hash='hash-1')


def test_resume_rejects_corrupted_occupied_episode_end_bit(tmp_path):
    learner = _learner()
    terminal = _row()
    terminal['done'] = True
    learner.ingest([terminal])
    checkpoint = learner.save_checkpoint(tmp_path / 'checkpoint.pt')
    assert load_checkpoint(checkpoint, expected_run_id='run-1',
                           expected_config_hash='hash-1').online_replay.episode_ends[0].item() is False
    payload = torch.load(checkpoint, weights_only=False)
    payload['online_replay']['episode_ends'][0] = True
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='episode end'):
        load_checkpoint(checkpoint, expected_run_id='run-1',
                        expected_config_hash='hash-1')


def test_large_float_reward_rejects_entire_batch_before_mutation():
    learner = _learner()
    bad = _row(1)
    bad['reward'] = 1e100
    before = learner.snapshot_counts()
    with pytest.raises(ValueError):
        learner.ingest([_row(), bad])
    assert learner.snapshot_counts() == before
    assert not learner.records and not learner.seen_transition_ids


class _StreamContext:
    def __init__(self, *, block_on_register=False):
        self.active = True
        self.callback = None
        self.registered = threading.Event()
        self.resume = threading.Event()
        if not block_on_register:
            self.resume.set()

    def is_active(self):
        return self.active

    def add_callback(self, callback):
        self.callback = callback
        self.registered.set()
        self.resume.wait(2.)
        return True

    def cancel(self):
        self.active = False
        if self.callback:
            self.callback()


@pytest.mark.parametrize('pending', (False, True))
def test_stream_cancel_during_wait_or_pending_envelope_stops_without_emit(pending):
    runtime = replace(_config().runtime, parameter_heartbeat_s=1.)
    learner = RealLearnerRuntime(config=_config(runtime=runtime), run_id='run-1')
    service = GrpcLearnerService(learner)
    context = _StreamContext(block_on_register=pending)
    if pending:
        service.publish(learner.publish_parameters())
    stream = service.StreamParameters(pb.Empty(), context)
    emitted = []

    def read_one():
        try:
            emitted.append(next(stream))
        except StopIteration:
            pass

    worker = threading.Thread(target=read_one)
    worker.start()
    assert context.registered.wait(1.)
    context.cancel()
    context.resume.set()
    worker.join(2.)
    assert not worker.is_alive()
    assert not emitted
    assert learner.stopped.is_set()


def test_transition_rpc_failure_wakes_waiting_parameter_stream():
    runtime = replace(_config().runtime, parameter_heartbeat_s=5.)
    learner = RealLearnerRuntime(config=_config(runtime=runtime), run_id='run-1')
    service = GrpcLearnerService(learner)
    context = _StreamContext()
    stream = service.StreamParameters(pb.Empty(), context)
    emitted = []

    def read_one():
        try:
            emitted.append(next(stream))
        except StopIteration:
            pass

    worker = threading.Thread(target=read_one)
    worker.start()
    assert context.registered.wait(1.)
    bad = _row()
    bad['complementary_info']['config_hash'] = 'wrong'
    packet = transitions_to_bytes([bad])
    with pytest.raises(ValueError):
        service.SendTransitions(send_bytes_in_chunks(packet, pb.Transition), None)
    worker.join(1.)
    assert not worker.is_alive()
    assert not emitted
    assert learner.stopped.is_set()
