"""Fake-only checks for the commissioned actor boundary."""

from types import SimpleNamespace
import threading
import grpc
import json
import numpy as np
import pytest
import torch

from g2_local.contract import EpisodeContext
from g2_local.operator_control import EpisodeContextInbox
from g2_local.real_actor import (GrpcActorTransport, ParameterEnvelope, RealActorRuntime,
                                 validate_parameter_envelope, validate_real_transition)


def observation():
    return {'state': np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            'left_wrist': np.zeros((128, 128, 3), dtype=np.uint8),
            'right_aux': np.zeros((128, 128, 3), dtype=np.uint8)}


class FakePolicy:
    def __init__(self):
        self.actor = self
        self.loaded_versions = []
        self.weight = torch.zeros(1)

    def state_dict(self):
        return {'weight': self.weight}

    def load_state_dict(self, state, strict=True):
        assert strict and set(state) == {'weight'}
        self.weight = state['weight'].clone()
        self.loaded_versions.append(int(state['weight'].item()))

    def eval(self):
        return self

    def select_action(self, state):
        return (torch.tensor([[0., .5, 0., 0., 0., 0.]]),)


class FakeTransport:
    def __init__(self, messages, blocked=False):
        self.messages = list(messages)
        self.sent = []
        self.blocked = blocked
        self.closed = False

    def receive_latest_parameters(self):
        return self.messages.pop(0) if self.messages else None

    def assert_alive(self):
        return None

    def send_transition_batch(self, rows):
        if self.blocked:
            raise TimeoutError('transition uplink backpressure')
        self.sent.extend(rows)

    def close(self):
        self.closed = True


class FakeCoordinator:
    def __init__(self):
        self.running = False
        self.context = None
        self.step_id = 0
        self.completed_step_token = None
        self.intervention = FakeIntervention()

    def offer_context(self, context):
        self.context = context

    def observe_start_frame(self):
        self.running = True
        self.step_id = 0
        return True

    def begin_step(self):
        token = SimpleNamespace(episode_id=self.context.episode_id, step_id=self.step_id)
        self.step_id += 1
        self.active_token = token
        return token

    @property
    def active_step_token(self):
        return getattr(self, 'active_token', None)

    def abort_step(self, token):
        self.running = False
        self.active_token = None

    def seal_episode(self, token):
        assert token.step_id == self.step_id - 1
        self.running = False
        self.context = None


class FakeEnv:
    def __init__(self, coordinator, *, completes_outcome=True):
        self.coordinator = coordinator
        self.completes_outcome = completes_outcome
        self.truncate_first = False
        self.step_calls = 0
        self.refresh_calls = 0
        self.reset_calls = 0
        self.during_step = None
        self.close_error = None
        self.backend = SimpleNamespace(stop_calls=0)
        self.closed = False

        def stop():
            self.backend.stop_calls += 1
        self.backend.stop = stop

    def reset(self, *, options):
        assert self.coordinator.running
        self.reset_calls += 1
        assert isinstance(options['context'], EpisodeContext)
        return observation(), {}

    def refresh_observation(self):
        self.refresh_calls += 1
        obs = observation()
        obs['state'][0] = self.refresh_calls / 100.
        return obs

    def step(self, action):
        self.step_calls += 1
        if self.during_step is not None:
            self.during_step()
        if self.completes_outcome:
            self.coordinator.completed_step_token = self.coordinator.active_token
            self.coordinator.active_token = None
        return observation(), 0., False, self.truncate_first and self.step_calls == 1, {
            'policy_action': tuple(action), 'human_action': (0., .25, 0., 0., 0., 0.),
            'selected_action': (0., .25, 0., 0., 0., 0.),
            'executed_action': np.array([0., .25, 0., 0., 0., 0.], dtype=np.float32),
            'is_intervention': True, 'reward_source': 'human', 'success_label': None,
            'target_offset_m': (0., 0., 0.), 'ee_reset_offset': (0.,) * 6,
        }

    def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class FakeIntervention:
    def __init__(self):
        self.gate = SimpleNamespace(fresh=True)
        self.polls = 0

    def __call__(self):
        self.polls += 1
        return False, None


def actor_rig(*, versions=((3, 8),), blocked=False, capacity=1):
    policy, coordinator = FakePolicy(), FakeCoordinator()
    env = FakeEnv(coordinator)
    envs = []
    messages = [ParameterEnvelope('run-1', 'hash-1', version, sequence,
                                  {'weight': torch.tensor([float(version)])})
                for version, sequence in versions]
    transport = FakeTransport(messages, blocked)
    context = EpisodeContext('episode-1', (0., 0., 0.), 'visual', 'grasp',
                             visual_reset_monotonic_ns=1)
    def env_factory(config, coordinator):
        assert coordinator.running
        new_env = env if not envs else FakeEnv(coordinator)
        envs.append(new_env)
        return new_env
    runtime = RealActorRuntime(
        config=SimpleNamespace(runtime=SimpleNamespace(queue_capacity=capacity,
            queue_put_timeout_s=.01, learner_silence_timeout_s=10.,
            operator_poll_interval_s=.001, device='cpu'),
            observation=SimpleNamespace(image_size=128)),
        run_id='run-1', config_hash='hash-1',
        coordinator=coordinator, context_source=SimpleNamespace(read_new=lambda: context),
        transport=transport, env_factory=env_factory, policy=policy)
    return SimpleNamespace(runtime=runtime, policy=policy, env=env,
                           envs=envs, coordinator=coordinator, transport=transport)


def test_actor_uploads_driver_confirmed_action_with_identity():
    rig = actor_rig()
    rig.runtime.run(max_completed_steps=1)
    row = rig.transport.sent[0]
    assert row['action'].tolist() == [0., .25, 0., 0., 0., 0.]
    info = row['complementary_info']
    assert info['policy_action'] != info['executed_action']
    assert info['selected_action'] == (0., .25, 0., 0., 0., 0.)
    assert info['transition_id'] == 'run-1/episode-1/0'
    assert info['synthetic'] is False
    assert rig.env.closed
    assert rig.transport.closed
    assert rig.coordinator.intervention.polls >= 1


def test_actor_emits_real_step_timing_outside_transition_payload():
    rig = actor_rig()
    events = []
    rig.runtime.telemetry = lambda kind, **fields: events.append((kind, fields))
    rig.runtime.run(max_completed_steps=2)
    steps = [fields for kind, fields in events if kind == 'step']
    assert len(steps) == 2
    assert steps[0]['inference_latency_s'] >= 0
    assert steps[1]['control_period_s'] > 0
    assert 'inference_latency_s' not in rig.transport.sent[0]['complementary_info']


def test_actor_reports_freshness_reject_and_unconfirmed_stop():
    rig = actor_rig()
    events = []
    rig.runtime.telemetry = lambda kind, **fields: events.append((kind, fields))
    def reject():
        raise RuntimeError('Source observation freshness not confirmed: camera_stale')
    rig.env.refresh_observation = reject
    def failed_stop():
        events.append(('command_stop', {}))
        raise RuntimeError('physical stop unconfirmed')
    rig.env.backend.stop = failed_stop
    original_close = rig.env.close
    def close_env():
        events.append(('env_close', {}))
        original_close()
    rig.env.close = close_env
    original_transport_close = rig.transport.close
    def close_transport():
        events.append(('transport_close', {}))
        original_transport_close()
    rig.transport.close = close_transport
    with pytest.raises(RuntimeError, match='freshness not confirmed'):
        rig.runtime.run(max_completed_steps=1)
    assert any(kind == 'freshness_reject' and fields['code'] == 'camera_stale'
               for kind, fields in events)
    assert rig.runtime.freshness_rejects == 1
    assert rig.runtime.stop_confirmed is False
    assert rig.transport.closed
    kinds = [kind for kind, _ in events]
    assert kinds.index('command_stop') < kinds.index('env_close') < kinds.index('transport_close')


def test_actor_counts_feedback_lease_reject():
    rig = actor_rig()
    events = []
    rig.runtime.telemetry = lambda kind, **fields: events.append((kind, fields))
    def reject(action):
        raise RuntimeError('Feedback freshness not explicitly confirmed')
    rig.env.step = reject
    with pytest.raises(RuntimeError, match='Feedback freshness'):
        rig.runtime.run(max_completed_steps=1)
    assert rig.runtime.freshness_rejects == 1
    assert ('freshness_reject', {'episode_id': 'episode-1', 'code': 'feedback_lease'}) in events


def test_freshness_telemetry_failure_still_attempts_stop_and_keeps_unconfirmed_result():
    rig = actor_rig()
    events = []
    def telemetry(kind, **fields):
        if kind == 'freshness_reject':
            raise OSError('evidence write failed')
    rig.runtime.telemetry = telemetry
    def reject():
        raise RuntimeError('Source observation freshness not confirmed: camera_stale')
    rig.env.refresh_observation = reject
    def failed_stop():
        events.append('command_stop')
        raise RuntimeError('physical stop unconfirmed')
    rig.env.backend.stop = failed_stop
    rig.env.close_error = RuntimeError('physical stop unconfirmed in cleanup')
    with pytest.raises(RuntimeError, match='Source observation freshness not confirmed') as result:
        rig.runtime.run(max_completed_steps=1)
    assert events == ['command_stop']
    assert rig.runtime.stop_confirmed is False
    assert result.value.stop_unconfirmed is True
    assert rig.transport.closed


def test_actor_rejects_rolled_back_parameters_before_loading():
    rig = actor_rig(versions=((3, 8), (2, 9)))
    rig.runtime.accept_latest_parameters()
    with pytest.raises(RuntimeError, match='rolled-back parameter version'):
        rig.runtime.accept_latest_parameters()
    assert rig.policy.loaded_versions == [3]


def test_same_version_heartbeat_refreshes_sequence_without_reload():
    rig = actor_rig(versions=((3, 8), (3, 9)))
    rig.runtime.accept_latest_parameters()
    rig.runtime.accept_latest_parameters()
    assert rig.policy.loaded_versions == [3]
    assert rig.runtime.last_message_sequence == 9


def test_same_version_heartbeat_cannot_change_actor_state():
    rig = actor_rig(versions=((3, 8), (3, 9)))
    rig.runtime.accept_latest_parameters()
    rig.transport.messages[0] = ParameterEnvelope(
        'run-1', 'hash-1', 3, 9, {'weight': torch.tensor([4.])})
    with pytest.raises(ValueError, match='heartbeat state mismatch'):
        rig.runtime.accept_latest_parameters()
    assert rig.policy.loaded_versions == [3]


def test_wrong_parameter_identity_rejected_before_loading():
    envelope = ParameterEnvelope('other', 'hash-1', 3, 8,
                                 {'weight': torch.tensor([3.])})
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_parameter_envelope(envelope, run_id='run-1', config_hash='hash-1',
                                    current_version=-1, current_sequence=-1)


def test_transport_backpressure_stops_motion():
    rig = actor_rig(blocked=True)
    with pytest.raises(TimeoutError, match='transition uplink backpressure'):
        rig.runtime.run(max_completed_steps=1)
    assert rig.env.backend.stop_calls >= 1
    assert rig.transport.closed


def test_replayed_parameter_sequence_is_rejected():
    rig = actor_rig(versions=((3, 8), (3, 8)))
    rig.runtime.accept_latest_parameters()
    with pytest.raises(RuntimeError, match='non-increasing'):
        rig.runtime.accept_latest_parameters()
    assert rig.policy.loaded_versions == [3]


def test_transition_validator_rejects_unknown_identity_and_nonfinite_action():
    rig = actor_rig()
    rig.runtime.run(max_completed_steps=1)
    row = rig.transport.sent[0]
    row['complementary_info']['run_id_override'] = 'other'
    with pytest.raises(ValueError, match='provenance'):
        validate_real_transition(row, 'run-1', 'hash-1')
    del row['complementary_info']['run_id_override']
    row['action'][1] = float('nan')
    with pytest.raises(ValueError, match='Executed action'):
        validate_real_transition(row, 'run-1', 'hash-1')


def test_timed_out_uplink_retries_identical_transition_ids():
    from lerobot.transport.utils import bytes_to_transitions, receive_bytes_in_chunks
    closed = threading.Event()
    ids = []

    class Deadline(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.DEADLINE_EXCEEDED

    class Stub:
        def StreamParameters(self, request):
            while not closed.wait(.01):
                yield from ()

        def SendTransitions(self, chunks, timeout):
            payload = receive_bytes_in_chunks(chunks, None, closed)
            ids.append(bytes_to_transitions(payload)[0]['complementary_info']['transition_id'])
            if len(ids) == 1:
                raise Deadline('deadline')

    transport = GrpcActorTransport('127.0.0.1:9999', queue_capacity=1,
                                   timeout_s=.1, stub=Stub(),
                                   channel=SimpleNamespace(close=closed.set))
    try:
        transport.send_transition_batch([{'complementary_info': {'transition_id':
                                                                  'run-1/episode-1/0'}}])
        assert ids == ['run-1/episode-1/0', 'run-1/episode-1/0']
    finally:
        transport.close()


def test_parameter_stream_outlives_single_rpc_deadline():
    closed = threading.Event()

    class Stub:
        def StreamParameters(self, request):
            while not closed.wait(.005):
                yield from ()

    transport = GrpcActorTransport('127.0.0.1:9999', queue_capacity=1,
                                   timeout_s=.02, stub=Stub(),
                                   channel=SimpleNamespace(close=closed.set))
    try:
        closed.wait(.06)
        transport.assert_alive()
        assert transport.receive_latest_parameters() is None
    finally:
        transport.close()


def test_actor_never_uploads_step_without_coordinator_outcome():
    rig = actor_rig()
    rig.env.completes_outcome = False
    with pytest.raises(RuntimeError, match='outcome not completed'):
        rig.runtime.run(max_completed_steps=1)
    assert rig.transport.sent == []
    assert rig.env.backend.stop_calls >= 1


def test_environment_factory_failure_still_closes_transport():
    rig = actor_rig()
    def fail_factory(config, coordinator):
        raise RuntimeError('factory failed')
    rig.runtime.env_factory = fail_factory
    with pytest.raises(RuntimeError, match='factory failed'):
        rig.runtime.run(max_completed_steps=1)
    assert rig.runtime.stop_event.is_set()
    assert rig.transport.closed


def test_time_limit_seal_allows_fresh_next_episode():
    rig = actor_rig()
    timings = []
    rig.runtime.telemetry = lambda kind, **fields: timings.append(fields) if kind == 'step' else None
    rig.env.truncate_first = True
    contexts = iter((EpisodeContext('episode-1', (0., 0., 0.), 'visual', 'grasp',
                                    visual_reset_monotonic_ns=1),
                     EpisodeContext('episode-2', (0., 0., 0.), 'visual', 'grasp',
                                    visual_reset_monotonic_ns=2)))
    rig.runtime.context_source = SimpleNamespace(read_new=lambda: next(contexts))
    summary = rig.runtime.run(max_completed_steps=2)
    assert summary.episodes_completed == 1
    assert [row['complementary_info']['transition_id'] for row in rig.transport.sent] == [
        'run-1/episode-1/0', 'run-1/episode-2/0']
    assert len(rig.envs) == 2
    assert rig.envs[0] is not rig.envs[1]
    assert all(env.closed and env.reset_calls == 1 for env in rig.envs)
    assert [row['control_period_s'] for row in timings] == [None, None]


def test_factory_starts_after_chord_and_refresh_precedes_each_inference():
    rig = actor_rig()
    rig.runtime.run(max_completed_steps=2)
    assert len(rig.envs) == 1
    assert rig.env.reset_calls == 1
    assert rig.env.refresh_calls == 2
    assert [float(row['state']['observation.state'][0, 0])
            for row in rig.transport.sent] == pytest.approx([.01, .02])


def test_episode_close_failure_stops_before_next_factory():
    rig = actor_rig()
    rig.env.truncate_first = True
    rig.env.close_error = OSError('close failed')
    with pytest.raises(OSError, match='close failed'):
        rig.runtime.run(max_completed_steps=2)
    assert len(rig.envs) == 1
    assert rig.runtime.stop_event.is_set()


def test_parameters_cannot_load_during_in_flight_step():
    rig = actor_rig(versions=((3, 8),))
    def receive_during_step():
        rig.transport.messages.append(ParameterEnvelope(
            'run-1', 'hash-1', 4, 9, {'weight': torch.tensor([4.])}))
        rig.runtime.accept_latest_parameters()
    rig.env.during_step = receive_during_step
    with pytest.raises(RuntimeError, match='in-flight step'):
        rig.runtime.run(max_completed_steps=1)
    assert rig.policy.loaded_versions == [3]
    assert rig.transport.sent == []


def test_actor_waits_for_new_context_file_and_does_not_reuse_old_one(tmp_path):
    rig = actor_rig()
    path = tmp_path / 'context.json'
    rig.runtime.context_source = EpisodeContextInbox(path)
    assert rig.runtime._read_context() is None
    path.write_text(json.dumps({
        'episode_id': 'episode-1', 'target_offset_m': [0, 0, 0],
        'approach_source': 'visual', 'grasp_description': 'grasp',
        'visual_reset_monotonic_ns': 1}), encoding='utf-8')
    assert rig.runtime._read_context().episode_id == 'episode-1'
    assert rig.runtime._read_context() is None


def test_step_error_preserves_original_after_coordinator_aborts_itself():
    rig = actor_rig()
    def fail_during_step():
        rig.coordinator.abort_step(rig.coordinator.active_step_token)
        raise ValueError('driver read failed')
    rig.env.during_step = fail_during_step
    with pytest.raises(ValueError, match='driver read failed'):
        rig.runtime.run(max_completed_steps=1)
    assert rig.env.backend.stop_calls >= 1
