"""Fail-closed real Actor loop and bounded loopback learner uplink.

The environment factory is the only path to motion. This module never creates
GDK resources itself; tests inject a fake environment and transport.
"""

from dataclasses import asdict, dataclass
import logging
import math
import os
from queue import Empty, Full, Queue
import threading
import time
from typing import Mapping

import numpy as np
import torch

from .contract import CAMERA_KEYS, EpisodeContext, vector
from .policy import create_policy


def _identity_part(value, name):
    if (type(value) is not str or not 0 < len(value) <= 128 or
            not value.isascii() or
            not all(c.isalnum() or c in '_.-' for c in value)):
        raise ValueError(f'Invalid {name}')
    return value


@dataclass(frozen=True)
class TransitionIdentity:
    run_id: str
    episode_id: str
    step_id: int

    def __post_init__(self):
        _identity_part(self.run_id, 'run_id')
        _identity_part(self.episode_id, 'episode_id')
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError('Invalid step_id')

    @property
    def value(self):
        return f'{self.run_id}/{self.episode_id}/{self.step_id}'


@dataclass(frozen=True)
class ParameterEnvelope:
    run_id: str
    config_hash: str
    version: int
    message_sequence: int
    actor_state: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class ActorRunSummary:
    transitions_sent: int
    episodes_completed: int
    interventions: int
    final_parameter_version: int
    stop_reason: str


def validate_parameter_envelope(envelope, *, run_id, config_hash,
                                current_version, current_sequence):
    if type(envelope) is not ParameterEnvelope:
        raise ValueError('ParameterEnvelope required')
    if envelope.run_id != run_id or envelope.config_hash != config_hash:
        raise ValueError('parameter run/config identity mismatch')
    if (type(envelope.version) is not int or envelope.version < 0 or
            type(envelope.message_sequence) is not int or
            envelope.message_sequence < 0):
        raise ValueError('Invalid parameter version or message sequence')
    if envelope.message_sequence <= current_sequence:
        raise RuntimeError('non-increasing parameter message sequence')
    if envelope.version < current_version:
        raise RuntimeError('rolled-back parameter version')
    if (not isinstance(envelope.actor_state, Mapping) or
            any(type(k) is not str or type(v) is not torch.Tensor or
                v.layout != torch.strided or not torch.isfinite(v).all().item()
                for k, v in envelope.actor_state.items())):
        raise ValueError('Invalid Actor state')
    return envelope.version > current_version


def _policy_observation(obs, device, image_size):
    if image_size != 128:
        raise ValueError('SiLRI dual-RGB contract requires 128-pixel images')
    if type(obs) is not dict or set(obs) != {'state', *CAMERA_KEYS}:
        raise ValueError('Dual-RGB/state observation required')
    state = obs['state']
    if (type(state) is not np.ndarray or state.shape != (7,) or
            state.dtype != np.float32 or not np.isfinite(state).all() or
            abs(np.linalg.norm(state[3:]) - 1.) > .01):
        raise ValueError('Invalid observed state')
    result = {'observation.state': torch.from_numpy(state.copy()).unsqueeze(0).to(device)}
    for key in CAMERA_KEYS:
        image = obs[key]
        if (type(image) is not np.ndarray or image.dtype != np.uint8 or
                image.shape != (image_size, image_size, 3)):
            raise ValueError('Invalid dual-RGB image')
        result[f'observation.images.{key}'] = (torch.from_numpy(image.copy())
            .permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.)
    return result


def validate_transition_provenance(row, run_id, config_hash):
    """Validate identity, actions, outcomes and scene provenance."""
    if type(row) is not dict or set(row) != {
            'state', 'next_state', 'action', 'reward', 'done', 'truncated',
            'complementary_info'}:
        raise ValueError('Invalid real transition fields')
    info = row['complementary_info']
    required = {'run_id', 'config_hash', 'transition_id', 'episode_id', 'step_id',
                'actor_version', 'synthetic', 'policy_action', 'human_action',
                'selected_action', 'executed_action', 'is_intervention',
                'reward_source', 'success_label', 'gate_summary',
                'target_offset_m', 'ee_reset_offset', 'approach_source',
                'grasp_description', 'visual_reset_monotonic_ns',
                'visual_confidence', 'upstream_frame_id'}
    if type(info) is not dict or set(info) != required:
        raise ValueError('Invalid transition provenance fields')
    if info['run_id'] != run_id or info['config_hash'] != config_hash:
        raise ValueError('transition run/config identity mismatch')
    identity = TransitionIdentity(info['run_id'], info['episode_id'], info['step_id'])
    if info['transition_id'] != identity.value or info['synthetic'] is not False:
        raise ValueError('Invalid real transition identity')
    if type(info['actor_version']) is not int or info['actor_version'] < 0:
        raise ValueError('Invalid Actor version')
    if (type(info['is_intervention']) is not bool or
            type(info['gate_summary']) is not dict or
            set(info['gate_summary']) != {'fresh'} or
            info['gate_summary']['fresh'] is not True):
        raise ValueError('Invalid intervention or freshness gate summary')
    EpisodeContext(info['episode_id'], info['target_offset_m'],
                   info['approach_source'], info['grasp_description'],
                   info['ee_reset_offset'], info['visual_reset_monotonic_ns'],
                   info['visual_confidence'], info['upstream_frame_id'])
    if type(row['done']) is not bool or type(row['truncated']) is not bool:
        raise ValueError('Invalid terminal flags')
    if type(row['reward']) not in (int, float) or not math.isfinite(row['reward']):
        raise ValueError('Nonfinite transition reward')
    if (type(info['success_label']) not in (bool, type(None)) or
            info['reward_source'] not in ('human', 'classifier', 'environment', 'unknown')):
        raise ValueError('Invalid reward provenance')
    executed = vector(info['executed_action'], 6)
    selected = vector(info['selected_action'], 6)
    vector(info['policy_action'], 6)
    if info['human_action'] is not None:
        vector(info['human_action'], 6)
    if any(abs(x) > 1 for x in executed + selected):
        raise ValueError('Action outside normalized contract')
    action = row['action']
    if (type(action) is not torch.Tensor or action.shape != (6,) or
            not torch.isfinite(action).all().item() or
            not np.array_equal(action.cpu().numpy(), np.asarray(executed, dtype=np.float32))):
        raise ValueError('Executed action mismatch')
    return identity


def validate_real_transition(row, run_id, config_hash):
    """Validate full real provenance before learner replay mutation."""
    identity = validate_transition_provenance(row, run_id, config_hash)
    for field in ('state', 'next_state'):
        obs = row[field]
        if type(obs) is not dict or set(obs) != {
                'observation.state', *(f'observation.images.{k}' for k in CAMERA_KEYS)}:
            raise ValueError('Invalid policy observation keys')
        state = obs['observation.state']
        if (type(state) is not torch.Tensor or state.shape != (1, 7) or
                state.dtype != torch.float32 or not torch.isfinite(state).all().item()):
            raise ValueError('Invalid policy state tensor')
        for key in CAMERA_KEYS:
            image = obs[f'observation.images.{key}']
            if (type(image) is not torch.Tensor or image.dtype != torch.float32 or image.ndim != 4 or
                    image.shape[:2] != (1, 3) or image.shape[2:] != (128, 128) or
                    not torch.isfinite(image).all().item() or
                    bool((image < 0).any()) or bool((image > 1).any())):
                raise ValueError('Invalid policy RGB tensor')
    return identity


class _SendRequest:
    def __init__(self, rows):
        self.rows = tuple(rows)
        self.done = threading.Event()
        self.error = None


class GrpcActorTransport:
    """One bounded sender and one bounded parameter stream over loopback gRPC."""

    def __init__(self, address, *, queue_capacity, timeout_s,
                 queue_put_timeout_s=None, stub=None, channel=None):
        if (type(address) is not str or not address.startswith('127.0.0.1:') or
                not address.removeprefix('127.0.0.1:').isdigit()):
            raise ValueError('Loopback learner address required')
        if (type(queue_capacity) is not int or queue_capacity <= 0 or
                type(timeout_s) not in (int, float) or
                not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError('Positive transport bounds required')
        if queue_put_timeout_s is None:
            queue_put_timeout_s = timeout_s
        if (type(queue_put_timeout_s) not in (int, float) or
                not math.isfinite(queue_put_timeout_s) or queue_put_timeout_s <= 0):
            raise ValueError('Positive uplink queue timeout required')
        import grpc
        from lerobot.transport import services_pb2_grpc as rpc
        self.timeout_s = timeout_s
        self.queue_put_timeout_s = queue_put_timeout_s
        self.channel = channel or grpc.insecure_channel(address)
        self.stub = stub or rpc.LearnerServiceStub(self.channel)
        self.parameters = Queue(maxsize=queue_capacity)
        self.outgoing = Queue(maxsize=queue_capacity)
        self.stopped = threading.Event()
        self.error = None
        self.sender = threading.Thread(target=self._send_loop, daemon=True)
        self.receiver = threading.Thread(target=self._receive_loop, daemon=True)
        self.sender.start()
        self.receiver.start()

    def _fail(self, error):
        if self.error is None:
            self.error = error
        self.stopped.set()

    def _receive_loop(self):
        from lerobot.transport import services_pb2 as pb
        from lerobot.transport.utils import bytes_to_state_dict
        data = bytearray()
        try:
            for chunk in self.stub.StreamParameters(pb.Empty()):
                if self.stopped.is_set():
                    return
                if chunk.transfer_state == pb.TransferState.TRANSFER_BEGIN:
                    data.clear()
                data.extend(chunk.data)
                if chunk.transfer_state == pb.TransferState.TRANSFER_END:
                    payload = bytes_to_state_dict(bytes(data))
                    if type(payload) is not dict or set(payload) != {
                            'run_id', 'config_hash', 'version', 'message_sequence',
                            'actor_state'}:
                        raise ValueError('Invalid parameter payload')
                    self.parameters.put(ParameterEnvelope(**payload), timeout=self.timeout_s)
                    data.clear()
            if not self.stopped.is_set():
                raise ConnectionError('Learner parameter stream ended')
        except BaseException as exc:
            if not self.stopped.is_set():
                self._fail(exc)

    def _send_loop(self):
        import grpc
        from lerobot.transport import services_pb2 as pb
        from lerobot.transport.utils import send_bytes_in_chunks, transitions_to_bytes
        while not self.stopped.is_set():
            try:
                request = self.outgoing.get(timeout=.05)
            except Empty:
                continue
            try:
                packet = transitions_to_bytes(list(request.rows))
                # Retry only the same serialized packet, hence the same IDs.
                for attempt in range(2):
                    try:
                        self.stub.SendTransitions(
                            send_bytes_in_chunks(packet, pb.Transition),
                            timeout=self.timeout_s)
                        break
                    except (TimeoutError, grpc.RpcError) as exc:
                        deadline = isinstance(exc, TimeoutError) or (
                            isinstance(exc, grpc.RpcError) and
                            exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED)
                        if attempt or not deadline:
                            raise
            except BaseException as exc:
                request.error = exc
                self._fail(exc)
            finally:
                request.done.set()
                self.outgoing.task_done()

    def assert_alive(self):
        if self.error is not None:
            raise RuntimeError('Learner transport failed') from self.error
        if self.stopped.is_set():
            raise ConnectionError('Learner transport stopped')

    def receive_latest_parameters(self):
        self.assert_alive()
        try:
            return self.parameters.get_nowait()
        except Empty:
            return None

    def send_transition_batch(self, rows):
        self.assert_alive()
        request = _SendRequest(rows)
        try:
            self.outgoing.put(request, timeout=self.queue_put_timeout_s)
        except Full as exc:
            self._fail(TimeoutError('transition uplink backpressure'))
            raise TimeoutError('transition uplink backpressure') from exc
        if not request.done.wait(self.timeout_s * 3):
            self._fail(TimeoutError('transition uplink backpressure'))
            raise TimeoutError('transition uplink backpressure')
        if request.error is not None:
            raise request.error
        self.assert_alive()

    def close(self):
        self.stopped.set()
        self.channel.close()
        for worker in (self.sender, self.receiver):
            if worker is not threading.current_thread():
                worker.join(timeout=self.timeout_s)


class RealActorRuntime:
    def __init__(self, *, config, run_id, config_hash, coordinator,
                 context_source, transport, env_factory, policy=None,
                 clock=time.monotonic, telemetry=None):
        self.config = config
        self.run_id = _identity_part(run_id, 'run_id')
        self.config_hash = _identity_part(config_hash, 'config_hash')
        self.coordinator = coordinator
        self.context_source = context_source
        self.transport = transport
        self.env_factory = env_factory
        self.policy = (policy or create_policy(config.runtime.device)).eval()
        self.clock = clock
        self.telemetry = telemetry
        self.stop_event = threading.Event()
        self._env = None
        self.current_observation = None
        self.parameter_version = -1
        self.last_message_sequence = -1
        self.last_parameter_at = self.clock()
        self.transitions_sent = 0
        self.episodes_completed = 0
        self.interventions = 0
        self.stop_reason = ''
        self.stop_confirmed = None
        self.stop_error = None
        self.freshness_rejects = 0
        self._step_active = False

    def _emit(self, kind, **fields):
        if self.telemetry is not None:
            self.telemetry(kind, **fields)

    def accept_latest_parameters(self):
        if self._step_active:
            raise RuntimeError('Cannot load parameters during in-flight step')
        envelope = self.transport.receive_latest_parameters()
        if envelope is None:
            if self.clock() - self.last_parameter_at > self.config.runtime.learner_silence_timeout_s:
                raise TimeoutError('learner parameter silence timeout')
            return False
        changed = validate_parameter_envelope(
            envelope, run_id=self.run_id, config_hash=self.config_hash,
            current_version=self.parameter_version,
            current_sequence=self.last_message_sequence)
        if changed:
            expected = self.policy.actor.state_dict()
            actual = envelope.actor_state
            if (set(actual) != set(expected) or any(
                    actual[k].shape != target.shape or actual[k].dtype != target.dtype
                    for k, target in expected.items())):
                raise ValueError('Actor parameter shape or dtype mismatch')
            self.policy.actor.load_state_dict(actual, strict=True)
            self.parameter_version = envelope.version
        elif (set(envelope.actor_state) != set(self.policy.actor.state_dict()) or
              any(not torch.equal(envelope.actor_state[k].cpu(), value.detach().cpu())
                  for k, value in self.policy.actor.state_dict().items())):
            raise ValueError('heartbeat state mismatch')
        self.last_message_sequence = envelope.message_sequence
        self.last_parameter_at = self.clock()
        return changed

    def _infer(self, observation):
        state = _policy_observation(observation, self.config.runtime.device,
                                    self.config.observation.image_size)
        with torch.no_grad():
            action = self.policy.select_action(state)[0].squeeze(0).cpu().numpy()
        return vector(action, 6)

    def _read_context(self):
        path = getattr(self.context_source, 'path', None)
        if path is not None and not os.path.lexists(path):
            return None
        try:
            return self.context_source.read_new()
        except ValueError as exc:
            if str(exc) == 'duplicate episode context':
                return None
            raise

    def _build_confirmed_transition(self, before, after, reward, terminated,
                                    truncated, info, token, context, policy_action):
        if type(info) is not dict or 'executed_action' not in info:
            raise ValueError('Driver-confirmed executed action required')
        executed = vector(info['executed_action'], 6)
        policy = vector(info.get('policy_action', policy_action), 6)
        if policy != policy_action:
            raise ValueError('Policy action provenance mismatch')
        human = info.get('human_action')
        human = None if human is None else vector(human, 6)
        selected = vector(info['selected_action'], 6)
        if selected != tuple(max(-1., min(1., x)) for x in
                             (human if info['is_intervention'] else policy)):
            raise ValueError('Selected action provenance mismatch')
        identity = TransitionIdentity(self.run_id, token.episode_id, token.step_id)
        if identity.episode_id != context.episode_id:
            raise ValueError('Step/context identity mismatch')
        context_info = asdict(context)
        metadata = dict(context_info, run_id=self.run_id, config_hash=self.config_hash,
                        transition_id=identity.value, episode_id=identity.episode_id,
                        step_id=identity.step_id, actor_version=self.parameter_version,
                        synthetic=False, policy_action=policy, human_action=human,
                        selected_action=selected, executed_action=executed,
                        is_intervention=bool(info['is_intervention']),
                        reward_source=info['reward_source'],
                        success_label=info['success_label'],
                        gate_summary={'fresh': self.coordinator.intervention.gate.fresh})
        row = {'state': _policy_observation(before, 'cpu', self.config.observation.image_size),
               'next_state': _policy_observation(after, 'cpu', self.config.observation.image_size),
               'action': torch.tensor(executed, dtype=torch.float32),
               'reward': float(reward), 'done': bool(terminated),
               'truncated': bool(truncated), 'complementary_info': metadata}
        validate_real_transition(row, self.run_id, self.config_hash)
        return row

    def summary(self):
        return ActorRunSummary(self.transitions_sent, self.episodes_completed,
                               self.interventions, self.parameter_version,
                               self.stop_reason)

    def stop(self, reason):
        if self.stop_event.is_set():
            return
        self.stop_reason = reason
        self.stop_event.set()
        try:
            if self._env is not None:
                self._env.backend.stop()
        except BaseException as error:
            self.stop_confirmed = False
            self.stop_error = error
            self._emit('stop', reason=reason, confirmed=False,
                       error=f'{type(error).__name__}: {error}'[:512])
            raise
        else:
            if self.stop_confirmed is not False:
                self.stop_confirmed = True
            self._emit('stop', reason=reason, confirmed=self.stop_confirmed)

    def run(self, *, max_completed_steps=None):
        if max_completed_steps is not None and (type(max_completed_steps) is not int or
                                                max_completed_steps <= 0):
            raise ValueError('Positive completed-step bound required')
        env = None
        completed = 0
        previous_step_at = None
        primary_error = None
        try:
            while not self.stop_event.is_set():
                self.transport.assert_alive()
                self.accept_latest_parameters()  # boundary: never inside env.step
                if not self.coordinator.running:
                    if self.parameter_version < 0:
                        time.sleep(self.config.runtime.operator_poll_interval_s)
                        continue
                    if self.coordinator.context is None:
                        context = self._read_context()
                        if context is not None:
                            if not isinstance(context, EpisodeContext):
                                raise ValueError('EpisodeContext required')
                            self.coordinator.offer_context(context)
                    if self.coordinator.context is not None:
                        # The chord reader consumes a freshly polled HID frame.
                        self.coordinator.intervention()
                        if self.coordinator.observe_start_frame():
                            context = self.coordinator.context
                            env = self.env_factory(self.config, self.coordinator)
                            self._env = env
                            self.current_observation, _ = env.reset(options={'context': context})
                            _policy_observation(self.current_observation, 'cpu',
                                                self.config.observation.image_size)
                    time.sleep(self.config.runtime.operator_poll_interval_s)
                    continue
                if env is None:
                    raise RuntimeError('Running episode has no commissioned environment')
                before = env.refresh_observation()
                self.current_observation = before
                context = self.coordinator.context
                inference_started = self.clock()
                policy_action = self._infer(before)
                inference_latency_s = self.clock() - inference_started
                control_period_s = (None if previous_step_at is None else
                                    inference_started - previous_step_at)
                previous_step_at = inference_started
                token = self.coordinator.begin_step()
                self._step_active = True
                try:
                    after, reward, terminated, truncated, info = env.step(policy_action)
                except BaseException:
                    if self.coordinator.active_step_token == token:
                        try:
                            self.coordinator.abort_step(token)
                        except BaseException:
                            logging.exception('Coordinator abort failed after environment step error')
                    raise
                finally:
                    self._step_active = False
                if self.coordinator.completed_step_token != token:
                    raise RuntimeError('coordinator outcome not completed')
                row = self._build_confirmed_transition(
                    before, after, reward, terminated, truncated, info,
                    token, context, policy_action)
                self._emit('step', episode_id=token.episode_id, step_id=token.step_id,
                           inference_latency_s=inference_latency_s,
                           control_period_s=control_period_s)
                self.transport.send_transition_batch((row,))
                self.transitions_sent += 1
                self.interventions += int(row['complementary_info']['is_intervention'])
                completed += 1
                self.current_observation = after
                if terminated or truncated:
                    self.episodes_completed += 1
                    previous_step_at = None
                    if truncated and self.coordinator.running:
                        self.coordinator.seal_episode(token)
                    completed_env = env
                    env = None
                    self._env = None
                    self.current_observation = None
                    completed_env.close()
                if max_completed_steps is not None and completed >= max_completed_steps:
                    return self.summary()
            return self.summary()
        except BaseException as error:
            primary_error = error
            message = str(error)
            if (message.startswith('Source observation freshness not confirmed') or
                    message.startswith('Feedback freshness not explicitly confirmed')):
                self.freshness_rejects += 1
                code = ('feedback_lease' if message.startswith('Feedback freshness') else
                        message.partition(': ')[2] or 'unknown')
                context = self.coordinator.context
                try:
                    self._emit('freshness_reject',
                               episode_id=context.episode_id if context is not None else None,
                               code=code[:128])
                except BaseException:
                    logging.exception('Freshness evidence write failed; stopping Actor')
            if message.startswith('physical stop unconfirmed'):
                self.stop_confirmed = False
                self.stop_error = error
            try:
                self.stop('actor_failure')
            except BaseException:
                logging.exception('Actor command stop unconfirmed')
            if self.stop_confirmed is False:
                error.stop_unconfirmed = True
            raise
        finally:
            cleanup_error = None
            try:
                if env is not None:
                    env.close()
            except BaseException as error:
                cleanup_error = error
                self.stop_confirmed = False
                self.stop_error = error
            finally:
                close_transport = getattr(self.transport, 'close', None)
                if callable(close_transport):
                    try:
                        close_transport()
                    except BaseException as error:
                        if cleanup_error is None:
                            cleanup_error = error
                        else:
                            logging.exception('Transport close failed after environment close error')
            if cleanup_error is not None:
                if primary_error is not None:
                    primary_error.stop_unconfirmed = self.stop_confirmed is False
                    logging.error('Actor cleanup failed after primary error: %s', cleanup_error)
                else:
                    cleanup_error.stop_unconfirmed = self.stop_confirmed is False
                    raise cleanup_error
