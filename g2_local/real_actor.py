"""Fail-closed real Actor loop and bounded loopback learner uplink.

The environment factory is the only path to motion. This module never creates
GDK resources itself; tests inject a fake environment and transport.
"""

from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
import uuid
import logging
import math
import os
from queue import Empty, Full, Queue
import threading
import time
from typing import Mapping

import numpy as np
import torch

from .console import RateLimit, say
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
    if type(info) is dict and 'automatic_reset_monotonic_ns' in info:
        required.add('automatic_reset_monotonic_ns')
    if type(info) is not dict or set(info) != required:
        raise ValueError('Invalid transition provenance fields')
    if info['run_id'] != run_id or info['config_hash'] != config_hash:
        raise ValueError('transition run/config identity mismatch')
    identity = TransitionIdentity(info['run_id'], info['episode_id'], info['step_id'])
    if info['transition_id'] != identity.value or info['synthetic'] is not False:
        raise ValueError('Invalid real transition identity')
    if type(info['actor_version']) is not int or info['actor_version'] < 0:
        raise ValueError('Invalid Actor version')
    gate = info['gate_summary']
    fresh = type(gate) is dict and gate == {'fresh': True} and gate['fresh'] is True
    idle = (type(gate) is dict and set(gate) == {'fresh', 'verified_neutral'} and
            gate['fresh'] is False and gate['verified_neutral'] is True)
    if type(info['is_intervention']) is not bool or not (fresh or idle):
        raise ValueError('Invalid intervention or freshness gate summary')
    EpisodeContext(info['episode_id'], info['target_offset_m'],
                   info['approach_source'], info['grasp_description'],
                   info['ee_reset_offset'], info['visual_reset_monotonic_ns'],
                   info['visual_confidence'], info['upstream_frame_id'],
                   info.get('automatic_reset_monotonic_ns'))
    if type(row['done']) is not bool or type(row['truncated']) is not bool:
        raise ValueError('Invalid terminal flags')
    if type(row['reward']) not in (int, float) or not math.isfinite(row['reward']):
        raise ValueError('Nonfinite transition reward')
    if (type(info['success_label']) not in (bool, type(None)) or
            info['reward_source'] not in ('human', 'classifier', 'environment', 'unknown')):
        raise ValueError('Invalid reward provenance')
    executed = vector(info['executed_action'], 6)
    selected = vector(info['selected_action'], 6)
    policy = vector(info['policy_action'], 6)
    human = None if info['human_action'] is None else vector(info['human_action'], 6)
    if idle and info['is_intervention'] and (human != (0.,) * 6 or selected != (0.,) * 6):
        raise ValueError('Silent neutral cannot authorize nonzero human action')
    if (info['is_intervention'] != (human is not None) or
            selected != tuple(max(-1., min(1., value)) for value in
                              (human if info['is_intervention'] else policy))):
        raise ValueError('Inconsistent action provenance')
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
            # Keep the cause in the text: this reaches the operator terminal
            # through a generic handler that prints only the message.
            raise RuntimeError(f'Learner transport failed: '
                               f'{type(self.error).__name__}: {self.error}') from self.error
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
                 clock=time.monotonic, telemetry=None, demonstration=False,
                 exit_after_labeled_demo=False):
        self.config = config
        self.run_id = _identity_part(run_id, 'run_id')
        self.config_hash = _identity_part(config_hash, 'config_hash')
        self.coordinator = coordinator
        self.context_source = context_source
        self.transport = transport
        self.env_factory = env_factory
        self.demonstration = demonstration
        if type(exit_after_labeled_demo) is not bool:
            raise ValueError('exit_after_labeled_demo must be boolean')
        # NOTE: the real Actor role passes policy=None on purpose (the runtime
        # loads parameters from the Learner over gRPC), so a None policy here is
        # not an error. The CLI restricts this flag to the demo and actor roles.
        
        self.exit_after_labeled_demo = exit_after_labeled_demo
        self._policy_warmed = False
        if demonstration and policy is not None:
            raise ValueError('Demonstration must not load a policy')
        self.policy = None if demonstration else (policy or create_policy(config.runtime.device)).eval()
        self.clock = clock
        self.telemetry = telemetry
        self.stop_event = threading.Event()
        self._env = None
        self.current_observation = None
        self.parameter_version = 0 if demonstration else -1
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
        if self.demonstration:
            return False
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
            first = self.parameter_version < 0
            self.parameter_version = envelope.version
            if first:
                say(f'已收到 Learner 策略参数：version=v{envelope.version}；'
                    'Actor 继续等待场景复位写入 EpisodeContext。')
        elif (set(envelope.actor_state) != set(self.policy.actor.state_dict()) or
              any(not torch.equal(envelope.actor_state[k].cpu(), value.detach().cpu())
                  for k, value in self.policy.actor.state_dict().items())):
            raise ValueError('heartbeat state mismatch')
        self.last_message_sequence = envelope.message_sequence
        self.last_parameter_at = self.clock()
        return changed

    def _infer(self, observation):
        if self.demonstration:
            return (0.,) * 6
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
        if self.demonstration and info.get('is_intervention') is not True:
            raise ValueError('Demonstration requires human control on every step')
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
                        gate_summary=({'fresh': True}
                                      if self.coordinator.intervention.gate.fresh else
                                      {'fresh': False, 'verified_neutral':
                                       getattr(self.coordinator.intervention, 'verified_neutral', False)}))
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

    def _context_window_text(self):
        window = getattr(self.config.runtime, 'context_max_age_s', None)
        return f'{window:.0f} 秒' if isinstance(window, (int, float)) else '限定的时效'

    def _automatic_reset(self, reference, previous_context):
        from .auto_reset import run_reset
        def poll():
            if self.stop_event.is_set():
                raise RuntimeError('Actor stopped during automatic reset')
            self.transport.assert_alive()
            self.accept_latest_parameters()
            intervention = self.coordinator.intervention
            active, action = intervention()
            if not (intervention.gate.fresh or intervention.verified_neutral):
                raise RuntimeError('SpaceMouse neutral state unconfirmed during reset')
            if (action is not None and any(abs(v) > 0 for v in action)) or any(
                    abs(v) > intervention.config.release_deadzone
                    for v in intervention.last_frame.axes):
                raise RuntimeError('SpaceMouse motion cancels automatic reset')
            if self.coordinator.keys.poll() is not None:
                raise RuntimeError('Operator key cancels automatic reset')
        self.coordinator.keys.drain()
        poll()
        reset_coordinator = SimpleNamespace(outcome=lambda obs: (0., False), intervention=None)
        env = self.env_factory(self.config, reset_coordinator)
        self._env = env
        try:
            self._emit('reset_started', previous_episode_id=previous_context.episode_id)
            auto = getattr(getattr(self.config, 'motion', None), 'auto_reset', None)
            lift = getattr(auto, 'lift_m', None)
            limit = getattr(auto, 'timeout_s', None)
            lift = (f'抬升 {lift * 100.:.0f} cm 后'
                    if isinstance(lift, (int, float)) else '')
            limit = (f'（{limit:.0f} 秒上限）'
                     if isinstance(limit, (int, float)) else '')
            say(f'开始自动归位：{lift}直线回到本回合起始位姿{limit}；'
                '过程中请不要碰 SpaceMouse、不要按键，否则会取消复位。')
            # run_reset closes resources before any new context becomes eligible.
            run_reset(env, reference, self.config.motion, poll=poll, emit=self._emit)
        finally:
            env.close()
        self._env = None
        say(f'自动归位完成，下一回合 context 已备好；请在 '
            f'{self._context_window_text()}内按住双键再全部松开。')
        context = replace(previous_context, episode_id='auto-'+uuid.uuid4().hex,
                          approach_source='automatic_lift_return',
                          visual_reset_monotonic_ns=None, visual_confidence=None,
                          upstream_frame_id=None, automatic_reset_monotonic_ns=time.monotonic_ns())
        self.coordinator.offer_context(context)
        self._emit('reset_completed', episode_id=context.episode_id,
                   automatic_reset_monotonic_ns=context.automatic_reset_monotonic_ns,
                   next_episode_requires_start_chord=True)

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
        episode_start_pose = None
        primary_error = None
        pending = None
        episode_steps = 0
        episode_interventions = 0
        waiting_context = RateLimit(10.)
        waiting_reminder = RateLimit(60.)
        waiting_announced = False
        heartbeat = RateLimit(2.)
        gate_wait_reminder = RateLimit(60.)
        gate_wait_announced = False
        def upload(row):
            self.transport.send_transition_batch((row,))
            self.transitions_sent += 1
            self.interventions += int(row['complementary_info']['is_intervention'])
        def finish_episode(token, context, truncated, *, labeled=False, success=None):
            nonlocal env, previous_step_at, episode_steps, episode_interventions
            self.episodes_completed += 1
            previous_step_at = None
            if truncated and self.coordinator.running:
                self.coordinator.seal_episode(token)
            completed_env = env
            env = None
            self._env = None
            self.current_observation = None
            completed_env.close()
            outcome = ('成功' if success is True else '失败' if success is False
                       else '截断' if truncated else '未标注')
            say(f'回合 #{self.episodes_completed} 结束：steps={episode_steps} 结果={outcome} '
                f'人工接管={episode_interventions}/{episode_steps} 已提交转移={self.transitions_sent}')
            episode_steps = 0
            episode_interventions = 0
            reset_config = getattr(getattr(self.config, 'motion', None), 'auto_reset', None)
            if reset_config is not None and reset_config.enabled:
                say('自动复位已启用：抬升后回到本回合起始位姿，不调用上游视觉复位。')
                self._automatic_reset(episode_start_pose, context)
            if self.exit_after_labeled_demo and labeled:
                # Training loop: exit cleanly right after Y/F so the launcher can
                # run the upstream reset (it needs the SpaceMouse) and start the
                # next episode. The Learner is not affected.
                self.stop('demo_labeled' if self.demonstration else 'labeled_episode')
        def interrupt_camera_episode(error, token=None):
            nonlocal env, pending, previous_step_at, episode_start_pose, episode_steps
            nonlocal episode_interventions
            from .gdk_backend import CameraUnavailable
            is_camera_freshness_error = getattr(error, 'code', None) in (
                'camera_stale:left_wrist', 'camera_stale:right_aux', 'camera_skew')
            if (not isinstance(error, CameraUnavailable) and not is_camera_freshness_error) or env is None:
                return False
            backend = getattr(env, 'backend', None)
            if getattr(backend, 'stop_confirmed', None) is not True:
                return False
            context = self.coordinator.context
            if context is None:
                return False
            self.coordinator.abandon_episode_after_camera_fault(token)
            discarded_step = (None if token is None else
                              dict(episode_id=token.episode_id, step_id=token.step_id))
            preserved_transition = None
            if pending is not None:
                row, _, _ = pending
                row['done'] = False
                row['truncated'] = True
                validate_real_transition(row, self.run_id, self.config_hash)
                upload(row)
                preserved_transition = row['complementary_info']['transition_id']
                pending = None
            self._emit('camera_episode_interrupted',
                       episode_id=context.episode_id,
                       step_id=None if token is None else token.step_id,
                       error=(f'{type(error).__name__}: {error}')[:512],
                       stop_confirmed=True,
                       discarded_inflight_step=discarded_step,
                       preserved_transition_id=preserved_transition,
                       requires_operator_scene_reset=True)
            try:
                env.close()
            finally:
                env = None
                self._env = None
                self.current_observation = None
                previous_step_at = None
                episode_start_pose = None
            self.interrupted_episodes = getattr(self, 'interrupted_episodes', 0) + 1
            episode_steps = 0
            episode_interventions = 0
            say('相机观测中断：已确认测量保持，当前回合已截断；Actor 继续运行，'
                '等待现场复位并提交新的 EpisodeContext 后再按双键开始。')
            return True
        def terminal_before_next_action(forced=None):
            nonlocal pending
            if pending is None:
                return False
            keys = getattr(self.coordinator, 'keys', None)
            label = forced.label if forced is not None else (keys.poll() if keys is not None else None)
            if label is None:
                return False
            read_ns = forced.read_ns if forced is not None else time.monotonic_ns()
            row, token, context = pending
            env.backend.stop()
            self.coordinator.seal_episode(token)
            row['done'], row['truncated'] = True, False
            row['reward'] = (self.config.task.success_reward if label == 'success'
                             else self.config.task.failure_reward)
            row['complementary_info'].update(reward_source='human', success_label=label == 'success')
            validate_real_transition(row, self.run_id, self.config_hash)
            self._emit('terminal_label', label=label, read_monotonic_ns=read_ns,
                       transition_id=row['complementary_info']['transition_id'],
                       attribution='previous_successor_before_next_action')
            upload(row)
            pending = None
            finish_episode(token, context, False, labeled=True, success=label == 'success')
            return True
        try:
            while not self.stop_event.is_set():
                if terminal_before_next_action():
                    continue
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
                            waiting_announced = False
                            say(f'已读取 EpisodeContext：{context.episode_id}；'
                                f'请确认现场已复位，并在 {self._context_window_text()}内'
                                '按住双键再全部松开。')
                        elif waiting_context.due():
                            # Announce the actionable hint once, then stay quiet
                            # apart from a slow reminder: this state can last
                            # minutes while the operator resets the scene.
                            path = getattr(self.context_source, 'path', None)
                            if not waiting_announced:
                                waiting_announced = True
                                say(f'等待下一回合的 EpisodeContext：{path}')
                                say('现场复位后执行 '
                                    'bash run_g2_python.sh scripts/start_training_actor.py --write-context '
                                    f'写入新 context，再在 {self._context_window_text()}内按住双键；'
                                    '或者 Ctrl+C 退出 Actor 后重新启动。')
                            elif waiting_reminder.due():
                                say(f'仍在等待新的 EpisodeContext：{path}')
                    if self.coordinator.context is not None:
                        # The chord reader consumes a freshly polled HID frame.
                        self.coordinator.intervention()
                        if self.coordinator.observe_start_frame():
                            context = self.coordinator.context
                            say(f'双键已识别：第 {self.episodes_completed + 1} 个回合开始 '
                                f'（episode_id={context.episode_id}）'
                                + ('，正在连接观测与运动后端…' if self.demonstration else '。'))
                            env = self.env_factory(self.config, self.coordinator)
                            self._env = env
                            try:
                                self.current_observation, _ = env.reset(options={'context': context})
                                if not self.demonstration and not self._policy_warmed:
                                    started = self.clock()
                                    # CUDA initialization must not consume the freshness
                                    # budget of the observation used for a command.
                                    for _ in range(10):
                                        self._infer(self.current_observation)
                                    self._policy_warmed = True
                                    self._emit('policy_warmup', episode_id=context.episode_id,
                                               discarded_actions=10,
                                               duration_s=self.clock() - started)
                                    # The running branch obtains a NEW observation;
                                    # none of these actions may reach env.step/Replay.
                            except Exception as error:
                                error.episode_id = context.episode_id
                                error.step_id = None
                                if interrupt_camera_episode(error):
                                    continue
                                raise
                            episode_start_pose = tuple(self.current_observation['state'])
                            gate_wait_announced = False
                            if self.demonstration:
                                say('人工采集已开始：SpaceMouse 控制，Y 成功 / F 失败。')
                            _policy_observation(self.current_observation, 'cpu',
                                                self.config.observation.image_size)
                    time.sleep(self.config.runtime.operator_poll_interval_s)
                    continue
                if env is None:
                    raise RuntimeError('Running episode has no commissioned environment')
                # Peek only: env.step consumes the report immediately before command
                # submission. A pre-read here would lose that motion.
                intervention = self.coordinator.intervention
                if self.demonstration:
                    if not (intervention.has_new_report() or intervention.verified_neutral):
                        time.sleep(self.config.runtime.operator_poll_interval_s)
                        continue
                else:
                    # A step may only be submitted and recorded while the human-input
                    # gate is verifiable: either a fresh report or an observed
                    # exact-zero hold. Those are exactly the two shapes the
                    # transition validator accepts, so any other state either loses
                    # the intervention provenance or aborts the episode. This is
                    # reachable on hardware: the upstream HID reports `ready` only
                    # after BOTH axis channels have been seen, and the start chord
                    # deliberately reads buttons only, so pressing the double key
                    # without ever having moved the knob used to reach the first
                    # env.step with gate.fresh == verified_neutral == False and die
                    # with 'Invalid intervention or freshness gate summary' after a
                    # policy action had already been submitted. Wait for the device
                    # instead, and say exactly what the operator has to do.
                    #
                    # This role reads the raw HID *inside* the callable (there is no
                    # background reader as in the demonstration role), so the gate
                    # can only become verifiable by calling it. A genuine input
                    # fault still raises here and stops the Actor, as designed.
                    gate = getattr(intervention, 'gate', None)
                    if not (getattr(gate, 'fresh', False) or
                            getattr(intervention, 'verified_neutral', False)):
                        intervention()
                        if not (getattr(gate, 'fresh', False) or
                                getattr(intervention, 'verified_neutral', False)):
                            if not gate_wait_announced:
                                gate_wait_announced = True
                                say('SpaceMouse 还没有上报过轴向数据（只按双键不算）：'
                                    '请轻拨一下旋帽再松手，收到轴向报告后本回合立即继续。')
                            elif gate_wait_reminder.due():
                                say('仍在等待 SpaceMouse 轴向数据：轻拨一下旋帽再松手。')
                            time.sleep(self.config.runtime.operator_poll_interval_s)
                            continue
                try:
                    before = env.refresh_observation()
                except Exception as error:
                    if interrupt_camera_episode(error):
                        continue
                    raise
                self.current_observation = before
                context = self.coordinator.context
                inference_started = self.clock()
                policy_action = self._infer(before)
                inference_latency_s = self.clock() - inference_started
                control_period_s = (None if previous_step_at is None else
                                    inference_started - previous_step_at)
                previous_step_at = inference_started
                if terminal_before_next_action():
                    continue
                token = self.coordinator.begin_step()
                self._step_active = True
                try:
                    after, reward, terminated, truncated, info = env.step(policy_action)
                except BaseException as error:
                    from .real_episode import TerminalBeforeCommand
                    if isinstance(error, TerminalBeforeCommand) and pending is not None:
                        self._step_active = False
                        terminal_before_next_action(forced=error)
                        continue
                    if interrupt_camera_episode(error, token):
                        continue
                    error.episode_id = token.episode_id
                    error.step_id = token.step_id
                    error.inference_latency_s = inference_latency_s
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
                if pending is not None:
                    upload(pending[0])
                    pending = None
                row = self._build_confirmed_transition(
                    before, after, reward, terminated, truncated, info,
                    token, context, policy_action)
                self._emit('step', episode_id=token.episode_id, step_id=token.step_id,
                           inference_latency_s=inference_latency_s,
                           control_period_s=control_period_s,
                           execution_timing=getattr(env.backend, 'last_execution_timing', {}))
                if terminated and info.get('success_label') is not None:
                    self._emit('terminal_label', transition_id=row['complementary_info']['transition_id'],
                               read_monotonic_ns=getattr(self.coordinator, 'last_terminal_read_ns', None),
                               attribution='during_command_or_successor_read')
                completed += 1
                episode_steps += 1
                episode_interventions += int(info.get('is_intervention') is True)
                if heartbeat.due():
                    say(f'回合 #{self.episodes_completed + 1} 进行中：step={episode_steps} '
                        f'人工接管={episode_interventions} '
                        f'推理={inference_latency_s * 1000:.1f}ms')
                self.current_observation = after
                if terminated or truncated:
                    upload(row)
                    finish_episode(
                        token, context, truncated,
                        labeled=(row['complementary_info'].get('success_label')
                                 in (True, False)),
                        success=(row['complementary_info'].get('success_label')
                                 if terminated else None))
                else:
                    pending = (row, token, context)
                if max_completed_steps is not None and completed >= max_completed_steps:
                    if not terminal_before_next_action() and pending is not None:
                        upload(pending[0])
                        pending = None
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
                code = getattr(error, 'code', None) or code
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
            if pending is not None:
                # Do not invent a terminal label or retry an ambiguous delivery.
                # Preserve the identity and uncertainty in the existing journal.
                try:
                    self._emit('pending_transition_discarded', count=1,
                               transition_id=pending[0]['complementary_info']['transition_id'],
                               reason='actor_exit_before_pending_delivery_confirmed',
                               delivery_status='not_confirmed',
                               error=(type(primary_error).__name__ if primary_error else None))
                except BaseException as error:
                    cleanup_error = error
                    logging.exception('Could not journal pending transition discard')
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
