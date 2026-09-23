"""Validated real SiLRI learner with dual replay and atomic local resume."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import tempfile
import threading
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import torch

from lerobot.transport import services_pb2 as pb, services_pb2_grpc as rpc
from lerobot.transport.utils import (bytes_to_transitions, send_bytes_in_chunks,
                                     state_to_bytes)
from lerobot.utils.buffer import ReplayBuffer, concatenate_batch_transitions

from .contract import CAMERA_KEYS
from .policy import create_policy
from .real_actor import ParameterEnvelope, validate_real_transition
from .runtime import train_batch
from .training_config import OptimizationConfig, RuntimeConfig, _open_owned_regular


@dataclass(frozen=True)
class IngestResult:
    accepted: int
    duplicates: int


@dataclass(frozen=True)
class LearnerSnapshot:
    run_id: str
    config_hash: str
    version: int
    message_sequence: int
    software_state_restored: bool
    physical_episode_state: str = 'WAITING_FOR_RESET'
    runtime: object = None

    def __getattr__(self, name):
        if self.runtime is not None:
            return getattr(self.runtime, name)
        raise AttributeError(name)


def _training_row(row):
    with np.errstate(over='ignore'):
        stored_reward = np.float32(row['reward'])
    if not np.isfinite(stored_reward):
        raise ValueError('Reward cannot be represented in replay float32 storage')
    return {**{key: row[key] for key in ('state', 'next_state', 'action',
                                        'done', 'truncated')},
            'reward': float(stored_reward),
            'complementary_info': {
                'is_intervention': float(row['complementary_info']['is_intervention'])}}


def _replay_state(buffer):
    state = {'capacity': buffer.capacity, 'position': buffer.position,
             'size': buffer.size, 'initialized': buffer.initialized}
    if buffer.initialized:
        for key in ('states', 'next_states', 'actions', 'rewards', 'dones',
                    'truncateds', 'complementary_info', 'episode_ends'):
            state[key] = getattr(buffer, key)
    return state


def _restore_replay(buffer, state):
    if (type(state) is not dict or state['capacity'] != buffer.capacity or
            type(state['position']) is not int or
            not 0 <= state['position'] < buffer.capacity or
            type(state['size']) is not int or
            not 0 <= state['size'] <= buffer.capacity or
            type(state['initialized']) is not bool or
            state['initialized'] != (state['size'] > 0)):
        raise ValueError('Invalid replay checkpoint')
    if not state['initialized']:
        if state['position'] != 0:
            raise ValueError('Empty replay has invalid insertion position')
        return
    if state['position'] != state['size'] % buffer.capacity and state['size'] < buffer.capacity:
        raise ValueError('Invalid replay insertion position')
    for key in ('states', 'next_states', 'actions', 'rewards', 'dones',
                'truncateds', 'complementary_info', 'episode_ends'):
        if key not in state:
            raise ValueError('Incomplete replay checkpoint')
    observation_shapes = {'observation.state': (buffer.capacity, 7),
                          **{f'observation.images.{key}': (buffer.capacity, 3, 128, 128)
                             for key in CAMERA_KEYS}}
    def check_tensor(value, shape, dtype, *, finite=False):
        if (type(value) is not torch.Tensor or value.shape != shape or
                value.dtype != dtype or value.device.type != 'cpu' or
                value.layout != torch.strided):
            raise ValueError('Replay tensor contract mismatch')
        occupied = value[:state['size']]
        if finite and not torch.isfinite(occupied).all().item():
            raise ValueError('Nonfinite occupied replay tensor')
    for field in ('states', 'next_states'):
        if type(state[field]) is not dict or set(state[field]) != set(observation_shapes):
            raise ValueError('Replay camera contract mismatch')
        for key, shape in observation_shapes.items():
            check_tensor(state[field][key], shape, torch.float32, finite=True)
            if key.startswith('observation.images.'):
                occupied = state[field][key][:state['size']]
                if bool((occupied < 0).any() or (occupied > 1).any()):
                    raise ValueError('Replay camera value mismatch')
    for field, shape, dtype, finite in (
            ('actions', (buffer.capacity, 6), torch.float32, True),
            ('rewards', (buffer.capacity,), torch.float32, True),
            ('dones', (buffer.capacity,), torch.bool, False),
            ('truncateds', (buffer.capacity,), torch.bool, False),
            ('episode_ends', (buffer.capacity,), torch.bool, False)):
        check_tensor(state[field], shape, dtype, finite=finite)
    if (type(state['complementary_info']) is not dict or
            set(state['complementary_info']) != {'is_intervention'}):
        raise ValueError('Replay intervention contract mismatch')
    check_tensor(state['complementary_info']['is_intervention'],
                 (buffer.capacity,), torch.float32, finite=True)
    buffer.position, buffer.size, buffer.initialized = (state['position'], state['size'], True)
    for key in ('states', 'next_states', 'actions', 'rewards', 'dones',
                'truncateds', 'complementary_info', 'episode_ends'):
        setattr(buffer, key, state[key])
    buffer.has_complementary_info = True
    buffer.complementary_info_keys = list(state['complementary_info'])


def _check_replay_records(buffer, records):
    if len(buffer) != min(len(records), buffer.capacity):
        raise ValueError('Replay size and provenance mismatch')
    if buffer.position != len(records) % buffer.capacity:
        raise ValueError('Replay position and provenance mismatch')
    for index in range(max(0, len(records) - buffer.capacity), len(records)):
        row = records[index]
        slot = index % buffer.capacity
        if not torch.equal(buffer.actions[slot].cpu(), row['action'].cpu()):
            raise ValueError('Replay executed action and provenance mismatch')
        for field, replay_field in (('state', 'states'), ('next_state', 'next_states')):
            for key, value in row[field].items():
                if not torch.equal(getattr(buffer, replay_field)[key][slot].cpu(),
                                   value.squeeze(0).cpu()):
                    raise ValueError('Replay observation and provenance mismatch')
        if (buffer.rewards[slot].item() != torch.tensor(row['reward'], dtype=torch.float32).item() or
                buffer.dones[slot].item() != row['done'] or
                buffer.truncateds[slot].item() != row['truncated']):
            raise ValueError('Replay outcome and provenance mismatch')
        intervention = float(row['complementary_info']['is_intervention'])
        if (set(buffer.complementary_info) != {'is_intervention'} or
                buffer.complementary_info['is_intervention'][slot].item() != intervention):
            raise ValueError('Replay intervention and provenance mismatch')


class RealLearnerRuntime:
    def __init__(self, *, config, run_id, config_hash=None, policy=None,
                 publish=None, checkpoint_path=None, manifest_digest=None):
        self.config = config
        self.run_id = run_id
        self.config_hash = config_hash or config.config_hash
        self.manifest_digest = manifest_digest or self.config_hash
        opt = config.optimization
        for name in ('online_capacity', 'human_capacity', 'min_online_transitions',
                     'online_batch_size', 'human_batch_size', 'utd_ratio',
                     'target_update_interval', 'publish_interval', 'checkpoint_interval'):
            if type(getattr(opt, name)) is not int or getattr(opt, name) <= 0:
                raise ValueError(f'Invalid {name}')
        if (opt.online_batch_size > opt.online_capacity or
                opt.human_batch_size > opt.human_capacity or
                opt.min_online_transitions > opt.online_capacity):
            raise ValueError('Impossible replay batch')
        self.policy = policy or create_policy(config.runtime.device)
        self.optimizers, _ = self.policy.get_optimizer_and_scheduler()
        for name, lr in (('actor', opt.actor_lr), ('critic', opt.critic_lr),
                         ('expert', opt.expert_lr), ('lagrange', opt.lagrange_lr)):
            if not np.isfinite(lr) or lr <= 0:
                raise ValueError(f'Invalid {name} learning rate')
            for group in self.optimizers[name].param_groups:
                group['lr'] = lr
        def buffer(capacity):
            return ReplayBuffer(capacity, device=config.runtime.device,
                                storage_device='cpu',
                                state_keys=list(self.policy.config.input_features),
                                use_drq=False, optimize_memory=False)
        self.online_replay = buffer(opt.online_capacity)
        self.human_replay = buffer(opt.human_capacity)
        self.seen_transition_ids = set()
        self.records = []
        self.version = 0
        self.message_sequence = -1
        self._published_version = 0
        self._published_actor_state = {
            key: value.detach().cpu().clone()
            for key, value in self.policy.actor.state_dict().items()}
        self.accepted_transitions = 0
        self.update_count = 0
        self._interaction_budget = 0
        self.publish = publish
        self.checkpoint_path = checkpoint_path
        self.stopped = threading.Event()
        self._lock = threading.RLock()

    def snapshot_counts(self):
        with self._lock:
            return {'online': len(self.online_replay), 'human': len(self.human_replay),
                    'accepted': self.accepted_transitions, 'updates': self.update_count,
                    'budget': self._interaction_budget,
                    'online_position': self.online_replay.position,
                    'human_position': self.human_replay.position}

    def ingest(self, rows):
        with self._lock:
            rows = tuple(rows)
            prepared = [(validate_real_transition(row, self.run_id, self.config_hash).value,
                         _training_row(row)) for row in rows]
            new = set()
            accepted = duplicates = 0
            for row, (identity, training) in zip(rows, prepared):
                if identity in self.seen_transition_ids or identity in new:
                    duplicates += 1
                    continue
                self.online_replay.add(**training)
                if row['complementary_info']['is_intervention']:
                    self.human_replay.add(**training)
                self.seen_transition_ids.add(identity)
                new.add(identity)
                self.records.append(row)
                accepted += 1
            self.accepted_transitions += accepted
            self._interaction_budget += accepted * self.config.optimization.utd_ratio
            return IngestResult(accepted, duplicates)

    def _envelope(self, version, state):
        self.message_sequence += 1
        return ParameterEnvelope(self.run_id, self.config_hash, version,
                                 self.message_sequence, state)

    def publish_parameters(self):
        with self._lock:
            if self.stopped.is_set():
                raise RuntimeError('Learner stopped')
            state = {key: value.detach().cpu().clone()
                     for key, value in self.policy.actor.state_dict().items()}
            envelope = self._envelope(self.version, state)
            if self.publish is not None:
                try:
                    self.publish(envelope)
                except BaseException:
                    self.stopped.set()
                    raise
            self._published_version = self.version
            self._published_actor_state = state
            return envelope

    def heartbeat_parameters(self):
        with self._lock:
            if self.stopped.is_set():
                raise RuntimeError('Learner stopped')
            envelope = self._envelope(self._published_version,
                                      self._published_actor_state)
            if self.publish is not None:
                try:
                    self.publish(envelope)
                except BaseException:
                    self.stopped.set()
                    raise
            return envelope

    def update_once(self):
        with self._lock:
            if self.stopped.is_set():
                raise RuntimeError('Learner stopped')
            opt = self.config.optimization
            if len(self.online_replay) < max(opt.min_online_transitions,
                                              opt.online_batch_size):
                return None
            online = self.online_replay.sample(opt.online_batch_size)
            has_human = len(self.human_replay) >= opt.human_batch_size
            data = (concatenate_batch_transitions(
                online, self.human_replay.sample(opt.human_batch_size))
                if has_human else online)
            names = ('critic', 'actor', 'lagrange', 'expert', 'actor_bc') if has_human else (
                'critic', 'actor', 'lagrange')
            metrics = train_batch(self.policy, self.optimizers, data, names)
            self.version += 1
            self.update_count += 1
            if self._interaction_budget:
                self._interaction_budget -= 1
            if self.version % opt.target_update_interval == 0:
                self.policy.update_target_networks()
            if self.version % opt.publish_interval == 0:
                self.publish_parameters()
            if self.checkpoint_path is not None and self.version % opt.checkpoint_interval == 0:
                self.save_checkpoint(self.checkpoint_path)
            return metrics

    def update_for_interactions(self):
        with self._lock:
            results = []
            while self._interaction_budget:
                result = self.update_once()
                if result is None:
                    break
                results.append(result)
            return results

    def _payload(self):
        return dict(schema=1, run_id=self.run_id, config_hash=self.config_hash,
                    manifest_digest=self.manifest_digest,
                    camera_keys=CAMERA_KEYS, image_size=128, action_size=6,
                    optimization=asdict(self.config.optimization),
                    runtime=asdict(self.config.runtime),
                    policy=self.policy.state_dict(),
                    optimizers={key: optimizer.state_dict()
                                for key, optimizer in self.optimizers.items()},
                    online_replay=_replay_state(self.online_replay),
                    human_replay=_replay_state(self.human_replay),
                    seen_transition_ids=sorted(self.seen_transition_ids),
                    records=self.records, version=self.version,
                    message_sequence=self.message_sequence,
                    published_version=self._published_version,
                    published_actor_state=self._published_actor_state,
                    accepted_transitions=self.accepted_transitions,
                    update_count=self.update_count,
                    interaction_budget=self._interaction_budget,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    numpy_rng=np.random.get_state(), python_rng=random.getstate())

    def save_checkpoint(self, path: Path) -> Path:
        with self._lock:
            path = Path(path)
            if not path.parent.is_dir() or path.is_symlink():
                raise ValueError('Checkpoint parent must exist and target cannot be a symlink')
            fd, temporary_name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp',
                                                  dir=path.parent)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    torch.save(self._payload(), stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                return path
            finally:
                temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, *, expected_run_id: str, expected_config_hash: str,
                    expected_manifest_digest: str | None = None) -> LearnerSnapshot:
    """Load only an owned regular local checkpoint; torch pickle requires trust."""
    fd, metadata = _open_owned_regular(Path(path))
    if metadata.st_mode & 0o022 or metadata.st_nlink != 1:
        os.close(fd)
        raise ValueError('Owned private trusted checkpoint required')
    with os.fdopen(fd, 'rb') as stream:
        payload = torch.load(stream, map_location='cpu', weights_only=False)
    if (type(payload) is not dict or payload.get('schema') != 1 or
            payload.get('run_id') != expected_run_id or
            payload.get('config_hash') != expected_config_hash or
            payload.get('manifest_digest') != (expected_manifest_digest or expected_config_hash) or
            tuple(payload.get('camera_keys', ())) != CAMERA_KEYS or
            payload.get('image_size') != 128 or payload.get('action_size') != 6):
        raise ValueError('Checkpoint identity or camera/action contract mismatch')
    config = SimpleNamespace(
        optimization=OptimizationConfig(**payload['optimization']),
        runtime=RuntimeConfig(**payload['runtime']),
        config_hash=expected_config_hash)
    learner = RealLearnerRuntime(config=config, run_id=expected_run_id,
                                 manifest_digest=payload['manifest_digest'])
    learner.policy.load_state_dict(payload['policy'], strict=True)
    if set(payload['optimizers']) != set(learner.optimizers):
        raise ValueError('Checkpoint optimizer contract mismatch')
    for key, optimizer in learner.optimizers.items():
        optimizer.load_state_dict(payload['optimizers'][key])
    _restore_replay(learner.online_replay, payload['online_replay'])
    _restore_replay(learner.human_replay, payload['human_replay'])
    learner.seen_transition_ids = set(payload['seen_transition_ids'])
    learner.records = payload['records']
    for row in learner.records:
        validate_real_transition(row, expected_run_id, expected_config_hash)
    record_ids = {row['complementary_info']['transition_id'] for row in learner.records}
    if learner.seen_transition_ids != record_ids or len(record_ids) != len(learner.records):
        raise ValueError('Checkpoint transition dedup mismatch')
    _check_replay_records(learner.online_replay, learner.records)
    human_records = [row for row in learner.records
                     if row['complementary_info']['is_intervention']]
    _check_replay_records(learner.human_replay, human_records)
    for name in ('version', 'message_sequence', 'accepted_transitions',
                 'update_count', 'interaction_budget'):
        if type(payload[name]) is not int or payload[name] < (-1 if name == 'message_sequence' else 0):
            raise ValueError('Invalid checkpoint counter')
    learner.version = payload['version']
    learner.message_sequence = payload['message_sequence']
    published = payload['published_actor_state']
    expected_actor = learner.policy.actor.state_dict()
    if (type(payload['published_version']) is not int or
            not 0 <= payload['published_version'] <= learner.version or
            type(published) is not dict or set(published) != set(expected_actor) or
            any(type(published[key]) is not torch.Tensor or
                published[key].shape != expected_actor[key].shape or
                published[key].dtype != expected_actor[key].dtype or
                not torch.isfinite(published[key]).all().item()
                for key in expected_actor)):
        raise ValueError('Invalid published Actor checkpoint state')
    learner._published_version = payload['published_version']
    learner._published_actor_state = published
    learner.accepted_transitions = payload['accepted_transitions']
    learner.update_count = payload['update_count']
    learner._interaction_budget = payload['interaction_budget']
    torch.set_rng_state(payload['torch_rng'])
    if payload['cuda_rng'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    np.random.set_state(payload['numpy_rng'])
    random.setstate(payload['python_rng'])
    return LearnerSnapshot(expected_run_id, expected_config_hash, learner.version,
                           learner.message_sequence, True, runtime=learner)


class GrpcLearnerService(rpc.LearnerServiceServicer):
    """Loopback transition ingress and monotonic parameter heartbeat stream."""

    def __init__(self, learner: RealLearnerRuntime):
        self.learner = learner
        self._condition = threading.Condition()
        self._latest = None
        learner.publish = self.publish

    def publish(self, envelope):
        with self._condition:
            if self.learner.stopped.is_set():
                raise RuntimeError('Learner stopped')
            self._latest = envelope
            self._condition.notify_all()

    def _stop(self):
        self.learner.stopped.set()
        with self._condition:
            self._condition.notify_all()

    def SendTransitions(self, request_iterator, context):  # noqa: N802
        data = bytearray()
        ended = False
        try:
            for chunk in request_iterator:
                if self.learner.stopped.is_set():
                    raise RuntimeError('Learner stopped')
                if ended:
                    raise ValueError('Unexpected transition chunk after batch end')
                data.extend(chunk.data)
                if chunk.transfer_state == pb.TransferState.TRANSFER_END:
                    ended = True
            if not ended:
                raise ValueError('Incomplete transition batch')
            rows = bytes_to_transitions(bytes(data))
            self.learner.ingest(rows)
            self.learner.update_for_interactions()
        except BaseException:
            self._stop()
            raise
        return pb.Empty()

    def StreamParameters(self, request, context):  # noqa: N802
        heartbeat = self.learner.config.runtime.parameter_heartbeat_s
        last_sequence = -1
        try:
            if hasattr(context, 'add_callback'):
                context.add_callback(self._stop)
            while context.is_active() and not self.learner.stopped.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda: self.learner.stopped.is_set() or
                        not context.is_active() or
                        (self._latest is not None and
                         self._latest.message_sequence > last_sequence),
                        timeout=heartbeat)
                    if self.learner.stopped.is_set() or not context.is_active():
                        return
                    if self._latest is not None and self._latest.message_sequence > last_sequence:
                        envelope = self._latest
                    else:
                        envelope = None
                if envelope is None:
                    envelope = self.learner.heartbeat_parameters()
                if self.learner.stopped.is_set() or not context.is_active():
                    return
                with self._condition:
                    last_sequence = envelope.message_sequence
                payload = dict(run_id=envelope.run_id, config_hash=envelope.config_hash,
                               version=envelope.version,
                               message_sequence=envelope.message_sequence,
                               actor_state=envelope.actor_state)
                for chunk in send_bytes_in_chunks(state_to_bytes(payload), pb.Parameters):
                    if self.learner.stopped.is_set() or not context.is_active():
                        return
                    yield chunk
        finally:
            self._stop()

    def Ready(self, request, context):  # noqa: N802
        return pb.Empty()

    def SendInteractions(self, request_iterator, context):  # noqa: N802
        raise NotImplementedError('Real learner uses transition provenance')
