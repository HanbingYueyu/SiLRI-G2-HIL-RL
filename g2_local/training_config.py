"""Versioned, fail-closed configuration for the real SiLRI training runtime.

This module parses data and records provenance. It never opens a robot command
port, and its motion permission is only one input to later live safety gates.
"""

from dataclasses import dataclass, fields, replace
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from types import MappingProxyType
from typing import Mapping, Sequence

from .config import HingeInsertTaskConfig, LocalTaskConfig
from .contract import CAMERA_KEYS
from .freshness import FreshnessLimits


_REQUIRED_EVIDENCE = frozenset({
    'freshness_approval', 'xyz_rpy_direction_scale',
    'software_stop_lease_expiry', 'hardware_estop',
})


def _keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f'{label}: exact keys required: {sorted(expected)}')
    return value


def reject_unknown_or_missing_keys(payload, expected):
    """Require exactly the named keys in a JSON object."""
    return _keys(payload, expected, 'configuration')


def _fields(cls):
    return tuple(field.name for field in fields(cls))


def _number(value, name, *, positive=False, nonnegative=False):
    if type(value) not in (int, float):
        raise ValueError(f'{name}: finite number required')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f'{name}: invalid finite number')
    return float(value)


def _integer(value, name, *, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name}: integer out of range')
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f'{name}: boolean required')
    return value


def _string(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(f'{name}: nonempty string required')
    return value


def _path(value, name):
    result = Path(_string(value, name))
    if not result.is_absolute() or '..' in result.parts:
        raise ValueError(f'{name}: absolute, normalized path required')
    return result


def _numbers(value, count, name):
    if type(value) is not list or len(value) != count:
        raise ValueError(f'{name}: {count} numeric values required')
    return tuple(_number(item, name) for item in value)


def _integers(value, count, name, *, minimum=0):
    if type(value) is not list or len(value) != count:
        raise ValueError(f'{name}: {count} integers required')
    return tuple(_integer(item, name, minimum=minimum) for item in value)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Nonfinite JSON number: {value}')


class _SymlinkPathError(ValueError):
    pass


def _open_no_symlink(path: Path, flags: int, mode: int | None = None):
    """Walk every directory through pinned descriptors without following links."""
    path = Path(path)
    parts = path.parts
    if '..' in parts:
        raise ValueError(f'Unsafe path: {path}')
    anchor = '/' if path.is_absolute() else '.'
    names = parts[1:] if path.is_absolute() else parts
    if not names:
        return os.open(anchor, flags | os.O_NOFOLLOW)
    directory_fd = os.open(anchor, os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for name in names[:-1]:
            try:
                next_fd = os.open(name, os.O_PATH | os.O_DIRECTORY |
                                  os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory_fd)
            except OSError as exc:
                try:
                    metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError:
                    metadata = None
                if metadata is not None and stat.S_ISLNK(metadata.st_mode):
                    raise _SymlinkPathError(f'Symlinked path component: {path}') from exc
                raise
            os.close(directory_fd)
            directory_fd = next_fd
        name = names[-1]
        try:
            if mode is None:
                return os.open(name, flags | os.O_NOFOLLOW, dir_fd=directory_fd)
            return os.open(name, flags | os.O_NOFOLLOW, mode, dir_fd=directory_fd)
        except OSError as exc:
            try:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError:
                metadata = None
            if metadata is not None and stat.S_ISLNK(metadata.st_mode):
                raise _SymlinkPathError(f'Symlinked path component: {path}') from exc
            raise
    finally:
        os.close(directory_fd)


def _open_owned_regular(path: Path):
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        fd = _open_no_symlink(path, flags)
    except OSError as exc:
        raise ValueError(f'Cannot open owned regular file: {path}') from exc
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        os.close(fd)
        raise ValueError(f'Owned regular file required: {path}')
    return fd, metadata


def read_owned_regular_json(path: Path, *, max_bytes: int = 131072):
    """Read a bounded JSON file without following any symlink component."""
    fd, metadata = _open_owned_regular(Path(path))
    try:
        if metadata.st_size > max_bytes:
            raise ValueError('Configuration file is too large')
        with os.fdopen(fd, 'rb') as stream:
            fd = -1
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError('Configuration file is too large')
        return json.loads(data.decode('utf-8'), object_pairs_hook=_unique_pairs,
                          parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid UTF-8 JSON configuration') from exc
    finally:
        if fd >= 0:
            os.close(fd)


def canonical_json(payload) -> bytes:
    """Canonical UTF-8 JSON bytes used for both identity and manifests."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _rename_noreplace(directory_fd: int, source: str, target: str):
    """Atomically publish a new directory without replacing any existing name."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(directory_fd, os.fsencode(source), directory_fd,
                 os.fsencode(target), 1) != 0:  # RENAME_NOREPLACE on Linux
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), target)
        raise OSError(error, os.strerror(error), target)


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ObservationConfig:
    camera_keys: tuple[str, str]
    image_size: int
    camera_rois: Mapping[str, tuple[int, int, int, int]]
    raw_rgb_logging: bool


@dataclass(frozen=True)
class InterventionConfig:
    axis_map: tuple[int, int, int]
    left_button: int
    right_button: int
    engage_deadzone: float
    release_deadzone: float
    release_hold_s: float
    report_max_age_s: float


@dataclass(frozen=True)
class OptimizationConfig:
    online_capacity: int
    human_capacity: int
    min_online_transitions: int
    online_batch_size: int
    human_batch_size: int
    utd_ratio: int
    actor_lr: float
    critic_lr: float
    expert_lr: float
    lagrange_lr: float
    target_update_interval: int
    publish_interval: int
    checkpoint_interval: int


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    device: str
    learner_host: str
    learner_port: int
    queue_capacity: int
    queue_put_timeout_s: float
    transport_timeout_s: float
    context_max_age_s: float
    parameter_heartbeat_s: float
    learner_silence_timeout_s: float
    operator_poll_interval_s: float


@dataclass(frozen=True)
class MotionRuntimeConfig:
    limits: LocalTaskConfig
    control_mode: int
    command_timeout_s: float
    send_timeout_s: float
    stop_timeout_s: float
    reader_timeout_s: float
    command_lifetime_s: float
    send_rate_hz: float
    adapter_root: Path


@dataclass(frozen=True)
class CommissioningEvidence:
    kind: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class CommissioningConfig:
    profile: str
    clock_socket: Path
    expected_master: str
    evidence: Sequence[CommissioningEvidence]

    def verify_files_and_hashes(self, freshness: FreshnessLimits) -> bool:
        kinds = set()
        valid = True
        for item in self.evidence:
            kinds.add(item.kind)
            try:
                fd, _ = _open_owned_regular(item.path)
            except _SymlinkPathError:
                raise
            except ValueError:
                valid = False
                continue
            digest = hashlib.sha256()
            with os.fdopen(fd, 'rb') as stream:
                if item.kind == 'freshness_approval':
                    data = stream.read(131073)
                    if len(data) > 131072:
                        raise ValueError('approved freshness artifact is too large')
                    digest.update(data)
                    if digest.hexdigest() != item.sha256:
                        valid = False
                        continue
                    artifact = json.loads(data.decode('utf-8'), object_pairs_hook=_unique_pairs,
                                          parse_constant=_reject_constant)
                    if (type(artifact) is not dict or type(artifact.get('schema')) is not int or
                            artifact['schema'] != 1 or artifact.get('thresholds_approved') is not True or
                            artifact.get('motion_authorized') is not False):
                        raise ValueError('Invalid approved freshness artifact')
                    limits = _keys(artifact.get('limits'), _fields(FreshnessLimits),
                                   'approved freshness limits')
                    approved = FreshnessLimits(**{key: _number(value, key, positive=True)
                                                  for key, value in limits.items()})
                    if approved != freshness:
                        raise ValueError('Live limits differ from approved freshness limits')
                    continue
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            valid &= digest.hexdigest() == item.sha256
        return self.profile == 'approved' and kinds == _REQUIRED_EVIDENCE and valid


@dataclass(frozen=True)
class LoadedTrainingConfig:
    schema: int
    mode: str
    task: HingeInsertTaskConfig
    motion: MotionRuntimeConfig
    freshness: FreshnessLimits
    observation: ObservationConfig
    intervention: InterventionConfig
    optimization: OptimizationConfig
    runtime: RuntimeConfig
    commissioning: CommissioningConfig
    requested_motion: bool
    motion_permitted: bool
    config_hash: str
    canonical_payload: Mapping[str, object]

    def write_manifest(self, output: Path, *, run_id: str, role: str) -> Path:
        """Publish a new manifest under an owned, non-group/world-writable parent.

        Linux mkdirat does not return an FD; the private parent is the trust
        boundary until the random staging directory is opened and pinned.
        """
        _string(run_id, 'run_id')
        _string(role, 'role')
        output = Path(output)
        if output.name in ('', '.', '..'):
            raise ValueError('A new manifest output directory is required')
        data = {'schema': self.schema, 'run_id': run_id, 'role': role,
                'config_sha256': self.config_hash,
                'config': _thaw(self.canonical_payload)}
        encoded = canonical_json(data) + b'\n'
        parent_fd = _open_no_symlink(output.parent,
                                     os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        stage_name = f'.{output.name}.manifest-{secrets.token_hex(16)}'
        stage_fd = None
        stage_identity = None
        manifest_created = False
        published = False
        try:
            parent_status = os.fstat(parent_fd)
            if (parent_status.st_uid != os.geteuid() or
                    parent_status.st_mode & 0o022):
                raise ValueError('Manifest output parent must be owned and not group/world writable')
            os.mkdir(stage_name, 0o700, dir_fd=parent_fd)
            stage_fd = os.open(stage_name, os.O_RDONLY | os.O_DIRECTORY |
                               os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
            stage_status = os.fstat(stage_fd)
            stage_identity = (stage_status.st_dev, stage_status.st_ino)
            if stage_status.st_uid != os.geteuid() or os.listdir(stage_fd):
                raise ValueError('New manifest staging directory must be owned and empty')
            os.fchmod(stage_fd, 0o700)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
            fd = os.open('run_manifest.json', flags, 0o600, dir_fd=stage_fd)
            manifest_created = True
            with os.fdopen(fd, 'wb') as stream:
                stream.write(encoded)
                stream.flush()
                os.fchmod(stream.fileno(), 0o400)
                os.fsync(stream.fileno())
            os.fsync(stage_fd)
            _rename_noreplace(parent_fd, stage_name, output.name)
            published = True
            os.fsync(parent_fd)
            return output / 'run_manifest.json'
        finally:
            if stage_fd is not None:
                if not published and manifest_created:
                    try:
                        current = os.stat(stage_name, dir_fd=parent_fd,
                                          follow_symlinks=False)
                    except OSError:
                        current = None
                    if (current is not None and
                            (current.st_dev, current.st_ino) == stage_identity):
                        try:
                            os.unlink('run_manifest.json', dir_fd=stage_fd)
                            os.rmdir(stage_name, dir_fd=parent_fd)
                        except OSError:
                            pass
                os.close(stage_fd)
            os.close(parent_fd)


SCHEMA_ONE_KEYS = frozenset({
    'schema', 'mode', 'requested_motion', 'task', 'motion', 'freshness',
    'observation', 'intervention', 'optimization', 'runtime', 'commissioning',
})


def parse_schema_one(payload) -> LoadedTrainingConfig:
    if _integer(payload['schema'], 'schema', minimum=1) != 1:
        raise ValueError('Only training schema 1 is supported')
    mode = _string(payload['mode'], 'mode')
    if mode not in ('train', 'eval'):
        raise ValueError('mode must be train or eval')
    requested_motion = _boolean(payload['requested_motion'], 'requested_motion')

    raw = _keys(payload['task'], _fields(HingeInsertTaskConfig), 'task')
    task_values = {key: _number(raw[key], key) for key in (
        'control_hz', 'success_reward', 'failure_reward', 'step_reward',
        'target_xy_range_m', 'ee_xyz_range_m', 'ee_rpy_range_rad')}
    task_values.update(max_episode_steps=_integer(raw['max_episode_steps'], 'max_episode_steps', minimum=1),
                       fix_gripper=_boolean(raw['fix_gripper'], 'fix_gripper'),
                       action_scale=_numbers(raw['action_scale'], 6, 'action_scale'),
                       reward_source=_string(raw['reward_source'], 'reward_source'))
    task = HingeInsertTaskConfig(**task_values)

    raw = _keys(payload['motion'], (
        'workspace_low', 'workspace_high', 'control_mode', 'command_timeout_s',
        'send_timeout_s', 'stop_timeout_s', 'reader_timeout_s',
        'command_lifetime_s', 'send_rate_hz', 'adapter_root'), 'motion')
    low = _numbers(raw['workspace_low'], 3, 'workspace_low')
    high = _numbers(raw['workspace_high'], 3, 'workspace_high')
    limits = task.motion_config(workspace_low=low, workspace_high=high)
    limits.validate_motion()
    motion = MotionRuntimeConfig(
        limits=limits, control_mode=_integer(raw['control_mode'], 'control_mode', minimum=0),
        command_timeout_s=_number(raw['command_timeout_s'], 'command_timeout_s', positive=True),
        send_timeout_s=_number(raw['send_timeout_s'], 'send_timeout_s', positive=True),
        stop_timeout_s=_number(raw['stop_timeout_s'], 'stop_timeout_s', positive=True),
        reader_timeout_s=_number(raw['reader_timeout_s'], 'reader_timeout_s', positive=True),
        command_lifetime_s=_number(raw['command_lifetime_s'], 'command_lifetime_s', positive=True),
        send_rate_hz=_number(raw['send_rate_hz'], 'send_rate_hz', positive=True),
        adapter_root=_path(raw['adapter_root'], 'adapter_root'))
    if (motion.send_timeout_s > motion.command_timeout_s or
            motion.command_lifetime_s > motion.command_timeout_s or
            1 / motion.send_rate_hz > motion.command_lifetime_s):
        raise ValueError('Invalid command timing relationship')

    raw = _keys(payload['freshness'], _fields(FreshnessLimits), 'freshness')
    freshness = FreshnessLimits(**{key: _number(raw[key], key, positive=True)
                                  for key in _fields(FreshnessLimits)})

    raw = _keys(payload['observation'], _fields(ObservationConfig), 'observation')
    camera_keys = raw['camera_keys']
    if type(camera_keys) is not list or tuple(camera_keys) != CAMERA_KEYS:
        raise ValueError('Exactly the two G2 camera keys are required')
    rois = _keys(raw['camera_rois'], camera_keys, 'camera_rois')
    parsed_rois = {}
    for key, value in rois.items():
        roi = _integers(value, 4, f'camera_rois.{key}')
        if roi[2] <= roi[0] or roi[3] <= roi[1]:
            raise ValueError(f'camera_rois.{key}: invalid bounds')
        parsed_rois[key] = roi
    observation = ObservationConfig(
        camera_keys=tuple(camera_keys),
        image_size=_integer(raw['image_size'], 'image_size', minimum=1),
        camera_rois=MappingProxyType(parsed_rois),
        raw_rgb_logging=_boolean(raw['raw_rgb_logging'], 'raw_rgb_logging'))

    raw = _keys(payload['intervention'], _fields(InterventionConfig), 'intervention')
    axis_map = _integers(raw['axis_map'], 3, 'axis_map', minimum=-3)
    if sorted(abs(axis) for axis in axis_map) != [1, 2, 3]:
        raise ValueError('axis_map must be a signed permutation of 1, 2, 3')
    intervention = InterventionConfig(
        axis_map=axis_map,
        left_button=_integer(raw['left_button'], 'left_button'),
        right_button=_integer(raw['right_button'], 'right_button'),
        engage_deadzone=_number(raw['engage_deadzone'], 'engage_deadzone', positive=True),
        release_deadzone=_number(raw['release_deadzone'], 'release_deadzone', nonnegative=True),
        release_hold_s=_number(raw['release_hold_s'], 'release_hold_s', positive=True),
        report_max_age_s=_number(raw['report_max_age_s'], 'report_max_age_s', positive=True))
    if (intervention.left_button == intervention.right_button or
            intervention.release_deadzone > intervention.engage_deadzone or
            intervention.engage_deadzone >= 1):
        raise ValueError('Invalid intervention buttons or deadzones')

    raw = _keys(payload['optimization'], _fields(OptimizationConfig), 'optimization')
    optimization = OptimizationConfig(**{
        key: (_number(raw[key], key, positive=True) if key.endswith('_lr') else
              _integer(raw[key], key, minimum=1))
        for key in _fields(OptimizationConfig)})
    if (optimization.online_batch_size > optimization.online_capacity or
            optimization.human_batch_size > optimization.human_capacity or
            optimization.min_online_transitions > optimization.online_capacity):
        raise ValueError('Optimization batch or warmup exceeds replay capacity')

    raw = _keys(payload['runtime'], _fields(RuntimeConfig), 'runtime')
    runtime = RuntimeConfig(
        seed=_integer(raw['seed'], 'seed'),
        device=_string(raw['device'], 'device'),
        learner_host=_string(raw['learner_host'], 'learner_host'),
        learner_port=_integer(raw['learner_port'], 'learner_port', minimum=1, maximum=65535),
        queue_capacity=_integer(raw['queue_capacity'], 'queue_capacity', minimum=1),
        queue_put_timeout_s=_number(raw['queue_put_timeout_s'], 'queue_put_timeout_s', positive=True),
        transport_timeout_s=_number(raw['transport_timeout_s'], 'transport_timeout_s', positive=True),
        context_max_age_s=_number(raw['context_max_age_s'], 'context_max_age_s', positive=True),
        parameter_heartbeat_s=_number(raw['parameter_heartbeat_s'], 'parameter_heartbeat_s', positive=True),
        learner_silence_timeout_s=_number(raw['learner_silence_timeout_s'], 'learner_silence_timeout_s', positive=True),
        operator_poll_interval_s=_number(raw['operator_poll_interval_s'], 'operator_poll_interval_s', positive=True))
    if (runtime.learner_host not in ('127.0.0.1', 'localhost', '::1') or
            runtime.parameter_heartbeat_s >= runtime.learner_silence_timeout_s or
            runtime.queue_put_timeout_s > runtime.transport_timeout_s or
            runtime.operator_poll_interval_s > intervention.report_max_age_s):
        raise ValueError('Invalid runtime transport or heartbeat timing')

    raw = _keys(payload['commissioning'], _fields(CommissioningConfig), 'commissioning')
    profile = _string(raw['profile'], 'profile')
    if profile not in ('unapproved', 'approved'):
        raise ValueError('Unknown commissioning profile')
    if type(raw['evidence']) is not list:
        raise ValueError('evidence must be a list')
    evidence = []
    for entry in raw['evidence']:
        entry = _keys(entry, _fields(CommissioningEvidence), 'evidence entry')
        kind = _string(entry['kind'], 'evidence kind')
        digest = _string(entry['sha256'], 'evidence sha256')
        if kind not in _REQUIRED_EVIDENCE or len(digest) != 64 or any(
                ch not in '0123456789abcdef' for ch in digest):
            raise ValueError('Unknown evidence kind or invalid SHA-256')
        evidence.append(CommissioningEvidence(kind, _path(entry['path'], 'evidence path'), digest))
    if len({item.kind for item in evidence}) != len(evidence):
        raise ValueError('Duplicate commissioning evidence kind')
    commissioning = CommissioningConfig(
        profile=profile, clock_socket=_path(raw['clock_socket'], 'clock_socket'),
        expected_master=_string(raw['expected_master'], 'expected_master'),
        evidence=tuple(evidence))

    return LoadedTrainingConfig(
        schema=1, mode=mode, task=task, motion=motion, freshness=freshness,
        observation=observation, intervention=intervention,
        optimization=optimization, runtime=runtime, commissioning=commissioning,
        requested_motion=requested_motion, motion_permitted=False,
        config_hash='', canonical_payload=MappingProxyType({}))


def load_training_config(path: Path, *, cli_allow_motion: bool) -> LoadedTrainingConfig:
    if type(cli_allow_motion) is not bool:
        raise ValueError('cli_allow_motion must be a boolean')
    payload = read_owned_regular_json(path, max_bytes=131072)
    reject_unknown_or_missing_keys(payload, SCHEMA_ONE_KEYS)
    parsed = parse_schema_one(payload)
    approved = parsed.commissioning.verify_files_and_hashes(parsed.freshness)
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    return replace(parsed,
                   motion_permitted=bool(cli_allow_motion and parsed.requested_motion and approved),
                   config_hash=digest, canonical_payload=_freeze(payload))
