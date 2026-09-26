"""Operator-facing, fail-closed composition for real SiLRI train and eval."""

import argparse
from concurrent import futures
from dataclasses import asdict
from functools import partial
import json
import logging
import os
from pathlib import Path
import re
import select
import stat
import sys
import time
from types import SimpleNamespace

from .training_config import load_training_config


_RUN_ID = re.compile(r'[A-Za-z0-9_.-]{1,128}\Z', re.ASCII)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('role', choices=('learner', 'actor', 'eval', 'demo'))
    cli.add_argument('--run-id', required=True)
    cli.add_argument('--config', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--allow-motion', action='store_true')
    cli.add_argument('--checkpoint', type=Path)
    cli.add_argument('--context', type=Path)
    cli.add_argument('--hid-device', type=Path)
    cli.add_argument('--demonstrations', type=Path, action='append', default=[],
                     help='Learner only: import a complete local demonstration dataset; repeatable')
    return cli


def bounded_error(error):
    return (f'{type(error).__name__}: {error}'.encode('ascii', 'backslashreplace')
            .decode('ascii')[:512])


class RunEvidenceWriter:
    """Write bounded, private lifecycle records in a new manifest directory."""

    def __init__(self, output, manifest, *, role):
        self.output = Path(output)
        self.manifest = Path(manifest)
        self.role = role
        self.finished = False
        self.events_path = self.output / 'events.jsonl'
        self.summary_path = self.output / 'episode_summaries.jsonl'
        info = self.output.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('Private owned evidence directory required')
        self.event('started', manifest=str(self.manifest))

    def _append(self, path, row):
        encoded = json.dumps(row, sort_keys=True, default=str, ensure_ascii=True,
                             separators=(',', ':')).encode('ascii') + b'\n'
        if len(encoded) > 16384:
            raise ValueError('Evidence event exceeds bound')
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(path, flags, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError('Owned private evidence file required')
            os.write(fd, encoded)
            os.fsync(fd)
        finally:
            os.close(fd)

    def event(self, kind, **fields):
        self._append(self.events_path, {'event': kind, 'role': self.role,
                                        'monotonic_ns': time.monotonic_ns(), **fields})

    def episode(self, summary):
        self._append(self.summary_path, summary)

    def finish(self, status, *, error=None):
        if self.finished:
            return
        row = {'role': self.role, 'status': status, 'monotonic_ns': time.monotonic_ns()}
        if error is not None:
            row['error'] = bounded_error(error) if isinstance(error, BaseException) else str(error)[:512]
        try:
            self.event('finished', status=status)
        except Exception:
            logging.exception('Final event append failed; writing run result independently')
        path = self.output / 'run_result.json'
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(row, stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.finished = True


class EvalTransitionSink:
    """Local eval sink: supplies frozen parameters and never opens an uplink."""

    def __init__(self, run_id, config_hash, version, actor_state, evidence):
        from .real_actor import ParameterEnvelope
        self.envelope = ParameterEnvelope(run_id, config_hash, version, 0, actor_state)
        self.evidence = evidence
        self._sequence = -1
        self._episodes = {}
        self._timing = {}
        self._clipping = 0
        self._freshness_rejects = 0

    def assert_alive(self):
        return None

    def receive_latest_parameters(self):
        from .real_actor import ParameterEnvelope
        self._sequence += 1
        return ParameterEnvelope(self.envelope.run_id, self.envelope.config_hash,
                                 self.envelope.version, self._sequence,
                                 self.envelope.actor_state)

    def _episode(self, episode_id, info=None):
        info = {} if info is None else info
        return self._episodes.setdefault(episode_id, dict(
            episode_id=episode_id, success=False, assisted=False, completed=False, length=0,
            intervention_ratio=0., action_clipping_count=0,
            freshness_rejects=0, stop_reason='', actor_latency_s=None,
            control_period_s=None, policy_version=info.get('actor_version'),
            target_offset_m=info.get('target_offset_m'),
            ee_reset_offset=info.get('ee_reset_offset')))

    def telemetry(self, kind, **fields):
        if kind == 'step':
            self._timing[(fields['episode_id'], fields['step_id'])] = fields
            return
        if kind == 'freshness_reject':
            self._freshness_rejects += 1
            episode_id = fields.get('episode_id')
            if episode_id is not None:
                self._episode(episode_id)['freshness_rejects'] += 1
            self.evidence.event(kind, **fields)
        elif kind == 'stop':
            self.evidence.event('command_stop', **fields)

    def finalize_unfinished(self, reason):
        for item in self._episodes.values():
            if not item['stop_reason']:
                item['stop_reason'] = reason
                if item['length']:
                    item['intervention_ratio'] /= item['length']
                self.evidence.episode(item)

    def send_transition_batch(self, rows):
        for row in rows:
            info = row['complementary_info']
            episode_id = info['episode_id']
            item = self._episode(episode_id, info)
            item['length'] += 1
            item['assisted'] |= bool(info['is_intervention'])
            item['intervention_ratio'] += int(info['is_intervention'])
            clipped = tuple(info['selected_action']) != tuple(info['executed_action'])
            item['action_clipping_count'] += int(clipped)
            self._clipping += int(clipped)
            timing = self._timing.pop((episode_id, info['step_id']), None)
            if timing is not None:
                item['actor_latency_s'] = timing['inference_latency_s']
                item['control_period_s'] = timing['control_period_s']
            if row['done'] or row['truncated']:
                label = info.get('success_label')
                item['success'] = label is True
                item['completed'] = True
                item['stop_reason'] = ('time_limit' if row['truncated'] else
                                       'success' if label is True else
                                       'failure' if label is False else 'terminal')
                item['intervention_ratio'] /= item['length']
                self.evidence.episode(item)

    def summary(self):
        completed = [item for item in self._episodes.values() if item['completed']]
        free = [item for item in completed if not item['assisted']]
        return {'episodes': len(completed),
                'successes': sum(item['success'] for item in completed),
                'intervention_free_success_rate_numerator': sum(item['success'] for item in free),
                'intervention_free_success_rate_denominator': len(free),
                'action_clipping_count': self._clipping,
                'freshness_rejects': self._freshness_rejects}

    def close(self):
        return None


class DemonstrationSink(EvalTransitionSink):
    """Persist full human trajectories locally without a Learner connection."""

    def __init__(self, run_id, config, evidence):
        from .demonstrations import DemonstrationWriter
        super().__init__(run_id, config.config_hash, 0, {}, evidence)
        self.writer = DemonstrationWriter(evidence.output / 'demonstrations',
                                           config=config, run_id=run_id)

    def send_transition_batch(self, rows):
        for row in rows:
            self.writer.append(row)
            super().send_transition_batch((row,))


class EvidenceActorTransport:
    """Record confirmed train transitions only after the learner accepts them."""

    def __init__(self, transport, evidence):
        self.transport = transport
        self.tracker = EvalTransitionSink('', '', 0, {}, evidence)

    def __getattr__(self, name):
        return getattr(self.transport, name)

    def send_transition_batch(self, rows):
        rows = tuple(rows)
        self.transport.send_transition_batch(rows)
        self.tracker.send_transition_batch(rows)

    def telemetry(self, kind, **fields):
        self.tracker.telemetry(kind, **fields)


def import_gdk_runtime():
    """Import HID only after the independent motion permission gate."""
    # The robot adapter is an operator-supplied, commissioned local checkout.
    from g2_adapter.spacemouse_input import CompactHID
    return CompactHID


def load_eval_checkpoint(path, *, run_id, config_hash):
    """Read the trusted frozen Actor snapshot without constructing optimizers."""
    import torch
    from .contract import CAMERA_KEYS
    from .training_config import _open_owned_regular
    fd, metadata = _open_owned_regular(Path(path))
    if metadata.st_mode & 0o022 or metadata.st_nlink != 1:
        os.close(fd)
        raise ValueError('Owned private trusted checkpoint required')
    with os.fdopen(fd, 'rb') as stream:
        payload = torch.load(stream, map_location='cpu', weights_only=False)
    if (type(payload) is not dict or payload.get('schema') != 1 or
            payload.get('run_id') != run_id or payload.get('config_hash') != config_hash or
            payload.get('manifest_digest') != config_hash or
            tuple(payload.get('camera_keys', ())) != CAMERA_KEYS or
            payload.get('image_size') != 128 or payload.get('action_size') != 6):
        raise ValueError('Checkpoint identity or camera/action contract mismatch')
    from .code_identity import algorithm_identity
    if (type(payload.get('runtime')) is not dict or
            payload.get('algorithm_identity') != algorithm_identity(payload['runtime']['device'])):
        raise ValueError('Algorithm code/version identity mismatch; explicit migration required')
    version = payload.get('published_version')
    state = payload.get('published_actor_state')
    if (type(version) is not int or version < 0 or
            type(payload.get('version')) is not int or version > payload['version'] or
            type(state) is not dict or not state or
            any(type(k) is not str or type(v) is not torch.Tensor or
                not torch.isfinite(v).all().item() for k, v in state.items())):
        raise ValueError('Invalid frozen Actor checkpoint state')
    return SimpleNamespace(version=version, actor_state=state,
                           physical_episode_state='WAITING_FOR_RESET')


class _TerminalInput:
    def __enter__(self):
        import termios
        import tty
        if not sys.stdin.isatty():
            raise RuntimeError('Interactive terminal required for Y/F outcomes')
        self.stream = sys.stdin
        self.fd = self.stream.fileno()
        self.original = termios.tcgetattr(self.fd)
        try:
            tty.setcbreak(self.fd, termios.TCSANOW)
        except BaseException:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.original)
            raise
        return self

    def __exit__(self, *exc):
        import termios
        termios.tcsetattr(self.fd, termios.TCSANOW, self.original)

    def read_available(self, *, limit):
        result = []
        while len(result) < limit and select.select([self.stream], [], [], 0)[0]:
            result.append(os.read(self.fd, 1).decode('ascii', 'ignore'))
        return result


def _validate_cli(args, loaded):
    if getattr(args, 'demonstrations', ()) and args.role != 'learner':
        raise ValueError('--demonstrations is only valid for learner')
    if args.role == 'learner' and (args.allow_motion or loaded.mode != 'train' or
                                   args.context or args.hid_device):
        raise ValueError('Learner accepts train mode only and cannot request motion or HID')
    if args.role == 'actor' and (loaded.mode != 'train' or args.checkpoint is not None):
        raise ValueError('Actor requires train mode without a checkpoint')
    if args.role == 'eval' and (loaded.mode != 'train' or args.checkpoint is None):
        raise ValueError('Eval requires the checkpoint training profile and one checkpoint')
    if args.role == 'demo' and (loaded.mode != 'train' or args.checkpoint is not None):
        raise ValueError('Demo requires train task profile without a policy checkpoint')
    if loaded.runtime.learner_host != '127.0.0.1':
        raise ValueError('Loopback learner address required')
    if args.role in ('actor', 'eval', 'demo') and loaded.motion_permitted:
        if args.context is None or args.hid_device is None:
            raise ValueError('Actor/eval require explicit context and HID device paths')
    if args.role != 'demo' and loaded.runtime.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise ValueError('CUDA required by configuration but unavailable')


def _run_learner(args, loaded, evidence):
    import grpc
    from lerobot.transport import services_pb2_grpc as rpc
    from .real_learner import GrpcLearnerService, RealLearnerRuntime, load_checkpoint
    if args.checkpoint:
        snapshot = load_checkpoint(args.checkpoint, expected_run_id=args.run_id,
                                   expected_config_hash=loaded.config_hash)
        learner = snapshot.runtime
        learner.config = loaded  # Restore task/ROI contract for optional demo import.
        learner.checkpoint_path = evidence.output / 'checkpoint.pt'
        evidence.event('resumed', physical_episode_state=snapshot.physical_episode_state)
    else:
        learner = RealLearnerRuntime(config=loaded, run_id=args.run_id,
                                     config_hash=loaded.config_hash,
                                     checkpoint_path=evidence.output / 'checkpoint.pt')
    if getattr(args, 'demonstrations', ()):
        from .demonstrations import import_demonstrations
        result = import_demonstrations(learner, args.demonstrations, evidence=evidence)
        learner.pretrain_behavior()
        learner.save_checkpoint(learner.checkpoint_path)
        evidence.event('demo_import_completed', **result)
        evidence.event('beta_pretrained', steps=learner.beta_pretrain_completed,
                       last_loss=learner.beta_last_loss,
                       human_samples=learner.human_transitions_total)
    service = GrpcLearnerService(learner)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    rpc.add_LearnerServiceServicer_to_server(service, server)
    if server.add_insecure_port(f'127.0.0.1:{loaded.runtime.learner_port}') == 0:
        raise RuntimeError('Learner loopback port unavailable')
    server.start()
    try:
        service.publish(learner.publish_parameters())
        evidence.event('ready', address=f'127.0.0.1:{loaded.runtime.learner_port}',
                       policy_version=learner.version,
                       checkpoint_path=str(learner.checkpoint_path))
        while not learner.stopped.wait(.2):
            pass
    finally:
        try:
            service.close()
        finally:
            try:
                server.stop(grace=loaded.runtime.transport_timeout_s).wait()
            finally:
                evidence.event('learner_stopped', **learner.snapshot_counts())
    if service.preservation_failure is not None:
        raise RuntimeError('Learner recovery checkpoint failed') from service.preservation_failure
    if service.failure is not None:
        raise RuntimeError('Background learner optimization failed') from service.failure
    return 0


def record_actor_failure(runtime, transport, evidence, role, error):
    """Keep physical stop outcome authoritative if evidence append fails."""
    if runtime.stop_confirmed is False:
        error.stop_unconfirmed = True
    tracker = transport.tracker if role == 'actor' else transport
    try:
        tracker.finalize_unfinished('stop_unconfirmed' if runtime.stop_confirmed is False
                                    else 'actor_failure')
    except BaseException:
        logging.exception('Actor episode evidence write failed after shutdown')
    try:
        evidence.event('actor_stopped', reason=runtime.stop_reason or 'actor_failure',
                       stop_confirmed=runtime.stop_confirmed is True,
                       freshness_rejects=runtime.freshness_rejects,
                       error=bounded_error(error))
    except BaseException:
        logging.exception('Actor stop evidence write failed after shutdown')


def _run_actor_or_eval(args, loaded, evidence):
    from .operator_control import EpisodeContextInbox
    from .real_episode import RealEpisodeCoordinator
    from .real_actor import GrpcActorTransport, RealActorRuntime
    from .spacemouse import AutomaticIntervention, DemonstrationIntervention
    from .motion_env import create_motion_env
    from .policy import create_policy
    frozen = None
    if args.role == 'eval':
        frozen = load_eval_checkpoint(args.checkpoint, run_id=args.run_id,
                                      config_hash=loaded.config_hash)
    adapter_root = str(loaded.motion.adapter_root)
    if adapter_root not in sys.path:
        sys.path.insert(0, adapter_root)
    hid_type = import_gdk_runtime()
    with _TerminalInput() as terminal, hid_type(str(args.hid_device)) as reader:
        intervention_type = DemonstrationIntervention if args.role == 'demo' else AutomaticIntervention
        intervention = intervention_type(reader, loaded.intervention)
        coordinator = RealEpisodeCoordinator(
            intervention, terminal, loaded.task,
            context_max_age_s=loaded.runtime.context_max_age_s,
            left_button=loaded.intervention.left_button,
            right_button=loaded.intervention.right_button)
        context = EpisodeContextInbox(args.context)
        if args.role == 'demo':
            policy = None
            transport = DemonstrationSink(args.run_id, loaded, evidence)
            evidence.event('demo_ready', policy_inference=False,
                           dataset_path=str(transport.writer.path))
        elif args.role == 'eval':
            policy = create_policy(loaded.runtime.device)
            policy.actor.load_state_dict(frozen.actor_state, strict=True)
            transport = EvalTransitionSink(args.run_id, loaded.config_hash,
                                           frozen.version, frozen.actor_state, evidence)
            evidence.event('checkpoint_loaded', policy_version=frozen.version,
                           physical_episode_state=frozen.physical_episode_state)
        else:
            policy = None
            transport = EvidenceActorTransport(GrpcActorTransport(
                f'127.0.0.1:{loaded.runtime.learner_port}',
                queue_capacity=loaded.runtime.queue_capacity,
                timeout_s=loaded.runtime.transport_timeout_s,
                queue_put_timeout_s=loaded.runtime.queue_put_timeout_s), evidence)
        runtime = RealActorRuntime(
            config=loaded, run_id=args.run_id, config_hash=loaded.config_hash,
            coordinator=coordinator, context_source=context, transport=transport,
            env_factory=partial(create_motion_env, cli_allow_motion=args.allow_motion),
            policy=policy,
            telemetry=transport.telemetry, demonstration=args.role == 'demo')
        evidence.event('ready', policy_version=runtime.parameter_version)
        try:
            summary = runtime.run()
        except BaseException as error:
            # RealActorRuntime stops the command path, then closes env and transport.
            record_actor_failure(runtime, transport, evidence, args.role, error)
            raise
        else:
            evidence.event('actor_stopped', **asdict(summary),
                           stop_confirmed=runtime.stop_confirmed is True,
                           freshness_rejects=runtime.freshness_rejects)
            if args.role in ('eval', 'demo'):
                evidence.event(f'{args.role}_summary', **transport.summary())
            return 0


def run_role(args, loaded, evidence):
    if args.role == 'learner':
        return _run_learner(args, loaded, evidence)
    return _run_actor_or_eval(args, loaded, evidence)


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        if not _RUN_ID.fullmatch(args.run_id):
            raise ValueError('Bounded ASCII run ID required')
        if args.role == 'learner' and args.allow_motion:
            raise ValueError('--allow-motion is illegal for learner')
        loaded = load_training_config(args.config, cli_allow_motion=bool(args.allow_motion))
        if args.role == 'learner' and loaded.mode != 'train':
            raise ValueError('Learner requires train mode')
        if args.role == 'actor' and loaded.mode != 'train':
            raise ValueError('Actor requires train mode')
        if args.role == 'eval' and loaded.mode != 'train':
            raise ValueError('Eval requires the checkpoint training profile')
        manifest = loaded.write_manifest(args.output, run_id=args.run_id, role=args.role)
        evidence = RunEvidenceWriter(args.output, manifest, role=args.role)
        if args.role in ('actor', 'eval', 'demo') and not loaded.motion_permitted:
            evidence.finish('motion_not_permitted')
            return 2
        try:
            _validate_cli(args, loaded)
        except (ValueError, PermissionError) as error:
            evidence.finish('preflight_failed', error=error)
            return 2
        try:
            result = run_role(args, loaded, evidence)
            evidence.finish('completed')
            return result
        except KeyboardInterrupt as error:
            if getattr(error, 'stop_unconfirmed', False):
                evidence.finish('stop_unconfirmed', error=error)
                return 1
            evidence.finish('operator_interrupt')
            return 130
        except BaseException as error:
            evidence.finish('stop_unconfirmed' if getattr(error, 'stop_unconfirmed', False)
                            else 'failed', error=error)
            return 1
    except (ValueError, PermissionError, OSError) as error:
        print(bounded_error(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
