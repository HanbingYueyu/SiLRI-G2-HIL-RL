"""G2 integration entrypoints using upstream gRPC, replay and real SiLRI.

The bounded synthetic run exercises software, not robot learning. GDK read-only
is exposed separately; no synthetic backend is ever substituted for real GDK.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import draccus
import json
from pathlib import Path
from queue import Queue
import threading
import time

import grpc
import numpy as np
import torch
from lerobot.scripts.rl.learner_service import LearnerService
from lerobot.transport import services_pb2 as pb, services_pb2_grpc as rpc
from lerobot.transport.utils import (bytes_to_state_dict, bytes_to_transitions,
    receive_bytes_in_chunks, send_bytes_in_chunks, state_to_bytes, transitions_to_bytes)
from lerobot.utils.buffer import ReplayBuffer, concatenate_batch_transitions
from .env import G2LocalEnv, SyntheticBackend
from .policy import create_policy


def emit(**data):
    print(json.dumps(data), flush=True)


def receive_parameters(stream, output, stop):
    try:
        receive_bytes_in_chunks(stream, output, stop)
    except grpc.RpcError as exc:
        if not stop.is_set():
            output.put(exc)


def train_batch(policy, optimizers, data, names):
    data = dict(data, is_intervention=data['complementary_info']['is_intervention'])
    metrics = {}
    for name in names:
        if name in ('expert', 'actor_bc') and not data['is_intervention'].any():
            continue  # Do not advance Adam moments on an empty human mask.
        policy.zero_grad(set_to_none=True)
        loss = policy(data, model=name)['loss_' + name]
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite {name} loss')
        loss.backward()
        optimizer = optimizers['actor' if name == 'actor_bc' else name]
        params = [p for group in optimizer.param_groups for p in group['params']]
        torch.nn.utils.clip_grad_norm_(params, 40., error_if_nonfinite=True)
        optimizer.step()
        metrics[name] = float(loss.detach())
    return metrics


def _buffer_transition(row):
    """Drop audit-only provenance before inserting into ReplayBuffer.

    ReplayBuffer intentionally stores tensor/scalar training fields only. The
    complete policy-vs-human-vs-executed action record remains in the learner
    checkpoint's ``records`` list for episode auditing and replay provenance.
    """
    result = dict(row)
    result.pop('provenance', None)
    return result


def learner(args):
    policy = create_policy(args.device)
    optimizers, _ = policy.get_optimizer_and_scheduler()
    incoming, parameters, interactions = Queue(), Queue(), Queue()
    stop = threading.Event()
    server = grpc.server(ThreadPoolExecutor(max_workers=3))
    rpc.add_LearnerServiceServicer_to_server(
        LearnerService(stop, parameters, .01, incoming, interactions), server)
    if not server.add_insecure_port(f'127.0.0.1:{args.port}'):
        raise RuntimeError('Cannot bind learner port')
    buffers = [ReplayBuffer(128, device=args.device, storage_device='cpu',
                            state_keys=list(policy.config.input_features), use_drq=False,
                            optimize_memory=False) for _ in range(2)]
    records = []
    version = 0
    if args.resume:
        # Local trusted checkpoints only: torch serialization can execute code.
        snapshot = torch.load(args.resume, map_location='cpu', weights_only=False)
        if snapshot['schema'] != 1 or snapshot['config'] != draccus.encode(policy.config):
            raise ValueError('Checkpoint schema or policy configuration mismatch')
        policy.load_state_dict(snapshot['policy'])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(snapshot['optimizers'][name])
        records = snapshot['records']
        for row in records:
            train_row = _buffer_transition(row)
            buffers[0].add(**train_row)
            if bool(row['complementary_info']['is_intervention']):
                buffers[1].add(**train_row)
        version = snapshot['version']
        if version >= args.updates:
            raise ValueError('--updates must exceed saved version when resuming')
        torch.set_rng_state(snapshot['rng'])
        if args.device == 'cuda' and snapshot.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(snapshot['cuda_rng'])
        emit(event='restored', version=version, records=len(records))
    def publish():
        parameters.put(state_to_bytes({'policy': {k: v.detach().cpu().clone()
                        for k, v in policy.actor.state_dict().items()}, 'version': version}))
    server.start()
    publish()
    emit(event='ready', port=args.port, policy='silri', cameras=2)
    deadline = time.monotonic() + args.timeout
    try:
        while version < args.updates:
            if time.monotonic() > deadline:
                raise TimeoutError('No completed actor/learner exchange before deadline')
            try:
                packet = incoming.get(timeout=.2)
            except Exception as exc:
                from queue import Empty
                if isinstance(exc, Empty):
                    continue
                raise
            for row in bytes_to_transitions(packet):
                records.append(row)
                train_row = _buffer_transition(row)
                buffers[0].add(**train_row)
                if bool(row['complementary_info']['is_intervention']):
                    buffers[1].add(**train_row)
            if not all(len(buffer) >= 2 for buffer in buffers):
                publish()
                continue
            human = buffers[1].sample(2)
            if version == 0:
                train_batch(policy, optimizers, human, ('expert', 'actor_bc'))
                policy.actor_target.load_state_dict(policy.actor.state_dict())
            data = concatenate_batch_transitions(buffers[0].sample(2), human)
            metrics = train_batch(policy, optimizers, data, ('critic', 'actor', 'lagrange', 'expert'))
            policy.update_target_networks()
            version += 1
            publish()
            emit(event='updated', version=version, online=len(buffers[0]),
                 human=len(buffers[1]), losses=metrics)
        args.output.mkdir(parents=True, exist_ok=True)
        snapshot = {'schema': 1, 'policy': policy.state_dict(),
                    'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
                    'version': version, 'records': records, 'rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if args.device == 'cuda' else None,
                    'config': draccus.encode(policy.config)}
        temporary = args.output / 'checkpoint.tmp'
        torch.save(snapshot, temporary)
        temporary.replace(args.output / 'checkpoint.pt')
        # Leave the final snapshot available long enough for the actor to load it.
        stop.wait(2)
    finally:
        stop.set()
        server.stop(0).wait()


def actor(args):
    from actor import make_policy_obs
    policy = create_policy(args.device).eval()
    channel = grpc.insecure_channel(f'127.0.0.1:{args.port}')
    grpc.channel_ready_future(channel).result(timeout=args.timeout)
    client = rpc.LearnerServiceStub(channel)
    stop, parameters = threading.Event(), Queue()
    stream = client.StreamParameters(pb.Empty(), timeout=args.timeout)
    receiver = threading.Thread(target=receive_parameters,
                                args=(stream, parameters, stop), daemon=True)
    receiver.start()
    env = G2LocalEnv(SyntheticBackend(), max_steps=4)
    version = -1
    obs, _ = env.reset()
    deadline = time.monotonic() + args.timeout
    try:
        for step in range(args.updates * 12):
            if time.monotonic() > deadline:
                raise TimeoutError('Actor did not receive updated policy')
            payload = parameters.get(timeout=args.timeout)
            if isinstance(payload, Exception):
                raise payload
            snapshot = bytes_to_state_dict(payload)
            policy.actor.load_state_dict(snapshot['policy'])
            version = int(snapshot['version'])
            emit(event='loaded', version=version)
            if version >= args.updates:
                break
            packet = []
            for j in range(4):
                state = make_policy_obs(obs, torch.device(args.device), 'g2')
                with torch.no_grad():
                    action = policy.select_action(state)[0].squeeze(0).cpu().numpy()
                # Explicit synthetic human input, never called a real demonstration.
                active = j % 2 == 0
                env.intervention = lambda: (active, np.zeros(6) if active else None)
                nxt, reward, terminated, truncated, info = env.step(action)
                executed = np.asarray(info['executed_action'], dtype=np.float32)
                human = info.get('human_action')
                provenance = dict(
                    policy_action=tuple(float(value) for value in action),
                    human_action=None if human is None else tuple(float(value) for value in human),
                    executed_action=tuple(float(value) for value in executed),
                    reward_source=info.get('reward_source', 'unknown'),
                    success_label=info.get('success_label'),
                    target_offset_m=tuple(float(value) for value in info['target_offset_m']),
                    ee_reset_offset=tuple(float(value) for value in info['ee_reset_offset']),
                )
                row = dict(state={k: v.cpu() for k, v in state.items()},
                           next_state=make_policy_obs(nxt, torch.device('cpu'), 'g2'),
                           action=torch.tensor(executed), reward=reward,
                           done=terminated, truncated=truncated,
                           complementary_info={'is_intervention': active,
                                               'actor_version': version,
                                               'step_id': step * 4 + j,
                                               'synthetic': True},
                           provenance=provenance)
                packet.append(row)
                obs = nxt
                if terminated or truncated:
                    obs, _ = env.reset()
            client.SendTransitions(send_bytes_in_chunks(transitions_to_bytes(packet), pb.Transition),
                                   timeout=args.timeout)
        if version < args.updates:
            raise RuntimeError('Actor never loaded final policy')
    finally:
        env.close()
        stop.set()
        stream.cancel()
        channel.close()
        receiver.join(timeout=2)


def main(role, argv):
    parser = argparse.ArgumentParser(description='Bounded dual-RGB SiLRI software integration run')
    parser.add_argument('--port', type=int, default=50175)
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--output', type=Path, default=Path('runtime/software_loop'))
    parser.add_argument('--resume', type=Path,
                        help='Learner checkpoint from this trusted local software run')
    args = parser.parse_args(argv)
    if args.updates < 1 or args.timeout <= 0:
        parser.error('Positive update budget and timeout required')
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    (learner if role == 'learner' else actor)(args)
