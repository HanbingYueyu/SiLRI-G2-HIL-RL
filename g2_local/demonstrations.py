"""Local human-only trajectories; complete episodes are the import boundary."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import uuid

import torch

from .real_actor import validate_real_transition
from .training_config import _open_owned_regular, _thaw, canonical_json


def demonstration_contract(config):
    """Allow optimizer/device changes, never silently reinterpret task data."""
    payload = _thaw(config.canonical_payload)
    contract = {key: payload[key] for key in ('task', 'observation', 'intervention')}
    contract['motion'] = {key: payload['motion'][key] for key in
                          ('workspace_low', 'workspace_high', 'control_mode')}
    if 'local_envelope' in payload['motion']:
        contract['motion']['local_envelope'] = payload['motion']['local_envelope']
    return hashlib.sha256(canonical_json(contract)).hexdigest()


def _publish(path, data):
    """Persist then publish; .partial files are never eligible for import."""
    temporary = path.with_name(path.name + '.partial')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    # Hard link gives no-overwrite publication within the same directory.
    os.link(temporary, path, follow_symlinks=False)
    temporary.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read(path, maximum):
    fd, stat = _open_owned_regular(path)
    with os.fdopen(fd, 'rb') as stream:
        if stat.st_size > maximum or stat.st_mode & 0o022:
            raise ValueError('Invalid demonstration file size or permissions')
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError('Demonstration file grew beyond its limit')
    return data


def _validate_demo(row, run_id, config_hash):
    validate_real_transition(row, run_id, config_hash)
    info = row['complementary_info']
    if info['is_intervention'] is not True or tuple(info['policy_action']) != (0.,) * 6:
        raise ValueError('Demonstrations require exclusively human-controlled steps')
    if row['done'] and (info['reward_source'] != 'human' or
                        type(info['success_label']) is not bool):
        raise ValueError('Terminal demonstration requires explicit Y/F label')
    if row['done'] and row['truncated']:
        raise ValueError('Ambiguous demonstration termination')
    if row['done'] != (info['success_label'] is not None):
        raise ValueError('Demonstration label must belong to a terminal step')


class DemonstrationWriter:
    """Each accepted step is durable; an episode closes only on a valid terminal."""

    def __init__(self, path, *, config, run_id):
        self.path = Path(path)
        self.path.mkdir(mode=0o700)
        self.run_id, self.config_hash = run_id, config.config_hash
        self.max_steps = config.task.max_episode_steps
        self.manifest = dict(schema=1, dataset_id=uuid.uuid4().hex,
                             run_id=run_id, config_hash=self.config_hash,
                             contract_sha256=demonstration_contract(config))
        _publish(self.path / 'dataset.json', canonical_json(self.manifest))
        self.episodes = {}

    def append(self, row):
        _validate_demo(row, self.run_id, self.config_hash)
        info = row['complementary_info']
        episode_id = info['episode_id']
        directory = self.path / hashlib.sha256(episode_id.encode()).hexdigest()
        if episode_id not in self.episodes:
            directory.mkdir(mode=0o700)
            self.episodes[episode_id] = {'closed': False, 'hashes': []}
        episode = self.episodes[episode_id]
        if (episode['closed'] or info['step_id'] != len(episode['hashes']) or
                info['step_id'] >= self.max_steps):
            raise ValueError('Non-contiguous or closed demonstration episode')
        buffer = io.BytesIO()
        torch.save(row, buffer)
        data = buffer.getvalue()
        _publish(directory / f'{info["step_id"]:08d}.pt', data)
        episode['hashes'].append(hashlib.sha256(data).hexdigest())
        if row['done'] or row['truncated']:
            _publish(directory / 'complete.json', canonical_json({
                'episode_id': episode_id, 'sha256': episode['hashes']}))
            episode['closed'] = True


def load_demo_episodes(path, *, config):
    """Yield validated complete episodes, with bounded per-episode memory.

    torch weights_only disables arbitrary pickle classes. Digests detect
    accidental corruption, not forgery by someone who can rewrite the dataset.
    """
    path = Path(path)
    manifest = json.loads(_read(path / 'dataset.json', 16384))
    if (set(manifest) != {'schema', 'dataset_id', 'run_id', 'config_hash', 'contract_sha256'} or
            manifest['schema'] != 1 or type(manifest['dataset_id']) is not str or
            len(manifest['dataset_id']) != 32 or
            any(c not in '0123456789abcdef' for c in manifest['dataset_id']) or
            manifest['contract_sha256'] != demonstration_contract(config)):
        raise ValueError('Demonstration task/ROI/action contract mismatch')
    found = False
    for directory in sorted(path.iterdir()):
        if not directory.is_dir():
            continue
        if directory.is_symlink():
            raise ValueError('Symlink demonstration directory')
        completion = directory / 'complete.json'
        if not completion.exists():
            continue  # Interrupted episodes remain on disk, excluded explicitly.
        completed = json.loads(_read(completion, 4 * 1024 * 1024))
        if (set(completed) != {'episode_id', 'sha256'} or
                type(completed['episode_id']) is not str or
                directory.name != hashlib.sha256(completed['episode_id'].encode()).hexdigest() or
                type(completed['sha256']) is not list or
                not 0 < len(completed['sha256']) <= config.task.max_episode_steps):
            raise ValueError('Invalid demonstration episode index')
        rows = []
        for index, digest in enumerate(completed['sha256']):
            data = _read(directory / f'{index:08d}.pt', 2 * 1024 * 1024)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('Demonstration checksum mismatch')
            row = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
            _validate_demo(row, manifest['run_id'], manifest['config_hash'])
            info = row['complementary_info']
            if (info['episode_id'] != completed['episode_id'] or info['step_id'] != index or
                    bool(row['done'] or row['truncated']) != (index == len(completed['sha256']) - 1)):
                raise ValueError('Invalid demonstration step/terminal boundary')
            expected_reward = (config.task.success_reward if info['success_label'] is True else
                               config.task.failure_reward if info['success_label'] is False else
                               config.task.step_reward)
            if row['reward'] != expected_reward:
                raise ValueError('Demonstration reward does not match task labels')
            rows.append(row)
        found = True
        yield manifest, rows
    if not found:
        raise ValueError('No complete demonstration episodes to import')


def import_demonstrations(learner, paths, *, evidence):
    """Import before network startup; preserve source IDs in the import journal.

    Full validation precedes ingestion. IDs remain stable on checkpoint resume,
    so reimporting the same dataset does not double-count replay or UTD credit.
    """
    paths = tuple(Path(path) for path in paths)
    for path in paths:
        for _manifest, _rows in load_demo_episodes(path, config=learner.config):
            pass
    accepted = duplicates = episodes = 0
    for path in paths:
        for manifest, source_rows in load_demo_episodes(path, config=learner.config):
            source_id = source_rows[0]['complementary_info']['episode_id']
            key = manifest['dataset_id'] + '/' + source_id
            episode_id = 'demo-' + hashlib.sha256(key.encode()).hexdigest()
            rows = []
            for source in source_rows:
                row = deepcopy(source)
                info = row['complementary_info']
                info.update(run_id=learner.run_id, config_hash=learner.config_hash,
                            episode_id=episode_id,
                            transition_id=f'{learner.run_id}/{episode_id}/{info["step_id"]}')
                rows.append(row)
            result = learner.ingest(rows)
            accepted += result.accepted
            duplicates += result.duplicates
            episodes += 1
            evidence.event('demo_import', dataset_id=manifest['dataset_id'],
                           source_run_id=manifest['run_id'], source_config_hash=manifest['config_hash'],
                           source_episode_id=source_id, episode_id=episode_id,
                           accepted=result.accepted, duplicates=result.duplicates,
                           dataset_path=str(path.resolve()))
    return dict(accepted=accepted, duplicates=duplicates, episodes=episodes)


def main(argv=None):
    """Offline dataset validation; no model, device, or command port creation."""
    import argparse
    from .training_config import load_training_config
    parser = argparse.ArgumentParser(description='Validate complete G2 human demonstrations offline')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_training_config(args.config, cli_allow_motion=False)
        episodes = transitions = successes = failures = truncated = 0
        for _, rows in load_demo_episodes(args.dataset, config=config):
            episodes += 1
            transitions += len(rows)
            successes += rows[-1]['complementary_info']['success_label'] is True
            failures += rows[-1]['complementary_info']['success_label'] is False
            truncated += rows[-1]['truncated']
        print(json.dumps(dict(episodes=episodes, transitions=transitions,
                              successes=successes, failures=failures, truncated=truncated,
                              motion_authorized=False)))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Demonstration validation failed: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
