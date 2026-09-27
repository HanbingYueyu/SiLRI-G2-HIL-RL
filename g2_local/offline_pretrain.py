"""Operator-invoked demonstration bootstrap; no GDK, HID or command ports."""
import argparse
import gc
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch

from .demonstrations import load_demo_episodes, import_demonstrations
from .real_learner import RealLearnerRuntime, load_checkpoint
from .training_config import load_training_config


def digest(module):
    result = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        result.update(key.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def load_offline_config(path, *, accept_motion_profile=False):
    config = load_training_config(path, cli_allow_motion=False)
    if config.motion_permitted or (config.requested_motion and not accept_motion_profile):
        raise ValueError('Motion profile requires explicit offline acceptance; motion remains disabled')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=10)
    parser.add_argument('--accept-motion-profile', action='store_true',
                        help='Use the exact future Actor profile offline; never grants motion')
    args = parser.parse_args()
    config = load_offline_config(args.config, accept_motion_profile=args.accept_motion_profile)
    if args.updates <= 0:
        raise ValueError('Read-only configuration and positive update count required')
    args.output.mkdir(mode=0o700, exist_ok=False)
    class Evidence:
        def event(self, event, **values):
            row = dict(event=event, **values)
            with (args.output / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            print(json.dumps(row, allow_nan=False), flush=True)
    evidence = Evidence()
    paths, episodes, steps = [], 0, 0
    for path in sorted(args.runtime_root.glob('demo-*/demonstrations')):
        if not any(path.glob('*/complete.json')):
            continue
        try:
            rows = list(load_demo_episodes(path, config=config))
        except ValueError as error:
            evidence.event('excluded_dataset', path=str(path), reason=str(error))
            continue
        if not all(batch[-1]['complementary_info']['success_label'] is True for _, batch in rows):
            raise ValueError('This bootstrap requires explicitly successful complete demonstrations')
        paths.append(path)
        episodes += len(rows)
        steps += sum(len(batch) for _, batch in rows)
        del rows
    if episodes < config.optimization.beta_min_demo_episodes or steps > config.optimization.human_capacity:
        raise ValueError('Insufficient episodes or human Replay cannot retain full dataset')
    with (args.output / 'datasets.json').open('x') as stream:
        json.dump(dict(paths=[str(p.resolve()) for p in paths], episodes=episodes, steps=steps,
                       config_hash=config.config_hash), stream, indent=2)
    evidence.event('dataset_ready', episodes=episodes, steps=steps, motion_authorized=False)
    torch.manual_seed(config.runtime.seed)
    np.random.seed(config.runtime.seed)
    random.seed(config.runtime.seed)
    learner = RealLearnerRuntime(config=config, run_id=args.output.name)
    imported = import_demonstrations(learner, paths, evidence=evidence)
    assert imported['accepted'] == steps and len(learner.human_replay) == steps
    before_actor = digest(learner.policy.actor)
    before_critic = digest(learner.policy.critic_ensemble)
    before_expert = digest(learner.policy.expert_network)
    evidence.event('pretrain_started', counts=learner.snapshot_counts())
    assert learner.pretrain_behavior()
    assert before_expert != digest(learner.policy.expert_network)
    evidence.event('pretrain_complete', counts=learner.snapshot_counts(), loss=learner.beta_last_loss)
    for index in range(args.updates):
        metrics = learner.update_once()
        if metrics is None:
            raise RuntimeError('Expected an actual optimizer update')
        evidence.event('update', index=index+1, metrics=metrics)
    assert before_actor != digest(learner.policy.actor)
    assert before_critic != digest(learner.policy.critic_ensemble)
    expected = learner.snapshot_counts()
    actor, critic = digest(learner.policy.actor), digest(learner.policy.critic_ensemble)
    checkpoint = args.output / 'checkpoint.pt'
    evidence.event('checkpoint_saving', counts=expected)
    learner.save_checkpoint(checkpoint)
    del learner
    gc.collect()
    torch.cuda.empty_cache()
    restored = load_checkpoint(checkpoint, expected_run_id=args.output.name,
                               expected_config_hash=config.config_hash).runtime
    restored.config = config
    assert restored.snapshot_counts() == expected
    assert digest(restored.policy.actor) == actor
    assert digest(restored.policy.critic_ensemble) == critic
    metrics = restored.update_once()
    assert metrics is not None and restored.update_count == args.updates+1
    evidence.event('restore_and_update_passed', counts=restored.snapshot_counts(), metrics=metrics,
                   saved_checkpoint_updates=args.updates, motion_authorized=False)
    with (args.output / 'result.json').open('x') as stream:
        json.dump(dict(status='passed', episodes=episodes, steps=steps,
                       checkpoint=str(checkpoint.resolve()), saved_counts=expected,
                       restored_counts=restored.snapshot_counts(), motion_authorized=False), stream, indent=2)


if __name__ == '__main__':
    main()
