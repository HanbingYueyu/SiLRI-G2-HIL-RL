"""GPU-free preflight for g2_local.offline_pretrain.

Repeats every check offline_pretrain performs *before* it constructs a model, so
an operator never discovers a config/demo/identity problem after a GPU rebuild
has already been paid for.  Lives under ``scripts/`` (gitignored) and is
therefore outside the algorithm source digest, so running it can never
invalidate a seed.
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from g2_local.code_identity import contract_digest, source_digest  # noqa: E402
from g2_local.demonstrations import load_demo_episodes  # noqa: E402
from g2_local.offline_pretrain import load_offline_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--actor-bc-steps', type=int, default=1000)
    args = parser.parse_args()

    problems = []
    config = load_offline_config(args.config, accept_motion_profile=True)
    print(f'config           : {args.config.resolve()}')
    print(f'config_hash      : {config.config_hash}')
    print(f'motion_permitted : {config.motion_permitted}')
    print(f'contract_sha256  : {contract_digest(config)}')
    print(f'source_sha256    : {source_digest(REPO)}')
    limits = config.motion.limits
    print(f'workspace_low    : {getattr(limits, "workspace_low", None)}')
    print(f'workspace_high   : {getattr(limits, "workspace_high", None)}')
    print(f'auto_reset       : {config.motion.auto_reset.enabled}')

    if args.output.exists():
        problems.append(f'output directory already exists: {args.output.resolve()}')
    if not 0 <= args.actor_bc_steps <= 100_000:
        problems.append('actor-bc-steps out of range 0..100000')
    if config.optimization.bc_weight and not config.optimization.bc_weight_decay_updates:
        problems.append('bc_weight set but bc_weight_decay_updates is 0 (never decays)')

    paths, episodes, steps, total = [], 0, 0, 0
    for path in sorted(args.runtime_root.glob('demo-*/demonstrations')):
        if not any(path.glob('*/complete.json')):
            continue
        total += 1
        try:
            rows = list(load_demo_episodes(path, config=config))
        except ValueError as error:
            print(f'excluded         : {path} :: {error}')
            continue
        if not all(batch[-1]['complementary_info']['success_label'] is True for _, batch in rows):
            problems.append(f'non-successful complete demo accepted by glob: {path}')
            continue
        paths.append(path)
        episodes += len(rows)
        steps += sum(len(batch) for _, batch in rows)
        del rows

    print(f'demo dirs seen   : {total}')
    print(f'datasets         : {len(paths)}')
    print(f'episodes / steps : {episodes} / {steps}')
    print(f'human_capacity   : {config.optimization.human_capacity}')
    print(f'beta_min_episodes: {config.optimization.beta_min_demo_episodes}')
    if episodes < config.optimization.beta_min_demo_episodes:
        problems.append(f'episodes {episodes} < beta_min_demo_episodes '
                        f'{config.optimization.beta_min_demo_episodes}')
    if steps > config.optimization.human_capacity:
        problems.append(f'steps {steps} > human_capacity {config.optimization.human_capacity}')

    if problems:
        print('\nPREFLIGHT FAILED')
        for item in problems:
            print(f'  - {item}')
        raise SystemExit(1)
    print('\nPREFLIGHT OK')
    print(json.dumps(dict(config_hash=config.config_hash,
                          contract_sha256=contract_digest(config),
                          source_sha256=source_digest(REPO),
                          datasets=len(paths), episodes=episodes, steps=steps), indent=2))


if __name__ == '__main__':
    main()
