"""Export the live experiment identity so a run can be audited from the repo.

The live training config is ``runtime/train-fixed-fridge-20260928-camera-relaxed.json``,
which is gitignored (the whole ``runtime/`` tree is). That made the reviewed
commit unable to see which parameters were actually used. This script writes:

  * ``configs/runtime-live.json``      byte-identical snapshot, meant to be committed
  * ``runtime/experiment-manifest-<timestamp>.json``  hashes, versions, commands,
    and the seed/training directories that currently exist

Usage:
    bash run_g2_python.sh scripts/export_run_manifest.py
    bash run_g2_python.sh scripts/export_run_manifest.py --print
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_ID = 'offline-pretrain-20260928-camera30-02'


def git_head(path):
    result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def newest(directory, pattern):
    matches = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return str(matches[-1].relative_to(ROOT)) if matches else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='runtime/train-fixed-fridge-20260928-camera-relaxed.json')
    parser.add_argument('--print', action='store_true', dest='print_only',
                        help='Print the manifest without writing any file')
    args = parser.parse_args(argv)

    source = Path(args.config)
    source = source if source.is_absolute() else ROOT / source
    raw = source.read_bytes()
    payload = json.loads(raw)

    from g2_local.code_identity import algorithm_digest, contract_digest, source_digest
    from g2_local.training_config import load_training_config
    config = load_training_config(source, cli_allow_motion=False)

    seed = ROOT / 'runtime' / f'offline-pretrain-{RUN_ID.split("offline-pretrain-")[1]}'
    manifest = {
        'exported_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'run_id': RUN_ID,
        'config_file': str(source.relative_to(ROOT)),
        'config_sha256_bytes': hashlib.sha256(raw).hexdigest(),
        'config_hash': config.config_hash,
        'contract_sha256': contract_digest(config),
        'algorithm_sha256': algorithm_digest(),
        'source_sha256': source_digest(ROOT),
        'git_head': git_head(ROOT),
        'python': platform.python_version(),
        'mode': config.mode,
        'control_hz': config.task.control_hz,
        'max_episode_steps': config.task.max_episode_steps,
        'rewards': {'success': config.task.success_reward,
                    'failure': config.task.failure_reward,
                    'step': config.task.step_reward},
        'capacity': {'online': config.optimization.online_capacity,
                     'human': config.optimization.human_capacity},
        'batch': {'online': config.optimization.online_batch_size,
                  'human': config.optimization.human_batch_size},
        'utd_ratio': config.optimization.utd_ratio,
        'actor_update_interval': config.optimization.actor_update_interval,
        'lagrange_lr': config.optimization.lagrange_lr,
        'motion_timeouts': {'command_timeout_s': config.motion.command_timeout_s,
                            'send_timeout_s': config.motion.send_timeout_s,
                            'reader_timeout_s': config.motion.reader_timeout_s,
                            'stop_timeout_s': config.motion.stop_timeout_s},
        'auto_reset': config.motion.auto_reset.enabled,
        'freshness': {'camera_age_s': config.freshness.camera_age_s,
                      'state_age_s': config.freshness.state_age_s,
                      'camera_skew_s_recorded_only': config.freshness.camera_skew_s,
                      'mapping_error_s_recorded_only': config.freshness.mapping_error_s},
        'seed_dir': str(seed.relative_to(ROOT)) if seed.is_dir() else None,
        'training_dir': newest(ROOT, 'runtime/fixed-fridge-training-*'),
        'commands': {
            'offline_pretrain': (
                'bash run_g2_python.sh -m g2_local.offline_pretrain '
                f'--config {source.relative_to(ROOT)} --runtime-root runtime '
                f'--output runtime/offline-pretrain-{RUN_ID.split("offline-pretrain-")[1]} '
                '--updates 10 --actor-bc-steps 3000 --accept-motion-profile'),
            'learner': (
                f'bash run_g2_python.sh -m g2_local.real_train learner --run-id {RUN_ID} '
                f'--config {source.relative_to(ROOT)} '
                f'--checkpoint runtime/offline-pretrain-{RUN_ID.split("offline-pretrain-")[1]}/checkpoint.pt '
                f'--checkpoint-dir runtime/fixed-fridge-training-{RUN_ID.split("offline-pretrain-")[1]} '
                '--output "runtime/learner-live-$(date +%Y%m%d-%H%M%S)"'),
            'actor_loop': ('bash run_g2_python.sh scripts/start_training_actor.py '
                           '--allow-motion --loop 20'),
        },
    }
    if seed.is_dir():
        result = seed / 'result.json'
        manifest['seed_result'] = (json.loads(result.read_text())
                                   if result.is_file() and result.stat().st_size else None)
        datasets = seed / 'datasets.json'
        if datasets.is_file():
            payload = json.loads(datasets.read_text())
            manifest['demo_sets'] = {'episodes': payload.get('episodes'),
                                     'steps': payload.get('steps'),
                                     'paths': payload.get('paths', [])}

    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    if args.print_only:
        print(text)
        return 0
    snapshot = ROOT / 'configs' / 'runtime-live.json'
    snapshot.write_bytes(raw)
    target = ROOT / 'runtime' / f'experiment-manifest-{datetime.now():%Y%m%d-%H%M%S}.json'
    target.write_text(text + '\n')
    print(f'已写入 {snapshot.relative_to(ROOT)}（逐字节副本，建议提交到仓库）')
    print(f'已写入 {target.relative_to(ROOT)}')
    print(f"config_hash={manifest['config_hash']}")
    print(f"source_sha256={manifest['source_sha256']}")
    print(f"control_hz={manifest['control_hz']} steps={manifest['max_episode_steps']} "
          f"rewards={manifest['rewards']} auto_reset={manifest['auto_reset']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
