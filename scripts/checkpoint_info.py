"""Print a checkpoint's identity and progress counters without loading it fully.

Read-only, mmap-backed: safe to run while a Learner is running and cheap for a
4 GB checkpoint. Use it to confirm "which update/episode am I resuming from" and
whether the checkpoint still matches the current source/config identity.

    bash run_g2_python.sh scripts/checkpoint_info.py runtime/.../checkpoint.pt

The last three lines are the verdict that matters before a long run: a
checkpoint whose ``source_sha256`` or ``config_hash`` differs from the values
printed here will be refused by the Learner with
``Algorithm code/version identity mismatch`` / ``parameter run/config identity
mismatch``. Rebuild the seed (Section 2) instead of working around it.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def current_identity(config_path):
    """The two enforced digests plus the audit-only source digest."""
    from g2_local.code_identity import algorithm_digest, contract_digest, source_digest
    from g2_local.training_config import load_training_config
    config = load_training_config(config_path, cli_allow_motion=False)
    return (contract_digest(config), algorithm_digest(), config.config_hash,
            source_digest(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoints', nargs='+', type=Path)
    parser.add_argument('--config',
                        default='runtime/train-fixed-fridge-20260928-camera-relaxed.json',
                        help='Config whose identity the checkpoint must match')
    parser.add_argument('--expect-run-id', default=None,
                        help='Refuse (non-zero exit) when the checkpoint run_id differs')
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(2)
    failures = []
    for path in args.checkpoints:
        target = path if path.is_absolute() else ROOT / path
        print(f'=== {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}')
        if not target.is_file():
            print('    不存在')
            failures.append(f'{target}: missing')
            continue
        try:
            payload = torch.load(target, map_location='cpu', weights_only=False, mmap=True)
        except BaseException as error:
            print(f'    无法读取：{type(error).__name__}: {error}')
            failures.append(f'{target}: unreadable')
            continue
        run_id = payload.get('run_id')
        print(f'    run_id            = {run_id}')
        print(f'    config_hash       = {payload.get("config_hash")}')
        print(f'    update_count      = {payload.get("update_count")}')
        print(f'    policy version    = {payload.get("version")}')
        print(f'    completed episodes= {len(payload.get("completed_episode_ids") or [])}')
        print(f'    accepted          = {payload.get("accepted_transitions")}')
        print(f'    human transitions = {payload.get("human_transitions_total")}')
        print(f'    imported demos    = {len(payload.get("imported_demo_episodes") or [])}')
        print(f'    beta pretrain     = {payload.get("beta_pretrain_completed")} '
              f'(loss {payload.get("beta_last_loss")})')
        print(f'    actor BC warm-up  = {payload.get("actor_bc_pretrain_steps", 0)} '
              f'(loss {payload.get("actor_bc_last_loss")})')
        optimization = payload.get('optimization') or {}
        print(f'    bc_weight         = {optimization.get("bc_weight")} '
              f'(decay {optimization.get("bc_weight_decay_updates")})')
        identity = payload.get('algorithm_identity')
        identity = identity if isinstance(identity, dict) else {}
        print(f'    contract_sha256   = {identity.get("contract_sha256")}')
        print(f'    algorithm_sha256  = {identity.get("algorithm_sha256")}')
        print(f'    source_sha256     = {identity.get("source_sha256")}  (审计用，不作废 checkpoint)')
    try:
        contract, algorithm, config_hash, source = current_identity(args.config)
    except BaseException as error:
        print(f'\n无法读取当前身份（--config {args.config}）：{type(error).__name__}: {error}')
        return failures and 1 or 0

    print(f'\n当前契约摘要 contract_sha256  = {contract}')
    print(f'当前算法摘要 algorithm_sha256 = {algorithm}')
    print(f'当前配置摘要 config_hash     = {config_hash}')
    print(f'当前源码摘要 source_sha256   = {source}  (审计用)')
    for path in args.checkpoints:
        target = path if path.is_absolute() else ROOT / path
        if not target.is_file():
            continue
        try:
            payload = torch.load(target, map_location='cpu', weights_only=False, mmap=True)
        except BaseException:
            continue
        identity = payload.get('algorithm_identity')
        identity = identity if isinstance(identity, dict) else {}
        stored_contract = identity.get('contract_sha256')
        stored_source = identity.get('source_sha256')
        name = target.relative_to(ROOT) if target.is_relative_to(ROOT) else target
        if stored_contract is None:
            print(f'✗ {name}：checkpoint 里没有 contract_sha256（本次改动之前建的）'
                  '→ 重建 seed 一次即可')
            failures.append(str(name))
        elif stored_contract != contract:
            print(f'✗ {name}：契约（任务/动作/观测 + 训练信号 + 策略配置）变了 → 迁移/重建 seed')
            failures.append(str(name))
        elif identity.get('algorithm_sha256') != algorithm:
            print(f'✗ {name}：训练语义实现变了（loss/target/优化器步）→ 迁移/重建 seed')
            failures.append(str(name))
        elif args.expect_run_id and payload.get('run_id') != args.expect_run_id:
            print(f'✗ {name}：run_id={payload.get("run_id")}，期望 {args.expect_run_id}')
            failures.append(str(name))
        else:
            notes = []
            if payload.get('config_hash') != config_hash:
                notes.append('config_hash 不同')
            if stored_source != source:
                notes.append('源码摘要不同')
            note = '' if not notes else '；' + '、'.join(notes) + '（仅审计，不影响续训）'
            print(f'✓ {name}：契约与 run_id 一致，可直接续训{note}')
    if failures:
        print(f'\n{len(failures)} 个 checkpoint 不能直接使用')
        return 1
    print('\n全部 checkpoint 身份一致')
    return 0



if __name__ == '__main__':
    raise SystemExit(main())
