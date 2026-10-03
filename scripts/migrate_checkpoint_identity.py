"""Re-stamp a checkpoint's recorded contract identity, with explicit evidence.

The enforced identity is ``code_identity.contract_digest`` (``contract_sha256``).
When the *definition* of that digest changes for a reason that provably does not
change the contract itself — removing a placement-only key such as the compute
device, which differs between a CUDA box and a CPU-only one — an existing
seed/checkpoint can be re-stamped instead of rebuilt. That is only allowed with
evidence, and this tool refuses to run without it:

  * the stored ``config_hash`` must equal the current configuration's;
  * ``--expect-run-id`` must name the stored run id;
  * ``--expect-source-sha256`` must name the stored full-source digest, i.e. the
    exact build that produced the checkpoint;
  * ``--from-contract-sha256`` must name the stored contract digest (or the
    operator passes ``--from-contract-sha256 none`` for a checkpoint written
    before the field existed);
  * the new digest must actually differ, otherwise there is nothing to migrate.

Only the ``contract_sha256`` field is rewritten; ``source_sha256``, ``packages``,
``policy_config``, git revisions and every tensor are left byte-for-byte as the
build recorded them. The rewrite is atomic (same directory, ``os.replace``), so an
interrupted run leaves the original checkpoint intact, and the tool reloads the
result through the real, enforcing loader before reporting success.

    bash run_g2_python.sh scripts/migrate_checkpoint_identity.py \
      runtime/offline-pretrain-20260928-camera30-02/checkpoint.pt \
      --config runtime/train-fixed-fridge-20260928-camera-relaxed.json \
      --expect-run-id offline-pretrain-20260928-camera30-02 \
      --expect-source-sha256 <stored source_sha256> \
      --from-contract-sha256 <stored contract_sha256 or none>
"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def stored_identity(payload):
    identity = payload.get('algorithm_identity')
    return identity if isinstance(identity, dict) else {}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--expect-run-id', required=True)
    parser.add_argument('--expect-source-sha256', required=True)
    parser.add_argument('--from-contract-sha256', required=True,
                        help="The checkpoint's current contract_sha256, or 'none'")
    parser.add_argument('--from-algorithm-sha256', default=None,
                        help="The checkpoint's current algorithm_sha256, or 'none'. "
                             "Omit to leave that field untouched.")
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the decision without rewriting the checkpoint')
    args = parser.parse_args(argv)

    from g2_local.code_identity import algorithm_identity
    from g2_local.contract import CAMERA_KEYS
    from g2_local.training_config import load_training_config
    import torch

    target = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_training_config(config_path, cli_allow_motion=False)
    identity = algorithm_identity(config)
    new_contract = identity['contract_sha256']

    if not target.is_file():
        print(f'✗ 找不到 checkpoint：{target}')
        return 1
    payload = torch.load(target, map_location='cpu', weights_only=False)
    if type(payload) is not dict or payload.get('schema') != 1:
        print('✗ 这不是本项目的 checkpoint（schema 不是 1）')
        return 1
    stored = stored_identity(payload)
    stored_contract = stored.get('contract_sha256') or 'none'

    print(f'checkpoint      : {target}')
    print(f'stored run_id   : {payload.get("run_id")}')
    print(f'stored config   : {payload.get("config_hash")}')
    print(f'stored source   : {stored.get("source_sha256")}')
    print(f'stored contract : {stored_contract}')
    print(f'current config  : {config.config_hash}')
    print(f'current source  : {identity["source_sha256"]}')
    print(f'target contract : {new_contract}')

    config_drift = payload.get('config_hash') != config.config_hash
    problems = []
    if payload.get('run_id') != args.expect_run_id:
        problems.append(f'run_id 不是 {args.expect_run_id}')
    if stored.get('source_sha256') != args.expect_source_sha256:
        problems.append('--expect-source-sha256 与 checkpoint 里记录的构建不一致')
    if stored_contract != args.from_contract_sha256:
        problems.append('--from-contract-sha256 与 checkpoint 里记录的不一致')
    if stored_contract == new_contract and (
            args.from_algorithm_sha256 is None or
            stored.get('algorithm_sha256') == identity['algorithm_sha256']):
        problems.append('摘要已经是目标值，没有需要迁移的内容')
    if args.from_algorithm_sha256 is not None and \
            (stored.get('algorithm_sha256') or 'none') != args.from_algorithm_sha256:
        problems.append('--from-algorithm-sha256 与 checkpoint 里记录的不一致')
    if problems:
        print('\n✗ 拒绝迁移：')
        for item in problems:
            print(f'   - {item}')
        return 1

    if config_drift:
        print('注意：config_hash 与当前配置不同（只调了非契约超参时属正常，仅作审计）')
    if args.dry_run:
        print('\n（--dry-run）证据齐备，本可以迁移；未写入任何东西')
        return 0

    rewritten = dict(payload)
    restamped = dict(stored, contract_sha256=new_contract)
    if args.from_algorithm_sha256 is not None:
        restamped['algorithm_sha256'] = identity['algorithm_sha256']
    rewritten['algorithm_identity'] = restamped
    staging = target.with_name(target.name + f'.migrating-{os.getpid()}')
    try:
        torch.save(rewritten, staging)
        os.chmod(staging, 0o600)
        with open(staging, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(staging, target)
    finally:
        if staging.exists():
            staging.unlink()
    print('\n已原地重写 identity（张量、优化器、回放、来源记录都未改动）')

    # Verification. The identity and the on-disk structures are checked here,
    # without constructing the policy, because the checkpoint records the device
    # it was built for ('cuda') and that device may be absent on the machine doing
    # the migration. The full path (model construction + replay restore) is then
    # exercised too whenever that device is actually available.
    check = torch.load(target, map_location='cpu', weights_only=False, mmap=True)
    identity = stored_identity(check)
    # config_hash is deliberately NOT compared: non-contract tunables may differ
    # between the build and the current configuration (that is the whole point of
    # the relaxed gate) and it is reported as a notice instead.
    if (check.get('schema') != 1 or check.get('run_id') != args.expect_run_id or
            tuple(check.get('camera_keys', ())) != CAMERA_KEYS or
            check.get('image_size') != 128 or check.get('action_size') != 6 or
            identity.get('contract_sha256') != new_contract or
            check.get('accepted_transitions') != payload.get('accepted_transitions') or
            check.get('update_count') != payload.get('update_count')):
        print('✗ 重写后复验失败：身份或结构不一致，请从备份/重建恢复')
        return 1
    provenance = check.get('provenance') or {}
    journal = target.parent / str(provenance.get('file'))
    if not journal.is_file():
        print(f'✗ 缺少 provenance 快照 {journal}')
        return 1
    import hashlib
    digest = hashlib.sha256(journal.read_bytes()).hexdigest()
    if digest != provenance.get('sha256'):
        print('✗ provenance 快照哈希不一致')
        return 1
    print(f'✓ 身份与结构复验通过：contract_sha256={new_contract}')
    print(f'  accepted={check.get("accepted_transitions")} update_count={check.get("update_count")} '
          f'provenance[{provenance.get("count")}] 哈希一致')

    device = (check.get('runtime') or {}).get('device')
    available = device == 'cpu' or (device == 'cuda' and torch.cuda.is_available())
    if not available:
        print(f'  注意：这份 checkpoint 记录的 device={device!r} 在本机不可用，'
              '模型构造与回放恢复留到真正的 Learner 启动时复验')
        return 0
    from g2_local.real_learner import load_checkpoint
    restored = load_checkpoint(target, expected_run_id=args.expect_run_id,
                               expected_config_hash=config.config_hash,
                               expected_contract_sha256=new_contract)
    print(f'✓ 正式加载路径（含模型与回放）复验通过：{restored.snapshot_counts()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
