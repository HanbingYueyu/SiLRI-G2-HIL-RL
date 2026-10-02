# configs/

仓库里的 JSON **不是**设备上正在用的训练配置——实况配置在 `runtime/` 下，而整个 `runtime/` 被 `.gitignore` 忽略（第 22 行）。这曾导致外部审查无法核实"这次训练到底用了哪些参数"。约定如下：

| 文件 | 状态 |
|---|---|
| `runtime/train-fixed-fridge-20260928-camera-relaxed.json` | **唯一实况配置**（gitignored，只在本机） |
| `configs/runtime-live.json` | 上面那份的**逐字节快照**，用来让仓库可审计；每次改配置后重新导出并提交 |
| `configs/g2_site_candidate.json` | **历史候选**：10 Hz / 200 步 / `human_capacity=1024` / `requested_motion=false`，**不是**当前训练配置；`tests/test_g2_review_20260926.py` 仍在引用它 |
| `configs/g2_real_training_readonly.json` | 只读模板（10 Hz、`requested_motion=false`），供 CLI/生命周期测试使用 |
| `configs/g2_freshness_limits.schema.json` | freshness 限制的 schema |

导出身份（配置 + 两份 SHA + 命令 + seed/训练目录 + BC 结果）：

```bash
bash run_g2_python.sh scripts/export_run_manifest.py
```

每轮实验应提交：`configs/runtime-live.json`、`runtime/experiment-manifest-*.json`，以及 seed 目录里的 `datasets.json` / `events.jsonl` / `result.json`（若需要审计）。

**身份分两层**（`g2_local/code_identity.py`）：`contract_sha256` 强制（示教契约键 + `g2_local/contract.py`/`config.py`/`demonstrations.py` 的字节 + 策略配置，剔除 `device`/`storage_device`/`use_amp` 这三个部署相关键，因此与本机是否插着 GPU 无关），变了才需要重建 seed；`source_sha256` 只记录全部算法源码、git HEAD 和依赖版本，用于审计，**不同也照常加载**（只在 stderr 提示）。所以"修运行时代码"不再作废 seed/checkpoint，`config_hash`（整份配置）仍然强制。

seed 相关的两个数字**不在配置里**，只在命令行和 `result.json` / `events.jsonl` 里，审计时必须一起看：

- `--actor-bc-steps 3000`：Actor 行为克隆预热步数（程序内默认 1000）。它**不改变 `config_hash`、不影响示教有效性**，但直接决定"初始策略有多好"。seed 的 `result.json` 会记 `actor_bc_pretrain_steps` 与 `actor_bc_last_loss`。
- `--updates 10`：seed 里先跑 10 次优化更新，再存 checkpoint（`update_count=10`、策略版本 v10）。

重建 seed 前先跑 GPU 免检（几秒钟，不需要 GPU，不写任何 seed 产物）：

```bash
bash run_g2_python.sh scripts/preflight_pretrain.py \
  --config runtime/train-fixed-fridge-20260928-camera-relaxed.json \
  --runtime-root runtime \
  --output runtime/offline-pretrain-20260928-camera30-02 \
  --actor-bc-steps 3000
```

建完后用 `scripts/checkpoint_info.py` 核对身份（`contract_sha256` / `config_hash` / `run_id` 三项一致才可续训；`source_sha256` 只作审计）：

```bash
bash run_g2_python.sh scripts/checkpoint_info.py \
  runtime/offline-pretrain-20260928-camera30-02/checkpoint.pt \
  --expect-run-id offline-pretrain-20260928-camera30-02
```

已知的两处"配置了但不生效"的字段（详见 `g2_local/freshness.py` 的 `FreshnessLimits` 注释）：`camera_skew_s` 只写入诊断、`mapping_error_s` 在本地新鲜度方案里已无对应实现；它们仍属于 commissioning 证据，改动它们只会改变 `config_hash`，不会改变拒绝行为。
