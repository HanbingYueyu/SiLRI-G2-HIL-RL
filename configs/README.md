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

**实测的裁剪与频率**（2026-10-02 真机 1336 步）：工作空间裁剪发生 **20 步（1.5%）**，最大裁剪 **1.21 mm**（一步 XYZ 满幅才 1.5 mm，所以触发时幅度不小）；`control_period_s` 中位 **0.102 s**，即**实测约 10 Hz，不是配置里的 30 Hz**。奖励/折扣的步数分析按"步"计，与频率无关；但"每回合约 11.7 秒"是按 30 Hz 算的，实测约 35 秒。`min_online_transitions=256` 是**回放池最小容量**（示教已经把它填满，所以第一条真机转移就能触发更新），不是"先攒 256 条真机数据"；日志里 `真机新增 N` 才是本次 run 新收集的真机转移数。

**内存（14 GB 机器的硬约束）**：回放里的相机帧现在按**采集精度 uint8** 存储（2026-10-02 改）。改动前每步的 `state`+`next_state` 图像是 float32，768 KB/步 → human 5632 步 4.22 GB + online 1024 步 0.77 GB ≈ 5.0 GB，加上存盘时的序列化峰值，Learner 常驻到 **11.6 GB** 并被内核 OOM 杀掉（`/var/log/syslog`: `Out of memory: Killed process … anon-rss:11560296kB`）。uint8 后每步 192 KB → 回放 ≈1.25 GB，checkpoint 从 4.9 GB 降到约 1.3 GB，常驻约 3–4 GB、存盘峰值约 5 GB。旧 checkpoint（float32 回放）在加载时会自动转换成 uint8，进度不丢。注意：`ReplayBuffer` 的存储 dtype 由**第一条** transition 决定，所以任何新的插入路径都必须先过 `_training_row`。

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
