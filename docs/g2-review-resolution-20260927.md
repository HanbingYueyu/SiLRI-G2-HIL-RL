# 2026-09-27 审查处理

## 已修正

- 离线示范导入不再授予在线 UTD 更新预算；预训练由显式 updates 控制。在线新转移仍按 UTD 累计，重复转移不重复计数。旧种子遗留的预算不迁移，重新预训练。
- Critic 目标动作按执行端同样的 `[-1+1e-6, 1-1e-6]` 裁剪，避免对无法执行的高斯尾部动作估值。
- checkpoint 复制 Replay 时不再持有参数心跳锁，去掉重复的 Replay 克隆；模型和 Replay 仍由更新锁保护一致性。磁盘保存仍暂停优化，不宣称完全异步；入口队列仍有容量上限。
- 更正旧报告关于绝对工作空间“未扩大”的错误表述。未擅自改变现场配置或补写批准。

## 保留的设计与待确认项

- 当前是 SiLRI 工程变体：Actor/Critic=1:2，λ 每次 Critic 更新，β 按当前门槛预训练/增量更新；不是论文逐项复现。
- batch=16 是两池各 8 条，不是保证 50% 人工。在线池也含接管，人工期望比例为 `(1+h)/2`，可能跨池重复。
- 首版离线 bootstrap 有意只收成功完整示范；通用导入可含失败，在线失败仍入 Replay。未悄悄改变 β 的示范语义。
- `HumanBinaryOutcome` 是未被正式运行入口引用的旧适配器，其失败奖励为 step_reward；正式入口使用自己的失败奖励配置。本次不改旧接口。
- 绝对 Z 上限 0.965 m（原 0.9389 m）：用户于 2026-09-27 明确确认可用；不是由局部 ±150 mm 的批准推断得出。
- 10 Hz 是配置目标，不是实测稳定吞吐。图像可观测性、接触行为及真实训练效果仍需现场闭环，不能由离线检查替代。

## 验证与清理

### 后续边界证据补充

**92504c5 复审的三项缺口已于本轮修正。** 验证不再只调用 `_check_local()`：使用实际 MotionBackend.execute/CommandStream、替身硬件验证发送前拒绝、已应答后 successor 越界、SDK 发送超时未确认、真实绝对 workspace 裁剪和 reset 绝对越界；Actor 运行失败/复位失败清空 context 后，事件仍保留 episode_id。相关检查共 84 项通过，未连接机器人。

当前启动种子为 `offline-pretrain-20260927-06`：30 条/4042 步重新导入，β500、Critic10、Actor5，预算0；保存、恢复及第11次更新通过。启动命令与脚本已同步，旧种子保留但不混用。以下 -04/-05 记录为历史验证，不是当前启动版本。

- `local_envelope_rejected` 记录局部边界；`absolute_workspace_rejected` 单独记录 reset/反馈位姿已超绝对空间。事件包含 episode_id、step_id、stage、被检查 pose、episode_reference（未锁定时为 null）、适用边界与原因。Actor 在 abort 清理前把身份附到异常，reset 失败也保留身份。
- `episode_reset` 的实际拒绝来源是绝对 workspace 校验，不再用 pose 对自身的局部校验作证明。`successor_read` 表示本步已获得发送应答后读取失败，不等于动作未发送，也不保证硬件完成动作。
- 动作映射在目标规划完成后、任何 submit 前构造。正常 step 写 `action_mapping`；失败写 `command_execution_failed`，边界事件也附 execution。字段含 selected_action、effective_action（计划的有效动作，不冒充物理实测动作）、origin_pose、裁剪距离、command_sequence 和发送应答时间。发送状态：not_submitted、submission_attempted_unconfirmed、submitted_unconfirmed、acknowledged，均指当前请求，不代表之前没有其他命令。
- 提交/应答异常不擅自断言“未发送”；保留未知状态。successor 失败记录不生成普通 RL transition。无规划结果时 action_mapping=null，不借用上一条动作。
- 几何统计保留 workspace_clipping_count/workspace_subtolerance_count，容差 1 nm；归一化动作差异容差 1e-7，微差另计。这些仅是诊断容差，不放宽边界。
- 48 项相关回归和 3 项新增检查通过。因代码身份变化，重新生成 `offline-pretrain-20260927-05`，30 条/4042 步、预算0，保存/恢复及后续更新通过；启动脚本和命令已切换。未改算法、输入/动作契约或训练参数，未启动真机。

- 针对导入、目标动作边界、保存期间心跳及恢复的相关检查：54 项通过；未运行机器人。
- 新种子 `runtime/offline-pretrain-20260927-04/`：30 条/4042 步，β500、Critic10、Actor5；恢复后第11次更新通过，预算保持0。
- 实际 Replay（在线1024、人工4042）单次保存测量：2.372 s，文件4,039,856,023字节；期间43次心跳，最大调用耗时0.0472 ms，进程峰值RSS约8.43 GiB（含加载/模型，不是保存增量）。不是p95/最坏延迟，也未测满队列持续上传；测量副本已移入回收站，正式种子保留。
- 只删除可再生缓存（移入回收站）。示范、checkpoint、故障证据、SDK/虚拟环境、论文/审查资料保留。旧 worktree 存在脏状态，未强制删除。

## 同域证明的真机核对（2026-10-01，只读探针）

`ObservationFreshnessGuard` 里那条"动作后必须出现时间戳更晚的帧"的严格证明，前提是 SDK 时钟
（`gdk.Clock.now_ns()`）与相机/关节/TF 时间戳属于**同一个时间基准**。这一点此前只是按设计推断，
没有真机数据。现在测了：

```
bash run_g2_python.sh scripts/gdk_clock_probe.py --reads 5
```

结果：`sdk_clock_ns − stamp` 五次采样分别为 182833.5 / 182827.1 / 182818.7 / 182814.2 / 182808.5 ms，
**全部远超实现里 500 ms 的 `SDK_SAME_DOMAIN_MAX_NS`**，判定 NOT same domain —— 两者相差一个约 183 秒的
常量偏移。PTP（`ptp_hard.sh`）只作用于日志/文件时间对齐与 GDK 的传感器延迟接口，改不掉这个偏移。

因此本机实际生效的是**退化后的本地接收保证**，而不是同域严格证明：

- `freshness.py:352` 的 `if same_domain:` 分支不进入，`not_after_command_sdk:*` 不会被触发；
- 实际用的是本地单调时间：`camera_age_s=0.25` / `state_age_s=0.05` 的接收年龄上限、相机时间戳
  严格前进（相同时间戳按缓存帧容忍）、以及 `not_after_command:*`（本地接收区间必须严格晚于发送时刻）；
- 每个样本在证据里记 `sdk_anchor.same_domain=false` 与 `sdk_anchor.margins_ns`，供后续审计或将来
  改成真正同域时再启用。

结论：P1-2 的严格同域证明在当前硬件上**不生效且不会造成误拒**（设计上就是"核实不了就退化并留证"）；
不需要为此调整 PTP 或放宽任何阈值。同时把 `scripts/gdk_clock_probe.py` 从失效的 `GdkBackend` API
修到现行的 `GdkReader`，`scripts/gdk_reinit_probe.py` 同样修正。

## 身份判定重构：契约摘要强制，源码摘要仅审计（2026-10-01）

原实现把 `source_digest`（`g2_local/**/*.py`、顶层 `*.py`/`*.sh`、`lerobot/src/lerobot/**/*.py`、
`rl_envs*`、`native/*.cpp`、`_gdk_safe_stop*.so` 的全部字节，外加 git HEAD 与依赖版本）作为唯一
身份，并在加载 seed/checkpoint 时**全等比较**。后果是：改一行日志、加一个注释、甚至只是换一次
git commit，都会让 4.9 GB 的 seed 和正在训练中的 checkpoint 一起失效 —— 一个诚实的 bug 修复要
付 15 分钟 GPU 重建的代价。这本身是设计缺陷。

现在拆成两层（`g2_local/code_identity.py`）：

- **`contract_sha256`（强制）**：示教契约（`task` 的 `action_scale`/`control_hz`/`ee_*_range*`/
  `fix_gripper`/`reward_source`/`target_xy_range_m`、整个 `observation`、整个 `intervention`、
  `motion` 的 `control_mode`/`local_envelope`/`workspace_low`/`workspace_high` 键）+ 定义与校验
  该契约的三个模块（`contract.py`、`config.py`、`demonstrations.py`）的字节 + 策略配置
  （`draccus.encode(create_policy_config(...))`）。只有它变了才拒绝加载。
- **`source_sha256`（审计）**：全部算法源码 + git HEAD + 依赖版本。不同时只在 stderr 打印
  `note: … written by different algorithm sources … the task/action contract and policy
  configuration are unchanged, so this checkpoint stays usable`，并写进 checkpoint 供审计。

`config_hash`（整份配置 JSON）仍然强制：存储的转移里带着当时的奖励值与回合标签，混用会静默出错；
`schema`、`run_id`、相机/动作契约与 replay 张量契约的检查一律保留。生产调用方
（`g2_local.real_train` 的 learner/eval、`g2_local.offline_pretrain`）显式传入
`expected_contract_sha256`；没有契约字段的旧 checkpoint 被拒绝而不是默默接受。

回归测试：`tests/test_g2_real_learner.py::test_contract_identity_is_enforced_while_source_drift_is_only_noticed`
（同契约不同源码 → 照常加载并打印审计提示；契约不同 → 拒绝）与
`tests/test_g2_real_actor.py::test_actor_never_steps_behind_an_unverified_input_gate`。

同一轮还修掉了 Actor 侧的一个真实缺陷：`AutomaticIntervention.__call__` 的"启动未就绪"早退分支
在回合已 `RUNNING` 时返回 `(False, None)`，而 `gate.fresh`/`verified_neutral` 都还是 `False`
（上游 `CompactReports.snapshot().ready` 要求两个轴向报告都出现过，而 `StartChord` 只看按钮），
于是第一个 step 会带着未经验证的人工输入闸门提交策略动作，随后被
`Invalid intervention or freshness gate summary` 拒绝、回合中断。现在策略分支在闸门可验证之前
只轮询 HID 并提示操作者拨一下旋帽，不提交动作也不记过渡。
