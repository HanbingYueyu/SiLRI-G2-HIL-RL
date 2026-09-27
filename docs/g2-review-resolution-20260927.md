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
