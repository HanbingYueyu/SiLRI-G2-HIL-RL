# 固定冰箱真机训练修复与启动交接（2026-09-28）

## 已完成的软件修复

- `MultivariateNormalDiag` 现在把 `scale_diag` 作为标准差传入 `scale_tril`。`fixed_std=0.05` 和 β 网络输出的 std 因而与采样分布、NLL 及约束使用的尺度一致。
- 真机训练策略频率已从 10Hz 调到 30Hz，让每个控制周期重新读取相机画面；动作发送线程仍为 50Hz。最大回合步数从 300 调到 900，保持原来的约 30 秒上限。
- 双相机时间差现在只记录为诊断，不再等待两路帧追齐，也不因两路时间差拒绝观测；这不会对齐图像。单帧年龄、时间戳倒退、PTP 映射、关节/TF 年龄和动作后观测检查仍生效。最新帧仍在年龄门限内时允许复用；动作后的相机帧若早于命令，会在 200ms 上限内重读，仍无新帧则作为可恢复的相机故障处理。
- 新相机配置 `runtime/train-fixed-fridge-20260928-camera-relaxed.json` 将单帧最大年龄从 100ms 调到 250ms；配置中的 `camera_skew_s=200ms` 仅用于审计/诊断，不再作为实时拒绝条件。关节/TF 状态年龄 50ms、PTP 映射误差 5ms、末端位置/姿态误差和动作租约保持原值。旧审计的相机年龄 P99 为 73.3ms、帧差 P99 为 38.3ms；近期失败日志观测到 198.2ms 年龄、133.6ms 帧差，帧差不再单独阻止运行。
- 左右手彩色流的相机配置目标已是 30 FPS（约每 33ms 一帧）。只读实测 3 秒：左右相机各收到 89 个不同时间戳，均为 30.00 FPS，最大帧间隔 33.6ms；SiLRI Actor 的端到端策略频率仍待真机运行验证。
- 活动回合的相机观测仍不合格时，后端先执行并确认测量保持。Actor 随后截断当前回合、丢弃没有有效后继观测的在途动作、上传此前已确认的有效转移，并保持进程运行等待现场复位和新的 `EpisodeContext`。不会自动重新发动作。测量保持未确认或 PTP/其他新鲜度检查失败时仍按原逻辑失败关闭。

## 30Hz 配置下需要重新采集示教并生成训练种子

旧种子目录 `runtime/offline-pretrain-20260928-camera-relaxed-01/` 保留，但它对应 10Hz/300 步配置，不能用于当前 30Hz 配置。示教数据的 `control_hz` 属于契约，原 10Hz 示教不能直接导入 30Hz 实验。

> **2026-09-29 修正：** 示教契约已改为只覆盖数据语义字段（`observation`、`intervention`、`action_scale`、`control_hz`、`fix_gripper`、目标/末端偏移范围、`reward_source` 以及 `motion` 的工作空间与 `control_mode`）。**回合步数上限和三个奖励值已移出契约**，导入时按当前配置从 `success_label` 重新推导，因此调整 reward 或 `max_episode_steps` 不再使已采示教失效。当前配置的 `max_episode_steps` 为 **350**。改动 `task`/`observation`/`intervention`/`motion` 工作空间仍会使旧示教被拒收。

下一步须用当前 30Hz 配置采集完整示教，再以这些数据创建新的 offline seed；新在线训练使用新的 run ID 和 checkpoint 目录。旧 10Hz seed、在线权重和 Replay 均保留，不得混入新实验。

在 clock monitor 健康、当前场景 context 已准备好且现场确认后，用下面命令开始 30Hz 人工示教；每次场景复位都提交新的 context。需达到配置要求的完整成功示教数量后，才运行 offline pretrain：

```bash
cd /home/flyfuture/桌面/hil-rRL/SiLRI-HIL-RL
bash run_g2_python.sh -m g2_local.real_train demo \
  --run-id demo-camera30-01 \
  --config runtime/train-fixed-fridge-20260928-camera-relaxed.json \
  --output runtime/demo-camera30-01 \
  --context /绝对路径/当前episode-context.json \
  --hid-device /dev/spacemouse-compact \
  --clock-socket /本轮clock-monitor目录/clock.sock \
  --allow-motion
```

新示教准备好后，生成与 30Hz 配置哈希匹配的 seed：

```bash
bash run_g2_python.sh -m g2_local.offline_pretrain \
  --config runtime/train-fixed-fridge-20260928-camera-relaxed.json \
  --runtime-root runtime \
  --output runtime/offline-pretrain-20260928-camera30-01 \
  --updates 10 --accept-motion-profile
```

## 启动前必须处理的时钟条件

Actor 配置内的旧 socket `/tmp/g2-demo-clock-20260926-03/clock.sock` 当前不存在；`runtime/shared-demo-clock/current.json` 指向的旧 socket 也不存在。检查时未发现 `clock_monitor`，但主机仍有独立的 `ptp4l` 与 `phc2sys` 进程（当时 PID 为 43445、43455）。因此本次已准备好代码和种子，**但此刻不能直接启动运动 Actor**。

先由现场负责人确认这两个 PTP 进程的所属服务和用途。不要在同一接口并行启动第二个 PTP 测量进程，也不要盲目结束现有系统进程。按 `docs/g2-clock-diagnostics.md` 的流程处理既有服务后，再在终端 A 启动唯一的健康监控：

```bash
cd /home/flyfuture/桌面/hil-rRL/SiLRI-HIL-RL
bash run_g2_python.sh -m g2_local.clock_monitor \
  --master 044052.fffe.000010 --max-seconds 43200
```

该进程会要求当前终端完成 `sudo -v`，创建全新监控目录并打印本轮 `clock.sock`。只有快照持续健康、主时钟身份正确、映射完成预热后才继续；将它打印的**本轮 socket 绝对路径**用于下面 Actor 命令。监控不健康、PTS 断流或 socket 不存在时停止，不运行 Actor。

## Learner 启动命令（新 seed 生成后）

30Hz 示教和对应 offline seed 尚未生成，因此当前不能运行此命令。新 seed 完成后，在终端 B 启动 Learner；确认其 `events.jsonl` 出现 `ready` 后再启动 Actor：

```bash
cd /home/flyfuture/桌面/hil-rRL/SiLRI-HIL-RL
bash run_g2_python.sh -m g2_local.real_train learner \
  --run-id offline-pretrain-20260928-camera30-01 \
  --config runtime/train-fixed-fridge-20260928-camera-relaxed.json \
  --checkpoint runtime/offline-pretrain-20260928-camera30-01/checkpoint.pt \
  --checkpoint-dir runtime/fixed-fridge-training-20260928-camera30-01 \
  --output runtime/learner-live-20260928-camera30-01
```

## Actor 启动命令

确认时钟健康、现场急停与隔离区按现场流程就绪、Learner 已 `ready` 后，在已加载 GDK 环境的终端 C 执行。把 `<本轮 clock_monitor 打印的路径>` 替换成实际 socket 所在目录；这是必要项，因为 JSON 里的默认路径已过期。

```bash
cd /home/flyfuture/桌面/hil-rRL/SiLRI-HIL-RL
source /home/flyfuture/.cache/agibot/app/env.sh /home/flyfuture/.cache/agibot/app
bash run_g2_python.sh -m g2_local.real_train actor \
  --run-id offline-pretrain-20260928-camera30-01 \
  --config runtime/train-fixed-fridge-20260928-camera-relaxed.json \
  --output runtime/actor-live-20260928-camera30-01 \
  --context /绝对路径/当前episode-context.json \
  --hid-device /dev/spacemouse-compact \
  --clock-socket /tmp/g2-clock-monitor-本轮目录/clock.sock \
  --allow-motion
```

以上 Actor 命令仅是 30Hz 配置的模板；在新 30Hz 示教、offline seed 和 Learner 就绪前不要运行。

相机故障后 Actor 会停在等待新上下文状态；现场先复位场景，再写入有新 episode ID 和新时间戳的 context。Learner 可以保持运行。Actor 不会把未确认的在途动作写入 Replay。

## 验证边界

本轮相关回归 **454 passed**。30Hz 配置已通过加载校验，步长为约 33.3ms；已有 10Hz 示教契约已实测确认不能导入新配置。30Hz offline seed 和完整训练运行尚未生成。此前 684 项回归及 10Hz seed 恢复验证对应的是旧配置，不代表本次 30Hz 示教/真机运行验证。

未启动 Learner 服务、时钟监控或真机 Actor，未发送机器人运动命令，也未做现场相机采集验证。PTP 服务冲突与现场硬件状态仍须现场检查。GDK SDK 及 `/home/flyfuture/g2_hinge_assembly` 均未修改。
