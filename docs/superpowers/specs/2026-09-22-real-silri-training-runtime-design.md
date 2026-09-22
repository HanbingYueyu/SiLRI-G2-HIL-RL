# G2 铰链插入正式 SiLRI 真机训练运行时设计

日期：2026-09-22

## 目标

在现有 G2、双 RGB、RTX 3090、持续 PTP/freshness 门控、MotionBackend 和
SiLRI Actor–Learner 软件闭环之上，建立一条正式、可调参、可恢复、可审计的
真机训练主链：

```text
上游视觉复位与 EpisodeContext
→ GDK 双相机/末端状态 + PTP freshness
→ SiLRI Actor
→ SpaceMouse 动作即接管
→ MotionBackend/CommandStream
→ 已确认执行动作与后继观测
→ 在线池/人工池
→ Critic、Actor、β、λ 更新
→ 权重发布、checkpoint、评估
```

任务只学习左手持铰链后的局部对准和插入。抓取、寻找孔和移动到孔附近由上游
视觉系统负责；RL 不执行固定回位轨迹，也不假设每回合具有同一初始位姿。

## 选定架构

采用本机双进程 Actor–Learner，复用现有 loopback gRPC、SiLRI policy、双
ReplayBuffer 和 checkpoint 结构：

1. **Real Actor 进程**独占 GDK reader、命令端口、SpaceMouse、键盘和 Gym
   回合状态机。策略推理使用 RTX 3090，动作只有在 freshness、输入、命令租约和
   工作空间检查全部通过后才能提交。
2. **Learner 进程**独占优化器与 replay，执行 Critic、Actor、expert/BC、β 和 λ
   更新，发布带单调版本号的 Actor 权重并原子保存 checkpoint。
3. `clock_monitor` 仍是独立只读前台服务。Actor 只能读取短租约 snapshot，不能
   启动、重配或修改 PTP，也不能通过时钟健康状态提升运动权限。

不采用单进程训练，因为 CUDA 反向传播和 checkpoint I/O 会干扰控制周期；不增加
分布式 replay/多 Actor 服务，因为当前单机器人本机训练不需要这层复杂度。

## 回合状态机

Real Actor 使用显式状态机，任何异常都只能向停止方向转移：

```text
WAITING_FOR_RESET
  └─ 上游完成视觉复位并提交 EpisodeContext
       └─ 左右键同时按下且松开形成一次新鲜 chord
            → RUNNING_POLICY

RUNNING_POLICY
  ├─ SpaceMouse 任一有效控制轴超过接管阈值 → RUNNING_HUMAN
  ├─ Y → TERMINATING_SUCCESS
  ├─ F → TERMINATING_FAILURE
  ├─ 时间上限 → TERMINATING_TIMEOUT
  └─ 任一安全/数据故障 → ABORTED

RUNNING_HUMAN
  ├─ 新鲜回中连续达到 release_hold_s → RUNNING_POLICY
  ├─ Y/F/时间上限 → 对应终止状态
  └─ HID 过期、拔出或数据异常 → ABORTED

所有终止状态
  → 停止命令流并封存回合
  → WAITING_FOR_RESET
```

`Y` 产生 `success_label=true` 和显式成功奖励；`F` 产生
`success_label=false` 和显式失败奖励。时间截断不伪装为失败标签。Y/F 后机器人
不自动回位，必须等待上游重新寻找孔、移动到新的局部起点并提交新上下文。

左右键 chord 只在 `WAITING_FOR_RESET` 有效；训练中同时按键不能重置或开始新
回合。chord 必须经历“两个按钮均新鲜按下→两个按钮均松开”，避免长按跨回合。

## SpaceMouse 动作即接管

左键继续选择输入含义：不按左键时控制 base_link XYZ；按住左键时控制
roll/pitch/yaw。右键不再是接管开关。

- 任一归一化控制分量超过 `intervention_engage_deadzone`，当前 step 起采用人工
  动作并标记 `is_intervention=true`。
- 接管后只有收到持续 `intervention_release_hold_s` 的**新鲜回中报告**才交还
  策略；短暂穿越死区不会在人工/策略之间抖动。
- HID 静默不是回中。报告过期、设备拔出、格式错误或按钮/轴时序异常时停止并
  废弃当前 step，绝不把失联解释成“不操作后交还策略”。
- policy proposal、human proposal、selected action 和驱动确认的 executed action
  分别保存；Critic 只训练 executed action。

接管死区、释放保持时间和 SpaceMouse 物理映射全部显式配置并记录进 run
manifest，不提供隐藏现场默认值。

## 上游复位与域随机化

上游视觉系统负责每回合：识别孔、移动铰链到孔的大致正上方、确认夹爪状态并
提供 `EpisodeContext`。上下文至少包含：

- 唯一 `episode_id`；
- 冰箱/目标相对基准的 XYZ 偏移，首版重点覆盖 XY 约 5 cm 候选范围；
- 实际局部起始末端 XYZ/RPY 相对参考的扰动；
- `approach_source`、抓取描述和视觉复位时间；
- 可选的视觉目标置信度及上游帧标识。

Real Actor 在 chord 前取得一份通过 freshness 的初始观测，将其与上下文绑定；
缺上下文、上下文重复、视觉复位尚未完成或初始观测不健康时不能开始回合。
训练运行时只记录并消费这些扰动，不自行移动冰箱或执行复位轨迹。

## 每步数据流

每个控制 step 的固定顺序为：

1. 接收最新单调 Actor 权重；权重加载只发生在安全边界，不中断正在发送的命令。
2. 读取并验证前观测：双相机、末端状态、TF、PTP 映射和 freshness。
3. Actor 生成归一化 6DoF `policy_action`。
4. 输入仲裁器决定是否使用新鲜 `human_action`；两者均保留。
5. MotionBackend 按显式动作尺度和 base_link 工作空间生成目标，提交给带租约的
   CommandStream，并等待首次成功发送回执。
6. 读取严格晚于发送回执的后继观测；任何不确定性都停止并丢弃当前 transition。
7. 结合 Y/F、时间截断和 outcome adapter 生成 reward、terminal、success label。
8. 构造 transition，记录策略版本、回合/步 ID、上下文、动作 provenance、门控
   摘要和接管累计状态。
9. transition 进入在线池；`is_intervention=true` 的样本同时进入人工池，并按
   有界批次发送给 Learner。

Actor 不把“命令已发送”解释为“目标已到达”；replay 中的 executed action 是
驱动确认接受且经过裁剪的候选动作，不伪装成实测位移。

## Learner、Replay 与权重发布

Learner 复用当前 SiLRI 更新顺序和两池采样，正式运行补齐以下生命周期：

- 在线池和人工池容量、人工样本比例、batch size、UTD、warm-up transition 数量、
  Actor/Critic/β/λ 学习率和 target update 参数均来自显式配置。
- 未达到最小 replay 或人工池为空时不执行依赖人工 mask 的更新，不推进对应优化器
  moments。
- 每个 transition 带唯一 run/episode/step ID；重复包按 ID 拒绝，断线重连不得
  重复训练同一批数据。
- Actor 权重带严格递增版本；Actor 只接受配置和 run identity 匹配的新版本。
- checkpoint 原子保存 policy、target、所有 optimizer、版本、两个 replay 的恢复
  索引、完整 provenance、随机数状态和不可变 run manifest。
- resume 必须验证 schema、配置哈希、相机/action 契约和可信本机 checkpoint；
  不恢复物理场景，恢复后仍从 `WAITING_FOR_RESET` 开始。

## 可调参数与运行清单

使用一份版本化训练配置，启动时解析、校验并复制为不可变 `run_manifest.json`。
至少包含：

- 任务：控制频率、最大步数、成功/失败/step reward、目标与末端扰动范围；
- 动作：6DoF 单步尺度、base_link 工作空间、控制模式、命令/send/stop 超时；
- 观测：两路相机 key、图像尺寸/ROI、六项 freshness 阈值及批准文件；
- 接管：axis map、按钮索引、engage deadzone、release deadzone、release hold、报告
  新鲜度；
- 训练：所有学习率、batch、UTD、replay 容量/比例、更新/发布/checkpoint 周期；
- 运行：seed、设备、端口、输出目录、checkpoint/resume、训练或评估模式；
- 安全：`allow_motion`、现场验收 profile 和对应证据身份。

配置加载不能自行把 `allow_motion` 从 false 提升为 true。真正运动必须同时满足：
显式 CLI 运动许可、现场验收 profile 完整、freshness 批准有效、GDK 状态/模式正确、
硬件急停可用。任一项缺失则可运行只读预检，但不能创建运动命令端口。

## 停止、故障与恢复

- Y/F、时间上限、Ctrl+C、Actor/Learner 断线、SpaceMouse 故障、PTP/freshness
  拒绝、GDK 读写异常和证据写入失败都会先停止当前命令流，再结束/废弃回合。
- stop 未确认时明确报告并禁止自动重建；硬件急停仍是独立最终边界。
- 当前 MotionBackend 实例发生任何失败后永久锁止。恢复必须创建新实例、重新读取
  健康初始观测、等待上游复位并重新执行 chord。
- Learner 退出时 Actor 停止而不是使用旧策略无限运行；Actor 退出不破坏已经原子
  保存的 checkpoint。
- 不自动重启 PTP、GDK、命令流或训练回合；每次恢复都有新的 run/session 证据。

## 训练、评估与日志

同一 Real Actor 入口支持两种互斥模式：

- `train`：上传 transition、接收权重并允许人工接管；
- `eval`：冻结指定 checkpoint，不更新 Learner，仍允许安全人工接管，但接管回合
  不计入无接管成功率。

每次 run 输出：不可变 manifest、JSONL 事件、每回合摘要、训练指标、策略版本、
checkpoint 和异常终止原因。指标至少区分成功率、无接管成功率、接管比例、回合
长度、动作裁剪率、freshness 拒绝、停止原因、Actor 推理/控制周期和 Learner 更新
耗时。日志不保存未显式启用的原始 RGB，以控制容量并保护现场数据。

## 实施与验收边界

实现顺序为：状态机与输入事件 → 配置/manifest → Real Actor 数据链 → Learner
可靠传输与恢复 → 训练/评估入口 → 安全装配 → 现场分级验收。

软件验收必须证明真实入口不再使用 `SyntheticBackend`，Actor/Critic/β/λ 的更新
来自带真实契约的 transition，并覆盖断线、重复包、错误权重、异常停止和 resume。
这些测试使用隔离后端，不冒充真机性能。

现场按以下门槛逐级开放：

1. PTP/双相机/真实 Actor 三轮只读审计及 freshness 批准；
2. 空夹爪、极小尺度的 XYZ/roll/pitch/yaw 方向、工作空间和控制模式验收；
3. 软件 stop、租约过期和硬件急停的停止时间/距离验收；
4. 上游复位、chord、Y/F、SpaceMouse 接管/回交的低速单回合验收；
5. 小规模数据采集、checkpoint/resume 和固定 checkpoint eval；
6. 最后才开放持续真机训练和约 5 cm 目标扰动。

完成软件运行时不等于自动获得运动许可。未通过对应现场门槛时，程序必须在最早
边界失败关闭，并清楚报告缺少的证据或参数。
