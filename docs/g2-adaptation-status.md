# G2 局部插入适配记录

## 真实 Actor 只读审计工具（2026-09-21，尚待三轮现场验收）

已增加可信本机 checkpoint 的 Actor-only 加载和 RTX 3090 同步计时；检查完整
配置、权重与 SHA-256，双腕 RGB 全帧转入 CUDA 后缩至 128×128，推理动作仅
记录形状/极值并丢弃。真实 GdkReader 保持 `allow_motion=False`；不创建 command、
motion、Gym 或 learner，不发保持、运动、夹爪或切模式指令。
checkpoint 固定为 `/home/flyfuture/桌面/hil-rRL/SiLRI-HIL-RL/runtime/software_loop/checkpoint.pt`。

合成 uint8 `(1056, 1280, 3)` 双图像的真实 CUDA smoke 已通过：schema 1、
version 3、RTX 3090、三次预热、有限 `(1, 6)` 丢弃动作和非零同步计时。
该测试没有实例化 GDK 或启动 PTP，不能证明真实双相机/GDK 下的年龄分布。
checkpoint SHA-256 为 `aa9bb8ad3b271745855d06d231db349b91f79a7750fbf59889353e1f7cf16f19`。

正式批准仍须 **3 个独立会话 × 至少 120 秒实际样本跨度**，每轮至少 1000
接受样本、零拒绝、同一 checkpoint/配置/GPU、健康且正常结束的独立 monitor。
建议每轮请求 125 秒正式采样（Actor 预热另外计算），监控上限 300 秒；
正常让 B 完成后 A Ctrl+C，提前停止时 B→A Ctrl+C，该轮中断不能 qualify。
每轮保留 audit JSONL/摘要、复制后的 monitor 原始证据和 `qualify` 生成的
`qualification.json`；三轮重算通过后，另行人工审查六项显式阈值才能 `approve`。
完整命令、全新证据路径、残留检查、收尾与批准模板见
[真实 Actor 三轮只读流程](g2-clock-diagnostics.md#真实-actorcuda-只读审计三轮现场流程)。

目前没有完成这三轮现场验收，没有批准阈值或运动；历史模拟等待数据不能补足。
审计/qualification 的许可字段固定 false；未来单独批准文件也仍然
`motion_authorized=false`，不修改生产配置、不解除其他现场门槛。

## 算法数据契约补齐（2026-09-22）

参考《SiLRI 铰链插入任务完整实现与部署指南》的 MVP 建议：固定夹爪、局部
6DoF 归一化动作、目标偏移与末端 reset 扰动分开记录，训练初期使用人工 reward，
后续再接二值 reward classifier。`EpisodeContext` 现在记录 `ee_reset_offset`；
每条 transition 同时保存 `policy_action`、`human_action`、`executed_action`、
`reward_source` 与 `success_label`。Critic 的 `action` 仍只使用已确认的实际执行动作，
不会把 policy 原始提案误当成执行结果。

这项改动已合并到 `main`（transition 契约提交 `eb9b2af`，Actor–Learner provenance
提交 `598935b`），不会创建 GDK
命令端口，也不会改变 `allow_motion=False`。训练入口仍分为：现有无运动软件闭环
（Actor/Learner/Critic/双 replay/checkpoint 已通）、只读 GDK/真实 Actor 审计、以及
尚未启用的受控真机 MotionBackend；三者不混用。

## 已完成：奖励与成功判定的可审计适配（2026-09-22，提交 `119cbc3`）

- 新增 `g2_local.outcome.OutcomeDecision` 和 `HumanBinaryOutcome`：人工/后续
  分类器只能显式返回 `None`（继续）、`True`（成功终止）或 `False`（失败终止），
  非布尔标签不会被悄悄当成成功。
- `MotionBackend` 兼容原有 `(reward, terminated)` 回调，同时接受带有
  `reward_source`、`success_label` 的结果；这些字段随 `StepResult`、Gym `info`
  和 transition 一起保留，Critic 仍只使用已确认的执行动作。
- `HingeInsertTaskConfig.success_reward/step_reward` 现在可通过该适配器进入
  执行链；失败标签使用显式 step reward 并立即终止，后续可在不改 transition
  契约的前提下替换为二值 classifier。
- Gym `succeed` 字段优先采用显式 `success_label`，只有旧回调没有标签时才按
  `done && reward > 0` 兼容推断，避免调高失败惩罚后误标成功。
- 定向回归 `39 passed`（含既有 Gym 的 2 条无限 Box warning）；未创建 GDK
  command port，未发送机器人命令，未改变 `allow_motion=False`。

## 持续时间映射与只读负载审计（2026-09-21，覆盖下方历史状态）

已实现独立前台 `clock_monitor`、只读短租约快照 IPC、GDK 四源时间戳/双向 TF
证据、有状态 freshness guard，以及 MotionBackend/Gym 的失败原因传递和永久锁止。
监控使用不校时的固定 PTP 命令，Actor 不控制 PTP；健康快照不能提升运动许可。
GdkReader 保持 `allow_motion=False`，运动后端尚未接入正式真机 Actor。

新增 `freshness_audit`：显式 30–1800 秒，只读采集双相机/GDK 和持续映射，
支持有限非负的模拟推理等待；保存有界原始 JSONL 和带明确单位的
min/p50/p95/p99/max 分布。断线、快照无效、源冻结/倒退或证据结构异常均失败，
已有输出在 reader/client 创建前拒绝。摘要固定 `motion_authorized=false`、
`thresholds_approved=false`；没有生产默认阈值，没有写入配置或启用运动建议。
审计不创建命令端口/运动后端，也不发送保持或其他命令。

不健康 snapshot 的机器可读原因现在由 `SnapshotClient` 保留到审计错误和摘要，
例如 PTP 停止续报时直接显示 `lease_expired`，不再丢失为笼统错误。原因码经过
长度和字符集限制；该改动只缩短现场诊断路径，不改变租约、阈值或失败关闭行为。

计划 Task 7 Step 6 已执行一次现场采集：监控最大期限 120 秒，审计 60 秒、
模拟推理等待 0.02 秒。审计证据位于
`runtime/freshness_audit/7bf7d98a6b7345f892573635aad109ce/`，摘要为
`status=completed`、717 个样本，无 `rejected` 行；`motion_authorized=false`、
`thresholds_approved=false`、`source_clock_identity_proven=false` 均保持不变。
快照 sequence 668→2599 严格递增，覆盖 31 个不同的最后 PTP 样本时间；
同一 `last_sample_mono_ns` 的 `valid_until_ns` 不变，没有靠重复读取延长租约。

左/右相机年龄上界最大 80.234 / 97.302 ms，p99 为 78.047 / 72.604 ms；
关节/TF 年龄上界最大 35.549 / 35.528 ms，相机时差最大 38.585 ms。
审计窗口映射误差最大 2.573 ms、漂移最大 15.316 ppm、残差最大 0.252 ms、
路径延迟最大 0.0392 ms；TF/motion 位置/旋转差最大 `5.77e-7 m` / `2.38e-6 rad`，
GDK 单次读取最大 37.594 ms。这些是观测统计，不是批准后的生产阈值。

监控证据位于 `/tmp/g2-clock-audit-20260921-02/`。审计结束后出现原始 PTP 行
`master offset 56328758690 s0 freq +705874 path delay 42102`，ClockWindow 随后
锁存 `mapping_invalid`、发布不健康状态并以 `returncode=2` 退出，记录了
fail-closed 行为；不能声称监控全程健康或据此批准运动。收尾已核对无残留
`ptp4l`、`phc2sys`、`clock_monitor`、`freshness_audit`。此前 `-01` 目录的
`sudo: 需要密码` 失败证据仍保留，该次退出码 1、未启动 PTP。

模拟等待不是实际 Actor/GPU 推理负载，最终六项阈值仍须在真实只读负载下分别批准。
两终端启动、B→A 停止顺序、证据位置和残留检查见
[持续时钟与负载审计说明](g2-clock-diagnostics.md)。SDK 同步调用卡死时不能保证
软件采样期限或 Ctrl+C 立即返回，不能从其他线程强制释放正在使用的 SDK。

操作员已报告硬件急停可访问；下方“仍不能操作硬件急停”为历史状态。
受控急停、停止距离、固件命令过期行为仍未验证；工作空间、动作尺度、控制模式、
速度/姿态/碰撞约束和任务判据也仍需独立确认。禁止据此启动真实运动或真机训练。

## 时间映射诊断工具（2026-09-20，本轮新增）

- 新增 `g2_local.clock_probe`：同期采集 PTP 偏差/时间属性与 GDK 双相机、
  关节、双向 TF、motion 位姿、本机 wall/monotonic 时间；不校时、不运动。
- 新增 `clock_mapping`：两种尺度假设检验、偏差/漂移、经验误差区间、
  boot/session 限制及最后 PTP 样本后 2.5 秒有效期；拒绝冻结、跳变、间断、
  主时钟改变和不一致的 TF。仅诊断，不接入 Gym/运动新鲜度许可。
- 首次联合运行暴露 TF 订阅缓存启动竞态：创建 TF 后立即查询时可能还没有
  `arm_l_end_link`。现改为 PTP 启动前最多等待 10 秒，确认两个查询方向均可用；
  非零退出也生成监督拒绝摘要。失败会保留证据，不会校时或授权运动。
- 已有输出目录在启动子进程前即被拒绝，避免向旧证据目录写入新会话结果。
- TDD 与独立只读复核后，时间诊断相关 34 项测试通过，全套 110 项通过；
  现场只读 GDK 3 次采样成功，TF 与 motion 位姿一致。
- sudo PTP + GDK 联合采集仍待用户在终端运行，不能宣称完成现场校准。
  命令、输出解释和限制见 [时间映射诊断说明](g2-clock-diagnostics.md)。
- 仍不能操作硬件急停，不启动实际运动或训练。

### PTP/GDK 联合现场诊断结果（2026-09-20）

- 用户完成 45 秒联合采集；证据位于
  `runtime/clock_probe/3b9712a68f1e4c3c9ee274dc79f8be6c/`，包含 254 条记录：
  220 组 GDK、8 次 PTP 时间属性、25 条 PTP 日志及 1 条会话元数据。
- 诊断唯一匹配 `raw_ptp` 尺度：linuxptp 报告的约 55.067 秒偏差包含
  37 秒 UTC correction；GDK 源时间戳相对本机 wall 的对应偏差约 18.067 秒。
  37 秒是时间尺度换算，不是链路延迟。
- 拟合漂移 12.515 ppm、最大残差 0.117 ms；诊断经验误差余量 2.130 ms。
  这是本次软件时间戳测量的经验余量，不是硬件保证，也不能跨会话长期复用。
- 175 组数据处于 PTP 拟合覆盖区间。估计年龄范围：左相机 8.96–46.92 ms、
  右相机 9.57–46.92 ms、关节 0.65–11.12 ms、TF 0.55–10.58 ms；
  加经验误差后的最大上界分别约 49.05、49.05、13.25、12.71 ms。
- 双相机最大源时间戳差 33.406 ms。全部 220 组 GDK 数据覆盖 44.825 秒，
  最大采样间隔 312.229 ms；仅 3 组相机时间戳不相同。
- TF 与 motion 位姿最大位置差约 0.31 µm、姿态差约 1.28 µrad；这确认本次
  查询方向和数值一致性，但 motion 状态仍没有自身源时间戳。
- PTP 属性稳定为 `ptpTimescale=1`、`currentUtcOffset=37`、无闰秒公告；
  `currentUtcOffsetValid=0`，因此只确认显式 37 秒 fallback 下的内部一致性，
  不宣称 UTC 可追溯或所有传感器物理时钟身份已经证明。
- 结果为 `diagnostic_consistent`，但映射在最后 PTP 样本后 2.5 秒即失效，
  `valid_for_live_use=false`、`motion_authorized=false`。系统时间未修改，采集后
  无残留 `ptp4l`、`phc2sys` 或 clock probe 进程。

## 已确认任务契约（2026-09-20）

- 左臂持有铰链，固定夹爪；策略仅负责局部对准与插入。
- 上游视觉负责抓取及接近，不假设必须使用历史 P4 绝对位姿。
- 左腕 RGB 与固定右手 RGB 双视角；第三视角后续按可观测性决定。
- 演示采集和训练首版即覆盖目标位置变化与抓取/起始姿态扰动。
- 约 5 cm 是目标移动候选范围；轴向和范围定义尚未冻结，不能用作动作尺度。
- 移动冰箱后由上游重新接近；每回合记录移动量与抓取条件。

## 现场资产

- GDK：`/home/flyfuture/.cache/agibot/app`，现场文档记录 4.1.5。
- 控制/相机/SpaceMouse：`/home/flyfuture/g2_hinge_assembly/g2_adapter`。
- 参考工程提交：`083dffd`，独立项目，保持原状。
- 本会话只读 preflight 已通过，14 关节可读，motion_mode=1、error_code=0。
- SpaceMouse 枚举为 `/dev/hidraw0`；枚举不等于轴、按钮和失联测试通过。
- GDK 末端目标采用 base_link、米、xyzw；命令有效期不代表到位。
- GDK 模式全局生效；现有手动接管会改变模式，接入 RL 前需定义统一模式。

## 本次基础代码

`g2_local/contract.py` 提供无硬件依赖的六维动作仲裁、回合随机化元数据和
transition 构造。显式人工控制允许零动作；transition 使用驱动确认的有效动作。
缺失后继观测、非有限动作和错误维度被拒绝；终止与截断分别保存。

`g2_local/config.py` 校验双视角配置、六维物理动作尺度和 base_link 工作空间。
默认不提供运动边界；未填写时运动配置校验失败。该校验不代表现场运动许可。
动作仲裁字段为 `selected_action`，驱动确认后的动作才进入 transition。

`g2_local/episode.py` 新增回合生命周期核心：reset 后才能 step；后端返回实际
执行动作；回合内接管标志累计；时间截断保留末帧；执行异常停止并要求重新 reset。
reset 只接收上游/人工已复位后的观测，不调用旧项目的运动回位流程。
`g2_local/env.py` 已提供 Gymnasium 环境，包含双 RGB、7 维末端位姿、6 维动作、
接管回调、终止/截断区分。初始观测校验失败后禁止 step。SyntheticBackend 仅用于
软件测试，不模拟接触物理、插入成功或真实演示。

## 已验证接通的部分

- 独立 `.venv`：Python 3.10.20、torch 2.7.1、torchvision 0.22.1、
  Gymnasium 0.29.1；使用仓库内 `lerobot/src`，不修改其他项目环境。
  完整已安装版本快照见 `requirements-g2-tested.txt`（不包含厂商 GDK）。
- `g2_local/gdk_backend.py` 的 GdkReader 已实际只读连接机器人，读到左末端
  pose 与左右腕 RGB；连续 3 组图片均为 1056×1280×3，时间戳递增。
  证据位于 `runtime/gdk_probe/`。读取不改变控制模式，不发送运动或夹爪命令。
- 图像仅检查时间戳递增；首帧绝对年龄、跨时钟同步、状态新鲜度和双目时差
  尚未构成安全门控。GdkReader 不是 execute/stop 运动后端，不能直接用于真机 step。
- `actor.py --g2-software` / `learner.py --g2-software` 使用真正 SiLRI 网络、
  上游 gRPC 服务和 ReplayBuffer，双图像合成输入，在线池与人工池分开。
  人工输入明确标注 synthetic，只有对应接管 transition 进入人工池。
- RTX 3090 已完成 3 次 β/Q/Actor/λ 更新，Actor 收到版本 0→3；
  learner 保存 `runtime/software_loop/checkpoint.pt`，两进程退出码均为 0。
- CPU 集成测试覆盖保存后启动新进程、从版本 2 恢复到 3、经验重建和优化器
  step 连续。恢复不是物理场景恢复，也不承诺 Actor 随机流逐位复现。
- 修复 SiLRI 连续动作优化器、空人工 mask 的非有限 loss、target encoder
  与在线 encoder 别名、固定标准差写死 cuda:0；空人工批次跳过 BC/β 更新。

软件模式采用从零训练的小 CNN（无预训练视觉权重、无数据增强），用于接口验证，
不是最终真机训练配方。原 Hydra 训练路径仍存在，尚未替换其 make_env 为 G2。
gRPC 仅绑定本机回环；pickle 通信和 checkpoint 仅接受可信本机数据。

验证命令（主仓库目录）：

最新完整回归：76 passed，包含发送循环、看门狗和 Gym 执行/关闭故障测试。Gym 对未限定状态空间给出 2 条无限边界警告。这不是
运动空间许可：真实动作边界必须另外明确设置。只读代码审查未发现短程软件
闭环阻塞问题；当前恢复后的 step_id 会重新计数，跨会话溯源标识仍待补齐。

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests -q
```

只读真机采集（不会运动）：

```bash
bash run_g2_python.sh -m g2_local.probe --frames 3
```

两个终端分别启动软件闭环（不要误认为真机训练）：

```bash
PYTHONPATH=lerobot/src .venv/bin/python learner.py --g2-software --device cuda --updates 3
PYTHONPATH=lerobot/src .venv/bin/python actor.py --g2-software --device cuda --updates 3
```

恢复时 learner 增加 `--resume runtime/software_loop/checkpoint.pt`，双方将
`--updates` 设为更大的累计版本（例如 4）；建议使用新的 `--output` 目录。
恢复要求配置一致，包括 device。软件模式保留全部接收记录供重建，因此只适合
有界短运行，不是长期采集存储系统。

## 后续实现顺序与证据边界

### 最新：发送循环、看门狗、Gym 执行链路（软件实现）

本节覆盖下面历史记录中“发送循环/完整 execute 尚未实现”的状态。

- `command_stream.CommandStream`：单写线程默认 50 Hz 重复发送固定目标，
  不重复累加位移；目标带序号和更新期限，首次成功发送后给出对应回执时间。
  不允许覆盖尚未回执的目标，停止/故障后不自动重启。
- 独立看门狗检查目标租约和发送心跳；发送线程自身也在发送前检查期限及
  漏发心跳，防止调度暂停恢复后抢在看门狗之前续发。SDK 超时的迟到返回
  不算成功回执；默认不是硬实时调度，不承诺实际每周期精确 20 ms。
- 超时或异常后由唯一写线程调用命令端口 stop，健康时尝试实测保持；
  SDK 发送/保持卡住时 bounded stop 明确报错“停止未确认”，不伪装成停稳。
- `motion_backend.MotionBackend` 已实现 observe/execute/stop/close，可注入
  `G2LocalEnv`。步骤为：校验前观测→计划/裁剪目标→提交→等待发送回执→
  等待配置的观测间隔→校验回执之后采集的后继观测→任务判据→StepResult。
  replay 动作是已确认发送的有效候选，不是目标到位或实际位移证明。
- 相机阻塞不阻塞独立写线程；若相机/策略不再及时产生新步骤，目标租约到期
  终止继续发送。前/后观测或任务判据出错则丢弃该 transition 并停止回合。
- 关闭操作等待发送线程和观测读取结束，超时保留 reader/SDK 资源；不能在
  卡住的调用下面释放 SDK。停止后需要显式重建，不通过 reset 自动重新使能。

构造时必须提供 `config`、`observation_guard(obs, info, after)`、
`outcome(obs)`、`command_timeout`、`send_timeout`、`step_period`；默认
`allow_motion=False`。观测 guard 必须根据实际采集时钟确认数据新鲜且
后继观测晚于回执，不能只看本机读取完成时间，也不能用恒真回调代替。
端口的源反馈 guard 仍须独立有效，控制器自身运动许可也不会被后端修改。

已用隔离端口/读者测试：重复发送、序号回执、租约过期、SDK 报错/卡住、
晚回执拒绝、默认禁用、裁剪动作写入 Gym、终止停止、相机失败/阻塞、
并发关闭保护、看门狗未获调度时发送线程自检。没有调用真实 GDK 运动接口。

**现场剩余门槛**：真实时间戳 guard（含双相机）、工作空间/动作尺度/模式、
姿态及碰撞/速度约束、固件命令过期行为和保持/急停验收。正式 Actor 尚未
切换为这个真实运动后端；物理接管开关与任务成功/复位判据仍待确定。
这次补齐软件执行路径，不意味着真机可启动训练。

### 当前增量：静置修复与低层 GDK 命令边界（覆盖下方旧状态）

静置问题已在输入逻辑中修复：只有本 gate 收到过新鲜回零报告、报告时间戳
与左键模式未改变、当前前三轴仍在死区内时，才允许缓存回零继续输出零。
新输入源、换模式、invalidate 或读设备错误不能沿用旧确认；非零输入仍要求
250 ms 内的新报告。静默回零不是连接心跳，不能证明 USB 未断连或设备未卡死。
HumanInput 使用同一有效性判断，不再将已确认的静置回零误判为活动输入故障。
RotationCheck 阶段推进仍要求新鲜报告，避免用旧回零充当新的校准证据。

三份真实校准日志离线回放分别处理 960/225/329 帧，允许的缓存静置回零样本
分别为 221/0/20，全部动作严格为零；测试覆盖长静置后重新输入及非零过期。
这不是新的现场拔插验收，真实 USB 断连/按钮报告丢失仍待验证。

新增 `motion.plan_target`：实测 XYZ+xyzw → base_link 平移/旋转向量增量；
检查实测位置在显式工作空间内，限制目标位置，并返回裁剪后的归一化候选动作。
候选不是已执行动作；只有驱动接收后才可写 replay。未提供物理尺度/空间则拒绝。
尚不包括碰撞、姿态包络、速度/加速度限制、状态新鲜度或执行时序。

新增 `GdkCommandPort`（低层诊断接口，不是完整 Gym 后端）：

- 默认 allow_motion=False；启用必须显式给出模式 1/3 和反馈新鲜度检查回调。
  回调必须明确返回 True；False/None 或异常均拒绝发送并锁定接口，不能用
  未实现的新鲜度检查或恒真占位回调启用实际运动。
- 发送前检查机器人状态/模式；不自动切模式，不发右臂/夹爪命令。
- 使用已检查的本地参考适配器 `_send_left_cartesian_pose` 原语，避免公开
  setpoint API 隐式准备并切换阻抗模式。参考工程没有被修改。
- stop 先锁定接口，健康时尝试一次当前实测位姿保持，不重发旧运动目标；
  反馈/模式异常不发保持，异常上报。停止或发送失败后不自动恢复。
- 所有发送测试使用隔离控制器替身，没有对真实 GDK 发送任何运动/保持命令。
  GdkReader 仍创建 allow_motion=False 的控制器，未改变这一默认值。

**仍未完成**：50 Hz 独立发送循环与看门狗、真实反馈时间戳门控、阻塞 SDK
调用处理、命令过期后的固件行为验收、完整 GDK execute/stop→Gym 路径。
Python 锁无法中断卡住的 SDK 调用；单次保持不是硬件急停，不能承诺立即停止。
真实工作空间/尺度/模式仍未冻结，不能启动真机训练。

### 三轴现场记录核验与 Gym 接管接口

已读取用户独立运行的完整证据文件，三轴均以 passed 结束：

- yaw：`runtime/calibration/70154c92c72149da8a17ca933f4629d1.jsonl`（962 行）。
- roll：`runtime/calibration/4692d4a616a94cde84e4cf980ab13204.jsonl`（227 行）。
- pitch：`runtime/calibration/8ea8d59385e945638f0ac8566d78c9af.jsonl`（331 行）。

证明手柄→动作提案正反向与回中/松键流程通过，不是持续单轴精度、拔插或
机器人运动验收。pitch 曾有串轴，不放宽阈值或自动加入主轴锁定。

新增 `HumanInput(reader, axis_map=..., left_button=...)`，可传给
`G2LocalEnv(..., intervention=source)`。默认不接管，必须由外部显式
`source.set_active(True)`；左键仅负责平移/旋转，未指定物理接管开关。
读设备异常、格式异常或人工接管期间报告过期会锁定输入源；修复后需要
创建新的输入源并 reset 回合，禁止自动回退策略或自动续跑。
reader 的打开/关闭由调用者管理，source 不隐式重连。

修复 Gym 回调异常绕过 EpisodeRunner 停止逻辑的问题：回调抛错会调用后端
stop、使回合失效，禁止下一次 step；停止失败记录日志并保留原始输入异常。
软件回归覆盖策略/人工标签、执行动作替换、人工零动作、过期输入和设备报错。
测试后端仍为 SyntheticBackend：不证明实际 GDK 停止行为或真实拔插已验证。
现有 250 ms 超时在静置时仍可能终止人工回合，接口尚不适合长期真机操作。
没有将 HumanInput 挂入正式真机 Actor，也没有实现 GDK 运动后端。

### 逐阶段旋转校验工具已实现（2026-09-20）

新增 `g2_local.calibration`，复用现有 LiveInputGate，不调用 GDK。
终端按事件推进：左键和轴报告→回中→正向→回中→反向→回中→松键。
每阶段默认最多 60 秒，只有实际达到条件才推进；超时、提前松键、读设备异常
均失败退出，不自动重试。不会把“没有输出”记录为通过。

```bash
# 在 SiLRI-HIL-RL 目录，确认其他遥操作已退出后运行：
.venv/bin/python -m g2_local.calibration --axis=yaw --axis-map=-2,-1,-3 --left-button=0
```

按终端提示操作，无需等待聊天提示；roll/pitch 可更换 `--axis`。
每次独立保存 `runtime/calibration/<唯一标识>.jsonl`：配置、原始六轴、按钮、
报告时间戳、预览动作及阶段/原因。指定 `--output` 时拒绝覆盖已有文件。
中文提示的正反方向基于本次校准摆放；换映射/摆放后需重新确认。

正反向检查要求归一化目标分量达到 ±0.2、其他分量绝对值不超过 0.15；
这仅是预览流程的样本筛选阈值，不是物理动作幅度或单轴精度验收标准。
仍保持现有 250 ms 轴报告新鲜度规则：静置超时后须轻拨回中再操作。
尚未解决事件驱动 HID 静默与断连的区分，也未验证按钮新鲜度。

新增测试覆盖完整正反流程、串轴拒绝、回中顺序、过期报告不推进、超时与
异常终止、原始证据记录。真实设备只做了 0.1 秒无操作超时 smoke，按预期
以 `timeout_at_press` 退出码 1 结束，证据为
`runtime/calibration/7f4aa003154647959c9cd18eff818484.jsonl`。
该 smoke 不是 yaw 实测通过；新工具下的操作者实测和拔插验收仍待进行。

### roll/yaw 现场复核补充（2026-09-20）

- 初次 roll 35 秒窗口：旋转模式且解锁的样本数为 0，结果无效，不能标通过。
- roll 重试 60 秒窗口：1664 次解锁旋转采样，XYZ 始终为零；roll 范围
  [-0.8127, 0.7175]，但 pitch [-0.9333, 0.8222]、yaw [-0.4254, 0.1746]
  也明显变化。确认 roll 有响应，不确认单轴精度；也未凭范围确认每个方向符号。
  观察到原始自身旋转通道大幅变化，需区分整体平推与旋帽倾斜操作。
- yaw 40 秒窗口：解锁旋转采样为 0，结果无效；末帧轴全零、按钮松开。
- 所有窗口均正常退出，退出时 invalidate 输出零提案，无 GDK 调用。
- 暂停重复倒计时实测：后续需要基于按钮/回中/目标轴输入事件逐阶段确认的
  校准工具，保留有界超时和原始证据，避免把错过窗口或门控未解锁误判为设备异常。
  当前尚不能确定 yaw 窗口无有效样本的具体原因，不据此调整安全阈值。

### 校准映射后的实时模式复核（2026-09-20）

使用 axis_map=(-2,-1,-3)、left_button=0，直接 CompactHID + LiveInputGate
进行了 35 秒平移、45 秒旋转只读观测；均退出码 0，无 GDK 控制调用。

- 平移段：所有旋转分量为零，X 提案范围约 [-0.689, 0.863]；
  同时出现 Z 最小 -0.254，说明手柄垂直串轴仍存在。第一段结束时输入未回零，
  因此不把这一段记录为松手停止验证。
- 旋转段：旋转模式内 XYZ 平移分量始终为零，pitch 范围约
  [-0.740, 0.879]；yaw 约 [-0.156, 0.038]，不能宣称纯 pitch 输出。
- 观察到按键切换后 blocked 零提案，收到回中报告后解锁；松开左键后回到
  平移模式。第二段最终按钮松开、提案全零，退出额外 invalidate 清零。
- 静置后多次因 250 ms 报告超时重新锁住：这验证了当前保守行为，也暴露
  交互不连续的问题。尚未解决事件驱动 HID 静默与断连的区分，不能直接将
  本预览门控当作真机使能/看门狗；不通过简单扩大超时掩盖该问题。
- 尚需复核按住左键时的左右→roll、上下→yaw，以及真实拔插故障。

### 分方向只读校准结果（2026-09-20）

用户依次按提示完成向前、向左、上提、按物理左键，以及左推复核。
均直接读取 CompactHID，未调用 GDK；各段结束正常、轴回零/按钮松开。

| 操作 | 主要平移通道读数 | 结论 |
| --- | --- | --- |
| 向前（远离身体） | 第 2 轴最小 -0.6514 | forward = -raw[1] |
| 向左 | 第 1 轴最小 -0.8571；复核 -0.7771 | left = -raw[0] |
| 上提 | 第 3 轴最小 -1.0 | up = -raw[2] |
| 物理左键 | 仅索引 0 收到按下与松开 | left_button = 0 |

因此此次设备摆放下，预览参数为 `--axis-map=-2,-1,-3 --left-button=0`。
这是手柄方向到逻辑 forward/left/up 的校准，不是机械臂运动验证；base_link
轴方向与姿态增量实际应用仍需受控真机测试。移动/旋转手柄摆放后应重新校准。

注意：左推初次伴随第 3 轴 +0.68；复核仍伴随第 2 轴 +0.30、第 3 轴
约 ±0.18。映射符号一致，但操作者单轴输入存在串轴；默认 0.1 死区不会全部
消除这些分量。不擅自加入主轴锁定，不据此宣称精确单轴控制通过。
本轮未复核反方向、未测试拔插，未执行真实动作。

### 已确认的 SpaceMouse 映射与实现

`g2_local/spacemouse.py` 提供归一化动作提案，不发送控制命令。
不按左键时，前后/左右/上下对应 base_link XYZ；按住左键时，
左右→roll(X)、前后→pitch(Y)、上下→yaw(Z)。旋转输出采用现有契约的
基坐标系旋转向量分量，不是绝对欧拉角。自身倾斜/扭转三个通道不参与输出。

- 原始平移轴到前后/左/上的带符号排列必须显式指定，无默认现场映射。
- 启动、切换模式、输入无效或非法数值后，必须先回中才能恢复输出。
- 死区默认 0.1 仅用于软件预览，可配置；没有确定现场死区或物理动作尺度。
- 左键不是接管/使能键。尚未将该映射器接入 Gym 的实时人工接管回调。
- `valid` 是调用方提供的有效性标志，不是已实现的 HID 断连/新鲜度检测。
- 离线预览命令：`.venv/bin/python -m g2_local.spacemouse --axis-map=2,-1,3`。
  从标准输入逐行读取 JSON：`{"axes":[0,0,0,0,0,0],"left_pressed":false,"valid":true}`。
  示例排列不是校准结果；CLI 不打开 HID、不导入 GDK，输出带 preview_only 标记。
  左键物理按钮索引确认、轴符号校准和完整断连验收仍待完成。

实时只读入口已增加：`g2_local.live_preview`。必须显式传入 `--axis-map`
和 `--left-button`（0 或 1）；`--seconds` 限制采集时长，最多 300 秒。
例如 `.venv/bin/python -m g2_local.live_preview --axis-map=1,2,3 --left-button=0 --seconds=30`。
这组参数仅演示语法，不代表现场校准结果；输出只用于预览，绝不调用 GDK。

`LiveInputGate` 检查两类轴报告时间戳，缺失、未来或过期时阻止输出；默认
250 ms 是预览阈值，不是已验证的机器人安全参数。恢复要求新的回中报告。
读设备异常时输出 blocked 零提案并抛出异常，CLI 关闭设备；不会自动重连。
静置设备可能不持续发送报告，因此静置也可能被阻止输出。参考解码器没有
按钮时间戳，此门控不是完整的断连检测或使能机制。

现场 0.1 秒静置 smoke 已执行，设备可打开、无报告时全程 blocked、退出清零，
退出码 0；未执行拔插故障实验，未校准映射，未将预览接入 Gym 或真实运动。

### 本轮现场只读检查（2026-09-20）

- 用户确认运动边界尚未测定；继续保持只读，不启用动作，不切换控制模式。
- 运行 probe 采集 10 组双 RGB 和末端位姿，退出码 0。原始记录与 20 张图片
  位于 `runtime/readonly_check_20260920/`，未发送机械臂或夹爪命令。
- 两路均为 1280×1056 RGB，时间戳各自递增；同组原始时间戳差最大约
  33.33 ms。未验证时钟来源/同步，不将时间戳差等同于确定的曝光时差。
- 左末端 base_link XYZ 约 `[0.422756, 0.345735, 0.855273]` m，
  四元数 xyzw 约 `[-0.496458, 0.509227, -0.492121, 0.502030]`；
  当前反馈 mode=1、control_mode=1、error_code=0。这只是当前姿态，不是安全边界。
- 人工查看采集图片：左腕主要为托盘/夹爪，右腕为冰箱顶部边缘/铰链。
  当前未处于最终插入姿态，尚不能验收插入目标的可观测性或决定第三相机。
- SpaceMouse `/dev/hidraw0` 可只读打开；即时 poll 未收到轴/按钮报告，
  axis_times 为 None、ready=False。不把默认零值当作六轴/按钮测试通过。
  本机有 spacenavd；待用户确认其他遥操作程序未连接后，再进行交互输入测试。

### SpaceMouse 交互只读验证（2026-09-20）

- 用户回复“准备好了”后，直接复用参考工程 CompactHID，O_RDONLY 读取
  `/dev/hidraw0` 60 秒；未导入 GDK 或遥操作控制模块，进程退出码 0。
- 六个原始轴均观测到正、负变化。按解码器归一化后的最小值为
  `[-0.8171, -0.8971, -1, -0.9943, -1, -0.2371]`，最大值为
  `[0.7686, 0.5657, 1, 0.74, 1, 0.4029]`。这些是输入读数，不是机器人动作尺度。
- 两个按钮均观测到按下和松开；结束时六轴全零、两按钮均松开，ready=True。
  两类轴报告各观测到 1405 次时间戳更新（poll 批次，不等于底层报文总数）。
- 本轮仅验证输入通路、双向轴变化、按钮和松手回零；未验证原始轴到
  base_link 的方向映射、解耦、死区、断连安全行为，也未启用机器人运动。

1. 配置与观测契约：相机采集时刻、状态新鲜度、物理尺度和工作空间。
2. G2 驱动和 Gym 环境：复用现场 API，完成测试后端、只读模式与显式运动门控。
3. 接管与奖励：复用 HID 解码；固定左臂，禁止旧电批/夹爪手势进入 RL。
4. SiLRI 网络修复与回归：优化器、人工 mask、设备、β/Q/Actor/λ、target。
5. 正式 Actor/Learner、双池、保存恢复和独立评估。

下一步真机门槛：确认 base_link 工作空间 XYZ 上下限、单步平移/旋转尺度、
统一模式（1 位控或 3 阻抗）、暂停/保持和硬件急停操作；随后实现受控运动后端。
SpaceMouse 真实失联、最终 ROI/任务成功现场判据、只读相机时钟校验和无接管独立
评估仍需完成；软件奖励/成功标签契约已接入。没有发送真实动作，不宣称完成正式
真机 Actor–Learner 或插入训练。
