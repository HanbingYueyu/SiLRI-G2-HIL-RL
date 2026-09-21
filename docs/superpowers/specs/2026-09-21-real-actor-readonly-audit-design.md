# G2 真实 Actor 只读推理负载审计设计

## 目标与边界

在 RTX 3090 上使用现有双 RGB SiLRI Actor 和可信本机 checkpoint，对 GDK
左腕相机、右侧辅助相机和状态执行真实预处理与 `policy.select_action()`
前向，同时持续验证 PTP 映射和四源观测新鲜度。输出 action 仅做形状、
有限性和范围校验，随后丢弃。

本流程始终：

- `allow_motion=False`；
- 不构造 `GdkCommandPort`、`MotionBackend`、Gym 环境或 learner client；
- 不执行 Gym step，不发送 hold、运动、夹爪或模式指令；
- 不调整系统时间或 PHC；
- 时序阈值获批也不代表运动、任务或训练获批。

## 选定方案

在现有 `freshness_audit` 的只读采样循环中注入一个独立的 Actor
推理器。不复用现有 `actor.py` 进程入口，因为该入口与 gRPC、Gym step
和 transition 上传耦合，会扩大只读验收边界。不使用独立图像 IPC 进程，
避免额外复制和调度歪曲未来单进程 Actor 的负载。

## Actor 推理器

新增一个小型、无运动依赖的 `ActorInferenceProbe`：

1. 要求显式 checkpoint 路径和 `device=cuda`，拒绝 CPU 伪装为 3090 验收。
2. 只接受信任的本机 PyTorch checkpoint；加载后验证 schema、SiLRI 类型、
   双相机 key、`state=(7,)`、`action=(6,)`、输入图像声明为 `3x128x128`。
3. 通过当前 `create_policy('cuda')` 创建与正式 Actor 相同的 SiLRI 网络，严格
   提取并加载 checkpoint 中的 Actor 权重，不连接 learner。
4. 确认 CUDA 可用且设备名称精确包含 `NVIDIA GeForce RTX 3090`。
5. 对每份 GDK observation 执行：
   - 验证 state 和两幅 HWC `uint8` RGB 图像；
   - 从 NumPy 创建 tensor，转为 NCHW float `[0,1]`；
   - 将原始 `1056x1280` 整幅图像传入 GPU，使用确定性 bilinear
     缩放为 `128x128`；
   - 在 `torch.inference_mode()` 下调用 `policy.select_action()`；
   - 调用 `torch.cuda.synchronize()` 后才记录耗时；
   - 验证输出精确为 `(1,6)`、全部有限且位于 SiLRI clamp 范围；
   - 仅将 action 的范围摘要写入证据，绝不传给环境或命令端。

整幅缩放仅用于本次时延审计，不构成插入任务最终 ROI、视觉精度
或策略性能验收。

## 审计数据流

1. 操作员独立启动已验证的 `clock_monitor`。
2. 审计进程预检输出目录、snapshot socket、checkpoint 和 CUDA，然后创建
   `GdkReader(allow_motion=False)` 和 Actor probe。
3. 用第一份完整 observation 执行显式次数的 warm-up；该阶段另行记录，
   不进入正式分布，不使用过时图像做新鲜度结论。
4. 正式循环每次按以下顺序：

   ```text
   GDK observe
   → Actor 预处理/H2D/resize/forward/CUDA synchronize
   → 读取当前 ClockSnapshot
   → 转换四源时间区间
   → 计算含完整真实推理时延的年龄/偏差
   → 写入有界证据
   ```

5. checkpoint、CUDA、图像、action、snapshot、源时间或证据写入任一失败，
   审计立即失败关闭，保留 `motion_authorized=false`。

## 计时与证据

每帧使用 monotonic nanoseconds 记录：

- GDK 读取时长；
- CPU tensor 准备时长；
- H2D 与 resize 时长；
- Actor forward 及 CUDA 同步时长；
- 完整 inference 端到端时长；
- 推理后的相机、joint、TF 年龄区间和双相机偏差；
- mapping 误差、残差、漂移、路径延迟与 snapshot gap；
- TF/motion 位姿差；
- CUDA allocated/reserved/peak allocated bytes；
- checkpoint schema/version/hash、模型配置摘要、GPU 名称和 warm-up 次数。

`summary.json` 使用明确单位报告 min/p50/p95/p99/max，并固定包含：

```json
{
  "motion_authorized": false,
  "thresholds_approved": false,
  "source_clock_identity_proven": false,
  "actions_discarded": true
}
```

审计程序不自动生成或改写生产阈值。

## 阈值批准规则

候选阈值只能在真实 Actor 现场证据完成后由独立审查步骤提出。最低证据
要求为三个独立 120 秒会话，每个会话至少 1000 个完整样本，且：

- 每个会话的快照 sequence 严格递增，租约不被读取续期；
- 无源冻结、倒退、未来时间、无效映射或 action 校验失败；
- 监控器、审计器和 PTP 子进程全部正常回收；
- 三次证据的 checkpoint hash、模型配置和 GPU 身份一致。

六项阈值的提案使用三次会话的最坏上界，再增加显式、可解释的余量；
不得仅使用平均值或 p50。相机、state、skew 和 mapping 阈值须同时检查
p99 与 max；TF/motion 阈值还须保留任务几何容差，不能只按浮点噪声
收紧。

批准结果作为一份显式配置与证据清单，不成为 `FreshnessLimits` 的默认值。
配置须记录三个证据目录、checkpoint SHA-256、GPU 身份、批准日期和六项值。
调用方仍须显式加载它。

## 失败和中断

- 输出目录已存在时，在加载 checkpoint、CUDA、GDK 或 socket 前拒绝。
- Actor 初始化、warm-up 或正式推理的任何 `BaseException` 都必须清理已获取
  资源；清理失败不得标记 completed。
- `Ctrl+C` 顺序仍为先审计器、后监控器。
- PyTorch CUDA 调用或厂商 GDK 同步调用若卡死，Python 无法保证截止期或
  `Ctrl+C` 立即返回；不从其他线程强制释放正在使用的 SDK/CUDA 资源。

## 验证

离线测试必须覆盖：

- checkpoint schema/config/actor state 严格校验；
- CUDA 设备身份、预处理尺寸/类型/范围与 action 输出校验；
- warm-up 不进入正式分布；
- 真实推理耗时取代固定 sleep，且时间顺序使年龄包含推理；
- 任何 action 都被丢弃，无 Gym/command/learner 依赖；
- 证据文件有界、有限数 JSON、中断和关闭失败语义；
- 阈值审查在少于三份合格现场证据时必须拒绝。

现场按三次独立会话执行，每次先完成软件预检，再启动只读
monitor/audit；每次结束后核对无残留进程。
