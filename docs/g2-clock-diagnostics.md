# G2 时间映射只读诊断

本工具不调整 CLOCK_REALTIME 或 PHC，不切换机器人模式、不发送运动/夹爪指令。
原有 probe、Gym 和运动许可保持不变。`diagnostic_consistent` 不是运动验收。

## 一次并行采集

先结束之前手动运行的 PTP 测量，保持机器人网线连接；不要同时启动其他校时服务。
在普通用户终端执行（仅 `sudo -v` 输入本机密码，不要用 sudo 运行 Python）：

```bash
cd /home/flyfuture/桌面/hil-rRL/SiLRI-HIL-RL
sudo -v && bash run_g2_python.sh -m g2_local.clock_probe \
  --seconds 45 --master 044052.fffe.000010
```

自动创建 `runtime/clock_probe/<随机会话>/`，不覆盖旧证据。
可用 `--output runtime/clock_probe/自定义新目录` 指定未存在的目录。

- `evidence.jsonl`：会话 ID（含 boot ID）、PTP 原始日志及本机接收时间、
  GDK 相机/关节/TF 源时间戳、本机 wall/monotonic 采样边界、SDK Clock、
  两种 TF 查询方向和运动状态位姿、重复的 PTP 时间属性 GET 响应。
- `summary.json`：诊断通过时包含候选时间尺度、线性偏差/漂移拟合、样本年龄区间；
  拒绝时说明原因。无论结果如何，`motion_authorized`、`valid_for_live_use` 均为 false。
- `supervisor.json`：SDK 等调用导致采集子进程整体超时时生成拒绝记录。

PTP 使用 `enp3s0`、二层 E2E、软件时间戳、client-only、`free_running=1`，
使用本会话专属 UDS 路径，明确设置 UTC correction fallback 为 37 秒。
仅固定的 `timeout → stdbuf → ptp4l` 和 `pmc GET TIME_PROPERTIES_DATA_SET`
经 sudo 执行；GDK 运行在普通用户进程。
这会发送 PTP 测量/查询报文，不是纯被动抓包，但不会校时。

正常采集约 45 秒，另有 GDK 初始化及收尾时间。主进程整体上限为 seconds+20 秒，
超时会终止自己的非特权采集进程；特权 PTP 子进程还有独立 seconds 秒超时及
3 秒强制终止余量。异常后不要立刻启动另一轮，应先确认旧测量已退出。
GDK 采集发生异常时保存已取得的证据，不保证一定产生 summary；查看终端错误。

## 计算含义与限制

测得的 linuxptp 软件时间戳偏差包含 UTC correction；分别检验：

- raw_ptp 假设：`本机 wall - 原始主时钟 = master offset - 37 秒`。
- utc 假设：`本机 wall - 主时钟 UTC = master offset`。

37 秒是已观测报文和本次显式配置对应的尺度转换，**不是网络延迟补偿**。
采集期间反复核验 `ptpTimescale=1`、`currentUtcOffset=37`、无闰秒公告；
属性缺失或改变即拒绝。`currentUtcOffsetValid=0` 会原样保留，说明使用显式
fallback，不宣称已验证机器人 UTC 正确或可追溯到外部标准时间。

使用 PTP 日志中的单调时间拟合，不使用 stdout 到达时刻，也不从相机年龄反推偏差。
迟到超过 500 ms 的 PTP 日志被拒绝；日志时间不是精确报文时间，纳入经验余量。
至少 8 个 PTP 样本、跨度 10 秒、相邻间隔不超过 4 秒。
映射限定本次 boot/session；有效区间截止最后 PTP 测量后 2.5 秒，不能持久化后
当作下一次训练的有效映射。离线报告仅回顾同期数据，不允许未来持续使用。

诊断误差余量 = 2 ms 软件时间戳/日志余量 + 最大拟合残差 + 最大路径延迟估计；
外推另外按 100 ppm 增长。这是经验容差，不是可靠硬件误差上界，也无法排除
链路不对称、采样至发布时间未知等问题。需要至少 8 组重叠 GDK 数据，仅在两种
时间尺度中恰有一个符合样本年龄范围时报告 `diagnostic_consistent`。

当前仅供诊断的拒绝条件：PTP 残差 >1 ms、漂移 >100 ppm、路径延迟 >1 ms
或负值、本机 wall/monotonic 差变化 >1 ms、GDK 一次读取 >500 ms、
GDK 间断 >5 秒、任一源冻结/倒退、双相机时间戳差 >100 ms、
TF 与 motion 位姿差 >5 mm 或 >0.02 rad、估计年龄（含误差）>500 ms。
这些宽松诊断阈值**不是训练时的新鲜度、安全或相机曝光同步验收阈值**。

当前 SDK 中查询 `lookup_transform_latest('arm_l_end_link', 'base_link', True)`
与 motion 位姿一致；保留反向查询原始证据，并逐样本复核，不悄悄交换参数。
motion 状态本身没有源时间戳，TF 一致性不是其采集新鲜度证明。
即使各源与主时钟数值一致，仍不能仅凭时间戳证明传感器来自同一物理时钟，
因此报告固定 `source_clock_identity_proven=false`。

## 验证

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m pytest \
  tests/test_g2_clock_mapping.py tests/test_g2_clock_probe.py -q
```

2026-09-20：首次联合运行时，TF 对象刚创建便查询，缓存尚未出现
`arm_l_end_link`，证据被拒绝且 PTP 按固定超时退出；未校时。现已在启动 PTP
前增加最多 10 秒的双向 TF 就绪预检，失败会快速退出并生成拒绝摘要。
手动指定的输出目录若已存在，会在启动采集前拒绝，避免污染旧会话证据。
相关测试共 34 项通过，全套 110 项通过（原有 Gym Box 两条 warning）。
只读 GDK 3 次真实采样成功，关节及两种方向 TF 时间戳可读，候选 TF 与 motion
位姿数值一致。随后完成 45 秒联合现场运行：结果为 `diagnostic_consistent`，
唯一匹配 `raw_ptp`，拟合漂移 12.515 ppm、残差 0.117 ms、经验误差余量
2.130 ms；详细结果记录在 `g2-adaptation-status.md`。这完成了本次回顾式时间域
关系确认，但映射已过期且固定 `valid_for_live_use=false`，不能作为训练或运动许可。
