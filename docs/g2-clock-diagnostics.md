# G2 时间映射只读诊断

本工具不调整 CLOCK_REALTIME 或 PHC，不切换机器人模式、不发送运动/夹爪指令。
原有 probe、Gym 和运动许可保持不变。`diagnostic_consistent` 不是运动验收。

## 持续时间映射监控（独立前台进程）

在操作员终端 A 执行：

```bash
cd /home/flyfuture/桌面/hil-rRL/SiLRI-HIL-RL
bash run_g2_python.sh -m g2_local.clock_monitor \
  --master 044052.fffe.000010 --max-seconds 43200
```

`--max-seconds` 必须显式提供，范围为 60–43200 秒，到期退出且不会自动重启。
监控在完成路径安全预检后，由普通用户 Python 主进程执行一次固定的
`/usr/bin/sudo -v`，需要密码时直接在当前终端输入。认证通过后才创建会话证据并
启动固定的 `sudo -n` PTP 测量进程组；认证失败不创建输出、不启动 PTP。
认证与测量保留同一控制终端和 TTY 会话，同时测量具有独立、已知的进程组。
仅共享父 PID 不够：若测量另建会话并丢失控制终端，TTY 票据仍可能失配。
外层 shell 预先 `sudo -v` 无需额外执行，也不能代替这一步。不要用 sudo
启动 Python/GDK；程序不读取、转存或记录密码。
每轮创建 `runtime/clock_monitor/<随机会话>/`，终端打印本轮 `clock.sock` 地址。
也可以用 `--output` 指定全新的输出目录；已有目录或 socket 会在启动 sudo 前拒绝。
输出路径不接受符号链接或不可信的可写祖先；目录身份若被替换会拒绝或停止本轮，
不会通过替换后的路径修改已有目录或证据。
使用较短路径，Unix socket 地址必须适合系统的路径长度上限。
目录权限为 0700、快照 socket 为 0600。普通用户 Actor/Gym 在另一终端独立启动，
只读取本轮 socket；不调用 sudo，不启动或停止 PTP。此次真实 `allow_motion=False`
保持不变，监控健康也不代表运动获准。

完成上述凭据验证后，监控以固定绝对路径启动 `sudo -n → timeout → stdbuf → ptp4l`，
接口固定 `enp3s0`，二层 E2E、软件时间戳、client-only、`free_running=1`，
不会改变系统时间或 PHC，也不调用 `phc2sys`。PMC 以普通用户运行，仅向会话专属
只读 UDS 发出 `GET TIME_PROPERTIES_DATA_SET`；可写管理 UDS 权限为 0600，
只读 UDS 为 0666。初次属性查询在启动后 8 秒开始，之后约每 5 秒查询一次。

至少 8 个有效 PTP 样本、覆盖 10 秒且属性通过后才可发布健康映射。租约固定截止
最后有效样本后 2.5 秒；读取快照不会延长租约。PTP 退出、属性查询失败、数据损坏、
socket 丢失或证据写入失败均进入不健康状态并退出。`evidence.jsonl` 记录原始 PTP
行、属性响应、映射变化和退出原因；IPC 服务线程停止或监听器失效同样会停止本轮。
单行 PTP 上限 4096 字节、属性响应上限
8192 字节、证据文件上限 64 MiB，达到限制停止本次会话，不覆盖已有证据。

停止时先结束独立 Actor/只读消费进程，再在终端 A 按 **Ctrl+C**。
监控先发布不健康状态，然后仅向自己创建的进程组发 SIGINT，并关闭本轮快照 socket；
不使用 `pkill`，不影响其他 PTP 服务。正常收尾等待最多 4 秒。若特权进程未响应或
权限导致信号无法传递，固定的 `timeout --signal=INT --kill-after=3s` 是最终回收
边界，日志会尽可能记录 `cleanup_pending`。此时等待本轮期限结束并确认旧进程已退出，
再开新会话。异常终止 Python 后，客户端租约仍会过期，特权命令仍受固定期限限制。

只读检查残留进程（该命令不停止任何进程）：

```bash
ps -eo pid,ppid,pgid,user,comm,args | rg 'ptp4l|phc2sys|clock_monitor'
```

现场停止/硬件急停验收与任务阈值批准仍是独立步骤；此监控不构成训练或运动授权。

## 两终端只读负载审计（2026-09-21 已执行一次现场采集）

操作员已报告硬件急停可访问；受控急停、停止距离和固件命令过期行为仍未验收。
本流程始终保持真实 `allow_motion=False`，不创建命令端口或运动后端，不发 hold、
运动、夹爪或切模式指令。审计成功也不能启动真机 Gym step 或训练。

终端 A：确认旧测量已退出后，启动独立监控。下面是已完成会话的命令记录；
复跑必须换一个全新目录名，不能删除旧证据后复用；不同时运行其他 PTP/校时服务。

```bash
cd /home/flyfuture/桌面/hil-rRL/SiLRI-HIL-RL
bash run_g2_python.sh -m g2_local.clock_monitor \
  --master 044052.fffe.000010 --max-seconds 120 \
  --output /tmp/g2-clock-audit-20260921-02
```

保留 `/tmp/g2-clock-audit-20260921-01` 的原始认证失败证据：`sudo: 需要密码`，
退出码 1，该次未启动 PTP。现场采集使用新的 `-02`，没有覆盖失败证据。

监控预热需至少 8 个有效样本且跨度至少 10 秒，并取得健康属性；留出约 15 秒，
以健康快照为准。审计遇到尚未预热、过期或断开的监控会失败退出，不自动重试。
若需要较长审计，必须提前显式选择足以覆盖预热、审计和收尾的监控期限；
监控最长 43200 秒，审计最长 1800 秒。不要通过重启监控延续同一审计。

终端 B：用 A 本轮打印的 socket 路径运行以下命令；不需要 sudo。

```bash
cd /home/flyfuture/桌面/hil-rRL/SiLRI-HIL-RL
bash run_g2_python.sh -m g2_local.freshness_audit \
  --socket /tmp/g2-clock-audit-20260921-02/clock.sock \
  --seconds 60 --inference-delay-s 0.02
```

`--seconds` 必须显式提供整数 30–1800；`--inference-delay-s` 必须是有限非负秒数，
默认 0。此参数是在读取双相机/GDK 后等待，再读取映射和计算年龄，用于模拟
推理造成的延迟；它不执行真实 Actor/GPU 推理，不能替代真实推理负载验收。
接近截止时间时等待会缩短，实际等待时长单独记录。两次采样间额外等待最多
50 ms，不保证固定采样频率。一次最多 36000 个成功样本、单行 16 KiB、
JSONL 总量 64 MiB；任何上限或证据写入失败均终止并标记失败。

每轮创建 `runtime/freshness_audit/<随机会话>/`；可用 `--output` 指定全新目录。
已有输出在创建 reader/client 之前拒绝；目录 0700，证据文件 0600，拒绝符号链接
祖先。`evidence.jsonl` 保存逐样本完整时间/双向 TF/位姿元数据、不可变时钟快照、
采集时刻区间和模拟推理计时，不保存 RGB 图像。有效可序列化的失败输入尽可能
保留为 `rejected` 行；不可序列化或超限输入由失败摘要说明。

`summary.json` 为完成、失败或 Ctrl+C 中断保留已接受样本统计：每项包含
`min/p50/p95/p99/max`，分位数使用线性插值。字段名明确单位：相机/状态年龄、
相机时差、映射误差/残差/路径延迟、GDK 读取时长/间隔、快照创建间隔和实际
模拟推理等待为 `ms`；漂移为 `ppm`，TF/motion 位置差为 `m`，旋转差为 `rad`。
`camera_age_ms` / `state_age_ms` 使用年龄区间上界，`*_age_lower_ms` 为下界；
`camera_skew_ms` 含映射不确定性，`source_camera_skew_ms` 是原始相机时间戳差。
无样本时没有分布；无论状态如何，`motion_authorized=false`、
`thresholds_approved=false`、`source_clock_identity_proven=false`。
不会输出启用运动的布尔建议，也不会生成或修改生产阈值配置。

停止顺序固定：先在 **终端 B 按 Ctrl+C**，等审计写摘要、关闭 reader/client 并退出
（退出码 130）；再在 **终端 A 按 Ctrl+C**，等监控处理自有 PTP 子进程并退出。
审计到期正常退出码 0，拒绝或写入/关闭失败退出码 1。关闭 client 不会停止监控。
时长限制采样循环；厂商同步 SDK 调用若卡死，软件不能保证此期限或 Ctrl+C 立即
返回，也不会从另一线程在卡住的 SDK 调用下释放资源。出现这种情况不启动新会话，
保留终端状态与证据并按现场异常流程处置，不能将进程退出等同于机器人停稳。

两边退出后只读检查（排除检查命令自身；不要用全局 `pkill`）：

```bash
ps -eo pid,ppid,pgid,user,comm,args | rg 'ptp4l|phc2sys|clock_monitor|freshness_audit'
```

若监控报告 `cleanup_pending`，按前文等待其固定期限及 kill-after 余量，复查自有
会话进程已退出后再运行。现场验收还须核对证据：快照 sequence 递增；同一最后 PTP
样本不能延长 `valid_until_ns`；无残留进程；摘要始终未授权运动。
本轮已执行上述最大 120 秒监控 + 60 秒审计；审计完成后监控因映射异常提前退出，
不能将本次结果表述为监控全程健康，详见下方现场记录。

后续阈值批准是单独流程：在双相机、GDK 与真实 Actor 推理同时运行的只读条件下
收集分布和异常证据，结合任务误差预算逐项审查并显式提供六个 FreshnessLimits。
当前模拟等待统计不批准任何候选阈值；仍需独立完成硬件急停/停止距离、运动空间、
尺度、模式及其他控制安全门槛，才能另行评估运动或训练许可。

### 2026-09-21 现场证据与结果

审计证据：`runtime/freshness_audit/7bf7d98a6b7345f892573635aad109ce/`。
本次 60 秒、模拟推理等待 0.02 秒，摘要为 `status=completed`，717 个样本，
没有 `rejected` 行。`motion_authorized=false`、`thresholds_approved=false`、
`source_clock_identity_proven=false` 均保持不变。

原始记录中的快照 sequence 从 668 到 2599 严格递增，覆盖 31 个不同的最后 PTP
样本时间；同一 `last_sample_mono_ns` 下的 `valid_until_ns` 没有变化，重复读取
没有延长旧样本租约。以下为此次审计窗口的统计，单位和年龄上界含义同前：

- 左相机年龄上界 p99 78.047 ms、最大 80.234 ms；右相机 p99 72.604 ms、
  最大 97.302 ms。
- 关节/TF 年龄上界最大分别为 35.549 / 35.528 ms；含不确定性的相机时差
  最大 38.585 ms。
- 映射经验误差最大 2.573 ms，拟合漂移最大 15.316 ppm，残差最大 0.252 ms，
  路径延迟最大 0.0392 ms。这些最大值只对应审计窗口，不包括之后的监控异常。
- TF/motion 位置差最大 `5.77e-7 m`、旋转差最大 `2.38e-6 rad`；
  GDK 单次读取最大 37.594 ms。

监控证据：`/tmp/g2-clock-audit-20260921-02/evidence.jsonl`。审计结束后，
监控收到原始 PTP 行 `master offset 56328758690 s0 freq +705874 path delay 42102`，
随后 ClockWindow 锁存 `mapping_invalid`，发布不健康快照并以 `returncode=2` 退出。
这记录了实际异常下的 fail-closed 行为，不是一次全程健康监控，也不批准阈值或运动。
收尾已核对无残留 `ptp4l`、`phc2sys`、`clock_monitor` 或 `freshness_audit` 进程。

此次为只读现场采集，不是实际 Actor/GPU 负载验收；同步 SDK 调用的中断/期限
限制仍在。硬件急停虽可访问，受控急停与停止距离仍未验证，禁止据此启动真机训练。

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
