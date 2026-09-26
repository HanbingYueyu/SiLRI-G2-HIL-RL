# G2 纯人工示范采集与导入

`g2_local.real_train demo` 复用正式 Gym、观测门控、回合协调器和动作后端，不创建策略网络、不请求策略参数、不连接 Learner。每一步始终为人工控制，SpaceMouse 回中时发零增量保持，不交还机器人策略。

软件入口已实现，尚未采集或验收现场示范。下列命令中的 `runtime/site-training.json` 是现场确认后生成的配置路径，并非仓库已经提供的可运行配置。现有 `configs/g2_real_training_readonly.json` 仍会拒绝运动。完成停止验收、任务 ROI/尺度和配置证据后再使用采集命令。

## 采集

在项目根目录运行；`--context` 由上游视觉/操作者在实际复位后更新，包含新的 episode_id、真实复位时间和扰动信息。时钟监控器须在运行，其 socket 路径与现场配置一致。

```bash
bash run_g2_python.sh -m g2_local.real_train demo \
  --run-id demo-001 \
  --config runtime/site-training.json \
  --output runtime/demo-001 \
  --context runtime/episode-context.json \
  --hid-device /dev/hidraw0 \
  --allow-motion
```

HID 路径以现场实际设备为准，output 必须是新目录，父目录必须已存在且符合现有运行器权限要求。无须同时运行 Learner。

1. 上游完成孔上方复位并提交本回合 context。
2. 同时按左右键并释放，开始回合。
3. 不按左键控制 XYZ；按左键控制 XYZ 和原生旋转三轴。松手保持，低于自动接管阈值但超过死区的人工输入也会记录。
4. `y` 成功、`f` 失败；通过有效后继观测结束该步并停止。配置 `motion.auto_reset.enabled=true` 时，在终止数据保存及旧环境关闭成功后，保持夹持、保持当前姿态先沿 base_link +Z 抬高 5 cm，再以直线平移/最短旋转返回本回合记录的起始位姿；不是固定的全局复位姿态。200 步正常超时也进入该流程。
5. 自动复位完成后创建新的机械复位 context，再按双键开始下一回合，不需伪造视觉 context。初次启动仍须真实复位 context；关闭自动复位时，每回合由上游/操作者提供新 context。自动复位期间保持旋帽回中：推动旋帽或按 Y/F 会中止复位，Ctrl+C 退出。异常/急停、停止未确认或终止数据保存失败时不自动复位。
6. Ctrl+C 退出。中断中的回合保留已落盘步骤，但不标成完整回合，也不导入。

自动复位不调用 Gym step、不记录 Replay、不改变夹爪。复位使用独立的新命令流，仍执行 GDK 模式/观测新鲜度、局部与绝对边界、命令租约检查。候选设定为合成平移 10 mm/s、旋转 0.2 rad/s，位置到达容差 1 mm、角度容差 0.01 rad、30 秒超时；均为软件设定，不是物理性能保证。完整抬升 5 cm 若超出任一范围，将在首条复位命令前拒绝，不裁短、不扩大边界。边界检查不证明路径无碰撞。首轮现场自动复位尚未验收。

数据位于 `runtime/demo-001/demonstrations/`：dataset.json 保存来源及任务配置摘要；每个回合独立目录，每步 `.pt` 保存实际 128×128 前后观测、有效动作、奖励、终止和来源信息；`complete.json` 是完整回合标记，记录逐步 SHA256。只保存训练用图像，不保存原始分辨率图像。超时回合保留 truncated，不伪造成功/失败标签。

## 离线检查

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m g2_local.demonstrations \
  --config runtime/site-training.json \
  --dataset runtime/demo-001/demonstrations
```

输出完整回合、步骤、成功、失败、时间截断计数。不连接 GDK，不创建模型。未完成回合不计入；没有完整回合、文件损坏、步号/终止不连续、非人工动作或配置不兼容都会拒绝。

## 导入 Learner

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m g2_local.real_train learner \
  --run-id train-001 \
  --config runtime/site-training.json \
  --output runtime/train-001-learner \
  --demonstrations runtime/demo-001/demonstrations
```

可重复传 `--demonstrations` 导入多套数据。Learner 启动入口在导入后先单独预训练行为策略 β（默认 500 步），记录步数和最后 loss，再保存初始 checkpoint。没有人工数据时不进行依赖 β 的 Actor/λ 更新。在线主循环只更新 Critic/Actor/λ；β 默认每新增 50 条人工样本更新 50 步，不再附加一次 Actor BC。少于 batch_size 的人工池采用有放回采样，不等于少量示范已足够。独立导入函数只导入，预训练由 Learner 启动/更新入口负责。

任务、图像 ROI、动作尺度、奖励、输入映射、控制模式和工作空间须与采集时兼容；可调整优化器参数、设备和服务配置。首次换训练 run 时生成新的训练 transition ID，来源映射写入 events.jsonl；同一 dataset/episode 再次导入会去重，恢复 checkpoint 后仍有效。修改 ROI/动作尺度后不能把旧示范悄悄重新解释。

当前 Replay 仍按容量预分配 float32 前后图像，两个池合计每 1,000 个容量约需 0.73 GiB；现场配置与只读模板已统一为 2048/1024，图像约 2.25 GiB。checkpoint 只序列化已占用槽位，恢复时重新分配容量。满池仍有容量级存储成本，运行时 uint8 存储尚未实现；按名义 10 Hz，在线池约覆盖 205 秒，人工池在连续人工控制下约覆盖 102 秒，实际以真实步频计算。

新 manifest/checkpoint 记录代码 SHA、实际源码摘要（包括未提交修改）、展开的策略配置和软件版本。新恢复/冻结评估入口拒绝缺失或不一致的身份；旧 checkpoint 不能直接作为修正算法的训练恢复点，需要明确迁移。物理周期不等于 `1/control_hz`：命令保持后还有观测、推理及传输耗时，应查看现场 `control_period_s` 分布。Y/F 在后继之后、下一动作前的轮询窗口内可结算暂存的最后样本；窗口外与物理按键时刻的对应仍需现场核验，不能称为硬实时归因。
