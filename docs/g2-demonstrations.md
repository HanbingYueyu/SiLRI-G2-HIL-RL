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
4. `y` 成功、`f` 失败；通过有效后继观测结束该步并停止。不会自动回到固定姿态；再次实际复位后提供新 context，开始下一回合。
5. Ctrl+C 退出。中断中的回合保留已落盘步骤，但不标成完整回合，也不导入。

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

可重复传 `--demonstrations` 导入多套数据。数据在服务启动前校验并进入在线池和人工池，保存初始 checkpoint。导入本身不做预训练：现有 Learner 在接收训练 transition 后触发更新，人工样本用于 Expert/BC 及混合批次。导入数据不等于已经学会插入。

任务、图像 ROI、动作尺度、奖励、输入映射、控制模式和工作空间须与采集时兼容；可调整优化器参数、设备和服务配置。首次换训练 run 时生成新的训练 transition ID，来源映射写入 events.jsonl；同一 dataset/episode 再次导入会去重，恢复 checkpoint 后仍有效。修改 ROI/动作尺度后不能把旧示范悄悄重新解释。

当前 Replay 按配置容量预分配 float32 前后图像，两个池合计每 1,000 个容量约需 0.73 GiB 图像内存，checkpoint 也受容量影响。只读模板的 100000/50000 容量不是现场内存配置建议；首次少量示范应按机器资源和样本量明确设置容量，不能直接用模板容量启动大规模导入。
