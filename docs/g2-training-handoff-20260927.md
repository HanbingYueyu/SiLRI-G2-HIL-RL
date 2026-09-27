# 固定冰箱首轮训练交接

## 当前边界

未启动真机 Actor。用户已提供低速停止后 3 秒 TF 稳定记录；下一步受监督闭环。不提供自动解锁。SDK 卡住或通信断开时软件不能保证停止，需现场急停。

## 当前训练产物

- **当前 1:2 配置已重新预训练并验证恢复：`actor_update_interval=2`，UTD=1；旧 `-01` 产物保留，不混用。**
- 配置：`runtime/train-fixed-fridge-20260927.json`
- run ID：`offline-pretrain-20260927-04`
- checkpoint：`runtime/offline-pretrain-20260927-04/checkpoint.pt`；审查修复后重新生成，旧 -01/-02/-03 不混用。
- 总 batch=16（在线 Replay 8＋人工 Replay 8）；累计真机回合不包含导入示范。每 10 回合和正常退出保存，原有每 1000 次更新的额外备份频率保留。
- 结果：同目录 `result.json` 和 `events.jsonl`，30 回合/4042 步，β=500、Critic=10、Actor=5、λ=10；恢复后第 11 次仅更新 Critic/λ，验证未覆盖文件。代码/配置身份及优化器步数已核对。
- 此配置与未来 Actor 共用，包含 `requested_motion=true`；离线预训练强制 `cli_allow_motion=False`，未授权运动。正式入口仍须 CLI 运动标志及有效现场证据。
- 修改配置或代码/原生绑定后，旧 checkpoint 可能身份不匹配；不能跳过检查。旧文件保留。

## 下一阶段 Learner 命令（本轮未启动服务）

```bash
cd /home/flyfuture/桌面/hil-rRL/SiLRI-HIL-RL
bash run_g2_python.sh -m g2_local.real_train learner \
  --run-id offline-pretrain-20260927-04 \
  --config runtime/train-fixed-fridge-20260927.json \
  --checkpoint runtime/offline-pretrain-20260927-04/checkpoint.pt \
  --checkpoint-dir runtime/fixed-fridge-training \
  --output "runtime/learner-live-$(date +%Y%m%d-%H%M%S)"
```

Learner 不接触机器人。以后重复同一命令：优先读取 `runtime/fixed-fridge-training/checkpoint.pt`；只有第一次没有该文件才读取 `--checkpoint` 种子。不因最新文件损坏或身份不匹配而偷偷回退。目录加单写者锁，checkpoint 原子替换；不要删训练目录或更改 run ID。正常 Ctrl+C 等待保存完成，强制杀进程/断电只能恢复最后一次保存。

终端每轮优化打印 `completed_episodes`、`learner_update`、`actor_loss`、`critic_loss`；累计回合 20 时 `next_episode=21`。Actor 未更新的轮次为 `actor_updated=false, actor_loss=null`，不复用旧值。第 10/20…回合和退出时打印 `checkpoint_saved`。尚未完成且没有终止转移的中断回合不算完成。

导入示范不再授予在线 UTD 预算，新种子预算为 0；在线新转移才授予更新预算。每次从 Replay 随机采样，不要求 16 条连续；在线池与人工池可能包含同一人工接管转移，人工比例不保证为 50%。

## Actor 启动前

2026-09-27 用户明确确认绝对 Z 上限 0.965 m 可用。配置未变，不改 checkpoint 身份；不代表助手实测或其他停止条件的新增验收。

使用同一配置、run ID 和新 output。当前健康时钟通过 `--clock-socket /实际路径/clock.sock` 传入，地址单独记入事件，不修改配置哈希；仍核对主时钟和映射。每回合上游复位并提供新鲜 context，双键开始。HID 使用 `/dev/spacemouse-compact`。

Actor 进程先加载本机 GDK 环境：`source /home/flyfuture/.cache/agibot/app/env.sh /home/flyfuture/.cache/agibot/app`。本轮不提供一键启动真机策略命令；现场安全停确认后再下发。

## 原生停止绑定

源码 `g2_local/native/safe_stop.cpp`；构建 `.venv/bin/python -m g2_local.build_safe_stop`，本机已完成。限定 CPython 3.10/x86_64、GDK 4.1.5，复用原 Robot 和 SDK 串行锁，不新增 Robot，不发送夹爪指令；仅暴露左臂安全停，不提供解除接口。生成 so/清单不进 Git，换机须重建。

正常停止使用验证后的实测保持；保持失败时请求 `SetControlModeFull(INPUT_GDK, TARGET_LEFT_ARM, 配置控制模式, SAFE_STOP, 默认优先级10)`。异常/非零返回不算应答，应答仍报告物理停止未确认。外层超时不能取消阻塞的原生调用。

## 验证范围

停止/入口相关 50 项通过；正式通信链用真实进程、gRPC、模型优化和恢复，硬件/HID 是替身，第二回合等待真实更新参数下发后开始。当前配置无运动 Actor 启动正确返回 `motion_not_permitted`。不替代现场停止或稳定性验收。
