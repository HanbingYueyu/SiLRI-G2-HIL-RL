# 固定冰箱 ROI 输入契约：fixed-fridge-roi-v1

记录日期：2026-09-27。此名称用于文档版本追溯，不改变配置 schema、数据或 checkpoint。

## 当前输入

| 项目 | 实际值 |
|---|---|
| 左腕 ROI，像素 xyxy，右/下端不含 | `[340,200,940,800]` |
| 右辅助 ROI，像素 xyxy，右/下端不含 | `[310,10,910,610]` |
| 处理 | 两路各裁 600×600，INTER_AREA 缩为 128×128；网络 CHW float32 /255 |
| 状态 | base_link 下末端 `[x,y,z,qx,qy,qz,qw]`，米、xyzw 四元数 |
| 动作 | 6 维，base_link 平移/旋转向量增量，每轴 0.0015 m / 0.026 rad |
| 图像存储 | 保存处理后的 ROI，无全幅原图 |

本轮固定冰箱，不新增域随机化；用户已查看裁剪样例并同意沿用，不等于所有接触阶段的可观测性已验证。

## 运行身份

- 本机配置：`runtime/train-fixed-fridge-20260927.json`。
- 配置 hash：`56293dcce0e4e1c24011d6e44a20981383313f1fa503f9b25e580c0c96e95cdd`。
- 当前种子/run ID：`offline-pretrain-20260927-06`，30 条兼容示范、4042 步。
- `runtime/` 不随仓库发布；此文档列出本机核对值，不冒充远端克隆可独立读取的产物。
- 示范导入校验 `demonstration_contract`（含 observation/ROI、task、intervention、相关 motion 字段）；不同完整配置 hash 可以在兼容契约下显式导入，不等于允许混用不同 ROI。
- checkpoint 恢复仍核对实际配置和算法身份，文档版本名不能绕过校验。改 ROI 后须重新确认数据兼容性及重建对应种子，不重解释旧像素。

本次只补文档，未修改输入、配置 hash 或现有模型。
