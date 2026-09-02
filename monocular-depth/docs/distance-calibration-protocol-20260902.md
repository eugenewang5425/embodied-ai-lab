# 距离变量单参考标定协议（60 cm 首点）

## 目标

把 Depth Anything V2 Metric Hypersim Small 输出的"预测光轴 Z"（米）在 60 cm 附近的工作区间做
单参考点尺度标定，并如实记录模型原始预测与真值的偏差；不把"拟合后的残差"当成精度证明。

## 真值来源

`monocular_depth/distance.py` 用已知 25 mm 格距的 ChArUco 板 + 已标定内参，对保存帧做
solvePnP（IPPE）恢复板平面。该平面是几何真值：

- 与深度模型无关（模型只影响 z_pred，不影响板位姿）；
- 与"镜头前端到板面"的尺量基准不同——尺量带镜头外壳偏移，而板位姿给的是相机投影中心的基准。
  程序同时记录两者，并给出推断的偏移量，但不把它当作已验证的量。

## 工具与用法

新增命令：`depth-distance-calibrate`（`monocular_depth/distance.py`）。

```bat
cd /d D:\项目\具身人工智能\monocular-depth
uv run --no-editable depth-distance-calibrate ^
  --metadata outputs\metric-live-20260902_205542\<capture>_metric_metadata.json ^
  --measured-distance-m 0.60
```

- `--measured-distance-m` 可选；传入时记录尺量值、推断投影中心偏移和"只用尺量"的备选尺度。
- 输出目录默认 `calibration/distance/session_<时间戳>/`，包含 `distance-calibration.json`
  与 `distance-calibration-debug.png`（板区拟合后相对残差图，±20% 截断）。
- 不写、不改任何默认内参文件；不自动应用尺度到实时管线。

## 一次收集的物理流程

1. 摄像头保持原位不动；板正对镜头、位于画面中央，约 60 cm 工作位置。
2. 用尺量"镜头前端 → 板面中心"的**实际**读数并记录（不是摆放目标值）。
3. 实时窗口确认绿色 `Focus LOCKED 293`、板面清楚且静止，按 S 保存。
4. 把保存的 capture 名（或时间）和实测距离（cm）交回分析。

## 当前限制（必须写进结果）

- 单参考点只能拟合**乘性尺度**：`z_true = scale * z_pred`，不能同时拟合偏移；
  偏移需要第二个独立距离。
- 拟合与"评估"用同一帧，不是独立验证；`metric_accuracy_validated` 保持 `false`。
- 尺度是否在其他距离成立（非线性、近距畸变）未知，需要另两档距离做 holdout。
- 仅用于室内粗对照，不用于机器人避障/碰撞/安全决策。

## 下一步（第二、三距离点）

- 40 cm 与 90 cm（或类似两档）各按同样流程保存一帧并记录实测距离。
- 用 60 cm 点拟合 scale，用 40/90 cm 点当 holdout 报告相对误差；
  若 scale-only 明显不足以描述，再在 3 点拟合 `z_true = a*z_pred + b` 并留一档做验证。
