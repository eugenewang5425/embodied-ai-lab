# 每帧全局尺度定尺管线（frame-scale pipeline，2026-09-02）

## 原理与依据

模型深度误差被实测为**整帧全局尺度**（80 cm 共面实验：同帧内 板↔真实物体 所需尺度比 1.00 ±1%；
逐帧尺度随图像内容摆动 ±10%）。因此不需要"距离→尺度"曲线，而是每帧用已知尺寸平面目标
（ChArUco 板）解算该帧的全局尺度，然后乘到全帧深度上。

## 已实现

- `src/monocular_depth/scaling.py`：`estimate_scale(depth, rgb, matrix, board_spec)` 返回
  {scale, mad, tilt, plane_distance, rms}；`apply_scale(depth, scale)`；CLI `depth-frame-scale`
  （--model da2-file|unidepth|depthpro）。
- `src/monocular_depth/depthpro_model.py`：Depth Pro 一等公民（variant "depthpro"，
  fp16@1536、fx 透传、SHA256 校验），与 unidepth 同一套 Pipeline 接口。
- `webcam_demo.py`：新增 `--frame-scale`；每帧检测板 → 定尺 → 校正深度 → 覆盖显示
  （黄色 "frame-scale s=… (stale)"）；板不在画面时沿用最近一次尺度并标记 stale；
  保存的 metadata 含 `frame_scale` 字段（scale/mad/板位姿/是否 stale）。
- 启动器：`run-scaled-depthpro.cmd`（Depth Pro + 定尺 + gain/WB 锁定）。

## 真机验证（80 cm 共面帧，Depth Pro）

- 定尺 s = 0.8526（该帧模型输出 0.943 m → 校正后 0.804 m）；
- 校正后板区相对误差：中位 0.0 %，RMS 0.24 %（帧内 MAD 0.14 %）。

## 已知限制（必须写进结果）

- 每帧必须有板（或至少一个已知尺寸平面目标）在场；板消失后沿用旧尺度并标记 stale，
  长时间无板时误差会回到模型原始水平（Depth Pro ±3–7 %，UniDepth +12 %，DA2 +67 %）。
- Depth Pro 实时约 1.3 FPS（0.76 s/帧，显存 3.97 GB）；追求实时用 `--variant unidepth`
  （~0.03 s/帧）配合 `--frame-scale`，精度档位低一档。
- 许可：Depth Pro 仅科研；UniDepth CC-NC；商用候选 Metric3D 待评测。
- 不用于避障/碰撞/安全决策。

## 使用

```bat
:: 实时（Depth Pro + 每帧定尺，~1 FPS）
run-scaled-depthpro.cmd

:: 实时（UniDepth + 每帧定尺，快）
uv run --no-editable depth-webcam --variant unidepth --frame-scale ^
  --camera 0 --width 3840 --height 2160 ^
  --calibration calibration\selected\focus293_16_20260902\selected-camera.json ^
  --gain 0 --wb-temp 5000

:: 离线：对已保存帧重算定尺深度
uv run --no-editable depth-frame-scale --metadata <...>_metadata.json --model depthpro
```

## 下一步（世界模型基座）

校正后的深度记录满足现有 `geometry_from_record` 的全部校验（variant=metric、已标定、焦点一致），
可直接：`depth-cloud --metadata ...` 出相机坐标系点云（`metric_accuracy_validated` 仍为 false，
但"帧尺度误差"已消除）；后续：帧间/跨视角配准（现在尺度一致性已被板保证）→ 增量建图。
