# 现场点云：Depth Pro + 每帧比例标定 (2026-09-02)

## 结论

Depth Pro 实时管线（每帧 ChArUco 比例标定）已修通并通过无头冒烟；用 80cm 共面场景
的帧进行了比例校正，导出相机系点云。拟合主平面距离 0.798 m（板位姿 0.804 m），
面内点 RMS 8.9 mm；板/手机/箱面深度中值 0.81–0.83 m，与 80cm 工作距离一致。

**验证边界**：单帧、相机系、无世界位姿；"metric_accuracy_validated": false。
点云几何一致性只说明比例校正内部自洽，不构成独立尺度真值。

## 实时管线修复

1. `webcam_demo.py`：DepthPro 分支取设备行改为 `next(model.parameters()).device`
   （FOVNetwork 无 `.device` 属性，原代码崩溃）。
2. `exposure.py`：白平衡属性自动解析。此相机（DSHOW/UVC）只响应
   `CAP_PROP_WHITE_BALANCE_BLUE_U`(=17)，而 `CAP_PROP_WB_TEMPERATURE`(=45)
   读取 -1、写入被拒（Media Foundation 路径才支持 45）。新增
   `resolve_wb_property()` 按候选顺序探测驱动实际实现的属性；锁定时会先关
   AUTO_WB(`CAP_PROP_AUTO_WB`=44) 再写入，并带 5 次重试。
   实测 17 号属性 5000↔6000 读写往返一致。

冒烟（无头 2 帧，Depth Pro fp16@1536，3840x2160）：
- 焦点锁 293 验证通过；gain=0、WB=5000 锁定并写入元数据（wb_property: 17）；
- 保存了 raw/rgb/depth/comparison + 元数据，`variant: metric`。

## 点云

输入：`outputs/metric-coplanar-20260902/capture_20260902_150003_496647_utc_metric_scaled_depthpro_metadata.json`
（depth_file 指向 DepthPro 校正后 npy，s=0.8526）

命令：
```
uv run --no-editable depth-cloud --metadata <上述json> --output outputs/pointclouds/coplanar_150003_scaled_depthpro.ply --fit-plane
```

结果（`outputs/pointclouds/coplanar_150003_scaled_depthpro_fit.json`）：
- 体素前 518,400 点 → 导出 28,846 点（stride=4, voxel 1cm, max 5m）；
- 主平面：法向 [-0.080, -0.144, 0.986]，原点距离 **0.798 m**（板位姿距离 0.804 m）；
- 面内点 4,159（占比 14.4%），inlier RMS **8.9 mm**（阈值 2cm）；
- 边界 x[-0.53, 1.29] y[-0.83, 0.70] z[0.47, 2.40] m。

感兴趣区深度中值（校正后）：箱面 0.808 m、手机背面 0.805 m、
底座台面 0.826 m、板面约 0.80 m —— 全部落在 0.80–0.83 m，符合共面设置。

## 现场使用

`run-scaled-depthpro.cmd`（相机 0，3840x2160，焦点 293，gain 0，WB 5000）：
- 每帧必须有 ChArUco 板在画面内，否则比例走第上一帧的 stale 值（黄色提示）；
- Depth Pro 约 0.8–1.3 FPS（模型+预处理 EMA），保存按 S。

## 变更文件

- `src/monocular_depth/exposure.py`：WB 属性解析 + 自动 WB 关闭 + 重试
- `src/monocular_depth/webcam_demo.py`：DepthPro 设备行
- 版本 0.2.7 → 0.2.8（uv wheel 缓存规避）
- 输出：`outputs/pointclouds/coplanar_150003_scaled_depthpro.ply`（含 fit 版与 json 报告）
