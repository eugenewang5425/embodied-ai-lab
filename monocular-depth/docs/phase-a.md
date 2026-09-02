# A 阶段：从深度图到可审计的相机坐标系几何

当前软件已实现；真实相机标定、米制误差验证待采集。先固定摄像头和桌面，
不引入 SLAM、机器人控制或大模型训练。默认相对模型仍可用，米制模型在同一环境中切换。

## 现在需要你做的一步

打开 `calibration/board/print.html`，用 A4 纵向、100% / 实际大小打印，关闭适应页面和页眉页脚。
尺量每个方格 **25 mm**、整块图案 **175 × 125 mm**、校验线 **100 mm**。不符合就调整打印设置。
贴在不弯曲、平整的硬板上，不要直接拿软纸标定。不要用随意缩放的屏幕显示代替这一测量步骤。

这里是 OpenCV ChArUco 7×5 棋盘，DICT_4X4_50，marker 18 mm，非 legacy 模式。
生成器参数和图像一起保存；打印尺寸、board.json 和实物必须一致。

全部命令在 `D:\项目\具身人工智能\monocular-depth` 运行。

## 1. 环境与权重

```powershell
.\setup-metric.cmd
```

复用本项目 `.venv` 中的 PyTorch，只补充 Open3D、米制权重和本包；不改上层 MuJoCo 环境。
本次已安装，无需再次执行。以后修改源码用：

```powershell
uv sync --no-editable --reinstall-package monocular-depth-lab
```

米制模型：[Depth Anything V2 Metric Hypersim Small 官方代码](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)，
适用于室内实验，配置最大预测深度 20 m。不是测距传感器，不保证桌面厘米级精度。
权重从官方 Hugging Face 仓库的固定 revision 下载，加载前校验 SHA256。
不能给原 relative 权重的数值直接贴上 m 单位。

## 2. 采集真实相机标定数据

```powershell
.\capture-calibration.cmd
```

弹出摄像头窗口：先让板对清楚，按 F 锁焦，等待绿色 LOCKED，再按 S 保存；Q / Esc 退出。
第一张保存前可以按 A 恢复自动对焦后重试 F；第一张保存后禁止重新对焦，需要重调时退出并开启新会话。
当前默认 camera 0，3840×2160，MJPEG，请求 30 fps。
这是本机 EMEET SmartCam C60E 4K 驱动列出的最高分辨率；启动会检查实际首帧尺寸，拒绝静默降档。
预览窗口缩放显示，保存 PNG 仍是 3840×2160。换设备后编号可能改变，需重新确认设备和模式。
旧的 1280×720 照片要保留在旧会话，不能混入新的 4K 标定；4K 标定不能直接用于 720p 推理。
默认自动创建新的会话目录，不覆盖旧采集。结束时会打印目录路径。

- 推荐采集 20–40 张清晰的板图；尽量停止移动后再按 S，避免模糊。
- 从不同距离和角度拍：正面、左右倾斜、上下倾斜；覆盖图像中央和四角。
- 让棋盘占据足够像素，至少检测到 10 个非共线角点才允许保存。
- 不要只把板平移，不要连续保存同一姿态；离线解算还会拒绝近重复帧。
- 尽量固定焦距/对焦、分辨率、裁剪和数字变焦；标定后改变这些设置要重新标定。
- 启动会请求自动对焦；F 读取当前位置、关闭自动对焦并保持该位置。至少 8 帧、0.75 秒状态一致才显示 LOCKED。
- LOCKED 代表驱动状态验证，不保证图像光学清晰；保存前仍需目视确认。失败/漂移时保存被阻止。
- 更换相机可加 `--camera 1`；同一分辨率也不能复用另一台相机的内参。

采集物：原始 PNG、对应的逐帧 JSON（取帧前后焦点状态、时间、角点数量）、session.json。
每个新会话的照片必须有匹配的锁焦记录，且焦点目标一致，解算时会复核。
这些数据属于本地私人采集，`calibration/` 和 `outputs/` 均被 Git 忽略。

## 3. 解算内参与畸变

下面会话目录必须换成上一步实际输出路径：

```powershell
uv run --no-editable depth-calibrate solve --images "calibration/sessions/实际会话目录"
```

输出 `calibration/camera.json`，包含 K、畸变系数、图像尺寸、每帧重投影误差及接收/拒绝清单。
至少 12 个有效视角，整体 RMS ≤0.8 px 且单帧 RMS ≤1.5 px 才过软件门槛。
失败结果会保存以供分析，但后续推理拒绝使用；重试请指定新的 `--output calibration/camera-v2.json`。

**低重投影误差只是初筛**，不证明标定充分、不证明深度准确。还应检查覆盖和倾斜多样性，
用未参与解算的新板图做独立验证。角度单一、板弯曲、对焦改变会造成低误差但错误内参。
当前未实现自动留出集评估，不能将 quality_pass 当成机器人上线验收。

## 4. 用相同相机与模式生成米制预测

```powershell
.\run-metric-webcam.cmd --calibration calibration/camera.json
```

S 保存当前帧。图像先去畸变，再推理；程序用新内参 K' 对齐输出，边界无效深度标为 NaN。
输入分辨率必须和标定一致，不能静默缩放内参。默认模型处理尺寸 392；预测会恢复到采集尺寸。
显示的 model+prep FPS 只计预处理和模型推理，不包括采集/展示/存盘，不是系统端到端帧率。

若还没有标定，也可先运行 `.\run-metric-webcam.cmd` 观察预测，但其结果**不能导出点云**。
单图入口：

```powershell
uv run --no-editable depth-image --variant metric --input "自己的原始相机图.png" --calibration calibration/camera.json --output-dir outputs/my-metric-run
```

每次保存包含原始 RGB、去畸变 RGB、float32 深度 `.npy`、展示 PNG、metadata.json。
metadata 记录权重哈希、单位、K'、原始标定、分辨率和坐标空间。颜色图每帧 min/max 拉伸，
不能用于跨帧距离比较；请读取 `.npy`。相机时间是主机收到帧的时间，不是硬件曝光时间。
模型没有提供校准过的置信度，不会虚构一个 confidence 分数。

## 5. 导出相机坐标系点云

把路径换成上一步实际生成的 metadata 文件：

```powershell
uv run --no-editable depth-cloud --metadata "outputs/captures/实际文件_metric_metadata.json" --fit-plane
```

默认每 4 像素采样、只保留 0 < Z ≤5 m、1 cm 体素下采样，生成 PLY 和统计 JSON。
这些阈值都是**预测单位**，不是已验证的测量精度。可用 `--stride`、`--max-depth`、`--voxel-size` 修改。
Open3D 用于点云采样和可选 RANSAC 平面拟合；PLY 可在支持点云的查看器中打开。

计算：X=(u-cx)Z/fx，Y=(v-cy)Z/fy，Z=预测深度。
坐标轴为相机 OpenCV 约定：X 向右、Y 向下、Z 向前。**没有相机→桌面/世界外参**。
RANSAC 输出主平面方程、内点比例和拟合残差，不能自动认作“桌面”，也不能据此证明尺度准确。
若要世界坐标，下一阶段还需固定世界原点并估计 T_world_camera。

以下情况会拒绝：相对深度、缺标定、标定质量门槛失败、分辨率不匹配、内参与去畸变转换不匹配、
无有效点、试图覆盖已有结果。程序不会自动猜焦距或用相对数值冒充米制结果。

## 真正的阶段验收还缺什么

1. 用实物完成内参标定，并用额外板图检查几何一致性。
2. 在目标桌面工作距离范围，用**未用于调整参数**的已知平面距离或已知物体尺寸作独立对照。
   区分相机光轴 Z 与斜向直线距离；不能把尺量的斜距直接对比 Z。
3. 报告距离/尺寸绝对误差与相对误差、静止 30 秒深度抖动、遮挡边缘失真。
4. 若拟合尺度/偏移，单独保存修正参数和校准样本；在另一组距离验证，禁止同一组拟合又验收。
5. 通过以上后，再设计对象检测→稳定 ID→世界坐标/关系→运动状态的小闭环。

当前没有完成语义识别、多帧建图、物体背面重建或动作条件预测；这一阶段是世界模型的几何基础，
不是完整世界模型。所有预测暂不用于机械臂运动、避障或碰撞安全判断。

参考：[OpenCV ChArUco 标定](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html)、
[Open3D PointCloud API](https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)。
