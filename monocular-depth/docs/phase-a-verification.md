# A 阶段软件验收记录 — 2026-09-02

结论：标定工具、米制推理和带门槛的相机坐标系点云导出已实现并通过软件验证。
**真实相机标定、现场距离精度、实时 GUI 和物理打印尺寸尚未验收，不是完整世界模型。**

## 环境和来源

- 路径：`D:\项目\具身人工智能\monocular-depth`。
- Python 3.11.15；PyTorch 2.11.0+cu128；CUDA 12.8；OpenCV 4.14.0；Open3D 0.19.0。
- NVIDIA GeForce RTX 5070 Laptop GPU，8.0 GiB。
- 本次复用原独立 `.venv` 的 PyTorch，新增 Open3D 依赖及米制权重；本包升级到 0.2.0。
- 上游 `git rev-parse HEAD`：`a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`。
- 米制权重：[官方 Hypersim Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small)。
- revision：`3bc65d4e14a6786a61acec16453c50e12bf5f338`。
- 下载文件大小：99,222,290 bytes，约 94.6 MiB。
- 文件 SHA256 实测：`b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545`。
- 该哈希与固定 revision 的官方 LFS 元数据一致，下载后和加载时均强制检查。

## 验证结果

| 检查 | 实际结果 | 解释边界 |
| --- | --- | --- |
| `uv run --no-editable pytest -q` | 17 passed | 包含原有 4 项回归测试 |
| `uv run --no-editable ruff check .` | All checks passed | 不扫描第三方源码 |
| `uv pip check` | 91 packages，All installed packages are compatible | 已补齐 Windows wheel 声明的依赖 |
| 米制 / 相对模型各自 GPU 推理 | 成功 | 官方样例，两类权重独立 |
| 同进程加载两类模型并推理 | 成功，类和模块不同 | 检查 metric Sigmoid 头与 relative 头没有混用 |
| ChArUco 生成图检测 | 检出 24 角点 | 已目视检查 PNG；实物打印尚待尺量 |
| 30 个投影观测标定 | 内参恢复满足测试容差 | 加噪合成数据，不是实机精度 |
| 24 张多姿态渲染图→检测→解算 | 通过，至少 20 有效视角 | 全链路合成测试，在 pytest 临时目录 |
| 合成平面→PLY→Open3D 读取 | 通过，3072 点、平面内点率 1 | 不是实际桌面点云 |
| 米制样例无标定→点云 | 拒绝，退出码 2 | 未猜测内参 |
| 相对深度→点云 | 拒绝，退出码 2 | 未将相对值当米 |
| 单张生成板图→相机标定 | 拒绝，退出码 2 | 不能用单张数字棋盘冒充实机标定 |
| 重复帧、错误分辨率、错误 K'、覆盖已有输出 | 测试中拒绝 | 避免常见静默失误 |
| webcam / collect / cloud CLI 帮助 | 均可运行 | 本次未打开真实摄像头或交互窗口 |

非可编辑包在源码变更后已重新构建安装；pytest 配置显式指向 src，防止测试旧安装副本。
日志中的 `xFormers not available` 是上游提示，未阻断本机 CUDA 推理。
收尾检查发现 Open3D Windows wheel 声明了 `ipywidgets>=8.0.4`，但跨平台锁元数据未带入；
已将 Windows 专用依赖显式加入 pyproject，安装 ipywidgets 8.1.9 后重新同步并复验。
最终 pytest 17 passed（3.24 s），Ruff 和依赖一致性检查均通过。

## 真实推理产物

都位于 `outputs/phase-a-20260902/`，未覆盖原相对深度样例或摄像头采集。

- `official-metric/demo01_metric_*`：官方样例米制预测、RGB、对比图与元数据。
- `official-relative/demo01_relative_*`：相同样例相对深度回归产物。
- `saved-camera-metric/capture_20260824_034153_utc_rgb_metric_*`：复用 8 月 24 日已保存 RGB 的米制推理，**不是本次实时采集**。
- `dual-model-smoke.json`：同进程 GPU 双模型隔离检查结果。

官方样例输出分辨率 2048×1362，input-size=392。同进程检查相对值范围约 0–8.482，
米制预测范围约 3.587–19.661 m。两者不可直接比较数值；原样例是室外街景，
室内 Hypersim 权重在它上的结果只作运行冒烟，不评价准确率。

分别启动 CLI 的冷帧预处理+推理用时：metric 367.1 ms、relative 369.0 ms，
复用旧相机 RGB 的 metric 316.7 ms。这不是稳态 FPS 或端到端延迟基准。

当前产物均明确记录 `calibrated=false`、`metric_accuracy_validated=false`、`world_pose=null`。
**没有生成真实 `calibration/camera.json`，没有导出伪装成已标定的现场点云。**

## 使用与剩余工作

打印入口：`calibration/board/print.html`；采集入口：`capture-calibration.cmd`。
完整流程见 [phase-a.md](phase-a.md)。

下一步需要用户准备并尺量实物板，采集不同姿态图像；完成内参解算后，再用独立距离/尺寸真值
衡量米制误差、静止抖动和失效区域。通过之前，不连接机械臂或用于避障。

本次没有提交或推送 Git，没有启动后台服务，没有改动上游模型源码。
上层 MuJoCo 项目的 pyproject / lock 未作为本任务编辑对象；以下是本次收尾两次读取一致的当前哈希
（不是把历史快照当作开工基线）：

- 上层 pyproject.toml：`5a4ad37868bbc02a925e8544f2731a9eb857c81e6ca5242719bac8a4be44ff5c`。
- 上层 uv.lock：`6fb91329082f6f62a88c49f2809d682c2c058ce00a4848e2149f23efc238b376`。
