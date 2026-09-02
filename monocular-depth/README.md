# 普通摄像头单目深度实验

这是一个与上层 MuJoCo 项目隔离的本地实验环境。第一阶段使用
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) Small (`vits`)
完成普通 RGB 摄像头的相对深度估计。现已加入第二步基础工具：
ChArUco 标定、室内米制深度预测、带单位与内参的结果记录、相机坐标系点云导出。

下一步请看 [标定与米制点云操作指南](docs/phase-a.md)；
本次软件验收见 [2026-09-02 验收记录](docs/phase-a-verification.md)。

## 当前边界

- 默认 `--variant relative` 的输出仍是**相对深度**，不能解释为米。
- `--variant metric` 使用独立的 Hypersim Small 权重，输出**米制预测**，尚未实测其距离误差。
- 点云要求真实标定、匹配分辨率、去畸变图像；未标定时明确拒绝导出。
- 点云只在相机坐标系，尚无世界坐标、物体跟踪、遮挡记忆或动力学预测。
- 不直接把预测结果接入机械臂控制或碰撞安全逻辑。

## 已固定的环境

- Python 3.11
- Windows + NVIDIA CUDA 12.8 PyTorch wheel
- Depth Anything V2 Small
- 上游源码提交：`a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`

上游源码位于 `third_party/Depth-Anything-V2`，模型权重位于 `checkpoints`；两者均可按本说明重建，
不会写入上层项目的 `.venv`。

## 重建环境

在本目录打开 PowerShell：

```powershell
git clone https://github.com/DepthAnything/Depth-Anything-V2.git third_party/Depth-Anything-V2
git -C third_party/Depth-Anything-V2 checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
uv sync --no-editable
uv run --no-editable depth-download
uv run --no-editable depth-env-check
```

已有上游源码时，也可以直接运行 `setup.cmd`。这里使用非可编辑安装，是为了规避部分 Windows Python
在中文工作路径下无法加载 editable `.pth` 的问题。

修改本包源码后需要执行 `uv sync --no-editable --reinstall-package monocular-depth-lab`，
否则 CLI 可能仍执行已安装的旧副本。此操作只重装本地小包，不重装 PyTorch。
新增米制模式可运行 `setup-metric.cmd`，复用同一个 `.venv`。

## 单图推理

```powershell
uv run --no-editable depth-image --input .\照片.jpg
```

输出到 `outputs/images`：

- `*_comparison.png`：RGB 与深度伪彩图并排图；
- `*_depth.npy`：未经着色的浮点深度数组，单位以 `*_metadata.json` 为准；
- `*_depth.png`：归一化后的灰度深度图。
- `*_metadata.json`：模型、权重哈希、单位、图像尺寸、标定与坐标空间信息。

新输出文件名带 `_relative` / `_metric` 区分，已有结果不会覆盖。
可视化 PNG 每帧单独拉伸颜色，不可用来量距离。

## 普通摄像头实时推理

```powershell
uv run --no-editable depth-webcam
```

也可以双击或在 PowerShell 中运行 `run-webcam.cmd`。

窗口快捷键：

- `Q` 或 `Esc`：退出；
- `S`：保存当前 RGB、深度数组和对比图到 `outputs/captures`。

如默认摄像头不是目标设备，可尝试：

```powershell
uv run --no-editable depth-webcam --camera 1
```

降低 `--input-size` 可以提高速度，默认值 `518` 与官方仓库一致：

```powershell
uv run --no-editable depth-webcam --input-size 392
```

不打开窗口的摄像头冒烟测试：

```powershell
uv run --no-editable depth-webcam --headless --max-frames 5 --save-last --input-size 392
```

## 验收命令

```powershell
uv run --no-editable pytest -q
uv run --no-editable ruff check .
uv run --no-editable depth-env-check
```
