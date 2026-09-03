# 本机环境审计

初始审计日期：2026-08-09。机器人运行环境于 2026-09-03 补充核验；下方硬件、磁盘空间及无关工具仍是初始快照，不当作当前状态。

## 硬件与系统

| 项目 | 检测结果 |
| --- | --- |
| 操作系统 | Windows 11 家庭中文版，64 位 |
| 处理器 | Intel Core Ultra 9 275HX，24 核 / 24 逻辑处理器 |
| 内存 | 约 32GB |
| 显卡 | NVIDIA GeForce RTX 5070 Laptop GPU |
| 显存 | 8151 MiB，约 8GB |
| NVIDIA 驱动 | 610.88 |
| D 盘剩余空间 | 约 352.6GB |

## 开发工具

| 工具 | 状态 |
| --- | --- |
| Git | 已安装，2.53.0.windows.3 |
| Python | 已由 uv 安装并固定为 3.12.13 |
| uv | 已安装，0.12.3；用户级路径为 `C:\Users\eugen\.local\bin` |
| 项目虚拟环境 | 已创建于 `.venv` |
| Gymnasium | 已安装并锁定，1.3.0 |
| MuJoCo | 已安装并锁定，3.11.0 |
| Conda / Mamba | 未安装；本项目当前不需要 |
| CMake / GCC / G++ | 未安装 |
| Docker | 未安装 |
| WSL | 2026-09-03 核验：WSL2，已有 `Ubuntu-24.04`（Ubuntu 24.04.4） |
| ROS 2 / Gazebo | 2026-09-03 核验：已有 ROS 2 Jazzy / Gazebo Sim 8.15.0；本课实际运行 rclpy 节点与 tf2，不启动 Gazebo 世界 |
| WSL Python / NumPy | 3.12.3 / 1.26.4，与 Windows 项目虚拟环境分离 |

## 决策

### 第一阶段：MuJoCo + Gymnasium

MuJoCo 官方提供 Windows 预编译支持，Python 包可以直接通过 PyPI 安装，并自带 MuJoCo 库，不需要先手工构建物理引擎。它适合作为当前机器上的最小可靠起点。

### 第二阶段：WSL2 + ROS 2 Jazzy + Gazebo Harmonic

独立 Linux 机器人环境已存在，本轮没有重新安装。第二十课在 WSL2 中运行真实 ROS 节点、消息与 TF，Windows 只负责导出测量和只读教学回放。相关入口和验收见[第二十课](22-session-20-ros2-messages-and-tf.md)。Gazebo 虽已安装，但本课没有增加物理环境或传感器仿真。

### 暂缓：Isaac Lab 完整工作流

当前内存达到官方基础建议，但 8GB 显存低于完整 Isaac Lab 工作流建议的 16GB。它不适合作为初始环境，避免把时间消耗在显存不足与复杂安装问题上。

## 已完成的环境安装

1. 安装用户级 `uv`；
2. 由 `uv` 管理项目专用 Python 3.12；
3. 在项目目录创建 `.venv`；
4. 安装并锁定 MuJoCo、Gymnasium、NumPy、Matplotlib 和测试工具；
5. 完成 headless、图形窗口、自动测试和代码检查。

## 中文路径兼容说明

MuJoCo 在 Windows 上不能直接用原生路径加载当前中文项目目录下的 MJCF 文件。项目中的 `UnicodeSafeInvertedPendulumEnv` 先由 Python 读取 XML 内容，再通过 `MjModel.from_xml_string` 加载模型，因此无需重命名或移动项目目录。该兼容方案已通过自动测试和实际渲染验证。
