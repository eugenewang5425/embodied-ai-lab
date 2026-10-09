# 🤖 Embodied-AI Learning Lab

From GIS and remote sensing to robot perception, control and learning — small experiments with reproducible evidence and retained failures.

我以 GIS、遥感和空间感知为基础，通过本地仿真学习控制、三维感知、建图、导航与机器人学习。现在按基础学习阶段和研究方向组织，每次实验只回答一个可检验问题。[学习路线](docs/01-learning-roadmap.md)说明知识依赖；[文档与实验标准](docs/documentation-standard.md)规定讲义、对照和验收方式。

[阶段与编号索引](#课程索引) · [当前研究证据](#current-evidence) · [3 分钟巡检讲解](#microdinosaur-patrol) · [运行与回放](docs/running-experiments.md) · [完整图册](docs/experiment-gallery.md) · [论文与官方来源](docs/references.md)

<a id="current-evidence"></a>

## 当前研究证据（2026-10-09）

| 方向 | 本轮结论 | 适用范围与未通过项 | 证据 |
| --- | --- | --- | --- |
| DINO · S288 感知巡检 | 配对整圈两组均 8/8 点、99.60 m；固定障碍增加深度后停车指令提前 80 ms，正常整圈动作未变 | 保留天空与侧墙误停；已知初始位姿、平路和彩色标记。硬件安装/外参/部分延迟仍是假设，未实机部署 | [小结与原始对照](docs/microdinosaur-patrol-review-20261009.md) |
| NAV · 身体约束导航 | 历史 69 课 R12–R14：新验证 60/60 到达停稳、零接触，3 个无路入口正确拒绝 | 精确位姿、同步低速静态仿真；计算 P95 超过 100 ms 周期。旧跨场景自建图失败及分离平台边界保留 | [本轮报告](docs/69-physical-height-navigation.md) · [阶段复盘](docs/navigation-stage-review-69-71.md) |
| MAN · 三维接触操作 | 历史 73 课 R03：同 27 个新条件，指面校准 21/27 → 24/27，3 例救回、无退步 | 单独接触门控 11/27，组合 23/27；仍有 3 次掉落。已知初始位置，RGB 尚未参与控制，视觉阶段入口未通过 | [配对与失败](docs/73-so101-contact-feedback.md) |

这些是不同平台、不同任务的限定仿真证据，不合并成“项目总成功率”。物理重放、算法重放、软件测试、真实窗口和硬件验收分别报告；停止或没有接触也不等于巡检/抓取成功。

<a id="microdinosaur-patrol"></a>

## S288 巡检：3 分钟技术讲解

[![原 S288 动作、RGB 与 ToF、单目深度辅助的技术讲解](docs/img/microdinosaur-patrol-explainer.jpg)](https://www.bilibili.com/video/BV1Mtp46EEw9/)

[观看星瞳配音成片](https://www.bilibili.com/video/BV1Mtp46EEw9/) · [RGB＋ToF 配置与基准](docs/microdinosaur-campus-patrol.md) · [深度定尺、闭环收益及两次误停](docs/microdinosaur-patrol-depth.md)

保留原 19 个 S288、刚性尾巴和 50 Hz V07 ONNX 策略。国产品牌微雪 IMX219-77 为相机候选，规定 ToF 为 VL53L5CX（8×8、15 Hz、4 m）；UniDepth V2 Small 经 ToF 定尺与质量检查后增加辅助停车条件。算法和硬件适配、时序、未知处理与实机限制在专项报告展开，历史盲走与新园区不是同场景配对。[模型、声音和硬件来源](docs/references.md#assets)分别适用各自许可。

<a id="课程索引"></a>

## 学习阶段与研究方向

| 基础阶段 | 历史课程 | 核心内容 |
| --- | --- | --- |
| F1 | 1–7 | 状态、PD/LQR、扰动、测量与摆起 |
| F2 | 8–13 | FK/IK、Jacobian、路径、速度与力矩 |
| F3 | 14–21 | 坐标、里程计、标定、地标、ROS 2 与反馈 |
| F4 | 22–27 | 投影、单目定尺、内参、ICP 与视觉接地 |
| F5 | 28–42 | BC、PPO/SAC、示教、残差与块策略 |
| F6 | 43–56 | 栅格、规划、PF、回环、图优化与地图质量 |

基础讲义已形成实验记录，目标能力是否通过见每份讲义；学习掌握另见[理解检查](docs/learning-checklist.md)。

| 研究方向 | 实验入口 | 下一阶段的关键条件 |
| --- | --- | --- |
| NAV · 定位、建图与身体约束导航 | [NAV-01–10，历史基准 62 与课程 63–71](docs/experiment-index.md#nav) | [异步预算、位姿误差、连续 RGB-D 与融合](docs/multisensor-navigation-roadmap.md) |
| MAN · 三维接触操作 | [MAN-01–04，历史课程 72–73](docs/experiment-index.md#man) | [先验证动态接触，再进入视觉、恢复、学习](docs/manipulation-roadmap.md) |
| DINO · S288 感知巡检 | [DINO-01–02](docs/experiment-index.md#dino) | [标定与新布局配对；实物能力单独验收](docs/microdinosaur-patrol-review-20261009.md#下一项实验) |

课号、文档号、方向实验和轮次分开。文档 57–61 对应课 52–56，62 是基准；63–73 保留历史身份，新工作先按方向和问题登记，不自动续排课号。[完整讲义与编号对照](docs/experiment-index.md) · [讲义标准](docs/documentation-standard.md#numbering) · [历史学习快照](docs/learning-history.md)

<a id="lesson69-rounds"></a>

[第 69 课全部轮次（NAV-08 R01–R14）](docs/experiment-index.md#lesson69-rounds)保留旧入口。

<a id="latest-demos"></a>
<a id="quick-start"></a>
<a id="cn"></a>

## 最小启动与最新回放

```powershell
uv sync --locked
uv run python -m embodied_learning.env_check --steps 300 --seed 7
uv run python -m embodied_learning.viewer --policy pd --seconds 10 --seed 7
```

[最新导航/机械臂生成与回放命令](docs/running-experiments.md#latest) · [逐实验命令存档](docs/running-experiments.md#commands) · [窗口接口与同步要求](docs/replay-window.md)

主环境使用 Python 3.12、uv、MuJoCo 与 Gymnasium；作者的 ROS 2 Jazzy / Gazebo Harmonic 运行于 WSL2。安装与版本见[环境审计](docs/00-environment-audit.md)，本次没有升级依赖。`results/` 不入 Git，回放需先生成对应记录。DINO 还依赖本地机器人源资产与独立模型环境，公开报告不能代替完整执行环境。

## 测试流程

先运行相关快速测试，再按研究方向选择受影响的测试文件；随后按改动范围选择 `record` / `gui` / `training`。共享模块变更追加实际使用方，影响范围无法界定或阶段验收时运行全仓 `full`。当前公开入口接受测试文件参数；本机分方向自动选择入口尚未同步，公开命令不依赖它。

`quick` 排除窗口、完整记录与训练重放；`record`、`gui`、`training` 分别覆盖这三类，`slow` 是合集，`full` 覆盖所选文件的全部层。不同批次分别报告，快测不代替实验数据和窗口验收。

```powershell
uv run python scripts/test.py quick tests/test_session55_rgbd_loops.py tests/test_session56_map_localization.py tests/test_navigation_benchmark.py
uv run python scripts/test.py quick
uv run python scripts/test.py record tests/test_session55_rgbd_loops.py tests/test_session56_map_localization.py
uv run python scripts/test.py gui tests/test_session47_pose_graph.py
uv run python scripts/test.py training tests/test_session40_rl_arm_reaching.py
uv run python scripts/test.py full tests/test_session55_rgbd_loops.py  # 单课所有测试
uv run python scripts/test.py full          # 共用基础模块变更或阶段验收
uv run python scripts/test.py slow          # 三类慢测合集
uv run ruff check src tests scripts docs/img/make_readme_figures.py
```

`quick` 通过只说明快测层通过；实验数值、图表和本机窗口仍按各课记录核对。慢速标记及自动分层规则见 [tests/conftest.py](tests/conftest.py)。

2026-09-24 全仓 `slow` 实测 176 项通过、耗时 3657 秒。其中第 53 课 SciPy 生产求解器的记录准备阶段为 1785 秒，约占总耗时一半；该次运行另有 11 条警告，未造成失败。

## 文档维护与研究来源

[统一讲义标准](docs/documentation-standard.md)规定变量、对照、失败、图文、3–5 道思考题、复现和验收；[文献目录](docs/references.md)映射原始来源与本地适配，每份实验讲义就近补引。Guide 作知识地图，原论文和官方实现作机制来源，引用不等于完整复现。

[实验图册](docs/experiment-gallery.md)保留原主页全部逐课图文；[决策日志](docs/34-experiment-decision-log.md)只追加；[本次全面审查](docs/reviews/2026-10-09-homepage-and-documentation-audit.md)记录去重、编号、引用覆盖、检查方法和剩余限制。

最近一次**历史全仓代码验收**仍是 2026-10-05：1134 通过、1 项旧 RGB 断言失败；修正后 12 项回放、936 项快速层与 20 项抓取分别通过，没有重跑一次全仓 full。[原始交付证据](docs/benchmarks/so101-contact-v4-delivery.json)。本次是文档整理，不能把上述批次写成新的全量全绿成绩。
