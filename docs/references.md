# 原始来源、官方实现与本地采用关系

2026-10-09 更新，覆盖基础课程和 NAV / MAN / DINO 的实验讲义。这里记录机制出处与采用关系，不搬用论文成功率，也不把所有参考算法都写成已实现。书籍为作者来源，软件/硬件为官方项目或厂商；论文优先原作者预印本、正式出版页或作者存档。

Guide 是知识地图；代码、模型、音色和资产各自许可，主仓许可证不覆盖第三方权重。[讲义标准](documentation-standard.md#citations) · [编号索引](experiment-index.md) · [本轮审查与访问限制](reviews/2026-10-09-homepage-and-documentation-audit.md)

## 阅读与实现状态怎样区分

- **本地实现/简化适配**：讲义定义本机任务、输入权限、公式和取舍，再与原机制比较。
- **实际调用**：报告具体模型/权重/版本、运行记录与时序；官方项目存在不代表本机已跑过。
- **参考或候选**：用于理论解释或后续计划，明确未训练、未闭环或试跑中止。
- **实物规范**：厂商范围约束模拟；真实安装、环境退化和状态输出另行标定。

## 基础与控制

| 来源 | 作者 / 年份 | 本项目采用关系 |
| --- | --- | --- |
| <a id="mujoco"></a>[MuJoCo 官方文档](https://mujoco.readthedocs.io/en/stable/overview.html) | 维护团队，持续更新 | 主仿真器；具体版本由各批环境快照决定，不以 stable 页面当锁定版本 |
| <a id="gym"></a>[Gymnasium 官方接口说明](https://gymnasium.farama.org/introduction/basic_usage/) | Farama，持续更新 | 使用 reset/step 与终止契约；奖励不是任务成功判据 |
| <a id="lqr"></a>[Underactuated Robotics · LQR](https://underactuated.mit.edu/lqr.html) | Russ Tedrake，作者教材 | 本地离散 LQR、代价与反馈对照；噪声/扰动/摆起还需各课判据 |
| <a id="robotics"></a>[Modern Robotics: Mechanics, Planning, and Control](https://github.com/NxRLab/ModernRobotics) | Kevin Lynch / Frank Park，2017 | FK/IK/Jacobian/坐标与动力学的教材来源；平面 2R 与受限 SO-101 为本地实现，不称直接运行全书库 |

## 感知与导航

| 来源 | 作者 / 年份 | 本项目采用关系 |
| --- | --- | --- |
| <a id="prob"></a>[Probabilistic Robotics](https://robots.stanford.edu/probabilistic-robotics/) | Sebastian Thrun / Wolfram Burgard / Dieter Fox，2005 | 运动/观测模型、概率定位与占据图的教材来源；简化地标重置不是完整 EKF/SLAM |
| <a id="ros"></a>[ROS 2 geometry2 / tf2](https://github.com/ros2/geometry2) | ROS 2 维护团队 | 消息时间戳与坐标变换；本机为 Jazzy，不混用 rolling 默认参数 |
| <a id="zhang"></a>[A Flexible New Technique for Camera Calibration](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/) | Zhengyou Zhang，2000（技术报告1998） | 合成棋盘格、单应与闭式内参实验；不等于真实镜头已完成标定 |
| <a id="pnp"></a>[OpenCV · PnP pose computation](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html) | OpenCV 官方文档 | 三维/像素对应和姿态方向参考；不把相机显示或几何回放写成已经实现 PnP 导航 |
| <a id="icp"></a>[Open3D · ICP registration](https://open3d.org/docs/release/tutorial/pipelines/icp_registration.html) | Open3D 官方实现文档 | 点到点/点到面机制与原文线索；第26课为本地带噪对照，未宣称调用 Open3D 全流程 |
| <a id="astar"></a>[A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/astar.pdf) | Peter Hart / Nils Nilsson / Bertram Raphael，1968 | 本地 A* 栅格/状态搜索；离散最短路径不证明身体、速度、制动可执行 |
| <a id="pursuit"></a>[Implementation of the Pure Pursuit Path Tracking Algorithm](https://publications.ri.cmu.edu/implementation-of-the-pure-pursuit-path-tracking-algorithm) | R. Craig Coulter，CMU技术报告1992 | 本地差速适配纯追踪；不称运行 Nav2 Regulated Pure Pursuit |
| <a id="switch"></a>[Switchable Constraints for Robust Pose Graph SLAM](https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf) | Niko Sünderhauf / Peter Protzel，IROS2012 | 本地开关因子与λ扫描，绝对/相对边问题分开；阴性结果不能归为原文全方法失败 |
| <a id="mixture"></a>[Inference on Networks of Mixtures for Robust Robot Mapping](https://april.eecs.umich.edu/papers/details.php?name=olson2012rss) | Edwin Olson / Pratik Agarwal，RSS2012 | 本地 max-mixture/硬组件选择实验；死锁与退火阴性保留 |
| <a id="scipy"></a>[SciPy · optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) | SciPy 官方文档 | 第53课实际生产求解器同题对照；loss/尺度与固定记录匹配，在线最新版不代表本机版本 |
| <a id="amcl"></a>[Nav2 Jazzy · AMCL 源码与参数](https://api.nav2.org/nav2-jazzy/html/amcl__node_8cpp_source.html) | Nav2 维护团队，Jazzy | 第64课实际官方 AMCL 回放，参数回读有效；无效参数旧结论已撤回，不当作当前对照 |

## 深度与视觉模型

| 来源 | 作者 / 年份 | 本项目采用关系 |
| --- | --- | --- |
| <a id="da2"></a>[Depth Anything V2（含 Metric Depth）](https://github.com/DepthAnything/Depth-Anything-V2) | Lihe Yang 等，2024 | 第24课实际相对深度仿射检验；DINO 对照实际 Metric Hypersim Small，区分权重与任务，不互借精度 |
| <a id="sam"></a>[MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | Chaoning Zhang 等，2023 | 第27课实际掩码/身份关联；分割成功不等于三维位置或定位正确 |
| <a id="uni"></a>[UniDepth / UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth) | Luigi Piccinelli 等，2024/2025 | DINO 实际 UniDepth V2 Small；ToF定尺、测量确认与时效是本地扩展，不是原模型自带的安全保证 |
| <a id="depthpro"></a>[Depth Pro](https://github.com/apple-aiml-research/ml-depth-pro) | Apple 研究团队，2024 | DINO 中仅尝试并在少量帧后中止，不能列为完成的闭环算法或整批比较 |

## 机器人学习

| 来源 | 作者 / 年份 | 本项目采用关系 |
| --- | --- | --- |
| <a id="ppo"></a>[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | John Schulman 等，2017 | numpy PPO/clip 本地实验；小预算失败不外推为算法普遍无效 |
| <a id="gae"></a>[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) | John Schulman 等，2015预印本/ICLR2016 | PPO 优势估计来源；实际折扣、GAE参数和训练范围以讲义为准 |
| <a id="sac"></a>[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) | Tuomas Haarnoja 等，2018 | numpy SAC、回放、双Q与熵的本地实现；不同任务/预算分别报告 |
| <a id="residual"></a>[Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201) | Tobias Johannink 等，2018预印本/ICRA2019 | 借鉴底座+学习残差；限幅、保护性与增值性各自检验，未复现原机器人任务 |
| <a id="pbrs"></a>[Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf) | Andrew Ng / Daishi Harada / Stuart Russell，1999 | 使用 γΦ(s′)−Φ(s) 势函数形式；策略不变性有 MDP/终止等前提，不保证有限训练成功 |
| <a id="dapg"></a>[Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations](https://arxiv.org/abs/1709.10087) | Aravind Rajeswaran 等，2017预印本/RSS2018 | 第32课为 DAPG 式 PPO+示教/BC正则适配，未使用原文完整 NPG 求解和灵巧手系统 |
| <a id="goexplore"></a>[First return, then explore](https://arxiv.org/abs/2004.12919) | Adrien Ecoffet 等，2020预印本/Nature2021 | 第33课借鉴档案探索与返回；仿真状态重置、后续 BC 失败，未复现原文完整鲁棒化 |
| <a id="twophase"></a>[Balance Controller Design for Inverted Pendulum Considering Detail Reward Function and Two-Phase Learning Protocol](https://www.mdpi.com/2073-8994/16/9/1227) | Symmetry 16(9):1227，2024 | 第34课借鉴分阶段奖励/学习思想，自身任务与预算另报；本轮确认题名和出版页，全文重读遇429，未作新增论文结果解释 |
| <a id="dagger"></a>[A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html) | Stéphane Ross / Geoffrey Gordon / Drew Bagnell，2011 | 第36课实际在线教师标注/数据聚合；成功判据和本文理论条件分别看 |
| <a id="act"></a>[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) | Tony Z. Zhao 等，2023 | 第37课最小多峰块策略、第42课 Transformer块策略为机制适配；不是原文完整视觉/CVAE双臂系统 |
| <a id="attention"></a>[Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Ashish Vaswani 等，2017 | 第42课 Transformer 表示来源；架构使用不代表原论文任务/性能复现 |
| <a id="diffusion"></a>[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137) | Cheng Chi 等，2023 | 只作多峰动作与视觉学习的比较参考；本仓当前没有完成 Diffusion Policy 训练/闭环 |

<a id="assets"></a>

## 机器人与硬件资产

| 来源 | 作者 / 年份 | 本项目采用关系 |
| --- | --- | --- |
| <a id="manip"></a>[Robotic Manipulation · 作者教材](https://manipulation.mit.edu/intro.html) | Russ Tedrake，持续更新 | 接触、支持和操作任务背景；本地接触闭环需看 MuJoCo 配置与原始物理记录 |
| <a id="so101"></a>[SO-ARM100 / SO-101](https://github.com/TheRobotStudio/SO-ARM100) | TheRobotStudio 官方项目 | 开源机械结构参考；采购、实机标定和硬件性能未在本轮验收 |
| <a id="menagerie"></a>[MuJoCo Menagerie · SO-101](https://github.com/google-deepmind/mujoco_menagerie/tree/4d038b3feae26ec82b46a4d586379114012a8ac7/robotstudio_so101) | Google DeepMind / 模型贡献者 | 本地使用固定提交模型，资产许可单独核对，不默认切换最新模型 |
| <a id="camera"></a>[IMX219-77 Camera](https://www.waveshare.com/wiki/IMX219-77_Camera) | Waveshare / 微雪，产品规范 | 国产品牌候选，Sony感光芯片；外形/光学规格与模拟模式、外参及时延假设分开 |
| <a id="tof"></a>[VL53L5CX Datasheet](https://www.st.com/resource/en/datasheet/vl53l5cx.pdf) | STMicroelectronics，产品规范 | 规定 ToF 的8×8等硬件依据；中心射线模拟不代表真实区域混合回波/日光/反射与状态码 |
| <a id="tts"></a>[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | RVC-Boss 与贡献者 | 实际本地星瞳配音工具，代码许可与音色/声音权利分开 |
| <a id="voice"></a>[parrots-gpt-sovits-speaker · 星瞳来源集合](https://huggingface.co/shibing624/parrots-gpt-sovits-speaker) | shibing624；星瞳模型作者 XzJosh | 本次音色模型来源与署名见巡检小结；不因主仓许可而改成可商用 |

## 后续参考与知识地图

| 来源 | 作者 / 年份 | 本项目采用关系 |
| --- | --- | --- |
| <a id="lerobot"></a>[LeRobot](https://github.com/huggingface/lerobot) | Hugging Face 与贡献者 | 后续示教/视觉学习候选；尚未完成本项目SO-101视觉训练/实机部署 |
| <a id="guide"></a>[Embodied-AI-Guide](https://github.com/TianxingChen/Embodied-AI-Guide) | Tianxing Chen 与贡献者 | 知识地图，不作为 PPO/ACT/SLAM 等算法原始出处；历史章节序号保留在旧快照 |

## 讲义到原始来源的覆盖表

每行注明文件类型；计划、诊断和媒体小结都不计作新课程，也不代表实验完成。所有来源已在讲义补引段注明采用关系。部分教材覆盖多个基本机制，这不表示它们是每个工程门控的原始论文。

| 文件类型 | 讲义 / 计划 | 本轮补引 |
| --- | --- | --- |
| 实验讲义或基准  [第一次实验：随机动作为什么控制不了倒立摆？](03-session-01.md) | [mujoco](#mujoco) · [gym](#gym) |
| 实验讲义或基准  [第二次实验：PD 反馈控制](04-session-02-pd-control.md) | [lqr](#lqr) · [mujoco](#mujoco) |
| 实验讲义或基准  [第三课：杆扶住了，车为什么还要管？](05-session-03-lqr.md) | [lqr](#lqr) · [mujoco](#mujoco) |
| 实验讲义或基准  [第四课：慢下来，看清 R 怎样影响动作](06-session-04-lqr-weights-and-demo.md) | [lqr](#lqr) · [mujoco](#mujoco) |
| 实验讲义或基准  [第五课：被推一下之后——从稳定到抗扰动](07-session-05-disturbance.md) | [lqr](#lqr) · [mujoco](#mujoco) |
| 实验讲义或基准  [第六课：看错了，也可能真的动起来](08-session-06-measurement-noise.md) | [lqr](#lqr) · [mujoco](#mujoco) |
| 实验讲义或基准  [第七课：倒下以后，为什么要先摆起来，再扶稳？](09-session-07-swingup.md) | [lqr](#lqr) · [mujoco](#mujoco) |
| 实验讲义或基准  [第八课：两个关节角，怎样决定机械臂末端的位置？](10-session-08-planar-arm.md) | [robotics](#robotics) |
| 实验讲义或基准  [第九课：到达终点，不等于沿直线到达](11-session-09-jacobian-path.md) | [robotics](#robotics) |
| 实验讲义或基准  [第十课：换一批路径，它还可靠吗？](12-session-10-path-coverage.md) | [robotics](#robotics) |
| 实验讲义或基准  [第十一课：换参考算法，不换电机](13-session-11-waypoint-ik.md) | [robotics](#robotics) |
| 实验讲义或基准  [第十二课：能走这条路，不等于能按这个速度走](14-session-12-timing-and-torque.md) | [robotics](#robotics) |
| 实验讲义或基准  [第十三课：提前出力，再修正误差](15-session-13-model-feedforward.md) | [robotics](#robotics) |
| 实验讲义或基准  [第十四课：小车转了，地图里的方向变了吗？](16-session-14-mobile-frames.md) | [robotics](#robotics) · [prob](#prob) |
| 实验讲义或基准  [第十五课：小车走直线，里程计却觉得它在转弯](17-session-15-encoder-odometry.md) | [robotics](#robotics) · [prob](#prob) |
| 实验讲义或基准  [第十六课：先校准尺子，再用其他路线验收](18-session-16-encoder-calibration.md) | [robotics](#robotics) · [prob](#prob) |
| 实验讲义或基准  [第十七课：标定之后，为什么每次还是不一样？](19-session-17-random-noise.md) | [robotics](#robotics) · [prob](#prob) |
| 实验讲义或基准  [第十八课：已知地标（控制点）观测与里程计对照](20-session-18-landmark-observations.md) | [robotics](#robotics) · [prob](#prob) |
| 实验讲义或基准  [第十九课：观测怎样解出位置，又怎样与里程计融合？](21-session-19-landmark-fusion.md) | [robotics](#robotics) · [prob](#prob) |
| 实验讲义或基准  [第二十课：先看小车怎样走，再看定位程序怎样合作](22-session-20-ros2-messages-and-tf.md) | [ros](#ros) |
| 实验讲义或基准  [第二十一课：它说到了，小车真的停到目标了吗？](23-session-21-goal-feedback.md) | [prob](#prob) · [pursuit](#pursuit) |
| 实验讲义或基准  [第二十一课补充：停车门限越小，小车就越准吗？](23a-session-21-stopping-tolerance.md) | [prob](#prob) · [pursuit](#pursuit) |
| 实验讲义或基准  [第二十二课：从三维世界到像素，再回到点云](24-session-22-pinhole-projection.md) | [robotics](#robotics) · [pnp](#pnp) |
| 实验讲义或基准  [第二十三课：单目相对深度 ↔ 米制尺度标定](25-session-23-monocular-metric.md) | [da2](#da2) |
| 实验讲义或基准  [第二十四课：真实单目相对深度的仿射检验](28-session-24-real-depth-affine.md) | [da2](#da2) |
| 实验讲义或基准  [第二十五课：合成棋盘格张氏标定——估计相机内参 K](29-session-25-camera-intrinsics.md) | [zhang](#zhang) · [pnp](#pnp) |
| 实验讲义或基准  [第二十六课：两帧带噪点云的 ICP 配准](30-session-26-icp-registration.md) | [icp](#icp) |
| 实验讲义或基准  [第二十七课：视觉基础模型给地标"发身份"](31-session-27-visual-grounding.md) | [sam](#sam) · [da2](#da2) |
| 实验讲义或基准  [第二十八课：行为克隆——用数据替代模型](32-session-28-bc-imitation.md) | [dagger](#dagger) |
| 实验讲义或基准  [第二十九课：强化学习入口——手写 PPO 摆起，只凭奖励行吗？](33-session-29-ppo-swingup.md) | [ppo](#ppo) · [gae](#gae) |
| 实验讲义或基准  [第三十课：残差强化学习——底座管能量注入，PPO 只学限幅残差](35-session-30-residual-swingup.md) | [residual](#residual) · [ppo](#ppo) |
| 实验讲义或基准  [第三十一课：势函数奖励塑形（PBRS）——给悬崖修梯子，山顶不动](36-session-31-pbrs-shaping.md) | [pbrs](#pbrs) |
| 实验讲义或基准  [第三十二课：DAPG 式示教空投——把策略锚在教师身上，最后一公里仍未打通](37-session-32-dapg-swingup.md) | [dapg](#dapg) · [ppo](#ppo) |
| 实验讲义或基准  [第三十三课：Go-Explore 画地图过崖——山顶被找到并入档，路还是没修通](38-session-33-goexplore-swingup.md) | [goexplore](#goexplore) |
| 实验讲义或基准  [第三十四课：两阶段奖励——能量梯子处处有糖，首次到达变 2/3 种子，但首成仍未现](39-session-34-twophase-swingup.md) | [twophase](#twophase) · [ppo](#ppo) |
| 实验讲义或基准  [第三十五课：手写 numpy SAC——回放池+最大熵+孪生 Q 也未能直接学会摆起](40-session-35-sac-swingup.md) | [sac](#sac) |
| 实验讲义或基准  [第三十六课：DAgger 在线纠错——教师逐帧标注学生的状态分布，历史性 0→1 仍未出现](41-session-36-dagger-swingup.md) | [dagger](#dagger) |
| 实验讲义或基准  [第三十七课：ACT/扩散策略最小实验——动作块 + 多峰（MoE）表示检验，确定性路径首次到达直立区，0→1 仍未出现](42-session-37-act-swingup.md) | [act](#act) · [diffusion](#diffusion) |
| 实验讲义或基准  [第三十八课：倒立摆组合学习——能量整形底座 + 多峰块残差：组合防毁但不保全，0→1 仍未出现](43-session-38-combo-swingup.md) | [residual](#residual) · [act](#act) |
| 实验讲义或基准  [第三十九课：差速小车纯学习目标到达——全驱动没换来 0→1，0/60 对手工 20/20](44-session-39-rl-goal-reaching.md) | [sac](#sac) · [robotics](#robotics) |
| 实验讲义或基准  [第四十课：2R 机械臂纯学习到达——固定小目标盒没有换来 0→1，0/60 对解析 IK+PD 20/20](45-session-40-rl-arm-reaching.md) | [sac](#sac) · [robotics](#robotics) |
| 实验讲义或基准  [第四十一课：分阶段最小验证——先验底座，再验极小交接修正](46-session-41-staged-verification.md) | [residual](#residual) |
| 实验讲义或基准  [第四十二课：torch ACT 块策略冲击毫米级到达——最优教师喂到嘴边，0/60 仍跨不过 2 mm × 0.5 s](47-session-42-act-torch-reaching.md) | [act](#act) · [attention](#attention) · [diffusion](#diffusion) |
| 实验讲义或基准  [第四十三课：占据栅格建图 + A\* 路径规划——回到"感知-建图-规划"主线](48-session-43-grid-nav.md) | [prob](#prob) · [astar](#astar) · [pursuit](#pursuit) |
| 实验讲义或基准  [第四十四课：定位误差穿栈——建图-规划-追踪在位姿不确定下还成立吗？](49-session-44-nav-pose-error.md) | [prob](#prob) · [astar](#astar) · [pursuit](#pursuit) |
| 实验讲义或基准  [第四十五课：最小栅格 SLAM——扫描匹配是相对锚，不能给出绝对真值](50-session-45-scan-slam.md) | [prob](#prob) · [icp](#icp) |
| 实验讲义或基准  [第四十六课：回环闭合——绝对锚的第二形态，与轨迹修正](51-session-46-loop-closure.md) | [prob](#prob) · [icp](#icp) |
| 实验讲义或基准  [第四十七课：加权位姿图优化——修复漂移的形状](52-session-47-pose-graph.md) | [prob](#prob) · [icp](#icp) |
| 实验讲义或基准  [第四十八课：鲁棒位姿图——错误回环边与单核两难](53-session-48-robust-graph.md) | [prob](#prob) · [scipy](#scipy) |
| 实验讲义或基准  [第四十九课 · 回环匹配器工程化——量化“正确 vs 伪影峰”的边界](54-session-49-matcher-engineering.md) | [prob](#prob) · [icp](#icp) |
| 实验讲义或基准  [第五十课 · 特征化回环检测——评分重叠与旧判据勘误](55-session-50-feature-loops.md) | [prob](#prob) · [icp](#icp) |
| 实验讲义或基准  [第五十一课 · 可切换约束（SC/MM）——单开关两难：λ 可行域为空](56-session-51-switchable-graph.md) | [switch](#switch) · [mixture](#mixture) |
| 实验讲义或基准  [第五十二课 · 相对回环边——SC/MM 的真实工作域与它的边界](57-session-52-relative-loops.md) | [switch](#switch) · [mixture](#mixture) |
| 实验讲义或基准  [第五十三课 · 生产级求解器对照——失败是问题的属性，不是实现的产物](58-session-53-production-solver.md) | [scipy](#scipy) · [switch](#switch) · [mixture](#mixture) |
| 实验讲义或基准  [第五十四课 · 各向异性环境对照 + 度量勘误——"方向正确率 5%"是评估伪影](59-session-54-aniso-env.md) | [prob](#prob) · [icp](#icp) |
| 实验讲义或基准  [第五十五课 · 用模拟外观找回环，再用几何检查](60-session-55-rgbd-loops.md) | [prob](#prob) · [icp](#icp) |
| 实验讲义或基准  [第五十六课 · 地图怎样影响定位](61-session-56-map-localization.md) | [prob](#prob) · [icp](#icp) |
| 实验讲义或基准  [二维导航集成基准 · 从单场景结论走向跨场景复验](62-navigation-benchmark.md) | [prob](#prob) · [astar](#astar) · [pursuit](#pursuit) |
| 实验讲义或基准  [第 63 课：照片房间三维验证——从平面地图走向有高度的障碍物](63-photo-room-3d-validation.md) | [prob](#prob) · [astar](#astar) · [pursuit](#pursuit) |
| 实验讲义或基准  [第 64 课：闭环导航对照——修正协议下的四组定位来源对照（v4）](64-closed-loop-navigation.md) | [prob](#prob) · [amcl](#amcl) |
| 实验讲义或基准  [第 65 课：地图分离试跑——自建图失败发生在哪一层？](65-map-separation.md) | [prob](#prob) · [astar](#astar) · [pursuit](#pursuit) |
| 实验讲义或基准  [第 66 课：街区式园区巡检——定位误差怎样影响真实到点](66-campus-patrol.md) | [prob](#prob) · [astar](#astar) · [pursuit](#pursuit) |
| 实验讲义或基准  [第 67 课：先知道自己有多大，再根据雷达和深度避障](67-body-aware-obstacle-avoidance.md) | [robotics](#robotics) · [astar](#astar) · [pursuit](#pursuit) |
| 实验讲义或基准  [第 68 课：用旧扫描修正画歪的地图，再独立开一圈检查](68-scan-based-map-repair.md) | [prob](#prob) · [icp](#icp) |
| 计划  [第69课第三轮预登记：刹停模型、转弯减速与曲率速度预算](69-braking-round3-plan.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 实验讲义或基准  [第69课第三轮：车要停下来，为什么还得继续转弯？](69-braking-round3.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 计划  [第69课第十三轮：停车匹配后，为择路保留误差空间](69-clearance-navigation-round13-plan.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 诊断报告  [第六十九课接触复盘：为什么发了刹车请求，身体还是撞上了？](69-contact-failure-diagnosis-round11.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 计划  [第69课第六轮预登记：允许轻微擦碰，以到达为主](69-contact-round6-plan.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 实验讲义或基准  [第六十九课第六轮：允许轻擦，能不能真正穿过窄路？](69-contact-round6.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 实验讲义或基准  [第六十九课第十轮：近地面的旧墙，为什么相机看不清它已经移走？](69-depth-sampling-round10.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 计划  [第69课第五轮预登记：排队动作与执行端保护](69-execution-round5-plan.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 实验讲义或基准  [第六十九课第五轮：排队动作和执行端保护](69-execution-round5.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 实验讲义或基准  [第 69 课：车身明明放得下，为什么规划和执行仍过不去？](69-footprint-planning.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 实验讲义或基准  [第 69 课第二轮：有路却跟不住，问题出在转向还是刹车？](69-footprint-tracking-round2.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 计划  [第69课第十四轮：规划也必须保留障碍高度](69-height-planning-round14-plan.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 实验讲义或基准  [第六十九课第八轮：怎样用观测修改错误的局部地图？](69-online-map-round8.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 计划  [第69课第十二轮：匹配物理停车预测](69-physical-braking-round12-plan.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 实验讲义或基准  [第六十九课补充：停车要算真实身体，规划要保留真实高度](69-physical-height-navigation.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 实验讲义或基准  [第六十九课第九轮：为什么有路线，机器人却不往前走？](69-progress-round9.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 实验讲义或基准  [第六十九课补充：发现新障碍后，真的能绕过去吗？](69-reachable-obstacles-round11.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 计划  [第69课第四轮预登记：定位误差与真实执行延迟](69-robustness-round4-plan.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 实验讲义或基准  [第69课第四轮：认偏一点、晚刹一点，原来的成功还成立吗？](69-robustness-round4.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 实验讲义或基准  [第六十九课第七轮：慢一点，能不能真正改善导航？](69-speed-round7.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 计划  [第69课第二轮预登记：固定足印与规划，只改变路径跟踪](69-tracking-round2-plan.md) | [astar](#astar) · [pursuit](#pursuit) · [robotics](#robotics) |
| 实验讲义或基准  [第 70 课：算法说“到了”，身体真的到了吗？](70-arrival-decisions.md) | [prob](#prob) · [pursuit](#pursuit) |
| 实验讲义或基准  [第 71 课：规划说“无路”，怎样安全地尝试恢复？](71-bounded-recovery.md) | [prob](#prob) · [pursuit](#pursuit) |
| 实验讲义或基准  [第七十二课：从平面双关节臂走到SO-101三维机械臂](72-so101-geometry.md) | [robotics](#robotics) · [so101](#so101) · [menagerie](#menagerie) · [mujoco](#mujoco) |
| 计划  [第73课第二轮：接近路径与夹爪开口中心](73-approach-round2-plan.md) | [robotics](#robotics) · [manip](#manip) · [menagerie](#menagerie) · [mujoco](#mujoco) |
| 计划  [第73课第三轮：指面截面与双侧接触反馈（执行前登记）](73-contact-round3-plan.md) | [robotics](#robotics) · [manip](#manip) · [menagerie](#menagerie) · [mujoco](#mujoco) |
| 实验讲义或基准  [第七十三课第二轮：对准夹爪的点，为什么仍夹不住方块？](73-so101-approach-geometry.md) | [robotics](#robotics) · [manip](#manip) · [menagerie](#menagerie) · [mujoco](#mujoco) |
| 实验讲义或基准  [第七十三课第三轮：两根手指都有力，为什么方块还是会掉？](73-so101-contact-feedback.md) | [robotics](#robotics) · [manip](#manip) · [menagerie](#menagerie) · [mujoco](#mujoco) |
| 实验讲义或基准  [第七十三课：夹爪闭合以后，怎样证明方块真的被抓住？](73-so101-physical-grasping.md) | [robotics](#robotics) · [manip](#manip) · [menagerie](#menagerie) · [mujoco](#mujoco) |
| 实验讲义或基准  [S288 MicroDinosaur：彩色相机与 ToF 园区巡检](microdinosaur-campus-patrol.md) | [camera](#camera) · [tof](#tof) · [mujoco](#mujoco) · [prob](#prob) |
| 实验讲义或基准  [MicroDinosaur 巡检：单目深度与 ToF 校正](microdinosaur-patrol-depth.md) | [uni](#uni) · [da2](#da2) · [depthpro](#depthpro) · [tof](#tof) |
| 交付小结  [S288 小恐龙巡检实验小结（2026-10-09）](microdinosaur-patrol-review-20261009.md) | [uni](#uni) · [camera](#camera) · [tof](#tof) · [tts](#tts) · [voice](#voice) |

## 核验范围与历史来源

本轮核对上述原作者/官方入口、题名、基本机制及与现有讲义的关系，没有重新训练模型或全面复现论文。MDPI 两阶段论文的出版题名可检索，但全文重读受 429 限流；ROS Jazzy 文档曾遇反爬，坐标机制改用官方 geometry2 入口。波动的仓库主页不能代替本机锁定版本与归档哈希。

未来 NAV 可比较 RTAB-Map / robot_localization，MAN 可比较 LeRobot/更高自由度模型；具体候选链接及进入条件保留在各方向路线，不在主页宣称已采用。第29/35课的历史文献讨论仍保留，后续改动引用时核对其具体任务/条件，不能只从摘要移用改善比例。
