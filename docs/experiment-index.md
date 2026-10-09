# 学习阶段、研究方向与编号对照

2026-10-09。导航按问题组织，历史课号用于定位讲义、代码参数和记录。

**三种编号分开：** F1–F6 是基础学习阶段；NAV / MAN / DINO 是研究方向；方向实验如 NAV-08 的 R01–R14 是迭代轮次。路线图阶段 S0–S8 是能力依赖，不是课程号、完成比例或新的实验成绩。

旧课程 1–56 与 63–73、文件名、CLI 和结果目录保持不变。文件号 57–61 对应课号 52–56；62 是集成基准，不补造第 57–62 课。09-28 曾明确恢复 63–68 为新课程，见决策 D-2026-09-28-02；本次新增方向视图，不撤销这一历史决定。后续工作先登记方向、问题和轮次，确有独立教学目标才另定课号，不预先把阶段排成 74–76。

[学习路线](01-learning-roadmap.md) · [讲义与编号标准](documentation-standard.md) · [完整图册](experiment-gallery.md) · [运行手册](running-experiments.md) · [参考文献](references.md) · [决策日志](34-experiment-decision-log.md)

<a id="foundation"></a>

## 基础学习阶段（历史课程 1–56）

这些课程已有记录。“已做过实验”不等于学习掌握或目标能力通过，失败和判据见讲义；原主页带数字的旧索引保留在[历史快照](learning-history.md)。Guide 作为知识地图，原论文/教材作为机制来源，见每份讲义的补引段。

### F1 · 状态与反馈控制

| 历史课号 | 学习问题 | 讲义 |
| --- | --- | --- |
| 1 | 环境与实验规范 | [讲义](03-session-01.md) |
| 2 | PD 控制与随机基线对照 | [讲义](04-session-02-pd-control.md) |
| 3 | 离散 LQR 全状态对比 | [讲义](05-session-03-lqr.md) |
| 4 | LQR 权重与慢放对照 | [讲义](06-session-04-lqr-weights-and-demo.md) |
| 5 | 随机外部扰动 | [讲义](07-session-05-disturbance.md) |
| 6 | 真实状态与传感读数 | [讲义](08-session-06-measurement-noise.md) |
| 7 | 下垂摆起与强扰动恢复 | [讲义](09-session-07-swingup.md) |

### F2 · 运动学与执行

| 历史课号 | 学习问题 | 讲义 |
| --- | --- | --- |
| 8 | 两个关节与末端坐标 | [讲义](10-session-08-planar-arm.md) |
| 9 | 沿直线运动与 Jacobian | [讲义](11-session-09-jacobian-path.md) |
| 10 | 换一批路径后还可靠吗 | [讲义](12-session-10-path-coverage.md) |
| 11 | 逐点解析 IK 起步 | [讲义](13-session-11-waypoint-ik.md) |
| 12 | 动作时间与电机限制 | [讲义](14-session-12-timing-and-torque.md) |
| 13 | 模型前馈＋原 PD | [讲义](15-session-13-model-feedforward.md) |

### F3 · 运动估计与接口

| 历史课号 | 学习问题 | 讲义 |
| --- | --- | --- |
| 14 | 差速小车与世界/车体/传感器坐标 | [讲义](16-session-14-mobile-frames.md) |
| 15 | 编码器里程计与累积误差 | [讲义](17-session-15-encoder-odometry.md) |
| 16 | 固定比例标定与独立验证 | [讲义](18-session-16-encoder-calibration.md) |
| 17 | 标定之后的随机测量噪声 | [讲义](19-session-17-random-noise.md) |
| 18 | 已知地标（控制点）观测与里程计对照 | [讲义](20-session-18-landmark-observations.md) |
| 19 | 看观测 → 解位置 → 最简融合 | [讲义](21-session-19-landmark-fusion.md) |
| 20 | ROS 2 节点、消息与坐标链 | [讲义](22-session-20-ros2-messages-and-tf.md) |
| 21 | 根据估计位置驶向目标 | [讲义](23-session-21-goal-feedback.md) · [补充](23a-session-21-stopping-tolerance.md) |

### F4 · 三维感知

| 历史课号 | 学习问题 | 讲义 |
| --- | --- | --- |
| 22 | 针孔相机与投影-反投影 | [讲义](24-session-22-pinhole-projection.md) |
| 23 | 单目相对深度 ↔ 米制尺度标定 | [讲义](25-session-23-monocular-metric.md) |
| 24 | 真实 DA V2 仿射检验 | [讲义](28-session-24-real-depth-affine.md) |
| 25 | 张氏内参标定 | [讲义](29-session-25-camera-intrinsics.md) |
| 26 | 点云 ICP 配准 | [讲义](30-session-26-icp-registration.md) |
| 27 | MobileSAM 视觉接地标身份 | [讲义](31-session-27-visual-grounding.md) |

### F5 · 模仿与强化学习

| 历史课号 | 学习问题 | 讲义 |
| --- | --- | --- |
| 28 | 行为克隆（阶段 5 入口） | [讲义](32-session-28-bc-imitation.md) |
| 29 | PPO 摆起（RL 对照） | [讲义](33-session-29-ppo-swingup.md) |
| 30 | 残差 RL 摆起 | [讲义](35-session-30-residual-swingup.md) |
| 31 | PBRS 势函数塑形 | [讲义](36-session-31-pbrs-shaping.md) |
| 32 | DAPG 示教空投 | [讲义](37-session-32-dapg-swingup.md) |
| 33 | Go-Explore 画地图 | [讲义](38-session-33-goexplore-swingup.md) |
| 34 | 两阶段奖励（分阶段目标） | [讲义](39-session-34-twophase-swingup.md) |
| 35 | 手写 numpy SAC | [讲义](40-session-35-sac-swingup.md) |
| 36 | DAgger 在线纠错 | [讲义](41-session-36-dagger-swingup.md) |
| 37 | 多峰块策略（ACT 最小版） | [讲义](42-session-37-act-swingup.md) |
| 38 | 倒立摆组合学习 | [讲义](43-session-38-combo-swingup.md) |
| 39 | 差速小车纯学习 | [讲义](44-session-39-rl-goal-reaching.md) |
| 40 | 2R 臂纯学习 | [讲义](45-session-40-rl-arm-reaching.md) |
| 41 | 分阶段最小验证 | [讲义](46-session-41-staged-verification.md) |
| 42 | ACT/torch 到达 | [讲义](47-session-42-act-torch-reaching.md) |

### F6 · 建图与导航

| 历史课号 | 学习问题 | 讲义 |
| --- | --- | --- |
| 43 | 占据栅格建图 + A* 规划 + 纯追踪（回主线） | [讲义](48-session-43-grid-nav.md) |
| 44 | 定位误差穿栈（位姿不确定下） | [讲义](49-session-44-nav-pose-error.md) |
| 45 | 最小栅格 SLAM（帧间扫描匹配） | [讲义](50-session-45-scan-slam.md) |
| 46 | 回环闭合（绝对锚第二形态） | [讲义](51-session-46-loop-closure.md) |
| 47 | 加权位姿图优化（后端） | [讲义](52-session-47-pose-graph.md) |
| 48 | 鲁棒位姿图（毒化回环注入） | [讲义](53-session-48-robust-graph.md) |
| 49 | 回环匹配器工程化（量化方向退化） | [讲义](54-session-49-matcher-engineering.md) |
| 50 | 特征化回环检测（阴性记录） | [讲义](55-session-50-feature-loops.md) |
| 51 | 可切换约束 SC/MM（阴性记录） | [讲义](56-session-51-switchable-graph.md) |
| 52 | 相对回环边（SC/MM 真实工作域） | [讲义](57-session-52-relative-loops.md) |
| 53 | 生产求解器对照（问题属性判定） | [讲义](58-session-53-production-solver.md) |
| 54 | 度量勘误 + 各向异性对照 | [讲义](59-session-54-aniso-env.md) |
| 55 | 模拟标记外观检索 | [讲义](60-session-55-rgbd-loops.md) |
| 56 | 建图与定位分离 | [讲义](61-session-56-map-localization.md) |

<a id="nav"></a>

## NAV · 定位、建图与身体约束导航

[方向路线及进入门槛](multisensor-navigation-roadmap.md) · [69–71 阶段复盘](navigation-stage-review-69-71.md)

| 方向实验 | 历史编号 | 核心问题 | 主报告 |
| --- | --- | --- | --- |
| NAV-01 | 62（基准） | 跨场景集成门槛 | [报告](62-navigation-benchmark.md) |
| NAV-02 | 63 | 照片房间几何与扫描高度 | [报告](63-photo-room-3d-validation.md) |
| NAV-03 | 64 | 闭环定位来源与官方 AMCL 参数审计 | [报告](64-closed-loop-navigation.md) |
| NAV-04 | 65 | 固定扫描下的地图表示与落点分离 | [报告](65-map-separation.md) |
| NAV-05 | 66 | 园区多点巡检基线 | [报告](66-campus-patrol.md) |
| NAV-06 | 67 | 机身标定与观测驱动避障 | [报告](67-body-aware-obstacle-avoidance.md) |
| NAV-07 | 68 | 冻结扫描修图后独立导航 | [报告](68-scan-based-map-repair.md) |
| NAV-08 | 69，R01–R14 | 足印、执行、停车与三维通行 | [报告](69-physical-height-navigation.md) |
| NAV-09 | 70 | 估计到达与真实停稳分离 | [报告](70-arrival-decisions.md) |
| NAV-10 | 71 | 有限且有观测依据的恢复 | [报告](71-bounded-recovery.md) |

<a id="lesson69-rounds"></a>

### NAV-08 轮次对照（历史第 69 课）

| 轮次 | 这次改变什么 | 讲义 |
| --- | --- | --- |
| 1–3 | 车身足印 → 跟踪与路线保留 → 制动形状 | [第一轮](69-footprint-planning.md) · [第二轮](69-footprint-tracking-round2.md) · [第三轮](69-braking-round3.md) |
| 4–6 | 位姿与延迟边界 → 队列预测 → 允许轻微擦碰的接触物理 | [第四轮](69-robustness-round4.md) · [第五轮](69-execution-round5.md) · [第六轮](69-contact-round6.md) |
| 7–9 | 降低速度 → 观测更新地图 → 进度与执行门控 | [第七轮](69-speed-round7.md) · [第八轮](69-online-map-round8.md) · [第九轮](69-progress-round9.md) |
| 10–11 | 深度采样 → 确实能绕开的新障碍 | [第十轮](69-depth-sampling-round10.md) · [第十一轮](69-reachable-obstacles-round11.md) |
| 12–14 | 真实物理停车 → 择路余量 → 按部件高度判断通行 | [合并讲义](69-physical-height-navigation.md) |

<a id="man"></a>

## MAN · 三维接触与操作

[方向路线及进入门槛](manipulation-roadmap.md)。新增视觉、恢复、学习阶段先使用方向阶段标识；当前还没有完成视觉抓取。

| 方向实验 | 历史编号 | 核心问题 | 主报告 |
| --- | --- | --- | --- |
| MAN-01 | 72 | 三维工具、关节与相机坐标 | [报告](72-so101-geometry.md) |
| MAN-02 | 73 首轮 | 实际接触抓取与负对照 | [报告](73-so101-physical-grasping.md) |
| MAN-03 | 73 R02 | 接近参考点与夹爪中心 | [报告](73-so101-approach-geometry.md) |
| MAN-04 | 73 R03 | 指面校准、双侧接触与掉落 | [报告](73-so101-contact-feedback.md) |

<a id="dino"></a>

## DINO · S288 机器人感知巡检

单列机器人资产、动作策略和传感器硬件假设；NAV 的差速车成绩不能作为 S288 验收。公开源资产范围见具体报告。

| 方向实验 | 核心问题 | 主报告 |
| --- | --- | --- |
| DINO-01 | 固定原 S288 策略，RGB＋规定 ToF 能否支持巡检与降级停车 | [基准](microdinosaur-campus-patrol.md) |
| DINO-02 | 单目深度经 ToF 定尺后是否增加可用信息或改变控制 | [深度对照与失败](microdinosaur-patrol-depth.md) |

[2026-10-09 小结与 3 分钟星瞳讲解](microdinosaur-patrol-review-20261009.md)是上述实验的交付，不另算一课或新实验。下一轮先分离标定/延迟/外观并换布局；实机进入条件见基准与小结。
