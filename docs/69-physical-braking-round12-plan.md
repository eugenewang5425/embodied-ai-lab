# 第69课第十二轮：匹配物理停车预测

2026-10-01，基线21119c5。用户授权将当前导航改到稳定通过后继续机械臂。

## 第一项：先隔离停车预测

- 原十一轮及源码、成绩、开发反例保留。
- 只比较原运动学停车预测与无环境几何的限力驱动物理停车预测；两组都使用雷达＋320×240深度。
- 真实身体、质量/惯量、伺服、力限、摩擦、接触、地图更新、规划、跟踪、0.2m/s、0.2s排队和90s超时不变。
- 物理预测器只含身体和驱动模型，输入测得完整速度和当前已有队列。环境真值只用于传感器、真实物理和独立评分。
- 预测每2ms推进；保存完整预测状态，检查路径按累计1mm角点路程抽样，最大间隔为1mm加一个物理子步路程。真实评分检查全部500Hz状态。
- 预测速度尾部至各分量小于0.0001；任务仍使用真实距离<25cm、世界平移/角速度<0.01和原40N/5mm/连续0.5s轻擦口径。
- 开发固定种子0、四种未知可绕障碍；先检查箱体接触能否避免，停车后能否继续到达。可只运行指定开发条件，保留每版输出。

## 稳定性的验收范围

最终候选冻结后另登记独立验证：至少三个障碍足印/位置布局，四类障碍，每布局五个新噪声种子，至少60回合。每回合必须真实到达、实际停稳、接触符合原口径，零误报；可达任务的停车、无路和超时都算失败。单独保留确实无路的负例，拒绝不算可达任务成功。

若本项只避免接触但仍不能到达，继续分离观测表示、路线与跟踪。新的开发候选另存，不修改本项已冻结记录，不反复使用已看过的验证数据宣称留出成功。新布局的可达性只由独立评分端采样检查，参考路线不传给控制器。

达到这个有限仿真范围的全通过门槛后，再继续73课第二轮接近几何诊断和独立位置/朝向抓取。位姿误差、移动障碍、实机和真实异步计算仍须单独验证。

## 原始来源与本地适配（2026-10-09补引）

[A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/astar.pdf)（Peter Hart / Nils Nilsson / Bertram Raphael，1968）；[Implementation of the Pure Pursuit Path Tracking Algorithm](https://publications.ri.cmu.edu/implementation-of-the-pure-pursuit-path-tracking-algorithm)（R. Craig Coulter，CMU技术报告1992）；[Modern Robotics: Mechanics, Planning, and Control](https://github.com/NxRLab/ModernRobotics)（Kevin Lynch / Frank Park，2017）。

在本地差速平台分离足印、动作队列、制动、观测地图和三维通行；这些受控扩展不是原 A* 或纯追踪自带的安全保证，各轮身体/权限/种子与失败分开。 [完整采用关系与引用规则](references.md)。
