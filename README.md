# 具身智能学习与本地仿真

本项目采用“学习一个概念，完成一个可运行实验，留下可复现实验记录”的方式学习具身智能。

## 项目目标

- 建立机器人学、控制、感知和学习之间的完整知识框架。
- 在本地仿真环境中验证每个核心概念，而不只停留在阅读材料。
- 利用 GIS、遥感和空间智能基础，逐步进入三维感知、建图、导航、机器人学习与具身智能。
- 保持项目小步迭代、Git 可追踪、结果可复现。

## 当前技术路线

1. **Windows 原生：MuJoCo + Gymnasium**
   - 用于动力学、控制、机械臂运动学和强化学习入门。
   - 第一批实验：环境自检、倒立摆、二维机械臂到达。
2. **WSL2 / Ubuntu 24.04：ROS 2 Jazzy + Gazebo Harmonic**
   - 用于机器人系统、传感器、LiDAR、SLAM、导航与多节点通信。
   - 等第一阶段形成稳定的 Python 与机器人学基础后再安装。
3. **Isaac Lab：暂不进入主线**
   - 当前电脑的 RTX 5070 Laptop GPU 具有 8GB 显存，低于完整 Isaac Lab 工作流建议的 16GB。
   - 后续可按具体任务评估轻量/headless 运行、远程 GPU 或云环境。

## 学习闭环

每个主题都按以下顺序推进：

1. 物理直觉与任务定义；
2. 坐标系、状态、动作与数学模型；
3. 在仿真中实现；
4. 记录指标、失败现象与原因；
5. 修改方法并复现实验；
6. 形成简短结论和下一步。

## 项目结构

```text
.
├── README.md
├── docs/
│   ├── 00-environment-audit.md
│   ├── 01-learning-roadmap.md
│   ├── 02-phase-1.md
│   ├── 03-session-01.md
│   ├── 04-session-02-pd-control.md
│   ├── 05-session-03-lqr.md
│   ├── 06-session-04-lqr-weights-and-demo.md
│   ├── 07-session-05-disturbance.md
│   ├── 08-session-06-measurement-noise.md
│   ├── 09-session-07-swingup.md
│   ├── 10-session-08-planar-arm.md
│   ├── 11-session-09-jacobian-path.md
│   ├── 12-session-10-path-coverage.md
│   ├── 13-session-11-waypoint-ik.md
│   ├── 14-session-12-timing-and-torque.md
│   ├── 15-session-13-model-feedforward.md
│   ├── 16-session-14-mobile-frames.md
│   ├── 17-session-15-encoder-odometry.md
│   └── 18-session-16-encoder-calibration.md
├── src/embodied_learning/
├── tests/
├── results/
├── pyproject.toml
├── uv.lock
└── .gitignore
```

## 快速运行

在项目目录打开一个新的 PowerShell：

```powershell
uv sync
uv run pytest -q
uv run python -m embodied_learning.env_check --steps 300 --seed 7
uv run python -m embodied_learning.experiments.pd_comparison
uv run python -m embodied_learning.viewer --policy pd --seconds 10 --seed 7
uv run python -m embodied_learning.experiments.lqr_comparison --output results/lqr_my_run
uv run python -m embodied_learning.viewer --policy lqr --seconds 15 --seed 7
```

这些命令分别同步依赖、运行自动测试、生成随机基线、比较 PD 控制，并打开 10 秒的 PD 控制图形窗口。

最后两条是第三课：40 秒全状态控制对比，以及 LQR 图形窗口。LQR 对比必须使用新的 `--output` 目录，不会覆盖已有结果。当前已验证报告在 `results/lqr_2026-09-02/`；该目录受 Git 忽略规则保护，保留结果需另行归档。运行 `uv sync --locked` 可严格使用锁定依赖。

模型单位校正：控制输入 `u ∈ [-3, 3]`，实际小车执行器水平力为 `100u N`；物理竖直对应关节角约 `-0.001667 rad`。旧结果文件仍保留，新报告已区分这些概念。

## 慢速教学演示

如果原来的仿真窗口太快，在项目目录运行：

```powershell
uv run python -m embodied_learning.teaching_demo
```

默认暂停、0.25 倍速，播放最有变化的前 8 秒。支持播放/暂停、单步、重播、时间轴、R 对照和真实状态数值；退出可关闭窗口或按 Esc。
如果存在 `results/lqr_r1_my_run`，自动读取其中的 R=1 实验，不会覆盖文件；其他 R 使用同一初态计算。没有该目录时，三个方案均由当前 MuJoCo 生成。

也可明确指定已有的 `lqr_comparison` 结果：

```powershell
uv run python -m embodied_learning.teaching_demo --results results/lqr_r1_my_run
```

这是实际仿真轨迹的二维正视示意回放，不是额外训练。默认摆角画面放大 8 倍，开关可以关闭；右侧数字始终是真实值，播放速度不改变物理步长。
详细操作和 R 对照结果见[第四课：慢放与输入代价](docs/06-session-04-lqr-weights-and-demo.md)。

图表区新增“叠加所有 R 曲线”（默认开启）：蓝色 R=0.1、橙色 R=1、绿色 R=10，共用时间与纵轴尺度。可切换位置、真实倾角、控制输入；关闭叠加只显示当前 R，颜色不变，也不会重置播放位置。

第五课已加入同条件随机外部推力，使用原来的 LQR、不新增学习算法或依赖：

```powershell
uv run python -m embodied_learning.teaching_demo --push-results results/lqr_push_2026-09-02 --seed 100
```

该命令只读取已有实验；完整复现使用新目录：

```powershell
uv run python -m embodied_learning.experiments.lqr_disturbance --output results/lqr_push_my_run
```

见[第五课：被推一下之后](docs/07-session-05-disturbance.md)。

## 第六课：真实状态与传感读数

固定 R=1，只增加测量噪声，比较 0×、1×、3× 三组；不与上一课推力混合。
已有结果在 `results/lqr_noise_2026-09-02/`，其中 `comparison.png` 对照真实倾角、传感读数和控制动作，`trajectories.npz` 保留逐步原始数据。

```powershell
uv run python -m embodied_learning.experiments.lqr_measurement_noise --output results/lqr_noise_my_run
```

复现命令须使用新目录。第六课现已补充慢放入口，直接读取已有数据：

```powershell
uv run python -m embodied_learning.teaching_demo --noise-results results/lqr_noise_2026-09-02 --seed 200 --seconds 10
```

默认暂停、0.25×、1× 噪声。实线杆是真实姿态，橙色虚线杆是同一时刻的传感读数（不是另一根杆）；右侧显示真实值→读数以及由读数产生的下一步动作。切换 0×/1×/3× 噪声，或勾选“三组真实曲线”比较实际运动；三组均固定 R=1，没有外力。角度 8× 仅为画面放大，数字和曲线不放大；单步仍是 0.04 s。不新增 GUI 框架或滤波器，不覆盖实验。
阅读[第六课：看错了，也可能真的动起来](docs/08-session-06-measurement-noise.md)，完成状态、读数、动作三者的辨别。按最新学习问题，先增加下面的摆起专题，再进入双关节机械臂运动学。

## 第七课：下垂摆起与强扰动恢复

新增独立的全转动环境：正下方 -180°、左右下方 ±120° 初态，以及在控制持续开启时仍能推倒杆的 ±400 N 扰动。远离直立时用能量摆起，接近直立时切入 LQR；过强的 +600 N 案例保留真实失败。

```powershell
uv run python -m embodied_learning.swingup_demo --results results/swingup_2026-09-02
```

默认暂停、0.25×，可单步或跳到“扰动开始 / 杆到下方 / 恢复稳定”。画面角度不放大，显示控制模式、真实状态、电机力和外力。

新任务取消角度结束条件并允许杆转整圈；轨道扩为 ±2.5 m，每步检查到 |x|≥2.4 m 会失败，另保留速度与数值边界。电机仍最大 ±300 N，原实验和学员已有结果不变。

已保存的正下方启动在 4.76 s 稳定；+400 N 于 5.00–5.40 s 施加，5.32 s 将杆推到下方，15.24 s 重新稳定。只是确定性教学案例，不代表任意扰动均可恢复。

详细原理、场景结果和新目录复现命令见[第七课：倒下以后，先摆起来](docs/09-session-07-swingup.md)。

## 第八课：两个关节与末端坐标

第八课已进入平面双关节机械臂：两个相对关节角决定末端世界坐标，解析 IK 给目标角，两个电机通过 PD 驱动真实 MuJoCo 运动。

```powershell
uv run python -m embodied_learning.arm_demo --results results/arm_reaching_2026-09-02_v2
```

第一页为几何滑块（直接设置姿态，不是动力学）；第二页为真实力矩驱动轨迹的 0.25× 慢放。三个固定案例均达到末端误差 ≤2 mm、关节误差和速度同时满足门限并持续至少 0.5 s；49 个姿态核验了公式和 MuJoCo 坐标一致。未加入随机目标、碰撞、噪声或 Jacobian 控制。

复现：`uv run python -m embodied_learning.experiments.arm_reaching --output results/arm_reaching_my_run`。输出目录须为新目录。学习顺序和验收记录见[第八课：两个关节角与末端坐标](docs/10-session-08-planar-arm.md)。早期未通过的 `results/arm_reaching_2026-09-02/` 保留作调试证据，当前使用 `_v2`。

## 第九课：沿直线运动与 Jacobian

沿用同一 2R 模型和电机限制，对比只给终点、关节角插值、Jacobian 直线路径。后两者使用相同 8 秒移动 + 3 秒停留与内层关节 PD。0–8 秒最大偏离线段分别为 76.154、46.241、0.197 mm；三组均停稳，只有 Jacobian 组满足本次 2 mm 路径门限。这是单条固定路径的理想仿真结果，不代表硬件精度。

```powershell
uv run python -m embodied_learning.arm_path_demo --results results/arm_path_2026-09-02
```

默认暂停、0.25×，三组曲线不同颜色、同一尺度。第一页是奇异位形的静态速度预测，第二页是真实力矩驱动的轨迹回放；数值阻尼不能恢复伸直位形缺失的瞬时运动方向。

复现：`uv run python -m embodied_learning.experiments.arm_path --output results/arm_path_my_run`。新目录输出包含三组实际状态、力矩、几何参考、对照图与奇异性探针；完整实验和讲解见[第九课：到达终点，不等于沿直线到达](docs/11-session-09-jacobian-path.md)。未增加噪声、碰撞、自由度或依赖；下一步才是有种子的多路径评估。

## 第十课：换一批路径后还可靠吗？

已完成 seed=400 的 24 条随机路径与 1 条固定奇异初态反例，三方法配对共 75 回合；控制器、2R 模型、电机限幅和时间安排不变。Jacobian 内部组路径通过 12/12，近伸直组 8/12；后者四个未通过案例均因关节角误差未连续满足末尾 0.5 s 要求，而非超过路径偏离门限。

完全伸直向内收的反例中，参考角和两电机力矩均为零，机械臂原地不动：虽偏离线段为零，仍距终点 200 mm，因此不误报成功。慢放入口：

```powershell
uv run python -m embodied_learning.arm_path_demo --results results/arm_path_batch_2026-09-02_v2/trials/singular_inward
```

同一窗口也可读取 `trials/interior_00` 或 `trials/near_extension_01`，新增整段结果与停稳未过条件。批量复现：

```powershell
uv run python -m embodied_learning.experiments.arm_path_batch --seed 400 --per-group 12 --output results/arm_path_batch_my_run
```

正式结果使用 `_v2`；早期目录存在 Windows 换行导致的清单哈希记录问题，保留但不作为正式报告。详见[第十课：多路径检验与失败分层](docs/12-session-10-path-coverage.md)。小样本分组计数不代表普遍成功率；未增加噪声、接触或依赖。

## 第十一课：逐点解析 IK 起步

只改变参考生成方式，不改变机械臂、电机、PD、时间表或验收。对原第十课清单重新比较关节插值、Jacobian 和逐点解析 IK；原有两种方法的轨迹逐数组完全一致。

逐点 IK 在原清单上通过内部组 12/12、近伸直组 12/12 和固定反例 1/1；完全伸直向内收时，实际最大偏离 1.342 mm，没有力矩饱和。不是任意任务上的成功保证：近伸直组中途偏离通常比 Jacobian 更大，最接近门限的案例约 1.891 mm。

```powershell
uv run python -m embodied_learning.arm_path_demo --results results/arm_ik_comparison_2026-09-02/trials/singular_inward
```

默认暂停、0.25×、逐点解析 IK。三种方案为橙/绿/紫色，复用现有窗口；旧结果仍可读取。复现与原理见[第十一课：换参考算法，不换电机](docs/13-session-11-waypoint-ik.md)。未增加依赖或自由度。

## 第十二课：动作时间与电机限制

同一批 25 条路径、同一逐点 IK/PD/电机，只把动作时间设为 8、4、2 秒，随后均停留 3 秒。8 秒通过 25/25；4 秒通过 13/25，却没有力矩截断；2 秒中 18 条规划速度超限而未执行，其余 7 条有 2 条通过、1 条发生力矩截断。不能把未执行、跟踪偏差和电机饱和混为一谈。

```powershell
uv run python -m embodied_learning.arm_path_demo --results results/arm_timing_2026-09-02/trials/interior_00
```

默认暂停、0.25×，三种动作时间不同颜色，右侧同时显示 PD 请求和实际施加力矩。将末尾换成 `interior_08` 可看实际截断，换成 `singular_inward` 可看 2 秒规划拒绝。旧回放入口保留；完整指标、复现命令和边界见[第十二课讲义](docs/14-session-12-timing-and-torque.md)。8 秒组所有原始数组与第十一课完全一致，没有改模型、增益、限幅或验收门限。

## 第十三课：模型前馈＋原 PD

这是机械臂基础阶段的收尾实验：沿用第十二课的 25 条路径，统一 4 秒动作＋3 秒停留；只加入参考轨迹逆动力学前馈，不改变 PD 增益、电机上限或参考。原 PD 全部数组与第十二课完全一致，路径通过数从 13/25 提高到 25/25。固定伸直案例仍有一次起步力矩截断，不代表模型失配或真实硬件同样可靠。

```powershell
uv run python -m embodied_learning.arm_path_demo --results results/arm_feedforward_2026-09-03_v2/trials/singular_inward
```

默认暂停、0.25×；橙色原 PD、绿色前馈＋PD。右侧拆分模型前馈、PD 修正、合计请求、实际施加。详见[第十三课讲义](docs/15-session-13-model-feedforward.md)与[学习路线回顾](docs/01-learning-roadmap.md)。全量 215 项测试通过。下一主线为差速移动机器人和坐标系，不继续无限扩展 2R 调参。

## 第十四课：差速小车与世界/车体/传感器坐标

进入移动机器人阶段：两个轮速输入决定直行、原地转向、左右圆弧，以及先左转 90° 再前进。新增独立的轻量二维运动学实验，不改变机械臂/倒立摆，也不安装 ROS 2。

```powershell
uv run python -m embodied_learning.mobile_demo --results results/mobile_frames_2026-09-03
```

默认暂停、0.25×，同时显示世界轴、车体轴、偏置安装的传感器轴和地标坐标。勾选“显示错误坐标变换”可看遗漏旋转和安装偏移后，固定地标为什么在地图里乱跑；蓝/橙曲线为同一地标的传感器 x/y，支持单步和时间轴。

先转再走例在世界 `[0, 0.400] m`、90° 结束，五例均与解析运动结果一致。地标完整转换最大往返误差约 `6.28e-16 m`，是已知位姿下的浮点数一致性，**不是定位精度**。轮速直接执行，无力矩、打滑、噪声或反馈控制。复现需新目录：

```powershell
uv run python -m embodied_learning.experiments.mobile_frames --output results/mobile_frames_my_run
```

见[第十四课讲义与观察练习](docs/16-session-14-mobile-frames.md)。新增 36 项测试，全量 251 项通过；已用真实桌面核验播放、单步、案例切换和错误变换叠加。下一步将真实位姿与编码器里程计分开，只加入单一可控偏差，暂不进入 SLAM。

## 第十五课，编码器里程计与累积误差

同一台理想差速车、同一轮速指令，只改变右轮编码器读数比例：0%、+1%、+2%。估计器从编码器增量独立累计位姿，不读取真实位置，不利用地标纠偏。直行 2.4 m 时，+2% 组位置误差 19.40 cm、朝向误差 9.17°；方形一圈后分别为 15.24 cm、15.82°。

```powershell
uv run python -m embodied_learning.odometry_demo --results results/mobile_odometry_2026-09-03
```

默认暂停、0.25×；蓝色为真值，彩色虚线是同一台车的估计轮廓，不是第二台真实车。三组误差曲线同轴叠加，支持位置/朝向/地标落图指标；切偏差保留当前时刻，切路线回起点，“下一段”可快速跳过等待。

复现需新目录：`uv run python -m embodied_learning.experiments.mobile_odometry --output results/mobile_odometry_my_run`。新增 31 项测试；第十四课直行前 4 s 真值与旧记录完全一致。上轮组合回归曾为 281 通过、1 项 Tk 初始化失败；本轮采用窗口测试进程隔离后重复通过，不宣称已确定或修复 Tcl 底层根因。历史证据保留于[第十五课讲义](docs/17-session-15-encoder-odometry.md)，当前验收见第十六课。

## 当前：第十六课，固定比例标定与独立验证

用新生成的 0.4/0.8/1.2/-0.6 m 独立直行测距数据拟合一个右轮修正系数，再验证冻结的第十五课路线。c≈0.980392 由数据求出，不是直接填入真实偏差。另保留“标定测距偏大 1%”反例，说明测量基准的重要性。

```powershell
uv run python -m embodied_learning.calibration_demo
```

新版默认从“① 对照测量”开始：先看 0.4000 m 与 0.4080 m 的差异，再到“② 求修正系数”点击计算，最后进入“③ 换路线验证”的暂停、0.25× 回放。换成“标定用的尺子偏大 1%”只改变参考距离，必须重新计算系数。灯杆落图默认隐藏，可勾选辅助显示；它不参与定位或控制。

验证页明确标出与旧实验的对应：紫色不修正 = 旧 +2%、绿色准确尺子标定 ≈ 旧 0%、橙色尺子偏大1%标定 ≈ 旧 +1%。切方法保留当前时刻；直行终点误差仍为 19.40 cm、数值舍入量级、9.69 cm。新增的是可见的标定过程，不是新运动能力；近零不代表实车精度。

正式结果仍在 `results/encoder_calibration_2026-09-03_v2/`，本次只改教学 UI、测试及文档，不重跑或覆盖旧实验。315 项全量测试连续两次通过，Ruff 静态及格式检查通过；computer-use 实际核验了测量表、两种尺子的计算步骤、验证页和可选灯杆显示。正式结果仍在 `results/encoder_calibration_2026-09-03_v2/`，本次只改教学 UI、测试及文档，不重跑或覆盖旧实验。315 项全量测试连续两次通过，Ruff 静态及格式检查通过；computer-use 实际核验了测量表、两种尺子的计算步骤、验证页和可选灯杆显示。完整原理、假设、复现命令与 Tk 隔离边界见[第十六课讲义](docs/18-session-16-encoder-calibration.md)。

## 当前：第十七课，标定之后的随机测量噪声

固定系数 c≈0.980392 已经完全抵消右轮多报的 2%：无噪声理想基准下终点误差小于 `1e-10 m`。本课只加一种变化——每个 0.04 s 区间右轮读数再叠加种子化零均值噪声 σ=0.008 rad（约为单步增量 0.16 rad 的 5%），同一条路线重复 20 次，把**系统偏差**（20 次平均）和**随机分散**（标准差/分位）分开统计。

```powershell
uv run python -m embodied_learning.mobile_noise_demo --results results/mobile_noise_2026-09-03
```

默认暂停、0.25×、样本 #0、固定标定组。样本 #0–19 选择不同的噪声实现；切换"无噪声 / 固定标定 / 未标定"只改变系数 c，ε 序列不变（公共随机数）。图表同时画 20 次均值 ±1σ 阴影带与当前样本曲线，可对比"统计规律"和"这一次结果"。

直行 2.4 m / 12 s 终点（N=20）：固定标定组平均 2.52 cm、标准差 2.55 cm、单次 0.32–10.16 cm；未标定组平均 18.65 cm（偏差向量 +18.61 cm 的 Y 分量与第十五课 19.40 cm 同源）。方形 24 s：固定标定 1.96 ± 1.19 cm；未标定 15.70 ± 2.08 cm。结论：**c 消除系统偏差（平均误差毫米级），但逐区间噪声仍以随机游走方式累积，单次结果只能用分布描述**；多做几遍不会把未标定的系统误差平均掉。

复现需新目录：`uv run python -m embodied_learning.experiments.mobile_noise --output results/mobile_noise_my_run --runs 20 --seed 0`。新增 16 项测试；全量 331 项连续两次通过，Ruff 静态及格式检查通过（62 个 Python 文件）。完整原理、假设、统计口径与停止点见[第十七课讲义](docs/19-session-17-random-noise.md)。是否引入外部观测/滤波由任务精度要求决定，不自动加卡尔曼或 SLAM。

## 当前：第十八课，已知地标（控制点）观测与里程计对照

本课增加第二种信息来源：装在车上的传感器每 2 s 对三个**已知世界坐标的地标**测一次距离和方位角（测距噪声 ±1 cm、测角 ±0.57°），
用 2D Procrustes 闭式解（旋转+平移刚体配准，无缩放）从观测反算车体位姿——就是 GIS 控制点配准的机器人版。里程计与地标观测各自独立估计，**不做融合**。

```powershell
uv run python -m embodied_learning.landmark_demo --results results/mobile_landmarks_2026-09-03
```

默认暂停、0.25×、直行、样本 #0。地图上蓝=真值、紫=里程计、绿点=观测时刻、黑三角=已知地标；"下一观测"按钮跳到取样点。
右侧面板区分两种估计的误差属性；图表可切位置/朝向误差。

三条路线 × 20 种子（直行 12 s / 方形 24 s / 长直行 32 s）：里程计终点误差 2.52 / 1.96 / 9.56 cm（累积，σ 同步增大）；
地标观测全采样均值 0.93 / 0.97 / 2.24 cm（不随路线增长；长直行末端因距地标 4–6.6 m，误差升至 6.57 cm——观测精度随距离下降）。
关键现象：两次观测之间估计"保持旧值"，误差按车行驶距离线性增大（锯齿峰 ≈ 每周期车速×2 s≈40 cm），观测时刻误差跳回小值——
**观测给绝对基准、里程计填观测间隙**，这是最简融合的动机；本课不实现融合。

复现需新目录：`uv run python -m embodied_learning.experiments.landmark_observations --output results/mobile_landmarks_my_run --runs 20 --seed 0`。
新增 15 项测试；全量 346 项连续两次通过，Ruff 静态及格式检查通过（63 个 Python 文件）。
完整原理、假设与停止点见[第十八课讲义](docs/20-session-18-landmark-observations.md)。下一课：最简融合（观测重置 + 里程计填充），不引入 SLAM/ROS 2。

- [x] 本机环境审计
- [x] 确定分层仿真技术栈
- [x] 建立学习路线与第一阶段验收标准
- [x] 安装 `uv` 与项目专用 Python 3.12
- [x] 创建项目虚拟环境并安装 MuJoCo/Gymnasium
- [x] 运行第一个可视化和无界面仿真实验
- [x] 实现 PD 控制器并与随机动作基线比较
- [x] 处理小车长期漂移：基于真实平衡点的离散 LQR（2026-09-02，40 秒仿真验证）
- [x] 学员已运行 R=1 实验，结果保留于 `results/lqr_r1_my_run/`
- [x] 提供可暂停、单步、调速的教学回放和同条件 R 对照
- [x] 多 R 彩色曲线叠加，以及 20 组配对随机推力实验
- [x] 测量噪声闭环实验：固定 R=1，0×/1×/3× 各 20 回合，分别记录真实状态和传感读数
- [x] 全转动摆起专题：下方初态、强扰动、能量摆起/LQR 切换和失败边界；独立慢放演示
- [x] 平面 2R 机械臂：FK/MuJoCo 几何核验、解析双分支 IK、关节 PD 到达和两页教学窗口
- [x] Jacobian/差分/MuJoCo 三方核验、直线路径三组对照、奇异性探针和多色慢放；140 项测试通过
- [x] 多路径配对评估、先验几何拒绝、奇异初态真实失败、参考/执行误差分解与停稳条件审计；153 项测试通过
- [x] 逐点解析 IK、连续分支与速度规划边界、原清单配对回归和三色慢放；170 项测试通过
- [x] 同路径 8/4/2 秒对照、规划拒绝与真实力矩截断分层、可变时长慢放；192 项测试通过（桌面交互验收受窗口置前失败限制，详见第十二课）
- [x] 模型逆动力学前馈＋原 PD、49 状态方程核验、25 条配对路径与四类力矩回放；215 项测试通过
- [x] 差速车五种运动、世界/车体/传感器变换、故意错误映射与独立慢放；251 项测试通过
- [x] 编码器里程计与真值分离、三档右轮读数比例偏差、直行/方形配对与三指标慢放；新增 31 项测试独立通过
- [x] Tk 集成测试改为独立进程执行，保留断言与失败传播；重复全量通过，底层异常根因仍未确定
- [x] 第十六课：对照测量→求系数→换路线验证、错误尺子反例与可选落图；315 项全量通过
- [x] 第十七课：标定后种子化逐区间噪声、20 次重复统计分离偏差与分散、样本/公共随机数切换慢放；331 项全量通过
- [x] 第十八课：已知地标观测 + 2D Procrustes 位姿解算、累积 vs 不累积对照、保持旧值锯齿与长直行距离效应；346 项全量通过
- [ ] 学员解释：为什么要相对传感轴测方位角；观测频率如何决定"保持窗口"的长短
- [ ] 学员说明：哪个来源负责"绝对基准"、哪个负责"填补观测之间"

- [ ] 学员解释：系统偏差与随机分散如何分开统计；为什么固定 c 无法消除逐区间噪声
- [ ] 学员说明：要"某一次"更准，额外信息从哪里来（而不是再调 c）
- [ ] 学员解释：修正系数从何而来、为何分离标定/验证、错误基准为何不能靠多测几次消除
- [ ] 学员区分：真实运动、编码器测量、估计位姿与地标落图误差；解释为什么误差不必单调增加
- [ ] 学员解释：车体前方与地图方向的区别，以及正确坐标转换为何不等于可靠定位
- [ ] 学员观察并解释：为什么先向右走？R 变大后哪些指标改变？
- [ ] 学员区分：真实状态、含噪读数、控制动作；解释摆起与扶稳的切换
- [ ] 学员理解：相对关节角、逆解分支、到达与路径跟踪的区别，以及奇异位形的瞬时方向限制

## 参考文档

- [MuJoCo Python 官方文档](https://mujoco.readthedocs.io/en/latest/python.html)
- [Gymnasium MuJoCo 环境](https://gymnasium.farama.org/main/environments/mujoco/)
- [ROS 2 Jazzy 官方文档](https://docs.ros.org/en/jazzy/)
- [Gazebo 与 ROS 的推荐组合](https://gazebosim.org/docs/jetty/ros_installation/)
- [Isaac Lab 安装与系统要求](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/index.html)
