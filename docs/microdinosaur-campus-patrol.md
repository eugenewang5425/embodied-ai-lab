# S288 MicroDinosaur：彩色相机与 ToF 园区巡检

[![巡检技术讲解封面，非定量指标图](img/microdinosaur-patrol-explainer.jpg)](microdinosaur-patrol-review-20261009.md)

动态相机/ToF/深度图见本机报告，定量结果见本文表和对应 benchmark；封面不替代科研结果图。


2026-10-09 的名义配置整圈实验完成 **8/8 点，97.89 m，427.72 s**；新种子组合压力配置也完成 **8/8 点，99.60 m，430.18 s**。两轮无误报到达。独立重放检查两轮全部 **686,320 个物理步**，园区接触与关节越限均为 0，位置和速度逐元素重放误差为 0。这里的“真实到点”指仿真物理位置符合评分要求。

这是使用实际 ONNX 策略驱动关节的闭环仿真。当前交付支持已知初始位置、平整道路、预布彩色巡检标记的导航与停留，以及传感器异常停车。尚未完成设备缺陷识别或实机部署。所有记录保持 `hardware_released: false`。

后续已加入真实单目深度模型、ToF 尺度校正，并完成同一种子压力配置的整圈 **8/8 点**以及障碍、相机/ToF 失效和短程对照，见[深度扩展记录](microdinosaur-patrol-depth.md)。深度版整圈 99.60 m、430.18 s，与原基线动作完全一致；两个误停版本也保留。下文仍记录原 RGB＋ToF 的基准成绩。

## 打开结果

公开入口：[3分钟技术讲解（B站）](https://www.bilibili.com/video/BV1Mtp46EEw9/) · [新实验小结](microdinosaur-patrol-review-20261009.md)。

本机报告：`results/microdinosaur_patrol/report.html`。用浏览器打开可以选择成功、异常与保留的失败试验；拖动回放时间时，地图上的实际位置和估计位置同步更新。全程统计与当前播放时刻分开显示。

```powershell
& 'D:/microduck_rl/.venv/Scripts/python.exe' scripts/serve_microdinosaur_patrol.py
# 浏览器打开 http://127.0.0.1:8770/report.html
```

每个目录保存 `summary.json`、`protocol.json`、`scene_audit.json`、`sources.json`、原始 RGB 图像、ToF 数组、关节状态、物理步控制指令和视频。最终验证汇总见同目录的 `validation_summary.json`、`media_validation.json`；实际视频关键帧为各试验中的 `visual_qa.png`。

报告逐点提供原始 RGB 图像链接，`inspection_records.json` 保留该图像的采样时间和到点事件。当前图像记录巡检标记及其周围画面，没有设备异常分类结果。

可随仓库保留的紧凑验证记录为[本轮基准 JSON](benchmarks/microdinosaur-campus-patrol-20261009.json)。它核对每轮入口快照、CAD/策略哈希、独立物理重放与预期结果，保留失败对照；大型原始数据仍在被 Git 忽略的 `results/` 中。

原始视频 HUD 在本帧到点评分前刷新；因此整圈视频的最后一帧显示第 8 点正在停留、先前完成 7 点。随后同一物理状态通过第 8 点评分，完整结果见报告表格与 `summary.json`。这个计数顺序没有改写原始视频或动力学记录。

## 调用的机器人与动作模型

机器人源项目为 `D:/项目/miro_dinosaur`，全程只读。本次新增实现留在本仓库的 `scripts/`，不覆盖现用 CAD、策略或训练源码。

| 输入 | 冻结来源与核验 |
| --- | --- |
| 现用 CAD | `design_source/current/MicroDinosaur_v1.blender`，19 个 S288、刚性尾巴；SHA256 `0a4f86aefea24e0d5260c837849a16164ec83cc2d364f58e6c03947de95ce286` |
| 运动策略 | `design_source/simulation/research/20260913_handoff/v7_reference.onnx`；SHA256 `53120a401d02f124b35361d791eb74748a7601550c701c511d62cbad9cddff96` |
| 物理基线 | 原 native v07；训练 XML 编译后，质量、惯量、关节坐标与范围、阻尼、驱动力等数组逐元素一致 |
| 质量与碰撞 | 1.0982148 kg；增加 20 个零质量 CAD 凸包，仅与园区障碍碰撞。地面及机器人原有自碰撞语义保留，未采用 v07.1/R7 候选 |
| 控制 | 原 50 Hz 动作策略、头部/躯干 IMU 姿态估计、航向控制、关节滤波、S288 协议量化与延迟；物理步长 1.25 ms |

每个试验在启动时保存 `entry_snapshot.py` 和哈希，之后脚本修改不会改变该试验的实现来源。源 CAD 与策略的运行前后哈希一致。

## 硬件模拟配置

用户允许自主选用体积合适的国产品牌彩色摄像头。本轮采用**微雪 Waveshare IMX219-77** 作为尺寸与光学候选，品牌模块使用 Sony IMX219 感光芯片。没有把它描述为国产感光芯片，也没有把 PCB 尺寸当成安装适配证明。

| 项目 | 规格来源 | 本轮仿真假设 |
| --- | --- | --- |
| 彩色相机 | [微雪官方资料](https://www.waveshare.com/wiki/IMX219-77_Camera)：25×24 mm，79.3° 对角视场 | 参考该模块的尺寸与视场 |
| 相机模式 | 待实物确认 | 640×480、4:3、30 Hz；按对角视场推导针孔内参，约 482.7 px 焦距；40 ms 延迟与 0.8 灰度级噪声是假设 |
| ToF | [ST VL53L5CX DS13754](https://www.st.com/resource/en/datasheet/vl53l5cx.pdf)：8×8、15 Hz、水平/垂直 45°、对角 65°、2 cm–4 m | 按规定的 8×8 / 15 Hz 配置 |
| ToF 模型 | 保留项目的多区测距基线 | 每区一个中心几何射线，毫米量化；状态 5 有效，255 无目标，其他状态未知；30 ms 延迟，误差标准差 `3+20*d/4` mm，`d` 用米 |
| 安装 | 相机来自 CAD `head_camera` 光学坐标 | ToF 相对光学坐标偏移 `(6,16,6)` mm、向下 35° 为独立候选；未新增传感器质量、未完成安装 CAD 验证 |
| S288 | 源项目研究配置 | 名义刚度 7、阻尼 0.8、峰值扭矩上限 0.6 N·m；已加入原项目暂定速度/电压包络和扭矩降额敏感性试验 |

RGB 提供像素、时间戳和估计外参，导航没有相机深度输入。ToF 保留无效与陈旧状态，不把未知区域当成畅通道路。30/15 Hz 采样在 50 Hz 控制网格上调度，时间抖动小于 20 ms，实际采样与交付时间保存于记录中。

相机畸变、裁切、ISP、曝光和滚动快门尚未标定。ToF 的中心射线与误差项是几何近似，未复现区内混合目标、红外信号强度、日光和反射率导致的失效。±5% / +11% 的测试只是相关测距误差敏感性，不能解释为所有室外条件的硬件精度保证。

## 巡检与停车逻辑

复用第 66 课的 24×24 m 米制园区几何和 8 个巡检点，原课程的轮式机器人、雷达定位及“知道真实位置”控制没有用于这台机器人。路线沿已知道路逐点行进；遇到障碍停车，尚未实现动态绕行。

定位使用延迟编码器的足端运动学、躯干/头部 IMU，以及已知尺寸和位置的彩色标记。近距离用 RGB 四角的平面单应性估计相机位姿，有限幅度地修正里程计。机器人抬头看向标记的动作继续经过原头部反馈与执行器限制。

到点门控要求估计距离约 25 cm 内、最近看到当前标记、停留 2 秒；已有停留采用 33 cm 滞回避免抖动。评分单独要求物理距离 ≤35 cm、末速度 ≤40 mm/s。仿真真值只用于评分、跌倒与接触终止，不送入导航器。标记帧和实际速度保留在每个到点事件中。

ToF 用下半区近距离有效回波拟合局部地面，再判断高于地面的障碍。中央有效分区不足、地面无法拟合、数据超过 200 ms 或相机断流时停车并锁存。本轮只验证明显障碍，不能据此放行 1–3 cm 台阶、细杆或坑洞。停车指令不是断电急停，原姿态控制仍有收敛动作。

## 验证批次

| 试验目录 | 条件 | 已验证结果 |
| --- | --- | --- |
| `20261009_campus_s31_fixed` | 名义整圈、种子 31、10 ms 指令延迟 | 8/8，97.89 m，427.72 s；全部物理步重放一致 |
| `20261009_campus_s73_derated` | 新种子 73、15 ms、10.8 V、暂定电机包络、80% 扭矩上限、ToF +5% | 8/8，99.60 m，430.18 s；各点真实距离 22.7–24.7 cm，全部物理步重放一致 |
| `20261009_courtyard_s71_fixed` | 新种子 71、15 ms、ToF −5% | 3/3，4.04 m；全部物理步重放一致 |
| `20261009_courtyard_s72_derated` | 新种子 72、15 ms、10.8 V、暂定电机包络、80% 扭矩上限 | 3/3，4.18 m；峰值 ≤0.48 N·m，全部物理步重放一致 |
| `20261009_tof_outage_s81_fixed` | 2–5 s 强制 ToF 无效 | 2.12 s 发出停车；后续净位移 8.30 cm，实际路径 15.48 cm |
| `20261009_camera_outage_s82_fixed` | 2–5 s 阻断相机交付 | 2.20 s 发出停车；后续净位移 1.60 cm，实际路径 5.34 cm |
| `20261009_obstacle_s83_fixed` | 未标入路线的 24 cm 高障碍、ToF +11%、15 ms | 1.98 s 发出停车；后续净位移 4.86 cm，实际路径 13.75 cm |
| `20261009_campus_dev31` | 修正前的完整路线 | 6/8 后定位失效并停车；保留全部失败记录 |

上述异常停车场景的巡检任务均未完成，不计入到点成功。新三项异常试验全部物理步接触与越限为 0，结束速度 <4 mm/s；单次功能验证不证明最坏停车距离。降额试验同时改变多个参数，只作为组合压力条件，不用来给某一个因素作因果归因。

修正前，标记材质发光造成 RGB 饱和，紫色标记被渲染为粉色。修正将发光量降为 0.1，并要求目标颜色与其他标记之间有识别余量。固定视角的实际渲染检查对 8 个标记均只识别正确身份，见 `marker_render_final/audit.json` 和 `rendered_tags.png`。脚本另有 12 项传感器契约检查，包括无效、陈旧、障碍、倾斜地面估计、RGB 位姿和不同标记身份。

名义整圈的位置误差 P95 **1.16 m**、最大 **1.56 m**，靠近标记后收敛到各点约 **23–25 cm** 的停止距离。它说明当前足端里程计仍会漂移，不能据路线成功声称厘米级全程定位。名义整圈左膝 RMS 扭矩约 **0.305 N·m**、饱和占比约 **12.1%**、高于 0.5 N·m 累计 **63.60 s**；这些是动力学筛查量，不能替代连续额定、温升、磨损和续航测试。

降额整圈的左膝 RMS 约 **0.269 N·m**，相对于降低后的动态扭矩边界，饱和占比约 **34.8%**。降低峰值后仍能完成路线，同时也更频繁地触及新的扭矩边界；它不证明存在充足的负载、热或扰动余量。

## 基准结果图（从冻结摘要绘制）

![两种历史配置的到点数、实际路径与仿真时间，非算法消融](img/microdinosaur-baseline-metrics.png)

数据直接来自[本轮基准JSON](benchmarks/microdinosaur-campus-patrol-20261009.json)。名义与压力配置的种子及参数不同，因此此图只展示两轮记录，不能把差值归因于某项算法。异常停车另列为降级对照，不加入巡检成功分子。

## 复现

以下命令依赖作者本机的实验脚本、机器人源 CAD/策略与独立环境。此次主页更新提供研究说明、指标与预览，完整执行环境和大型原始数据另存；新克隆不能仅凭本页命令直接重放。

使用现有 `D:/microduck_rl/.venv/Scripts/python.exe`，MuJoCo 3.10.0；主仓库另一个 MuJoCo 3.11 环境不能重放冻结模型。源目录可以通过 `--robot-root` 调整。输出目录必须新建，不覆盖旧试验；种子不因失败而重抽。

```powershell
& 'D:/microduck_rl/.venv/Scripts/python.exe' scripts/microdinosaur_patrol.py --scenario campus --seed 31 --seconds 600 --out results/microdinosaur_patrol/my_new_campus
& 'D:/microduck_rl/.venv/Scripts/python.exe' scripts/replay_microdinosaur_patrol.py results/microdinosaur_patrol/my_new_campus
& 'D:/microduck_rl/.venv/Scripts/python.exe' scripts/test_microdinosaur_patrol.py
& 'D:/microduck_rl/.venv/Scripts/python.exe' scripts/test_render_microdinosaur_markers.py results/microdinosaur_patrol/my_new_campus --output results/microdinosaur_patrol/my_new_tag_check
& 'D:/microduck_rl/.venv/Scripts/python.exe' scripts/report_microdinosaur_patrol.py --runs my_new_campus
```

组合压力配置追加 `--command-ms 15 --motor-curve --voltage 10.8 --effort-fraction 0.8 --tof-relative-bias 0.05`。异常场景分别为 `--scenario obstacle`、`tof_outage`、`camera_outage`。

独立重放直接使用保存的物理步控制、完整初始积分状态及动态扭矩边界，没有重新运行策略或导航。它验证动力学记录可复现，尚不是独立传感器/决策重放。

## 进入实物验证还需什么

下一阶段可以从受控平地的短程、预布标记巡检开始。当前缺少的实物输入具体为：摄像头板与现用 M12 载架的尺寸/镜头适配、相机内参及时间戳标定、ToF 安装与盖板校准、阳光/地材下的状态和误差记录、19 个 S288 的实测输出曲线及电流温升、连续行走电池压降和总线时延，以及实际传感器和执行器驱动。

获得这些测量后，用同一到点、停车路径、接触、姿态和电机负荷指标重跑短程；再扩展到长路线、光照变化、路面变化与设备缺陷检查。当前记录不支持直接在真实开放园区无人值守运行。

## 原始来源与本地适配（2026-10-09补引）

[IMX219-77 Camera](https://www.waveshare.com/wiki/IMX219-77_Camera)（Waveshare / 微雪，产品规范）；[VL53L5CX Datasheet](https://www.st.com/resource/en/datasheet/vl53l5cx.pdf)（STMicroelectronics，产品规范）；[MuJoCo 官方文档](https://mujoco.readthedocs.io/en/stable/overview.html)（维护团队，持续更新）；[Probabilistic Robotics](https://robots.stanford.edu/probabilistic-robotics/)（Sebastian Thrun / Wolfram Burgard / Dieter Fox，2005）。

保留外部项目原 CAD/ONNX 的只读哈希；国产品牌相机候选和规定 ToF 用硬件规范约束模拟，模式/外参/噪声/时延仍有假设，中心射线不等于实物区域回波。 [完整采用关系与引用规则](references.md)。


## 教学补充：变量怎样影响巡检

先从信息来源理解：ONNX 决定关节动作，编码器/IMU填补标记之间的运动，RGB标记提供当前位置校正与目标确认，ToF提供近场测距和未知停车。地图中预设目标不等于机器人已经观测到目标；测距安全条件不等于整点任务通过。

| 量 | 含义 / 单位 | 来源 | 改变它影响什么 |
| --- | --- | --- | --- |
| `dt` | 物理步长，s；本批0.00125 | 冻结 native v07 仿真 | 接触积分及物理重放，不能随视频倍速修改 |
| `f_policy` | 策略频率，Hz；50 | 原 ONNX 调用契约 | 控制更新与动作排队；不等于相机或 ToF 频率 |
| `T_rgb` / `T_tof` | RGB/ToF采样周期，s | 候选30Hz相机 / 规定15Hz ToF模拟 | 观测间隔、失效判断与停车可用时刻 |
| `r_i` | ToF第i分区径向测距，m | 射线与测距模拟 | 近场地面/障碍判断；不是光轴Z或完整区域回波 |
| `t_age` | 当前时刻减可用观测的采样时刻，s | 采样、交付与控制时间戳 | 陈旧观测应转未知；不能用未来数据减误差 |
| `e_pose` | 实际与估计位姿差，m/rad | 评分器与估计记录分开 | 校正是否有效及目标停靠偏差；评分真值不输入定位器 |

## 思考题

1. 为什么 50Hz 动作、30Hz RGB 和15Hz ToF不能视为同一个同步传感器？怎样判断控制此刻能用哪帧？
2. 彩色标记提供的位置锚点与 ToF提供的距离信息各自消除什么歧义？拿掉标记会留下哪些问题？
3. 两轮都没有接触，为什么仍要分别检查真实到点、误报、停车后路径和电机代价？
4. 模拟每分区一条中心射线与实物区域回波有什么差异？哪些室外条件可能破坏该近似？
5. 历史盲走和新园区并非同场景对照，应如何设计配对才能量化感知带来的增益？
