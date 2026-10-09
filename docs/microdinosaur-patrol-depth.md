# MicroDinosaur 巡检：单目深度与 ToF 校正

2026-10-09 已实际运行 UniDepth V2 Small 和 Depth Anything V2 Hypersim Small，并将 UniDepth 接入原 S288 策略的完整园区闭环仿真：**8/8 点，99.60 m，430.18 s，无新增深度停车、误报到达、园区接触或关节越限**。使用原 IMX219-77 彩色相机与 VL53L5CX 8×8 ToF 模拟配置。没有把 MuJoCo 深度缓冲伪装成算法输出。

[本轮小结与下一步](microdinosaur-patrol-review-20261009.md) · [公开技术讲解（B站，3分钟）](https://www.bilibili.com/video/BV1Mtp46EEw9/)

本机入口：<http://127.0.0.1:8770/depth_20261009_v3/report.html>。上半页切换模型和画面，显示 RGB＋有效 ToF 投影、原始深度、校正深度及独立评分真值；下半页观看动作、RGB、算法深度三栏视频。ToF 点是软件显示标记，940 nm 测距光不会直接成为彩色图像中的红色点；部分分区朝下且超出 RGB 视场，只绘制实际投影落在图内的有效点。

另有本地星瞳技术讲解版（`results/microdinosaur_patrol/20261009_xingtong_explainer/report.html`）：180 s、1920×1080、24 fps，配套旁白、字幕、逐镜时间线与来源核对。长距离精选片段 8×、关键到点 1×、停车对照 0.5×，跳段和静帧明确标出。历史 V07 盲走与 IMU 外环的收益、当前 RGB＋ToF 巡检闭环、新增深度的同条件收益分别说明；没有把不同场景换算为成功率提升，也没有声称新增深度降低整圈定位误差。原始录像不改写，音色模型与完整实验均沿用非商用研究范围。

## 同一输入的算法比较

冻结 39 张已有画面：园区 19、障碍 10、短程 6、ToF 失效 4。各案例预先交替分为开发 20 帧与留出 19 帧，仍来自相同世界与轨迹，不是独立实景泛化验证。模型只读 RGB 和模拟针孔内参。ToF 校正只读测距与估计外参。

| 留出画面 | 同帧原始近场 AbsRel | ToF 校正后近场 AbsRel | 可校正帧 | 推理中位 / P95 |
|---|---:|---:|---:|---:|
| UniDepth V2 Small | 82.2% | 14.5% | 17/19 | 53.8 / 60.2 ms |
| Depth Anything V2 Hypersim Small | 87.9% | 14.6% | 16/19 | 78.8 / 85.3 ms |

AbsRel 在 0.15–4 m 环境像素上计算，每帧取像素均值，再对帧平均。排除机器人、天空及测距点附近 7 px 区域；原始/校正比较使用同一批校正成功帧。完整原始结果包含全部 19 帧，失败校正不被填入成功误差。障碍表面留出 5 帧的 UniDepth 校正平均 AbsRel 为 8.4%。校正主要修复尺度，不能修复全部形状；UniDepth 留出远场 4–20 m 校正 AbsRel 仍为 44.1%，不能依赖它保证远距尺寸。

根据开发组的障碍误差和耗时选择 UniDepth 作闭环候选。模型权重沿用本地已校验文件，SHA256 `93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648`；DA-V2 权重 SHA256 `b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545`。Depth Pro 尝试运行后因显存压力与长耗时停止，5 张部分输出与 `aborted.json` 留存，不计入完整对照。

## 投影、校正与时延

1. 将每个有效 ToF 径向距离按 35° 下倾安装与候选平移变换到 RGB 光学坐标，再取光轴 Z 和像素位置。径向距离没有直接当作 Z。
2. 使用偶数分区拟合稳健中位尺度，奇数分区检查一致性；至少 8 个拟合与 6 个留出点，尺度在 0.25–4 内，log-MAD ≤0.35，留出中位相对误差 ≤25%。无效测距或质量不足时不生成校正图，不沿用旧尺度。
3. 原归档未保存 ToF 的完整估计位姿，对照用 ToF 采样前最近 RGB 的估计 FK/IMU 位姿，近似误差年龄单列。新闭环记录保存每个 ToF 样本的估计外参和位姿时间戳，按真实传感器采样/交付时间配对；不使用仿真机器人真值配准。
4. 模型运行在独立 Python 3.11/CUDA 环境，MuJoCo 3.10 保持原环境。最多 10 Hz，处理最近已交付 RGB 和不晚于它的已交付 ToF；不积压旧图。实测 IPC、推理、融合、显示图生成及数组写入耗时加入仿真交付延迟。相机 40 ms 假设与控制步量化另计。
5. ToF 测距拟合近场地面，密集深度中中心区域高出地面 >7.5 cm、Z 为 0.25–0.70 m、至少 80 像素支持的连通区域提供候选障碍。使用地面法向和估计相机姿态，只允许朝向地面的射线参与这一地面近障碍分支，避免模型将天空估成近距离时误停；相机俯仰和侧倾后的地平线按物理方向计算。
6. 每个候选区域必须另有至少两个中央 ToF 分区在相同角域确认地面以上返回，距离 <1.2 m，地面残差大于不确定性 `0.015 + d*sin(45°/16) + 3*(0.003+0.020*d/4) + abs(bias)*d`。角域按分区半角覆盖，不将单个中心射线视为整片区域。使用实际归档的径向测距与偏差配置。神经网络独自给出的近墙形状不能触发这一辅助停车条件。连续两次建议且最近结果年龄 ≤200 ms 才增加停车条件。原 ToF 未知/障碍与相机陈旧停车保留优先级。上述不确定性沿用原模拟假设，并非实测安全置信界；该分支不证明悬空物、完整机体净空或地平线以上目标可通行。

第一次闭环把原始和校正深度压缩保存，单次写盘约 122 ms，全部深度结果过期：短程结果年龄中位 293 ms，0/126 可用于控制。该失败保留在 `20261009_depth_courtyard_s72`，障碍初版也保留。后来仅将保存改为无损、非压缩数组，并按相同种子重跑；没有放宽 200 ms 门限。

保存格式修复后的首轮短程结果可用时年龄中位 169.7 ms、P95 270.0 ms；计入 20 ms 控制网格量化后实际交付年龄中位 180 ms、P95 280 ms，过期结果排除。这些是保留的旧版本记录。最终整圈实际交付 **3,897/3,897** 个新鲜结果，另有1个计算结果尚未交付就结束；每段路线均有新鲜深度，最长相邻新鲜结果间隔约200 ms。纯推理中位26.9 ms、P95 28.4 ms；实际交付结果年龄中位100 ms、P95 120 ms、最大180 ms。整圈模拟 430.18 s 耗费墙钟约 956.01 s，包含模型、模拟、绘制和归档；实际新鲜计数和浮点边界以最终 `live_audit.json` 为准。纯模型时间和完整结果年龄不能互换，此交付没有证明端到端实时运行或机器人板载速度。

## 闭环动作对照

同一 S288 策略、场景、种子、传感器噪声与执行器配置，仅增加深度辅助。没有改原 CAD 或 ONNX 权重，也没有重采样失败种子。

| 场景 | 原 RGB＋ToF | UniDepth＋ToF 辅助 | 接触 / 越限 |
|---|---|---|---|
| 全园区，种子73，15 ms / 10.8 V / 80% 暂定扭矩上限 / ToF +5% 偏差 | 8/8，99.595 m，430.18 s | 8/8；动作 trace 完全一致，无新增深度停车 | 0 / 0 |
| 短程，种子72，15 ms / 10.8 V / 80% 暂定扭矩包络 | 3/3，4.181 m | 3/3，4.181 m；动作 trace 完全一致 | 0 / 0 |
| 障碍，种子83，15 ms / ToF +11% 偏差 | t=1.98 s 发停车指令 | t=1.90 s 发停车指令，提前 80 ms；停后净位移 4.84 cm，路径 12.47 cm，末速度 0.140 mm/s | 0 / 0 |
| ToF 失效，种子81，15 ms | t=2.12 s 发停车指令 | t=2.12 s；动作 trace 完全一致，算法不解除未知停车 | 0 / 0 |
| 相机失效，种子82，15 ms | t=2.20 s 发停车指令 | t=2.20 s；动作 trace 完全一致，算法不解除陈旧停车 | 0 / 0 |

最终五项运行全部 **381,504 个物理步**独立重放，qpos/qvel 最大逐元素误差为 0，所有步接触与关节越限为 0。五个视频 **4,772 帧**完整解码，关键画面、失效期间的未知/过期深度显示与八个到点相机记录已实际查看。障碍和失效停车不计为巡检路线成功。整圈计算 3,898 次深度，3,857 次接受尺度校正，41 次按未知处理；67 帧密集候选缺少测量确认，248 帧排除了地平线以上候选。无误停车不等于模型没有形状误判。首轮旧版三项结果、32,240 步重放和405帧视频仍保留。

已完成这组硬件模拟与算法的整圈仿真验收。深度没有改变正常整圈动作，在固定近障碍案例中提供了较小的提前停车收益，并显示更密集的距离估计。未覆盖坑洼、细杆、透明/高反射目标、室外日光、设备缺陷识别、非平地定位或完整 CAD 碰撞。整圈定位误差 P95 0.433 m、最大0.684 m，与原基线一致，仍不能外推到无标记或复杂路面。左膝扭矩上限饱和约 34.84%、RMS 约 0.269 N·m；暂定包络通过不代表 S288 连续功率或温升合格。硬件安装、真实镜头标定、板载时延/功耗及实物停车仍需独立验收，所有运行保持 `hardware_released: false`。

## 整圈发现的误停与固定输入检查

首轮深度整圈 `20261009_depth_campus_s73_fast` 保持原种子73、15 ms、10.8 V、80% 暂定扭矩上限及 ToF +5% 偏差。只完成 1/8 点，在 t=94.02 s 被深度建议停车；末速度 0.275 mm/s、净位移 1.67 cm，无接触或关节越限。77,648 个物理步独立重放状态完全一致。该结果属于路线失败，记录与视频保留。

实际查看触发前两张图，模型把天空估成约 0.47–0.68 m；旧分支将地面以上的近距离连通区域当作地面近障碍。朝下 ToF 的尺度锚点均在地面，不能约束天空形状。修复只增加物理地平线过滤，没有修改 RGB、模型权重、尺度拟合、200 ms 门限或路线种子。

在四个冻结运行的 **1,119 张**已保存深度输出上重放旧函数与当前函数，传感器外参直接来自各帧已保存的估计 FK/IMU 位姿，无真值输入。旧状态与连通区域逐帧一致，最近距离允许 NumPy 版本造成的 ≤0.1 µm 插值差异。新函数排除整圈的全部 7 张天空障碍建议，同时保留障碍试跑 29 张中的 28 张；这只证明固定输出的反事实变化，不替代新闭环重跑。详见 `horizon_counterfactual.json`。新增三个独立几何检查分别覆盖近天空误判、地面近障碍与相机低头时图像上半部的有效障碍；加上既有尺度/年龄与传感器契约，共 21 项通过。

地平线过滤版 `20261009_depth_campus_s73_horizon` 越过原误停位置，但在侧墙路段 t=113.94 s 再次误停，只完成 2/8；93,584 个物理步重放一致，无接触或越限。实看图像显示模型将侧墙估得过近，而该方向的中央 ToF 仍观测地面，没有超出不确定性边界的正高度返回。最终增加上述同角域双分区确认。五个冻结运行的 **2,107 张**输出以不变模型、相机、测距和估计姿态再次比较，两个整圈失败中的障碍建议均被排除；早停后的原障碍片段也不再满足更严格的测量确认，详见 `measurement_counterfactual.json`，没有将其写成有效障碍回归通过。

因此另跑完整新障碍闭环 `20261009_depth_obstacle_s83_confirmed`，实际继续接近后获得四个分区确认，在 t=1.90 s 由深度分支发出停车，较原 ToF 基线 1.98 s 提前 80 ms，末速度 0.140 mm/s、净位移 4.84 cm，独立重放的停后路径 12.47 cm。3,936 个物理步状态一致，无接触或越限。初版 0.72 s 的提前量仅属于旧算法，不能当作最终版本收益。又增加无测量确认的墙面形状误判、单分区/偏差不确定性检查，最终契约总数为 23（原传感器12、深度11），不是全仓回归。

## 复现与文件

以下为作者本机的归档复现入口，依赖本机实验脚本、独立深度环境、机器人源 CAD 与策略。此次主页更新发布研究说明、指标与预览，不分发完整执行环境和大型原始数据；新克隆不能仅凭本页命令直接重放。

输出在 `results/microdinosaur_patrol/depth_20261009_v3/`，原始冻结输入、评分真值与算法输出分目录保存，完整数据为 `manifest.json`、`comparison.json`、`live_audit.json`。归档数组只在本机输出，不含发布模型权重。

```powershell
# MuJoCo 3.10 建立固定输入与独立评分数据；输出目录必须不存在
& 'D:/microduck_rl/.venv/Scripts/python.exe' scripts/microdinosaur_depth_dataset.py --out results/microdinosaur_patrol/depth_new

# 独立深度环境跑模型，不读 scoring_only 数据
& './monocular-depth/.venv/Scripts/python.exe' scripts/microdinosaur_depth_infer.py results/microdinosaur_patrol/depth_new --model unidepth
& './monocular-depth/.venv/Scripts/python.exe' scripts/microdinosaur_depth_infer.py results/microdinosaur_patrol/depth_new --model da2_metric

# 动作控制保持原物理环境，通过持久 stdio 进程调用深度模型
& 'D:/microduck_rl/.venv/Scripts/python.exe' scripts/microdinosaur_patrol.py --out results/microdinosaur_patrol/depth_trial_new --scenario obstacle --seed 83 --seconds 8 --command-ms 15 --tof-relative-bias .11 --depth-mode assist
& 'D:/microduck_rl/.venv/Scripts/python.exe' scripts/replay_microdinosaur_patrol.py results/microdinosaur_patrol/depth_trial_new

# 整圈与最终确认分支；输出使用新目录，不覆盖已保存的失败
& 'D:/microduck_rl/.venv/Scripts/python.exe' scripts/microdinosaur_patrol.py --out results/microdinosaur_patrol/depth_campus_new --scenario campus --seed 73 --seconds 600 --command-ms 15 --motor-curve --voltage 10.8 --effort-fraction .8 --tof-relative-bias .05 --depth-mode assist
& 'D:/microduck_rl/.venv/Scripts/python.exe' scripts/replay_microdinosaur_patrol.py results/microdinosaur_patrol/depth_campus_new

# 11 个深度契约与原 12 个传感器契约；不是全仓 full 回归
& 'D:/microduck_rl/.venv/Scripts/python.exe' scripts/test_microdinosaur_depth.py
& 'D:/microduck_rl/.venv/Scripts/python.exe' scripts/test_microdinosaur_patrol.py
```

上游：[UniDepth](https://github.com/lpiccinelli-eth/UniDepth)、[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)、[Depth Pro](https://github.com/apple-aiml-research/ml-depth-pro)。当前 UniDepth 软件为 CC-BY-NC 4.0，本次用于本地研究仿真；商业部署需另行确认可用许可和模型选型。
