# 运行、生成记录与回放

从仓库根目录运行 PowerShell 命令。Python 3.12 与主环境依赖以 `pyproject.toml` / `uv.lock` 为准；ROS 2 Jazzy、WSL 与外部模型环境见[环境审计](00-environment-audit.md)。本次仅整理文档，没有升级环境或重新运行历史批次。

```powershell
uv sync --locked
uv run python -m embodied_learning.env_check --steps 300 --seed 7
uv run python -m embodied_learning.viewer --policy pd --seconds 10 --seed 7
```

`results/` 不入 Git。克隆后先生成对应记录，再打开回放；新运行使用新的输出目录。历史命令中的固定 `results/...` 路径是原记录位置，已有目录时按讲义改用新目录，不覆盖失败。

<a id="latest"></a>

## 最新 NAV / MAN 窗口

**导航：第 69 课第十四轮。** 先生成 75 回合正式记录，再打开指定的街区箱体回合。窗口一键同步相机、三维场景、雷达点、当前地图、目标和统计；可切换参照与改进算法、查看失败、暂停和拖动时间。[讲义](69-physical-height-navigation.md)

```powershell
uv run python -m embodied_learning.experiments.navigation_height_study --workers 3 --output results/navigation_reinforcement_v14
uv run python -m embodied_learning.navigation_study_demo --lesson 69 --footprint results/navigation_reinforcement_v14 --case 69_street_crate_37__delay_2__bypass_crate --play
```

**机械臂：第 73 课第三轮。** 先生成 126 回合正式记录，再打开实际指面校准组。外部相机、腕部相机、三维场景、目标、接触力和统计一键共同切换；可以查看完成、掉落和接触拒绝。[讲义](73-so101-contact-feedback.md)

```powershell
uv run python -m embodied_learning.experiments.so101_contact_study --workers 3 --output results/so101_contact_v4
uv run python -m embodied_learning.manipulation_demo --results results/so101_contact_v4 --position p04a1 --method surface --play
```

已有上述完整记录时，直接运行对应第二条命令即可。导航 60/60 是限定仿真范围的验证；机械臂 24/27 仍未达到稳定抓取。两者 RGB 均为存档位姿重渲染，尚未参与控制，下一步见[导航阶段复盘](navigation-stage-review-69-71.md)和[机械臂路线](manipulation-roadmap.md)。逐课原理、变量、数据、思考题和复现步骤见下方课程索引。

<a id="commands"></a>

## 逐实验命令存档

以下保存原主页的课程命令，去除了完全相同的代码块。`--lesson` 等参数继续使用历史编号；先从[编号索引](experiment-index.md)进入对应讲义，核对具体参数、依赖与数据版本。某个早期脚本存在不代表可以跳过当前阶段门槛。

```powershell
uv sync
uv run python scripts/test.py quick
uv run python -m embodied_learning.env_check --steps 300 --seed 7
uv run python -m embodied_learning.experiments.pd_comparison
uv run python -m embodied_learning.viewer --policy pd --seconds 10 --seed 7
uv run python -m embodied_learning.experiments.lqr_comparison --output results/lqr_my_run
uv run python -m embodied_learning.viewer --policy lqr --seconds 15 --seed 7
```

```powershell
uv run python -m embodied_learning.teaching_demo
```

```powershell
uv run python -m embodied_learning.teaching_demo --results results/lqr_r1_my_run
```

```powershell
uv run python -m embodied_learning.teaching_demo --push-results results/lqr_push_2026-09-02 --seed 100
```

```powershell
uv run python -m embodied_learning.experiments.lqr_disturbance --output results/lqr_push_my_run
```

同一命令已列于本页最新窗口或前一命令组；参数含义见对应讲义。

同一命令已列于本页最新窗口或前一命令组；参数含义见对应讲义。

```powershell
uv run python -m embodied_learning.campus_patrol_demo --play
```

```powershell
uv run python -m embodied_learning.map_separation_demo --play
```

```powershell
uv run python -m embodied_learning.photo_room
uv run python -m embodied_learning.experiments.photo_room_validation --layout results/photo_room_v1/layout.json
uv run python scripts/render_photo_room_replay.py
```

```powershell
uv run python -m embodied_learning.experiments.so101_grasping --output results/so101_my_run
uv run python -m embodied_learning.manipulation_demo --results results/so101_my_run --play
uv run python scripts/validate_so101_grasping.py results/so101_my_run
uv run python scripts/test.py full tests/test_so101_manipulation.py
```

```powershell
uv run python -m embodied_learning.experiments.lqr_measurement_noise --output results/lqr_noise_my_run
```

```powershell
uv run python -m embodied_learning.teaching_demo --noise-results results/lqr_noise_2026-09-02 --seed 200 --seconds 10
```

```powershell
uv run python -m embodied_learning.swingup_demo --results results/swingup_2026-09-02
```

```powershell
uv run python -m embodied_learning.arm_demo --results results/arm_reaching_2026-09-02_v2
```

```powershell
uv run python -m embodied_learning.arm_path_demo --results results/arm_path_2026-09-02
```

```powershell
uv run python -m embodied_learning.arm_path_demo --results results/arm_path_batch_2026-09-02_v2/trials/singular_inward
```

```powershell
uv run python -m embodied_learning.experiments.arm_path_batch --seed 400 --per-group 12 --output results/arm_path_batch_my_run
```

```powershell
uv run python -m embodied_learning.arm_path_demo --results results/arm_ik_comparison_2026-09-02/trials/singular_inward
```

```powershell
uv run python -m embodied_learning.arm_path_demo --results results/arm_timing_2026-09-02/trials/interior_00
```

```powershell
uv run python -m embodied_learning.arm_path_demo --results results/arm_feedforward_2026-09-03_v2/trials/singular_inward
```

```powershell
uv run python -m embodied_learning.mobile_demo --results results/mobile_frames_2026-09-03
```

```powershell
uv run python -m embodied_learning.experiments.mobile_frames --output results/mobile_frames_my_run
```

```powershell
uv run python -m embodied_learning.odometry_demo --results results/mobile_odometry_2026-09-03
```

```powershell
uv run python -m embodied_learning.calibration_demo
```

```powershell
uv run python -m embodied_learning.mobile_noise_demo --results results/mobile_noise_2026-09-03
```

```powershell
uv run python -m embodied_learning.landmark_demo --results results/mobile_landmarks_2026-09-03
```

```powershell
uv run python -m embodied_learning.fusion_demo --results results/mobile_fusion_2026-09-03
```

```powershell
uv run python -m embodied_learning.ros2_system_demo
```

```powershell
uv run python -m embodied_learning.goal_demo
```

```powershell
uv run python -m embodied_learning.threshold_demo
```

```powershell
uv run python -m embodied_learning.pinhole_demo --results results/mobile_pinhole_2026-09-03
```

```powershell
uv run python -m embodied_learning.monocular_metric_demo --results results/mobile_monocular_2026-09-05
```

```powershell
monocular-depth/.venv/Scripts/python.exe monocular-depth/bench/affine_check_pinhole.py
uv run python -m embodied_learning.experiments.real_depth_affine --input <bench npz> --output results/real_depth_affine_my_run
uv run python -m embodied_learning.real_depth_demo --results results/real_depth_affine_2026-09-05
```

```powershell
uv run python -m embodied_learning.intrinsics_demo --results results/camera_intrinsics_2026-09-05
```

```powershell
uv run python -m embodied_learning.icp_demo --results results/point_cloud_icp_2026-09-05
```

```powershell
uv run python -m embodied_learning.grounding_demo --results results/visual_grounding_2026-09-05
```

```powershell
uv run python -m embodied_learning.bc_demo --results results/bc_imitation_2026-09-05
```

```powershell
uv run python -m embodied_learning.ppo_demo --results results/ppo_swingup_2026-09-06
```

```powershell
uv run python -m embodied_learning.residual_demo --results results/residual_swingup_2026-09-06
```

```powershell
uv run python -m embodied_learning.pbrs_demo --results results/pbrs_swingup_2026-09-06
```

```powershell
uv run python -m embodied_learning.dapg_demo --results results/dapg_swingup_2026-09-06
```

```powershell
uv run python -m embodied_learning.goexplore_demo --results results/goexplore_swingup_2026-09-06
```

```powershell
uv run python -m embodied_learning.twophase_demo --results results/twophase_swingup_2026-09-06
```

```powershell
uv run python -m embodied_learning.sac_demo --results results/sac_swingup_2026-09-06
```

```powershell
uv run python -m embodied_learning.dagger_demo --results results/dagger_swingup_2026-09-06
```

```powershell
uv run python -m embodied_learning.act_demo --results results/act_swingup_2026-09-06
```

```powershell
uv run python -m embodied_learning.combo_demo --results results/combo_swingup_2026-09-06
```

```powershell
uv run python -m embodied_learning.rl_goal_demo --results results/rl_goal_reaching_2026-09-06
```

```powershell
uv run python -m embodied_learning.rl_arm_demo --results results/rl_arm_reaching_2026-09-06
```

```powershell
uv run python -m embodied_learning.staged_demo --results results/staged_verification_2026-09-06
```

```powershell
uv run python -m embodied_learning.act_torch_demo --results results/act_torch_reaching_2026-09-06
```

```powershell
uv run python -m embodied_learning.experiments.grid_nav --output results/grid_nav_my_run --seed 0
uv run python -m embodied_learning.grid_nav_demo --results results/grid_nav_my_run
```

```powershell
uv run python -m embodied_learning.experiments.nav_pose_error --output results/nav_pose_error_my_run --seed 0
uv run python -m embodied_learning.nav_pose_error_demo --results results/nav_pose_error_my_run
```

```powershell
uv run python -m embodied_learning.experiments.scan_slam --output results/scan_slam_my_run --seed 0
uv run python -m embodied_learning.scan_slam_demo --results results/scan_slam_my_run
```

```powershell
uv run python -m embodied_learning.experiments.loop_closures --output results/loop_closure_my_run --seed 0
uv run python -m embodied_learning.loop_closures_demo --results results/loop_closure_my_run
```

```powershell
uv run python -m embodied_learning.experiments.pose_graph --output results/pose_graph_my_run --seed 0
uv run python -m embodied_learning.pose_graph_demo --results results/pose_graph_my_run
```

```powershell
uv run python -m embodied_learning.navigation_study_demo --lesson 69 --footprint results/navigation_reinforcement_v7 --case 69_narrow_crate_17__delay_4__normal --play
```

```powershell
uv run python -m embodied_learning.navigation_study_demo --lesson 69 --footprint results/navigation_reinforcement_v8 --case 69_street_crate_20__delay_2__shifted --play
```

```powershell
uv run python -m embodied_learning.navigation_study_demo --lesson 69 --footprint results/navigation_reinforcement_v10 --case 69_street_low_26__delay_2__removed_low --play
```

```powershell
uv run python -m embodied_learning.navigation_study_demo --lesson 69 --footprint results/navigation_reinforcement_v11 --case 69_street_low_29__delay_2__bypass_low --play
```

```powershell
uv run python -m embodied_learning.experiments.so101_approach_study --workers 3 --output results/so101_approach_v3b
uv run python -m embodied_learning.manipulation_demo --results results/so101_approach_v3b --position p04a1 --method center_cartesian --play
```

## S288 与独立深度环境

这一方向调用另一个本地项目的 CAD、ONNX 与独立深度环境。公开仓库提供报告、图和紧凑证据；执行脚本、机器人源资产及大型数组尚未完整分发。不能把主环境 `uv sync` 当作其完整安装说明。

[RGB＋ToF 基准复现与硬件假设](microdinosaur-campus-patrol.md#复现) · [深度记录与专用命令](microdinosaur-patrol-depth.md#复现与文件) · [配音成片与验收](microdinosaur-patrol-review-20261009.md#验收与视频交付)

## 测试与证据

当前公开入口见[主页测试流程](../README.md#测试流程)。按影响范围选择相关测试，再选择 record/gui/training；共享影响或阶段验收扩大范围。各次测试批次分开报告，文档链接检查不能代替实验重跑、GUI 验收或硬件验收。
