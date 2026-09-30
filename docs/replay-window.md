# 可复用实验回放窗口

该窗口将机器人第一视角、3D 全景、地图轨迹、误差曲线、统计表同步到同一个实验帧。当前有两个数据适配器：第 65 课 的 24 份配对定位记录，以及街区式园区巡检的 4 方法 × 3 轮独立闭环记录。相机常驻左上角，依据所选方法的实际位姿重新渲染。

推荐先打开[园区巡检窗口](campus-patrol-window.md)：双击 `scripts/open_campus_patrol_demo.cmd`。它包含存档雷达回波、各方法当时的规划与巡检目标、方法含义和任务结果解释。下面同时保留第 65 课 入口。

## 启动和操作

在仓库目录运行：

```powershell
uv run python -m embodied_learning.map_separation_demo
```

也可双击 `scripts/open_map_separation_demo.cmd`。它使用现有 `.venv`，打开原生窗口并开始播放。默认选择上一轮演示的 `iso_w1_s1 / 400`；这是用来观察失稳的回合，不代表所有回合的典型表现。

```powershell
# 指定记录和初始回合
uv run python -m embodied_learning.map_separation_demo --case aniso_w1_s0 --particles 100 --play
# 仅核验全部正式记录的哈希、地图几何和逐帧误差，不启动窗口
uv run python -m embodied_learning.map_separation_demo --verify
```

| 操作 | 行为 |
| --- | --- |
| 回合和粒子预算下拉框 | 切换 ISO/ANISO、场景、种子和 100/400 粒子；同回合切换预算保留当前模拟时刻并暂停 |
| 播放/暂停、左右单步、进度条 | 所有面板同步；支持 0.25–16 倍速；最后一帧自动停止 |
| Space / ← / → / Esc | 播放暂停 / 前一帧 / 后一帧 / 关闭 |
| 点击误差曲线 | 跳到对应模拟时刻并暂停 |
| 同步图层复选框 | 同时控制 3D、地图、曲线中的对应估计来源；统计表仍保留完整对照 |
| 单独跟随的方法按钮 | 一次切换相机、3D/地图轨迹、对应底图、雷达、目标、误差曲线和统计高亮；保持当前时间、播放状态与速度 |
| 汇总叠图 / 并排分图 | 一键恢复所有轨迹；前者叠加比较，后者在小地图中分别显示实际/估计路线、雷达点和目标；镜头保留最近选中的方法 |
| 自选图层 | 展开后手动选择要比较的轨迹；勾选后转入汇总模式 |
| 方法含义与效果 | 通俗解释输入、用途和限制，逐项列出本轮误差和任务结果 |
| 雷达探测点 | 青点为当前扫描的有效回波；园区使用原始存档，旧实验使用明确标注的几何重算 |
| 机器人相机 | 跟随真值位置和朝向；与暂停、单步、跳帧、换回合同步；隐藏诊断轨迹和估计标记 |
| 拖动 3D / 滚轮 / 双击 | 旋转 / 缩放 / 重置视角 |
| 拖动地图 / 滚轮 / 双击 | 平移 / 缩放 / 重置范围；可包含越界估计轨迹 |
| 地图下拉框 | 理想占据图、真值投影图、里程计自建图和地图叠加对照 |
| 拖动面板间分隔线 | 调整上下和左右面板的大小 |
| 导出当前统计 | 保存该帧位姿、位置/航向误差、全程统计、来源 SHA-256，以及重渲染相机的参数和来源标识 |

相机近裁剪距离固定为 1 cm，避免大场景的默认缩放裁掉近处障碍表面。园区示意相机高 55 cm、前偏 18 cm，位于 25 cm 半径包络内；机位属于展示参数。相机画面正常不能代替机身碰撞和传感覆盖验收。

## 模块和数据接口

| 文件 | 职责 |
| --- | --- |
| `src/embodied_learning/replay_viewer/model.py` | `ReplaySource`、回合目录、只读记录与轨迹；单位和数组校验 |
| `replay_viewer/map_separation.py` | 第 65 课 适配器；读取摘要/NPZ、核验哈希和地图足印、缓存记录 |
| `replay_viewer/campus.py` | 园区独立闭环适配器；核验协议与记录哈希，按实际有效长度评分 |
| `replay_viewer/clock.py` | 按模拟时间推进、暂停、跳转、倍速；独立于渲染速度 |
| `replay_viewer/scene.py` | MuJoCo 场景渲染、视角和 OpenGL 资源释放 |
| `replay_viewer/camera_panel.py` | 独立第一视角面板；固定光学参数，保持图像比例，跟随同一真值帧 |
| `replay_viewer/goal_overlay.py` | 巡检目标投影与视野外/后方/近距离方向提示 |
| `replay_viewer/analysis_panel.py` | 各方法分图、通俗解释、结果解读与误差曲线页签 |
| `replay_viewer/panels.py` | 独立场景、地图、曲线和统计面板 |
| `replay_viewer/window.py` | 窗口布局、选择器、共享时间轴、面板注册与关闭 |
| `map_separation_demo.py` | 实验入口、默认记录目录和命令行参数 |

接入其他实验时，提供一个 `ReplaySource`：

```python
class MySource:
    title = "我的实验"
    variant_label = "方案"
    entries = (ReplayEntry("trial_0", "A", "第一回合"),)

    def load(self, case, variant):
        return ReplayRecord(...)  # 读取该实验的真实存档

app = ReplayWindow(root, MySource())
```

`ReplayRecord` 的时间戳严格递增，单位为秒；位姿为 `(N,3)` 的 `x / y / yaw`，单位为米、米、弧度，首条轨迹为 `truth`。所有轨迹共用同一条时间轴。地图以行 y、列 x 存储，显式提供分辨率和原点。记录复制输入并将数组设为只读，避免视图修改实验数据。

独立闭环记录通过 `Track.reference_key` 指定每条估计对应哪条实际轨迹，`role="truth"` 标记实际轨迹。`lengths` 指定每种方法真正记录的帧数，后续补齐停留帧不计入统计。`lidar` 保存每帧距离/命中标志；`goals`、`missions` 保存目标与编号；`plans` 保存每帧实际规划，空位使用成对 NaN。不得拿一个方法的实际位置给另一个独立行驶的方法评分。

现有接口处理有限数值的平面位姿；错帧、重复时间、非有限位姿会被明确拒绝。若后续实验有不同采样时刻、无效视觉输出、RGB-D 或点云，应在适配器中提供明确对齐规则和有效性信息，并扩展记录字段，不能靠删帧或补零绕过边界。

面板只需实现 `set_record(record)` 和 `set_frame(frame, visible, trails)`；需要释放资源时实现 `close()`。使用 `app.add_panel(MyPanel, section="bottom")` 注册后，新面板会收到同样的换回合、跳帧和关闭事件。第一视角面板已使用这个生命周期，后续存档 RGB-D 面板也可复用。

`ReplayRecord.camera` 接收 `CameraSpec(height_m=0.30, forward_m=0.18, vertical_fov_deg=65)`，表示离地高度、相对机器人中心向前偏移和垂直视场。相机水平朝向机器人 +x，图像上方为世界 +z；通过 `reference_key` 绑定所选方法的实际轨迹。园区机位为高 0.55 m、前移 0.28 m，垂直视场 65°。当前渲染为 640×480，面板缩放采用留黑边的等比例显示，不改变视场或拉伸图像。这些值是观看设定，不能作为实验相机标定结果。

## 数字与展示边界

- 位置误差、航向差、全程均值、P95 和末端误差从完整存档位姿计算；只对绘图尾迹降采样，当前点和统计不降采样。
- 100/400 粒子的来源与每个回合的地图质量来自各自正式摘要；加载时检查 NPZ 哈希、误差数组及回合均值。
- 平面障碍几何重新生成后必须与存档真值占据图逐格相同，才允许展示。3D 高度仅为观看设定，运动按存档回放；没有重新做动力学实验。
- 地图是实验保存的固定地图，回放进度不代表正在实时建图。第 65 课 旧批次没有保存逐帧 RGB-D 图像和原始扫描，雷达点标为几何重算；园区批次保存了实际使用的雷达距离和命中标志。两种入口的第一视角都标注为重渲染，不能据此评估视觉定位或深度精度；相机画面不参与定位计算。
- 窗口展示原实验的成功和失败结果；不表示此前审核指出的评分和归因问题已修复。

## 验证

69课第三轮：本地存在`results/braking_navigation_v3`时默认打开最新五组比较（旧参照＋制动/限速2×2）。第六页“制动过程”显示进入帧速度、实际命令、名义速度上界和制动请求；失败尾段真实执行到停稳，表格区分到达与余量合格。来源按钮保持当前帧和第3/4/5索引的诊断页，五组全显示；车身页按所选制动模型画延迟分支，立即分支另说明。运行`uv run python scripts/check_braking_window.py`生成固定种子0、7秒、同比例组的本机截图。看第二轮请加`--footprint results/footprint_tracking_v2`，看第一轮指定`footprint_navigation_v1`。详见[第三轮讲义](69-braking-round3.md)。

69课第二轮：本地没有第三轮而有`results/footprint_tracking_v2/summary.json`时默认选该记录，四个中文方法按钮和第五页“跟踪与余量”继续可用。按钮一次同步相机、3D、目标、雷达、统计和诊断，并保持当前帧及诊断页；短回合只画到其真实末帧。左图比较相对首次规划的横向偏差，右图局部放大外扩足印分离量，负数表示预留空间重叠而非实体碰撞。定位误差零为实验受控假设。第二轮独立窗口截图由`uv run python scripts/check_footprint_tracking_window.py`生成。详见[第二轮讲义](69-footprint-tracking-round2.md)。

第69–71课继续使用同一个模块化窗口：`uv run python -m embodied_learning.navigation_study_demo --lesson 69`（也可70/71）。入口自动列出本地已有的课程档案，无需为了查看69课额外生成67/68全部记录。每个方法按钮一次同步相机、3D、目标、雷达点、地图、统计及“车身与决策诊断”，并保留时间。

69课诊断页显示身体、刹停与存档深度；70课显示距离和内部位置预算；71课显示近场整盘恢复预测、执行速度和粒子分散程度。预测图形、按真值重渲染的RGB与真实存档传感器都有明确区分。运行 `uv run python scripts/check_navigation_followup_windows.py` 可生成固定回合的三张本机窗口验收图（需要三课及67/68记录）。

```powershell
uv run python -m pytest tests/test_replay_viewer.py -q
uv run python -m embodied_learning.map_separation_demo --verify
uv run ruff check src/embodied_learning/replay_viewer src/embodied_learning/map_separation_demo.py tests/test_replay_viewer.py
uv run ruff format --check src/embodied_learning/replay_viewer src/embodied_learning/map_separation_demo.py tests/test_replay_viewer.py
```

已知答案测试覆盖角度跨 ±π、位置误差、非均匀时间轴、数组只读、播放暂停和结束边界。真实 Tk 子进程测试使用第二个独立实验适配器，检查进度条、换记录、同步图层、视角拖动、统计导出与资源关闭，不依赖本地大实验数据。

相机验收还检查 0°/90° 朝向的光轴与世界位置、转弯后的实际图像变化、诊断图层开关不污染相机画面，以及窗口相机和统计处于同一帧。双渲染器回归测试检查任一面板调整尺寸、释放旧 OpenGL 资源后，另一面板仍输出相同的有效图像，防止播放时相机变黑。
