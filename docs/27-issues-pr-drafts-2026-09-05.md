# Issue 与 PR 提交文稿（2026-09-05）

本文稿是可直接粘贴的提交内容。来源审查见 [docs/26-experiment-review-2026-09-05.md](26-experiment-review-2026-09-05.md)（编号 B1–B5、D1–D2 与下文 Issue 一一对应）。文中一律使用相对路径，敏感路径已打码。

---

## 一、Issue 草稿（共 7 条）

### Issue 1：README 英文区进度落后实际两轮（21 lessons / 443 tests → 实际 23 课 / 468 项）

**现象**
README 英文区（hero、状态行、Verified 行、badge）仍写 21 lessons / 443 passing。实际主线已完成 23 课（第二十二课针孔相机 + 第二十三课单目米制标定），全量测试实测 468 项通过。

**证据**
- README.md:9 "checked by **443 automated tests**"
- README.md:15 "21 lessons complete (Sep 2026)"
- README.md:16 "`uv run pytest -q` → **443 passing**"
- README.md:22 badge `tests-443%20passing`
- 对照：docs/25-session-23-monocular-metric.md 已存在，results/mobile_monocular_2026-09-05/ 有正式记录；git 提交 3add7cd（第二十二课演示修复）提交信息已写明"全量 455 项×2 通过"；当前分支实测全量 468 项通过。

**建议**
四处正文 + badge 同步为 23 lessons / 468 passing；状态行补 "pinhole camera projection / monocular metric calibration"；测试数以后每次提交时随 README 更新（或改为不带具体数字的表述，避免再次过期）。

**验收标准**
- [ ] README.md:9、15、16、22 四处数字与实测一致
- [ ] 状态行包含第二十二、二十三课内容
- [ ] `uv run pytest -q` 实测数字与 README 声明一致

---

### Issue 2：README 中文区未纳入第二十三课，"454"本身也已过期

**现象**
中文区"当前状态"停在 2026-09-03 的 1–22 课 / 454 项；项目结构清单与进度清单均缺第二十三课。且"454"在第二十二课演示修复提交（+1 项测试）后已应为 455，加第二十三课 13 项后实测 468 项。

**证据**
- README.md:61 "当前状态（2026-09-03）"（日期过期）
- README.md:63 "主线课程 1–22 课已完成"
- README.md:64 "全量 **454 项通过**（含本轮新增 11 项）"
- README.md:124 项目结构清单止于 `24-session-22-pinhole-projection.md`，缺 `25-session-23-monocular-metric.md`
- README.md:436（第 22 课段落）、README.md:469（进度清单）均写 454
- 对照：git 提交 3add7cd 信息"全量 455 项×2 通过"；docs/25:139-142 "新增 13 项测试"；455 + 13 = 468 与实测一致

**建议**
同步五处：日期改 2026-09-05、课数改 1–23、测试数改 468、结构清单补 docs/25、进度清单补第二十三课条目与对应"学员待解释"项。顺带把 README.md:428 "深度噪声压低到 σ=15 cm" 改为准确表述（是调大噪声、压低精度，见 docs/24:50）。

**验收标准**
- [ ] 上述五处全部更新且与 `uv run pytest -q` 实测一致
- [ ] 进度清单出现第二十三课条目
- [ ] 第 22 课段落噪声表述与 docs/24:50 一致

---

### Issue 3：学习路线图未更新第二十三课

**现象**
docs/01 的进度回顾（日期、表格、段落）止于第二十二课；第四阶段行的"monocular-depth 是独立目录，不等同于本阶段验收"在第二十三课把该子项目接进主线后已不准确。

**证据**
- docs/01-learning-roadmap.md:15 "当前进度回顾（2026-09-03）"
- docs/01:24 第四阶段行只写第二十二课产物
- docs/01:34-46 进度段落止于第二十一课补充
- 对照：docs/25 全文（第二十三课已完成"相对深度→米制标定"的主线验收，且 docs/25:143 声明主线实验不依赖 monocular-depth 的 .venv/torch）

**建议**
- 进度回顾表第四阶段补第二十二、二十三课产物；
- 段落补第二十二、二十三课各一段；
- "monocular-depth 是独立目录，不等同于本阶段验收"改写为："monocular-depth 为工具卫星仓（权重与交互工具），其主线验收已由第二十三课在 src/embodied_learning 内完成"。

**验收标准**
- [ ] 进度回顾表、段落包含 22、23 两课
- [ ] monocular-depth 的定位表述与 docs/25:143 一致
- [ ] 日期更新

---

### Issue 4：README 声称 "Python 3.12/3.13"，与 pyproject 锁定不符

**现象**
README 英文区 Stack 行与 badge 写 Python 3.12 / 3.13，但项目锁定为 3.12，两端环境均无 3.13。

**证据**
- README.md:17 "uv + Python 3.12/3.13"
- README.md:25 badge `Python-3.12%20%2F%203.13`
- pyproject.toml:6 `requires-python = ">=3.12,<3.13"`
- docs/00-environment-audit.md:22（Windows 侧固定 3.12.13）、docs/00:32（WSL 侧 3.12.3）

**建议**
改为 "Python 3.12"，badge 同步；若确有支持 3.13 的计划，先放宽 pyproject 并实测通过后再声明。

**验收标准**
- [ ] README 两处与 pyproject 一致
- [ ] 若保留 3.13 声明，则 pyproject 与实测（3.13 下全量测试）支持该声明

---

### Issue 5：18 处本机绝对路径与 1 处用户名进入公开文档

**现象**
多处讲义与 README 含本机绝对路径（项目目录、用户级安装路径、WSL vhdx 路径）和 Linux 用户名，违反"文档用相对路径"的惯例；公开仓库中它们还泄露本机环境信息。

**证据**（路径内容打码，行号为当前工作区）
- `C:\Users\<用户名>\.local\bin`：docs/00-environment-audit.md:23
- `C:\Users\<用户名>\.local\bin\uv.exe`：docs/05-session-03-lqr.md:133
- `D:\<项目目录>`：docs/09-session-07-swingup.md:11、docs/10-session-08-planar-arm.md:21、docs/11-session-09-jacobian-path.md:7、docs/12-session-10-path-coverage.md:15、docs/13-session-11-waypoint-ik.md:9、docs/14-session-12-timing-and-torque.md:94、docs/15-session-13-model-feedforward.md:90、docs/16-session-14-mobile-frames.md:97、docs/17-session-15-encoder-odometry.md:111、docs/18-session-16-encoder-calibration.md:79、docs/19-session-17-random-noise.md:106、docs/20-session-18-landmark-observations.md:98、docs/24-session-22-pinhole-projection.md:76（共 13 处）
- `D:\<WSL目录>`：README.md:65、README.md:75
- Linux 用户名：README.md:75

**建议**
- `D:\<项目目录>` 全部替换为"项目目录"（或 `.`）；
- docs/00、docs/05 的 uv 路径改为"用户级 bin 目录（加入 PATH 后可直接运行 `uv`）"；
- README 的 WSL vhdx 位置改为"位于本机自定义路径（约 8 GB）"，删除用户名；
- 在贡献/写作规范里补一条"文档只用相对路径"。

**验收标准**
- [ ] `grep -rn "C:\\Users\|D:\\\\" docs/ README.md`（排除 WSL 用途说明的相对化改写后）无本机路径残留
- [ ] 全文档无用户名
- [ ] 讲义命令在任意同名项目目录下可照抄运行

---

### Issue 6：README 摘要引用成功率时丢失"有限样本"限定，且对照无区间估计

**现象**
第二十一课讲义明确声明"有限样本下的模型内结果，不是总体成功率，也不是统计显著性结论"（n=20），但 README 与路线图摘要直引成功率对比，限定语丢失；n=20 下 5/20→11/20 这类差异抽样波动很大，跨文档摘引容易被读成"成功率翻倍"的稳定结论。

**证据**
- README.md:412 "近目标实际通过数 16/20→20/20，远目标 5/20→11/20"
- docs/01-learning-roadmap.md:44 同样直引
- docs/23-session-21-goal-feedback.md:119（原始限定语）

**建议**
二选一：
- (a) 在 README.md:412 与 docs/01:44 补一句"20 种子有限样本，方向性观察"（推荐，成本最低）；
- (b) 在 docs/23、23a 的结果摘要补 Wilson 区间或 Fisher 精确检验 p 值（纯计算，不重跑仿真），讲义同步引用。

**验收标准**
- [ ] 摘要引用处均携带有限样本限定（方案 a），或结果文件与讲义带区间/检验（方案 b）
- [ ] 不改动任何 results/ 既有数据

---

### Issue 7：第二十三课 N–σ 扫描表以均值为主列，均值反序行易被误读

**现象**
N=3 的均值反高于 N=2（1% 时 2.57 vs 1.79 cm），讲义正文已解释这不是理论反转（种子标准差与均值同量级，20 个种子分不出 N=2→3 差距），但表格本身没有标准差列，只看表容易得出"增加控制点可能变差"的误读。

**证据**
- docs/25-session-23-monocular-metric.md:50-53（表格无标准差列）
- docs/25:59-63（正文解释：种子间标准差 1% 时 2.39 cm、3% 时 12.7 cm）
- results/mobile_monocular_2026-09-05/summary.json 的 `group_std_abs_error_m`（数据已存在，仅未进表）

**建议**
表格补一列种子标准差，或将呈现改为"中位数（均值±标准差）"；正文解释保留。只改呈现，不改任何数据与结论。

**验收标准**
- [ ] 表格能独立支持"均值反序来自种子间波动"的解读
- [ ] 数字与 summary.json 一致；results/ 不变

---

## 二、PR 描述草稿（可直接粘贴）

**标题**

```
第 1–13 课讲义理论映射与思考题；第二十三课单目米制标定实验；主页与文档同步；审查报告
```

**描述**

```markdown
## 本轮改动

1. **过去课程讲义理论映射 + 思考题（docs/02–24）**
   - 为第一至二十二课（docs/02-phase-1 至 docs/24-session-22-pinhole-projection.md）统一补写
     「理论对应（Embodied-AI-Guide）」与「思考题」两节，把每课数据映射到指南
     Control 6.2.1/6.2.2/6.2.3、机器人学导论 6.3.2 及 Infrastructure Benchmarks 的对应位置；
     思考题全部回指本课实测数字（截至本文稿写作时 24 个文件、+530 行，纯新增，未改既有正文；
     最终行数以提交时 `git diff --stat` 为准）。

2. **第二十三课新实验：单目相对深度 ↔ 米制尺度标定**
   - 新增 `src/embodied_learning/experiments/monocular_metric.py`、
     `src/embodied_learning/monocular_metric_demo.py`、`tests/test_session23_monocular_metric.py`（13 项测试）、
     `docs/25-session-23-monocular-metric.md`；
   - 逆深度仿射歧义 r = a·(1/Z)+b 的最小二乘标定：无标定基线 193.2 cm → 标定后（N=10, σ=1%）1.10 cm；
     N×σ 扫描、近/远 Z² 分层、三明治协方差机制核对（8 组比值 0.990–1.018）；
   - 正式记录 results/mobile_monocular_2026-09-05/（受 Git 忽略）；
   - 相机、场景、近裁剪逐项沿用第二十二课，只变 N、σ 与标定与否。

3. **主页更新（README.md / docs/01-learning-roadmap.md）**
   - 英文 hero 与中文"当前状态"同步至 23 lessons / 468 passing；补第二十三课段落与进度清单条目；
   - 结构清单补 docs/25；路线图第四阶段与 monocular-depth 定位表述更新。

4. **审查报告与提交文稿**
   - docs/26-experiment-review-2026-09-05.md：23 个 results 目录抽查（数字逐项一致）、
     15 项发现（5 bug / 2 实验设计 / 3 文档一致性 / 5 学习路径建议）、
     与 Embodied-AI-Guide 的知识地图对照、第 24 课起的建议序列；
   - docs/27-issues-pr-drafts-2026-09-05.md：7 条 issue 草稿与本 PR 描述。

## 核查清单

- [ ] `uv run pytest -q` 全量 468 项通过（连续两次）
- [ ] `uv run ruff check src tests` 与 `uv run ruff format --check src tests` 通过
- [ ] results/ 既有目录未被改写（新记录只进新目录）
- [ ] docs/02–24 仅新增「理论对应」与「思考题」两节，原正文与数字未改动（以 `git diff` 逐文件核对）
- [ ] README/路线图中的课数、测试数与实测一致（21/443 → 23/468）
- [ ] 全部文档无本机绝对路径与用户名（含历史 18 处，见 Issue 5）
- [ ] 新讲义 docs/25 的每个数字都能在 results/mobile_monocular_2026-09-05/summary.json 中找到
```

---

## 三、如何提交（本机无 gh CLI，不使用任何代用凭据）

以下全部通过 GitHub 网页完成；先把本文稿里对应内容复制出来，再粘贴进网页表单。

**1. 提交 Issue（7 条，逐条提交）**

1. 在本地用文本编辑器打开本文档，复制某一条 `### Issue N` 的标题与正文（不要带 `### Issue N：` 前缀本身，可将其并入标题框）；
2. 浏览器打开新建 issue 页，格式：
   `https://github.com/eugenewang5425/embodied-ai-lab/issues/new?title=<URL编码的标题>&body=<URL编码的正文>`
   —— 建议先只打开 `https://github.com/eugenewang5425/embodied-ai-lab/issues/new`，把标题粘进 Title、正文粘进评论框（Markdown 直接渲染），比手工 URL 编码更稳；带参数的链接适合从笔记工具一键生成时使用；
3. 提交顺序建议：Issue 1（README 英文区）→ 2（中文区）→ 3（路线图）→ 4（Python 版本）→ 5（绝对路径）→ 6（成功率限定）→ 7（N–σ 表呈现）；Issue 1–5 可被同一个 PR 一起关闭，提交时可在正文末尾注明 "will be fixed by the docs-sync PR"。

**2. 提交 PR（compare 分支）**

1. 当前工作在 `fork/review-20260905` 分支（含 docs/02–13 修改、第二十三课新文件与本轮两个新文档）；确认已推送到 fork 仓库的同名分支；
2. 浏览器打开：
   `https://github.com/eugenewang5425/embodied-ai-lab/compare/main...fork/review-20260905`
   （若 fork 名不同，把 `fork` 替换为实际用户名/仓库名；方向是 base=上游 main，compare=fork 分支）；
3. 点击 "Create pull request"，把上文「PR 描述草稿」的标题与描述整段粘贴进对应输入框；
4. 提交后逐项勾选描述里的核查清单；Issue 1–5 对应的修复若已包含在本 PR，可在描述末尾补一行 "Closes #1, Closes #2, ..."（编号按实际 issue 创建后的序号调整）。

**3. 注意事项**

- 粘贴前自查一遍内容不含绝对路径、用户名、邮箱、token（本文稿已按此处理，但 issue 正文如需补充新证据，请沿用相对路径与打码写法）；
- 不要在 issue/PR 里粘贴 results/ 的 .npz 二进制或完整 JSON 长文，只引用 `文件:行号` 与关键数字；
- 本轮两个新文档（docs/26、docs/27）本身已按公开仓库标准撰写，可直接随分支提交。

---

## 四、补充草稿（2026-09-05 晚，演示验收轮）

### Issue 8：全量测试时长随课程增长，建议引入慢速标记分层

**现象**：全量 `uv run pytest -q` 从第 22 课时的 ~78 s 增长到 8 分 20 秒（512 项）。主要增量来自
重型端到端：第 26 课 ICP 测试单文件 6:27（经真机验收轮提速到 3:08，提交 fc6a3ff），第 24/25/27 课
各含子进程 CLI 端到端与进程隔离 Tk 演示测试。

**建议**：
1. 引入 pytest 标记（如 `@pytest.mark.slow`），开发口径默认 `-m "not slow"`（目标 <90 s），
   发布/推送前跑全量口径；
2. 或把重实验测试的扫描网格缩小（如 runs 20→3、迭代上限 200→40），保持断言结构不变；
3. pyproject 注册 marker，README 快速开始注明两种口径。

**验收标准**：`uv run pytest -q -m "not slow"` 开发口径 <90 s；全量口径保持全绿且时长记录在讲义。

**状态**：第 26 课测试已提速（fc6a3ff，6:23→3:08）；分层机制未实现，本 issue 为追踪入口。

## 五、缺陷登记与新增 issue（2026-09-06，演示验收轮与 27/28 课）

### 已随 PR 修复的缺陷（留档，无需单独开 issue）

真机演示验收（docs/26 第六节标准）连续三课抓出"测试全绿、渲染 FAIL"的实例，连同代码审查缺陷一并登记：

| # | 缺陷 | 证据 | 修复 |
| --- | --- | --- | --- |
| F1 | icp_demo ③模式切模式不清图：旧坐标轴与②模式收敛轨迹叠画、标题叠印、图例残留 | 真机截图（2026-09-05 验收记录） | fc6a3ff：draw_error 增加 fig.clear() |
| F2 | icp_demo ①模式面板平移向量全精度打印溢出窗口 | 同上 | fc6a3ff：格式化 3 位小数 |
| F3 | monocular_metric_demo ①模式左图标题在默认窗口被中图裁剪 | 同上 | fc6a3ff：缩短标题 + set_title(fontsize=10) |
| F4 | intrinsics_demo ②模式柱顶数值标签被轴顶裁剪 | 同上 | fc6a3ff：log 轴 ylim 顶部余量 |
| F5 | 第 26 课测试套件 6.5 min（4 项重型测试重跑全尺寸实验）拖慢全量 | pytest --durations | fc6a3ff：light 测试口径（正式记录逐字节不变；另见 Issue 8） |
| F6 | visual_grounding 实验+演示"机制/整链"纵轴标 mrad 实画毫度（57.3 倍） | 验收报告 | 6bd4c25：两处同步修复，讲义按"口径更正、原记录保留"加注 |
| F7 | grounding_demo 滑条切换位姿时右侧面板数字不跟随 | 同上 | 6bd4c25 |
| F8 | grounding_demo 未过 ruff format（提交了未格式化文件） | ruff format --check | 6bd4c25 |
| F9 | bc_imitation evaluate_trajectory 在物理失败的截断回合崩溃（tip−desired 形状不匹配，恰在最关键的失败案例上崩） | 测试报告 | c24095c：截断对齐 + 回归测试 |
| F10 | bc_demo points 数组 (N,3,2) 被按二维索引，专家/BC 线各画两条 | 真机验收 | c24095c：取末端修复 + loader 形状契约 |
| F11 | README 第 20 课测试数 401 未随改版同步为 406；第 22 课"噪声压低 σ=15cm"措辞方向写反 | 审查报告 B 类 | 3133423 等 |
| F12 | 测试遗留孤儿 Tk 进程/窗口（进程隔离方案的窗口未清理，停于屏外） | 进程清单 | 手工清理；低危运维项，流程改进归入 Issue 8 一并考虑 |

### Issue 9（新增，开放）：正式记录 source_sha256 与提交版源码漂移

- **现象**：第 26、27 课正式记录 summary.json 内的 source_sha256 与最终提交版源码不一致（正式跑完到 git 提交之间源码又被修改——修格式、修注释或后续小修）。
- **影响**：哈希锚定的"可复现证据链"出现断口；数字本身逐格复现通过，但审计口径不干净。
- **建议**：流程约定三选一——(a) 正式记录目录生成后冻结源码，只允许在新目录重跑；(b) 源码变更后重建 `*_v2` 目录（现有惯例）；(c) summary 增补 `source_sha256_final` 双锚。
- **验收标准**：最新两个课的正式记录哈希与 HEAD 源码一致，或存在明确 v2 链。

### 对既有 issue 的状态更新

- Issue 5（绝对路径）：docs 侧 15 处已在 PR 内修复；README WSL 路径与用户名已改相对表述。
- Issue 7（N–σ 表缺标准差）：已修复（补 ±1σ 列与口径说明）。
- Issue 1/2/3/4：已随 PR 修复（数字链最终为 28 课/540 项）。
- Issue 6/8：开放（n=20 区间估计属实验流程改进；慢测试分层 light 口径已试点 26 课，其余课待推广）。

### 补充登记 F13–F19（2026-09-06，全面清点各课讲义"如实记录"小节后的漏网项）

| # | 缺陷 | 证据 | 修复 |
| --- | --- | --- | --- |
| F13 | bench 首跑 `cv2.imwrite` 在中文路径下**静默失败**（PNG 缺失无报错） | docs/28 §7 如实记录 | imencode + 字节写入后重跑（360f412） |
| F14 | 第 24 课分析图跟随 matplotlib 默认 TkAgg，测试进程触发项目已记录的 Tcl 间歇错误 | docs/28 §7 | 绘图入口固定 Agg 后端（先于 D-2026-09-05-12 的先例，360f412） |
| F15 | 第 25 课把 r1、r2 装配成 R 的**行**（=用了转置 R）：σ=0 时真值 K 重投影达 10 px | docs/29 §7 诚实记录 | 改列装配后回 1e-12（cfd56b6） |
| F16 | 第 25 课测试种子确定性比较遇 NaN 直接判不等 | 子代理调试报告 | 改 JSON 文本比较（cfd56b6） |
| F17 | 第 28 课 matplotlib 3.11 `stairs` 要求 edges 比 values 长 1（崩溃） | 子代理调试报告 | 边界修正（c24095c） |
| F18 | 第 28 课复用半成品的三处缺陷：精选案例去重只比路径 id（train/gen 可撞名）；验证步数硬编码 350；无序采样破坏"从最大档回扫" | 子代理复用审查报告 | 三处修复 + 契约加固（c24095c） |
| F19 | 第 27 课演示面板比值用俯仰修正版 0.952，与讲义 δpx/f 口径 0.981 不一致 | 验收报告 | 面板统一为 δpx/f 口径（6bd4c25） |

**排查过、确认非缺陷**（留档避免后人重复排查）：第 28 课 FD 梯度探针发现 preact 恰为 0 的真 ReLU 折点——零初始化偏置+死区样本所致，属激活函数数学性质，以偏置抖动规避后梯度检查通过（c24095c）。

### 补充登记 F21–F25（2026-09-06，历史记录可视化补全包）

编号顺延说明：F20 已用于第 22 课演示标签叠压（见 docs/26 第七节台账），本表从 F21 起。来源：为第 20/21/21a 课从冻结记录生成图版（`record_figures.py`）过程中的数据探查与渲染自查。

| # | 缺陷 | 证据 | 修复 |
| --- | --- | --- | --- |
| F21 | 图版初版把 `ros_trace.jsonl` 的 `map_to_sensor` 字段当成固定的 base_link→sensor 安装边画水平线；实际该字段是随车移动的 map→sensor **整链查询结果**，逐帧不同 | results/ros2_system_2026-09-03_v2 trace 逐帧值从 (0.12, 0.04, +30°) 起持续变化（终帧 x≈0.92） | 图版只引用第 0 帧查询值 (0.12, 0.04, +30°) 并注明"整链查询"；docs/22 §8 面板 3 加注 |
| F22 | `stamp_sec` 只是整秒部分，0.04 s 步进的小数在 `stamp_nanosec`；按 stamp_sec 画时间线会把最多 25 个连续帧折叠进同一秒 | 同记录 576/601 帧 `stamp_sec ≠ time_s + 1`（帧 1–24 全为 1） | 图版时间轴一律用 `time_s`（0–24 s）；docs/22 §8 末段注明口径 |
| F23 | 图版 TF 面板初版把偏航（°，±7.6）直接画在平移轴（cm，−3.5–5.5）上，超出部分被裁剪、曲线断成数段 | lesson-20-ros2-timeline 首版渲染检查 | 偏航移到右轴（°刻度），图例合并双轴句柄 |
| F24 | 消息时间线初版把地标累计（每 2 s +1、共 12 条）画在 0–700 条的位姿计数轴上，阶梯完全不可见 | 同上首版渲染检查 | 地标阶梯移到右轴（0–13.5），单格步进清晰可见 |
| F25 | 并行补全包同名冲突：先行 goal_figures 工具包（commit 5b0a46f）已把 README 第 21/21a 课图行与 lesson-21a-thresholds.png 入册；本轮 record_figures 的同名输出覆盖了已入册 PNG，README 图行一度指向两套脚本 | git 5b0a46f 与工作区 diff（docs/img/lesson-21a-thresholds.png 47734→175974 字节） | README 三处统一指向 record_figures 命令与本轮图版；goal_figures.py 与 lesson-21-goal-reaching.png 保留在库但不再被 README 引用；取舍记 docs/34 D-2026-09-06-02 |

## 六、项目规划变更 issue 草稿（Issue 10–13，2026-09-06）

规划层面变更按"一事一 issue"沉淀，正文可直接粘贴（标题可在 New issue 页输入框直接使用，正文从本稿复制）：

### Issue 10：确立演示真机验收标准，并推广至全部历史演示

**正文**：
背景：进程隔离测试全绿不等于演示渲染正确。2026-09-05/06 真机验收连续在四课抓出此类缺陷（F1/F3/F4/F10，见第五节登记），故确立标准（docs/26 第六节）：每课演示逐模式真机过一遍；切模式必须清图；标题/标签不裁剪；面板数值格式化不溢出；面板数字与 summary 逐格一致；交互控件实点验证。
已完成：第 23/25/26/27/28 课演示已按标准验收并修复。
待办：第 1–22 课演示的逐课真机复验（代码级排查已确认无"切模式残轴"同款问题，风险集中在窗口缩放裁剪与文案）。
验收标准：全部含 Tk 演示的课在 docs/26 验收台账中登记"模式×截图×结论"。

### Issue 11：建立实验决策日志与口径变更约定（已实施，供追溯）

**正文**：
背景：第 22–28 课推进中多次发生"实验中途改变细节规划"（体素选点规则、重投影指标、残差形式、测试口径等），散见各讲义不利于追溯。已建立 docs/34 实验决策日志（append-only）：每条 D-<日期>-<序号> 含变更/触发原因/证据/影响；已登记 13 条。
配套约定：正式记录不可改写，口径变更后旧记录原样保留、新口径建 `*_v2` 目录（先例：第 22 课 _sigma5cm、第 24 课 _v1/_v2）；对外报告只反映最新口径。
待办：无（本 issue 为追溯登记；后续变更按约定直接追加 docs/34）。

### Issue 12：单目深度米制标定的技术跟进（第 23/27 课遗留）

**正文**：
三条已量化的跟进项，按优先级：
1. **质心表面/轴口径修正**（第 27 课）：模型掩码定位 7.35 cm 的瓶颈是掩码质心取在柱面上（−半径 6 cm），改用轴线/主轴估计可望回到 ~3.4 cm 基线水平；
2. **真实照片域仿射检验**（第 24 课遗留）：当前结论限合成域，需真实图像重做 r=a/Z+b 拟合，决定是否分段/逐区域标定；
3. **分段标定**：若真实模型残差仍为结构场，控制点布设需按深度分层采样。
验收标准：每项在同条件对照下给出前后误差表；正式记录新目录。

### Issue 13：阶段 5 路线确认（RL 入口与后续）

**正文**：
第 28 课 BC 的实证（开环 MSE 33×、闭环 0/75）构成 RL 的动机：奖励在环 vs 监督分布。规划：
1. 第 29 课：手写 numpy PPO 摆起（GAE/clip/value），基线=第 7 课能量整形+LQR（零样本），对照样本效率与扰动恢复；奖励设计作为"隐形手工"如实讨论；
2. 若 PPO 在算力预算内学不会，"未学会+失败分析"同样作为正式结果入库；
3. 其后：ACT/扩散策略最小本地实验（docs/26 第五节），再评估是否进入 RoboTwin 2.0 动手章（需 16 GB 显存，本机 8 GB，届时评估云端/远程方案）。
验收标准：每步保留基线对照与失败证据；不引入大型训练框架。

### 提交方式提醒

New issue 页：`https://github.com/eugenewang5425/embodied-ai-lab/issues/new?title=<URL编码标题>`（标题可用下面四条）：
- Issue 10 标题：`确立演示真机验收标准，并推广至全部历史演示`
- Issue 11 标题：`建立实验决策日志与口径变更约定（已实施，供追溯）`
- Issue 12 标题：`单目深度米制标定的技术跟进（口径修正/真实照片域/分段标定）`
- Issue 13 标题：`阶段 5 路线确认（RL 入口与后续）`
正文从上方各节复制；Issue 1–9 同理（见第一、四节）。

### Issue 14（新增，2026-09-06）：摆起探索难题的解法路线（第 29 课失败归因 → 第 30 课设计）

**标题**：`第 29 课 PPO 摆起 0/60 的文献对照与第 30 课残差 RL 方案`

**正文**：
第 29 课（docs/33）主结果：纯 numpy PPO 150 万步学到扶稳子技能但完整摆起 0/60（基线 20/20）。失败四机制与文献已对上（docs/33 §10 对照表）：
1. 直立区域从未被探索 → 硬探索/稀疏奖励问题，摆起为文献典型例；PPO 在连续摆起上被 MathWorks 官方文档标记不稳定；
2. 扶稳策略外推成下方恒满偏反射 → 策略分布移（Residual RL 论文的动机问题）；
3. 出界惩罚经 GAE 压制中间进步 → 任意塑形陷阱，Ng 1999 势函数塑形定理可解；
4. 下方回合早死、数据塌缩 → 起始态分布问题（反向课程/状态档案类解法）。

**提议的第 30 课设计（单变量、有现成基线）**：残差 RL——`u = u_energy(第 7 课) + clip(u_RL, ±a)`。能量注入交给物理控制器（解决机制①③），学习容量集中在近失案例暴露的抓取-回中协调（机制②④被底座兜住）。三方对照：纯基线（20/20 已知）/ 纯 PPO（0/60 已知）/ 残差（新）。学习问题：RL 残差能否在不破坏基线的前提下改善交接段（起摆时间/输入峰值）？
**验收标准**：同种子同口径三组对照；残差限幅扫描（0=退化为基线，作为守卫）；失败与"无改善"同样作为正式结论入库。
**参考**：Johannink et al. ICRA 2019（Residual RL）；Rajeswaran et al. RSS 2018（DAPG，示教替代路线，官方开源 hand_dapg）；Ng et al. ICML 1999（PBRS 定理）；Ecoffet et al. 2021（Go-Explore，detach/derail 概念与第 29 课机制②同源）。

