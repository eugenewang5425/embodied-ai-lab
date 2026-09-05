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
