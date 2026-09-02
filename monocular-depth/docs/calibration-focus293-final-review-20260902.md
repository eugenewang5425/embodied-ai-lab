# 焦点 293 补拍及双会话联合审查

## 结论

补拍会话 `focus293_20260902_194814` 已正常退出，共保存 **16 张** 3840×2160 照片。
16 张全部检测到 24 个 ChArUco 内角点，全部通过逐张锁焦核验。
与先前 `session_20260902_113833_342289` 的 9 张一起，共 25 张记录均为手动对焦、焦点 293。

两组的 camera index、实际/请求分辨率、FOURCC、棋盘规格一致，允许进行联合诊断。
这不是相机序列号核验，也不能仅靠相同驱动数值证明光学状态完全一致。
没有与焦点 286 或无逐帧锁焦记录的历史照片混合。

**未通过项目既有质量门槛，未创建默认 `calibration/camera.json`，未启用点云或世界坐标流程。**
所有原图保留，联合解算仅引用原图路径并记录 SHA256；没有复制重编码原图、没有覆盖或删除照片。
不建议此时继续增加同类照片，先核实纸张/底板平整性和成像条件。

## 数值结果

| 检查 | RMS / 结果 |
| --- | --- |
| 补拍 16 张单独标定 | 1.257965 px，未通过 |
| 补拍组五折留出 | 1.340076 px |
| 用补拍 16 张拟合内参，检查之前 9 张 | 0.821851 px |
| 25 张联合标定 | 1.117909 px，未通过 |
| 25 张五折留出 | 1.150496 px |
| 按原有单帧 1.5 px 门槛标记高误差照片 | 6 张，均来自补拍组 |
| 排除这 6 张的敏感性分析 | 19 张，0.841087 px，仍未通过；仅诊断 |
| 内参辅助角点插值后联合重拟合 | 1.115507 px，无明显改善 |
| 25 张角点凸包占图像面积 | 71.54%，不是均匀覆盖率 |
| 25 张候选估计板倾角 | 4.17°–41.43° |

原有门槛保持整体 ≤0.8 px、任一单张 ≤1.5 px；没有为了通过而改阈值。
排除高误差照片只做诊断，不更改正式候选的 25 张输入，不把筛选后结果称为独立验证。
仅用补拍 16 张排除 6 张后剩 10 张，不足 12 张，因此该敏感性分析明确跳过。

五折验证每张图不参与对应折的内参拟合，但其自身角点仍参与自身位姿估计。
跨会话验证也只估计留出图自身位姿；0.821851 px 是几何一致性指标，不是独立尺量误差，
也不能理解为米制深度模型精度合格。

## 高误差照片与观察

以下为联合候选下补拍照片的数值；文件编号对应补拍会话，不是之前 9 张：

| 文件 | 重投影 RMS | 去畸变后的平面单应性 RMS |
| --- | --- | --- |
| view_000.png | 1.597 px | 1.549 px |
| view_001.png | 1.905 px | 1.878 px |
| view_004.png | 1.819 px | 1.822 px |
| view_005.png | 1.543 px | 1.524 px |
| view_009.png | 1.577 px | 1.559 px |
| view_012.png | 1.669 px | 1.526 px |

已查看补拍全部 16 张局部预览，并查看 view_001、view_004 原分辨率局部；
之前 9 张的预览和重点局部已在上一轮检查。
本次仍有边缘发虚的图像；另有图案可辨但较大倾角下平面拟合残差偏高的照片。
角点数量齐全不等于角点位置准确，也不保证纸张处于同一理想平面。

在上述照片中，放宽到每张图单独拟合平面单应性，仍留下接近重投影 RMS 的残差。
这是优先检查板平面/格点几何/角点定位的依据，但**不是纸张弯曲的确诊**；
局部打印误差、运动/滚动快门、成像处理或未充分描述的畸变等尚未被单独排除。
候选内参本身也未通过门槛，不能拿它的去畸变结果当成独立地面真值。

需要用户先确认红色底板是什么材质，纸张是整面贴平、仅固定四角还是夹在封皮下，
并从侧面检查是否鼓起或随抓持弯曲。此前尺量确认的 25 mm 单格和 175×125 mm 外形仍有效，
但外形尺寸正确不等于板面平整，不能只据尺寸确认排除这一因素。

## 产物

- 补拍候选：`calibration/camera-focus293-supplement-candidate-20260902.json`
- 补拍逐图预检查：`outputs/calibration-focus293-supplement-precheck-20260902/trial.json`
- 补拍完整审查：`outputs/calibration-focus293-supplement-audit-v2-20260902/audit.json`
- 跨会话检查：`outputs/calibration-focus293-pair-20260902/pair-audit.json`
- 联合候选及源图路径/哈希：同目录 `combined-candidate.json`
- 联合完整审查：`outputs/calibration-focus293-combined-audit-20260902/audit.json`
- 图像局部：预检查目录的 `crop_view_001.png`、`crop_view_004.png` 等；原图不变。

审查脚本首次执行遇到“排除后不足 12 张”的诊断分支错误，未改变已生成的标定候选。
已修正为明确跳过该项，并在新的 v2 输出目录完整重跑；首次不完整输出保留、不作结论依据。
新加的 `scripts/evaluate_capture_pair.py` 仅校验/读取来源、计算候选和检查报告，不改采集程序。
项目测试 39 passed，Ruff 全部通过；这不构成物理标定精度保证。

## 复现

在 monocular-depth 项目目录运行，输出目录必须尚不存在：

```powershell
uv run --no-editable python scripts/evaluate_capture_pair.py --train calibration/sessions/focus293_20260902_194814 --heldout calibration/sessions/session_20260902_113833_342289 --output outputs/calibration-focus293-pair-rerun
uv run --no-editable python scripts/audit_calibration.py --images calibration/sessions --candidate outputs/calibration-focus293-pair-rerun/combined-candidate.json --output outputs/calibration-focus293-combined-audit-rerun --physical-size-confirmed
```
