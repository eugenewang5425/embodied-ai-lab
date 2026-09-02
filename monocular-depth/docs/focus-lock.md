# 锁焦采集

保持原先的 4K 设置与打印板规格。重新运行 `capture-calibration.cmd` 后：

1. 摄像头固定，板放在画面中央和计划使用的工作距离，等待棋格边缘清楚。
2. 点击采集窗口，英文输入法下按 **F**。先显示 VERIFYING，等到绿色 **LOCKED** 再保存。
3. 第一张保存前，若不够清楚，可按 **A** 重新自动对焦，再按 F。
4. 按 **S** 保存一张，Saved 数字应增加。锁焦未确认、焦点漂移或角点不足时不会保存。
5. 第一张照片后不能再按 A/F 改焦点；需要重新调焦时按 Q 结束，另开新会话。

建议本次先拍 15 张：中央 3 张、左右倾斜 4 张、上下倾斜 4 张、四角各 1 张。
使用约 15°–35° 倾角，停稳再拍；锁焦后不要大幅前后移动到模糊范围。
旧的 30 张照片保留，新数据先单独验证，不直接混合。

## 恢复指定焦点补拍

在项目目录运行 `capture-calibration.cmd --focus 293`，可在新会话直接请求手动焦点 293。
不会主动开启自动对焦；仍须读取连续至少 8 帧、0.75 秒的稳定状态后才能保存。
此模式 A/F 始终禁用，避免补拍时误改焦点；只需等绿色 LOCKED 后按 S。
启动验证失败会拒绝采集，后续漂移会禁止保存。`focus-lock.json` 记录首次锁定证据，
`session.json` 记录请求值，逐张 JSON 仍记录取帧前后状态。

补拍不覆盖旧会话。焦点数值一致只是必要条件，合并前还应核对设备、分辨率、板规格、
裁剪/变焦以及重投影一致性，不能仅凭 293 这个数值保证光学内参完全相同。

## 程序保护

- F 读取当前对焦位置，关闭自动对焦，将该位置设为手动焦点。
- 检查两个设置调用是否成功，并读取状态确认；连续至少 8 帧、0.75 秒一致后才进入 LOCKED。
- 在取帧前、取帧后、保存前检查焦点；漂移后进入 FAILED 并禁止保存。
- 每张 PNG 配套 `view_XXX.json`，记录规范化 autofocus、原始标志 autofocus_raw、焦点和目标、时间。
- 新会话包含 `focus_lock_required=true`；解算器拒绝缺记录、未锁焦或目标不一致的照片。
- 焦点数值是驱动单位，不是毫米。LOCKED 不代表画面必然清晰，更不代表米制深度精度已验证。

## Windows DirectShow 适配

本机实测自动对焦读回 1，手动对焦读回 2。OpenCV DirectShow 返回的是 CameraControlFlags，
不是通常的 0/1 布尔值。程序仅在 DirectShow 后端将 2 解释为手动；未知标志仍拒绝通过。

参考：[Microsoft CameraControlFlags](https://learn.microsoft.com/en-us/windows/win32/api/strmif/ne-strmif-cameracontrolflags)、
[OpenCV DirectShow 实现](https://github.com/opencv/opencv/blob/4.x/modules/videoio/src/cap_dshow.cpp)。
