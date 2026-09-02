@echo off
setlocal
cd /d "%~dp0"
uv run --no-editable depth-webcam --variant metric --input-size 392 --camera 0 --width 3840 --height 2160 --calibration calibration/selected/focus293_16_20260902/selected-camera.json %*
