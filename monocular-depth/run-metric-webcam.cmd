@echo off
setlocal
cd /d "%~dp0"
uv run --no-editable depth-webcam --variant metric --input-size 392 %*
