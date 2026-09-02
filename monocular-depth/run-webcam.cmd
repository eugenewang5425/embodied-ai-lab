@echo off
setlocal
cd /d "%~dp0"
uv run --no-editable depth-webcam %*
