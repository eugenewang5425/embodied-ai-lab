@echo off
setlocal
cd /d "%~dp0"
uv sync --no-editable || exit /b 1
uv run --no-editable depth-download || exit /b 1
uv run --no-editable depth-env-check
