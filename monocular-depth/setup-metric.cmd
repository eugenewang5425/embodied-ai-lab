@echo off
setlocal
cd /d "%~dp0"
uv sync --no-editable --reinstall-package monocular-depth-lab || exit /b 1
uv run --no-editable depth-download --variant metric || exit /b 1
uv run --no-editable depth-env-check --variant metric
