@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONPATH=%CD%\src"
start "" "%CD%\.venv\Scripts\pythonw.exe" -m embodied_learning.campus_patrol_demo --play
