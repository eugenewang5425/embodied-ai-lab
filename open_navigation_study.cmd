@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m embodied_learning.navigation_study_demo %*
if errorlevel 1 pause
