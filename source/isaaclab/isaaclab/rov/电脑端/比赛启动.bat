@echo off
cd /d "%~dp0pc_gamepad_bridge"
pythonw launch_competition.py
if errorlevel 1 (
  echo Startup failed. Run: python launch_competition.py --check
  pause
)
