@echo off
chcp 65001 >nul
cd /d "%~dp0"
python pc_gamepad_bridge\install_qgc_fly_panel.py
pause
