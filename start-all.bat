@echo off
rem Starts the bot (automatic mode), the order-book service and the price recorder. Use this after a PC restart.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File tools\start_all.ps1
echo.
echo Dashboard: http://127.0.0.1:8765/
echo You can close this window: everything keeps running.
pause
