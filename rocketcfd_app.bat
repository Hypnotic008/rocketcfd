@echo off
rem Double-click to start the rocketcfd desktop app (Windows).
cd /d "%~dp0"
where py >nul 2>nul && (py -m rocketcfd.app) || (python -m rocketcfd.app)
if errorlevel 1 pause
