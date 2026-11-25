@echo off
rem rocketcfd one-click install (Windows, NVIDIA GPU).
rem Creates a private Python environment in .venv, installs CUDA PyTorch and the requirements,
rem checks the GPU, and puts a "rocketcfd" shortcut on the desktop. Re-run to update.
setlocal
cd /d "%~dp0\.."
set ROOT=%CD%

where py >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install Python 3.11-3.13 from python.org ^(tick "Add to PATH"^), then run this again.
  pause
  exit /b 1
)
if not exist .venv (
  echo Creating the Python environment...
  py -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo Installing PyTorch with CUDA (about 2.5 GB)...
pip install torch --index-url https://download.pytorch.org/whl/cu126
echo Installing rocketcfd requirements...
pip install -r requirements.txt
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND - runs will use the CPU')"

echo Creating the desktop shortcut...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\rocketcfd.lnk');" ^
  "$s.TargetPath='%ROOT%\.venv\Scripts\pythonw.exe'; $s.Arguments='-m rocketcfd.app'; $s.WorkingDirectory='%ROOT%';" ^
  "$s.IconLocation='%ROOT%\rocketcfd\app\assets\rocketcfd.ico'; $s.Description='rocketcfd CFD app'; $s.Save()"
echo.
echo Done. Start rocketcfd from the desktop shortcut (or rocketcfd_app.bat).
pause
