@echo off
rem Claude Opus wrote this - one-click start for Windows: double-click this file.
rem First run: creates a private Python environment in .venv and installs panda3d, numpy, matplotlib
rem (needs internet, 1-3 minutes). Later runs start immediately and work offline.
setlocal
cd /d "%~dp0"
title LLM Pipeline Demo

rem ---- find Python 3.9 or newer (the "py" launcher first, then "python")
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul && set "PY=python"
if not defined PY goto nopython

rem ---- first run: set up .venv
if exist ".venv\ready.txt" goto run
echo.
echo  First start: installing what the demo needs. This takes 1-3 minutes and needs internet.
echo.
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv || goto fail
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt || goto fail
echo ok> ".venv\ready.txt"

:run
echo  Starting the demo... (keep this window open; close the demo window to quit)
".venv\Scripts\python.exe" main.py || goto crashed
exit /b 0

:nopython
echo.
echo  Python was not found.
echo  1. Install Python 3.9 or newer from  https://www.python.org/downloads/
echo  2. In the installer, tick  "Add python.exe to PATH"
echo  3. Double-click run_windows.bat again.
echo.
pause
exit /b 1

:fail
echo.
echo  Setup failed - see the messages above. Usually the internet connection or a proxy is the cause.
echo  Fix it and double-click run_windows.bat again; to start over, delete the .venv folder first.
echo.
pause
exit /b 1

:crashed
echo.
echo  The demo stopped with an error - please show the messages above to the teacher.
echo.
pause
exit /b 1
