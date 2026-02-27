@echo off
setlocal

set "ROOT=%~dp0"
set "ACTIVATE_BAT="

if defined CONDA_EXE (
    for %%I in ("%CONDA_EXE%") do set "ACTIVATE_BAT=%%~dpI..\Scripts\activate.bat"
)

if not defined ACTIVATE_BAT (
    if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "ACTIVATE_BAT=%USERPROFILE%\anaconda3\Scripts\activate.bat"
)
if not defined ACTIVATE_BAT (
    if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "ACTIVATE_BAT=%USERPROFILE%\miniconda3\Scripts\activate.bat"
)
if not defined ACTIVATE_BAT (
    if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" set "ACTIVATE_BAT=C:\ProgramData\anaconda3\Scripts\activate.bat"
)

if not defined ACTIVATE_BAT (
    echo [OpenDroneKit] Could not locate conda activate script.
    echo [OpenDroneKit] Install/initialize conda first.
    pause
    exit /b 1
)

call "%ACTIVATE_BAT%" cc-env
if errorlevel 1 (
    echo [OpenDroneKit] Failed to activate conda environment: cc-env
    pause
    exit /b 1
)

pythonw "%ROOT%main.py"
exit /b %errorlevel%

