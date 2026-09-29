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

rem Which environment: ODK_CONDA_ENV if set, else "odk" (the one requirements.txt is
rem installed into), else the older "cc-env". cc-env has no reportlab or python-docx, so
rem launching into it silently removed PDF and Word reports from the application.
set "ODK_ENV=%ODK_CONDA_ENV%"
if not defined ODK_ENV (
    for %%I in ("%ACTIVATE_BAT%") do if exist "%%~dpI..\envs\odk\python.exe" set "ODK_ENV=odk"
)
if not defined ODK_ENV set "ODK_ENV=cc-env"

call "%ACTIVATE_BAT%" %ODK_ENV%
if errorlevel 1 (
    echo [OpenDroneKit] Failed to activate conda environment: %ODK_ENV%
    pause
    exit /b 1
)

rem Say so up front when the environment cannot write report deliverables; the
rem application also refuses PDF/Word by name, but this is where it can be fixed.
python -c "import reportlab, docx" 1>nul 2>nul
if errorlevel 1 (
    echo [OpenDroneKit] Environment %ODK_ENV% cannot write PDF/Word reports.
    echo [OpenDroneKit] Fix: conda activate %ODK_ENV% ^&^& pip install -r "%ROOT%requirements.txt"
)
echo [OpenDroneKit] Using conda environment: %ODK_ENV%
if "%ODK_LAUNCH_CHECK_ONLY%"=="1" exit /b 0

pythonw "%ROOT%main.py"
exit /b %errorlevel%

