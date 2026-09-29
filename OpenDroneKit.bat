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

rem The conda environment to run in. Set ODK_CONDA_ENV to use a different one.
if not defined ODK_CONDA_ENV set "ODK_CONDA_ENV=cc-env"

call "%ACTIVATE_BAT%" %ODK_CONDA_ENV%
if errorlevel 1 (
    echo [OpenDroneKit] Failed to activate conda environment: %ODK_CONDA_ENV%
    pause
    exit /b 1
)

rem Features load their packages only when used, so a package missing from this
rem environment would otherwise first show up as a failing button (Fleet needs
rem SQLAlchemy, for one). Check now, and offer to install what is missing.
python "%ROOT%tools\check_environment.py"
if errorlevel 1 (
    choice /C YN /M "[OpenDroneKit] Install the missing packages into %ODK_CONDA_ENV% now"
    if not errorlevel 2 (
        python "%ROOT%tools\check_environment.py" --install
        if errorlevel 1 pause
    )
)

pythonw "%ROOT%main.py"
exit /b %errorlevel%

