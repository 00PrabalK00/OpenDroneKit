@echo off
setlocal EnableExtensions

rem OpenDroneKit launcher.
rem
rem   1. Starts in the toolkit folder, whatever folder it was run from.
rem   2. Picks the Python environment:  ODK_PYTHON (a python.exe)  >  .venv\  >
rem      the conda env named by ODK_CONDA_ENV (default: odk, see environment.yml).
rem   3. Verifies every package in requirements.txt is installed there, and offers to
rem      install what is missing instead of letting features fail one button at a time.
rem   4. Starts the window with pythonw. Startup errors are shown in a message box and
rem      written to final_toolkit_outputs\logs\startup-error.log.
rem
rem   Arguments are passed through:  OpenDroneKit.bat --ui cockpit
rem   ODK_ON_MISSING_DEPS=install^|launch^|cancel answers the missing-package question
rem   without asking; ODK_LAUNCH_DRYRUN=1 prints the launch command instead of running it.

set "ROOT=%~dp0"
cd /d "%ROOT%"
if not defined ODK_CONDA_ENV set "ODK_CONDA_ENV=odk"

set "PY="
set "PYW="

if defined ODK_PYTHON (
    set "PY=%ODK_PYTHON%"
    for %%I in ("%ODK_PYTHON%") do set "PYW=%%~dpIpythonw.exe"
    goto :have_python
)

if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PY=%ROOT%.venv\Scripts\python.exe"
    set "PYW=%ROOT%.venv\Scripts\pythonw.exe"
    goto :have_python
)

set "ACTIVATE_BAT="
if defined CONDA_EXE (
    for %%I in ("%CONDA_EXE%") do set "ACTIVATE_BAT=%%~dpI..\Scripts\activate.bat"
)
if not defined ACTIVATE_BAT if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "ACTIVATE_BAT=%USERPROFILE%\miniconda3\Scripts\activate.bat"
if not defined ACTIVATE_BAT if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "ACTIVATE_BAT=%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not defined ACTIVATE_BAT if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" set "ACTIVATE_BAT=C:\ProgramData\miniconda3\Scripts\activate.bat"
if not defined ACTIVATE_BAT if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" set "ACTIVATE_BAT=C:\ProgramData\anaconda3\Scripts\activate.bat"

if not defined ACTIVATE_BAT (
    echo [OpenDroneKit] No Python environment found.
    echo [OpenDroneKit] Either install Miniconda and run:  conda env create -f environment.yml
    echo [OpenDroneKit] or create .venv here and run:      .venv\Scripts\python -m pip install -r requirements.txt
    echo [OpenDroneKit] or set ODK_PYTHON to a python.exe that has requirements.txt installed.
    pause
    exit /b 1
)

call "%ACTIVATE_BAT%" "%ODK_CONDA_ENV%"
if errorlevel 1 goto :no_env
if /I not "%CONDA_DEFAULT_ENV%"=="%ODK_CONDA_ENV%" goto :no_env
set "PY=%CONDA_PREFIX%\python.exe"
set "PYW=%CONDA_PREFIX%\pythonw.exe"
goto :have_python

:no_env
echo [OpenDroneKit] Could not activate the conda environment "%ODK_CONDA_ENV%".
echo [OpenDroneKit] Create it from the toolkit folder with:
echo [OpenDroneKit]     conda env create -f environment.yml
echo [OpenDroneKit] or set ODK_CONDA_ENV to an environment that has requirements.txt installed.
pause
exit /b 1

:have_python
if not exist "%PY%" (
    echo [OpenDroneKit] Python not found: %PY%
    pause
    exit /b 1
)
if not exist "%PYW%" set "PYW=%PY%"

:check_deps
"%PY%" "%ROOT%main.py" --check-deps
if not errorlevel 1 goto :launch

echo.
set "ANSWER=%ODK_ON_MISSING_DEPS%"
if not defined ANSWER (
    choice /C ILC /M "Install the missing packages now (I), launch anyway with those features unavailable (L), or cancel (C)"
    if errorlevel 3 set "ANSWER=cancel"
    if not defined ANSWER if errorlevel 2 set "ANSWER=launch"
    if not defined ANSWER set "ANSWER=install"
)
if /I "%ANSWER%"=="cancel" exit /b 1
if /I "%ANSWER%"=="launch" goto :launch
if /I "%ANSWER%"=="install" (
    "%PY%" -m pip install -r "%ROOT%requirements.txt"
    if errorlevel 1 (
        echo [OpenDroneKit] pip install failed; see the messages above.
        pause
        exit /b 1
    )
    set "ODK_ON_MISSING_DEPS=launch"
    goto :check_deps
)
echo [OpenDroneKit] Unknown ODK_ON_MISSING_DEPS value: %ANSWER%
exit /b 1

:launch
if "%ODK_LAUNCH_DRYRUN%"=="1" (
    echo [OpenDroneKit] would launch: "%PYW%" "%ROOT%main.py" %*
    echo [OpenDroneKit] cwd: %CD%
    exit /b 0
)
start "OpenDroneKit" "%PYW%" "%ROOT%main.py" %*
exit /b 0
