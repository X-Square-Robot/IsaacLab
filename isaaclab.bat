@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Copyright (c) 2022-2025, The Isaac Lab Project Developers
rem (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
rem All rights reserved.
rem
rem SPDX-License-Identifier: BSD-3-Clause

set "ISAACLAB_PATH=%~dp0"
rem Remove trailing backslash.
if "%ISAACLAB_PATH:~-1%"=="\" set "ISAACLAB_PATH=%ISAACLAB_PATH:~0,-1%"

rem Find python to run CLI.
if defined VIRTUAL_ENV (
    set "python_exe=%VIRTUAL_ENV%\Scripts\python.exe"
) else if defined CONDA_PREFIX (
    set "python_exe=%CONDA_PREFIX%\python.exe"
) else if exist "%ISAACLAB_PATH%\_isaac_sim\python.bat" (
    set "python_exe=%ISAACLAB_PATH%\_isaac_sim\python.bat"
) else (
    rem Fallback.
    set "python_exe=python"
)

rem Add source/isaaclab to PYTHONPATH so we can import isaaclab.cli.
set "PYTHONPATH=%ISAACLAB_PATH%\source\isaaclab;%PYTHONPATH%"

rem Let Kit associate direct wrapper launches with the Isaac Sim desktop icon.
if "%RESOURCE_NAME%"=="" set "RESOURCE_NAME=IsaacSim"

rem If a local Isaac Sim binary is present, source its env setup so that
rem PYTHONPATH/PATH/EXP_PATH are correct without depending on a conda
rem activate.d hook (those don't fire under e.g. `conda run` on Windows).
if exist "%ISAACLAB_PATH%\_isaac_sim\" (
    rem The env setup script was renamed across Isaac Sim versions
    rem (setup_conda_env.bat -> setup_python_env.bat); probe both.
    if exist "%ISAACLAB_PATH%\_isaac_sim\setup_python_env.bat" (
        call "%ISAACLAB_PATH%\_isaac_sim\setup_python_env.bat" >NUL
    ) else if exist "%ISAACLAB_PATH%\_isaac_sim\setup_conda_env.bat" (
        call "%ISAACLAB_PATH%\_isaac_sim\setup_conda_env.bat" >NUL
    ) else (
        echo [WARNING] _isaac_sim is present but no Isaac Sim env setup script ^(setup_python_env.bat / setup_conda_env.bat^) was found; Isaac Sim env vars not exported. 1>&2
        echo [WARNING] Re-extract the Isaac Sim Windows zip if you intend to use the bundled binary. 1>&2
    )
    rem setup_python_env.bat only sets PATH/PYTHONPATH; the launcher vars
    rem (EXP_PATH/CARB_APP_PATH/ISAAC_PATH) are set by python.bat around the call,
    rem which we bypass under the venv/conda python. Derive them when unset so
    rem AppLauncher can resolve the experience (.kit) files.
    if not defined EXP_PATH set "EXP_PATH=%ISAACLAB_PATH%\_isaac_sim\apps"
    if not defined CARB_APP_PATH set "CARB_APP_PATH=%ISAACLAB_PATH%\_isaac_sim\kit"
    if not defined ISAAC_PATH set "ISAAC_PATH=%ISAACLAB_PATH%\_isaac_sim"
)

rem Execute CLI.
"%python_exe%" -c "from isaaclab.cli import cli; cli()" %*

if errorlevel 1 exit /b 1
endlocal
exit /b 0
