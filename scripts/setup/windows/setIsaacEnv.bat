@echo off
REM Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
REM All rights reserved.
REM
REM SPDX-License-Identifier: BSD-3-Clause

set "MODE=conda"
set "ENV_NAME=env_isaaclab"
set "REBUILD=0"

REM Isaac Sim root directory
set "ISAAC_BUILD_DIR=_build\windows-x86_64\release"
set "ISAACSIM_PATH=%ISAAC_SIM_DIR%\%ISAAC_BUILD_DIR%"

:parse_args
if "%~1"=="" goto begin
if /I "%~1"=="--help" goto usage
if /I "%~1"=="-h" goto usage
if /I "%~1"=="--rb" set "REBUILD=1" & shift & goto parse_args
if /I "%~1"=="--rebuild" set "REBUILD=1" & shift & goto parse_args
if /I "%~1"=="--conda" goto parse_conda
if /I "%~1"=="--uv" goto parse_uv
echo 错误: 未知参数 %~1
goto usage

:parse_conda
set "MODE=conda"
shift
if "%~1"=="" goto parse_args
set "NEXT=%~1"
if "%NEXT:~0,2%"=="--" goto parse_args
set "ENV_NAME=%~1"
shift
goto parse_args

:parse_uv
set "MODE=uv"
shift
if "%~1"=="" goto parse_args
set "NEXT=%~1"
if "%NEXT:~0,2%"=="--" goto parse_args
set "ENV_NAME=%~1"
shift
goto parse_args

:begin
echo Beginning setup...

if "%REBUILD%"=="1" (
    rd /s /q "%ISAACSIM_PATH%" 2>nul
    echo Isaac Sim build directory removed.
)

if not exist "%ISAACSIM_PATH%" (
    echo Error: Isaac Sim build directory not found at %ISAACSIM_PATH%
    echo Building...
    call "%ISAAC_SIM_DIR%\build.bat"
)

REM Isaac Sim python executable
set "ISAACSIM_PYTHON_EXE=%ISAACSIM_PATH%\python.bat"

REM Create symbolic link (requires admin privileges)
if not exist "_isaac_sim" (
    mklink /D "_isaac_sim" "%ISAACSIM_PATH%"
)

if /I "%MODE%"=="uv" (
    echo 创建/使用 uv 环境: %ENV_NAME%
    call isaaclab.bat --uv %ENV_NAME%
    echo 激活 uv 环境: %ENV_NAME%
    call "%ENV_NAME%\Scripts\activate"
) else (
    echo 创建/使用 Conda 环境: %ENV_NAME%
    call isaaclab.bat --conda %ENV_NAME%

    where conda >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo conda initialized
    ) else (
        echo 错误: conda 命令不可用，请先安装并初始化 Conda。
        exit /b 1
    )

    echo 激活 Conda 环境: %ENV_NAME%
    call conda activate %ENV_NAME%
)

REM Install all dependencies
call isaaclab.bat --install
exit /b %errorlevel%

:usage
echo 用法:
echo   scripts\setup\windows\setIsaacEnv.bat [--conda [环境名]] [--uv [环境名]] [--rb^|--rebuild]
echo.
echo  --conda    使用 Conda 创建^/配置环境；默认环境名为 env_isaaclab。
echo  --uv       使用 uv 创建^/配置环境；默认环境名为 env_isaaclab。
echo  --rb       删除 Isaac Sim build 目录后重新构建。
echo  --rebuild  同 --rb。
echo  -h, --help 显示帮助。
exit /b 0
