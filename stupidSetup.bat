@echo off
REM Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
REM All rights reserved.
REM
REM SPDX-License-Identifier: BSD-3-Clause
REM Set proxy settings
@REM set "http_proxy=http://10.101.10.6:7890"
@REM set "https_proxy=http://10.101.10.6:7890"

REM Isaac Lab root directory
set "ISAAC_LIB_DIR=%~dp0"
set "ISAAC_SIM_DIR=%ISAAC_LIB_DIR%\IsaacSim"

set "MODE=conda"
set "ENV_NAME=env_isaaclab"

:parse_args
if "%~1"=="" goto run
if /I "%~1"=="--help" goto usage
if /I "%~1"=="-h" goto usage
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

:run
if /I "%MODE%"=="uv" (
    echo 使用 uv 环境名: %ENV_NAME%
    call "%ISAAC_LIB_DIR%isaaclab.bat" -u %ENV_NAME%
) else (
    echo 使用 Conda 环境名: %ENV_NAME%
    call "%ISAAC_LIB_DIR%isaaclab.bat" -c %ENV_NAME%
)
exit /b %errorlevel%

:usage
echo 用法:
echo   stupidSetup.bat [--conda [环境名]] [--uv [环境名]]
echo.
echo  --conda   使用 Conda 创建^/配置环境；默认环境名为 env_isaaclab。
echo  --uv      使用 uv 创建^/配置环境；默认环境名为 env_isaaclab。
echo  -h, --help 显示帮助。
exit /b 0
