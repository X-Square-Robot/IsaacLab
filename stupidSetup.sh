#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Do not enable `set -e` here. This script is often sourced from zsh/bash to
# keep the environment activated in the current terminal; strict shell options
# would leak into the parent shell and can close the terminal on failures.

# Isaac Lab root directory.
ISAAC_LIB_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export ISAAC_LIB_DIR
readonly ISAACSIM_PIP_VERSION="6.0.1.0"

# shellcheck disable=SC1091
source "${ISAAC_LIB_DIR}/scripts/setup/linux/isaac_sim_build_utils.sh"

MODE="conda"
ENV_NAME="env_isaaclab"
SHOW_HELP=0
SKIP_ENV_CREATE=0
ENV_ALREADY_EXISTS=0
RUN_INSTALL=1
# OV 运行时 extra 选择: "" | ovrtx | ovphysx | all
OV_EXTRA=""

usage() {
    cat <<'EOF'
用法:
  source ./stupidSetup.sh [--conda [环境名]] [--uv [环境名]] [--ovrtx|--ovphysx|--ov]
  bash ./stupidSetup.sh [--conda [环境名]] [--uv [环境名]] [--ovrtx|--ovphysx|--ov]

说明:
  默认安装固定版本的 Isaac Sim 6.0.1.0 pip 包，不需要 IsaacSim 源码目录。
  安装完成后，本脚本会在当前 shell 激活环境。
  --conda      使用 Conda 创建/配置环境；默认环境名为 env_isaaclab。
  --uv         使用 uv 创建/配置环境；默认环境名为 env_isaaclab。

  OV（Omniverse 运行时后端）选项 —— 与主环境完全独立的 kitless 安装：
  --ovrtx      OVRTX 渲染器运行时（ov[ovrtx]）。
  --ovphysx    OVPhysX 物理后端运行时（ov[ovphysx]）。
  --ov         全部 OV 运行时（ov[all] = ovrtx + ovphysx）。
               指定任一 OV 选项时，本脚本不再触碰主环境，也不构建/链接 IsaacSim；
               而是创建/配置一个独立的 kitless 环境「<环境名>_ov」，并在其中安装
               core + newton + rl + visualizer + ov[...]（全程不依赖 kit/IsaacSim）。
               例：source ./stupidSetup.sh --uv myenv --ov  →  创建并配置 uv 环境 myenv_ov。
  --help, -h   显示帮助。
EOF
}

normalize_env_name() {
    name="$1"
    while :; do
        case "$name" in
            *。|*.) name="${name%?}" ;;
            *) break ;;
        esac
    done
    printf '%s' "$name"
}

conda_env_exists() {
    local env_name="$1"

    # Run the probe in a subshell with `_isaac_sim` cp312 pollution stripped:
    # if PYTHONPATH carries IsaacSim's cp312 stdlib (left by a prior env's
    # activate.d), even `conda __init__` import chain (json -> re -> _sre)
    # crashes with `SRE module mismatch` before plugins load, so CONDA_NO_PLUGINS
    # alone is not enough. The subshell keeps the parent shell's PATH vars intact.
    # CONDA_NO_PLUGINS additionally guards the plugin-load pydantic_core crash.
    if command -v conda >/dev/null 2>&1; then
        (
            strip_isaac_sim_pollution
            CONDA_NO_PLUGINS=true conda env list | awk -v name="$env_name" '$1 == name { found = 1 } END { exit(found ? 0 : 1) }'
        )
        return $?
    fi

    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        (
            strip_isaac_sim_pollution
            # shellcheck disable=SC1091
            source "$HOME/miniforge3/etc/profile.d/conda.sh"
            CONDA_NO_PLUGINS=true conda env list | awk -v name="$env_name" '$1 == name { found = 1 } END { exit(found ? 0 : 1) }'
        )
        return $?
    fi

    return 1
}

ensure_env_created() {
    if [ "$MODE" = "uv" ]; then
        if [ ! -f "${ISAAC_LIB_DIR}/${ENV_NAME}/bin/activate" ]; then
            echo "环境不存在，先创建 uv 环境: $ENV_NAME"
            env -u VIRTUAL_ENV -u CONDA_PREFIX "${ISAAC_LIB_DIR}/isaaclab.sh" --uv "$ENV_NAME"
            SKIP_ENV_CREATE=1
        else
            ENV_ALREADY_EXISTS=1
        fi
    else
        if ! conda_env_exists "$ENV_NAME"; then
            echo "环境不存在，先创建 Conda 环境: $ENV_NAME"
            env -u VIRTUAL_ENV -u CONDA_PREFIX "${ISAAC_LIB_DIR}/isaaclab.sh" --conda "$ENV_NAME"
            SKIP_ENV_CREATE=1
        else
            ENV_ALREADY_EXISTS=1
        fi
    fi
}

# 已存在的环境询问是否重新 install（不含 IsaacSim build/link 判断）。
# 设置全局 RUN_INSTALL：新建的环境保持默认值 1（必装）；已存在环境在交互终端下
# 询问 y/N，非交互终端默认跳过。主流程与 OV kitless 流程共用此逻辑。
prompt_reinstall_if_exists() {
    if [ "$ENV_ALREADY_EXISTS" != "1" ]; then
        return 0
    fi

    # 上游（setup_arena_local.sh -f / install.sh -f）可设此标志强制重装，
    # 跳过交互询问，并优先于「非交互终端默认跳过」逻辑。
    if [ "${ISAACLAB_FORCE_REINSTALL:-0}" = "1" ]; then
        echo "检测到 ISAACLAB_FORCE_REINSTALL=1，强制重新 install。"
        RUN_INSTALL=1
        return 0
    fi

    if [ ! -t 0 ] || [ ! -t 1 ]; then
        echo "检测到环境已存在，非交互终端下默认跳过重新 install。"
        RUN_INSTALL=0
        return 0
    fi

    while :; do
        printf '检测到环境已存在，是否重新 install？[y/N] '
        IFS= read -r answer || answer=""
        case "$answer" in
            [Yy]|[Yy][Ee][Ss])
                RUN_INSTALL=1
                return 0
                ;;
            [Nn]|[Nn][Oo]|"")
                RUN_INSTALL=0
                return 0
                ;;
            *)
                printf '请输入 y 或 n。\n'
                ;;
        esac
    done
}

prompt_reinstall_existing_env() {
    if ! python - "$ISAACSIM_PIP_VERSION" <<'PY' >/dev/null 2>&1
from importlib.metadata import PackageNotFoundError, version
import sys

try:
    installed = version("isaacsim")
except PackageNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if installed == sys.argv[1] else 1)
PY
    then
        echo "未检测到 Isaac Sim $ISAACSIM_PIP_VERSION pip 包，将执行安装。"
        RUN_INSTALL=1
        return 0
    fi

    prompt_reinstall_if_exists
}

parse_mode_and_env() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --conda)
                MODE="conda"
                shift
                if [ "$#" -gt 0 ]; then
                    case "$1" in
                        --*) ;;
                        *) ENV_NAME="$1"; shift ;;
                    esac
                fi
                ;;
            --uv)
                MODE="uv"
                shift
                if [ "$#" -gt 0 ]; then
                    case "$1" in
                        --*) ;;
                        *) ENV_NAME="$1"; shift ;;
                    esac
                fi
                ;;
            --rb|--rebuild)
                echo "提示: publication 安装使用 Isaac Sim $ISAACSIM_PIP_VERSION pip 包，已忽略 $1。"
                shift
                ;;
            --ovrtx|-ovrtx)
                OV_EXTRA="ovrtx"
                shift
                ;;
            --ovphysx|-ovphysx)
                OV_EXTRA="ovphysx"
                shift
                ;;
            --ov|-ov)
                OV_EXTRA="all"
                shift
                ;;
            --help|-h)
                SHOW_HELP=1
                return 0
                ;;
            *)
                shift
                ;;
        esac
    done
    ENV_NAME="$(normalize_env_name "$ENV_NAME")"
}

deactivate_current_env() {
    # Strip `_isaac_sim` cp312 pollution first: `conda deactivate` is a shell
    # function but internally invokes the cp313 `$CONDA_EXE` binary, which
    # crashes with `SRE module mismatch` if PYTHONPATH carries IsaacSim's cp312
    # stdlib (left by a prior env's activate.d).
    strip_isaac_sim_pollution

    if [ -n "${VIRTUAL_ENV:-}" ]; then
        deactivate >/dev/null 2>&1 || true
    fi

    if [ -n "${CONDA_PREFIX:-}" ] && command -v conda >/dev/null 2>&1; then
        conda deactivate >/dev/null 2>&1 || true
    fi
}

activate_uv_env() {
    env_path="${ISAAC_LIB_DIR}/${ENV_NAME}"

    if [ ! -f "${env_path}/bin/activate" ]; then
        echo "错误: uv 环境激活脚本不存在: ${env_path}/bin/activate"
        return 1
    fi

    deactivate_current_env
    # shellcheck disable=SC1091
    source "${env_path}/bin/activate"
    echo "已激活 uv 环境: ${env_path}"
}

activate_conda_env() {
    deactivate_current_env

    # Re-strip right before `conda activate`: deactivate_current_env above may
    # have re-polluted PYTHONPATH if the previously active env's deactivate.d
    # didn't fully restore it. The target env's activate.d re-adds correct paths.
    strip_isaac_sim_pollution

    if command -v conda >/dev/null 2>&1; then
        conda activate "$ENV_NAME"
        echo "已激活 Conda 环境: $ENV_NAME"
        return 0
    fi

    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
        echo "已激活 Conda 环境: $ENV_NAME"
        return 0
    fi

    echo "错误: conda 命令不可用，无法激活 Conda 环境。"
    return 1
}

run_pip_isaaclab_install() {
    echo "安装 Isaac Sim $ISAACSIM_PIP_VERSION pip 包与 IsaacLab..."
    "${ISAAC_LIB_DIR}/isaaclab.sh" --install isaacsim
}

activate_mode_env() {
    if [ "$MODE" = "uv" ]; then
        activate_uv_env
    else
        activate_conda_env
    fi
}

# OV（Omniverse 运行时）独立 kitless 安装流程。
#
# 与主流程的关键区别：
#   - 环境名切换为「<原环境名>_ov」，与主环境完全隔离；
#   - 不安装 Isaac Sim；
#   - 直接 `./isaaclab.sh --install newton,rl,visualizer,ov[...]`，全程不含 isaacsim token，
#     安装的依赖链均为 kitless（核心 isaaclab 依赖 usd-core 等，OV 运行时为 ovrtx/ovphysx wheel）。
#   - 设置 ISAACLAB_SKIP_PREBUNDLE_REPOINT，避免 install 末尾把共享的 _isaac_sim
#     prebundle 重定向到本环境的 site-packages（防跨环境污染）。
run_ov_kitless_setup() {
    # 切换到独立的 OV kitless 环境名。
    ENV_NAME="${ENV_NAME}_ov"
    local spec="newton,rl,visualizer,ov[${OV_EXTRA}]"

    if [ "$MODE" = "uv" ]; then
        echo "OV kitless 模式：使用独立 uv 环境名: $ENV_NAME"
    else
        echo "OV kitless 模式：使用独立 Conda 环境名: $ENV_NAME"
    fi
    echo "将安装（不依赖 kit/IsaacSim）: ${spec}"

    # 创建（若不存在）并激活独立环境；ensure_env_created/activate_* 均作用于上面改名后的 ENV_NAME。
    ensure_env_created || return $?
    # 环境已存在时询问是否重装（OV kitless 不构建 IsaacSim，无需 isaacsim_setup_required 判断）。
    prompt_reinstall_if_exists || return $?
    activate_mode_env || return $?

    if [ "$RUN_INSTALL" != "1" ]; then
        echo "已跳过重新 install。"
        return 0
    fi

    # 在独立环境内执行 kitless install（不安装 Isaac Sim）。
    export ISAACLAB_SKIP_PREBUNDLE_REPOINT=1
    if ! "${ISAAC_LIB_DIR}/isaaclab.sh" --install "$spec"; then
        unset ISAACLAB_SKIP_PREBUNDLE_REPOINT
        echo "错误: OV kitless install 失败，环境仍保持为当前已激活状态。"
        return 1
    fi
    unset ISAACLAB_SKIP_PREBUNDLE_REPOINT

    # install 可能改动环境，重新激活以反映最终状态。
    activate_mode_env
}

main() {
    parse_mode_and_env "$@" || return $?

    if [ "$SHOW_HELP" = "1" ]; then
        usage
        return 0
    fi

    # 指定任一 OV 选项时，走独立 kitless 流程，完全不触碰主环境与 IsaacSim。
    if [ -n "$OV_EXTRA" ]; then
        run_ov_kitless_setup
        return $?
    fi

    if [ "$MODE" = "uv" ]; then
        echo "使用 uv 环境名: $ENV_NAME"
    else
        echo "使用 Conda 环境名: $ENV_NAME"
    fi

    ensure_env_created || return $?
    activate_mode_env || return $?
    prompt_reinstall_existing_env || return $?

    export ISAACLAB_SKIP_ENV_CREATE="$SKIP_ENV_CREATE"
    unset ISAACLAB_SKIP_ENV_ACTIVATE

    if [ "$RUN_INSTALL" = "1" ]; then
        if ! run_pip_isaaclab_install; then
            echo "错误: Isaac Sim pip 包或 IsaacLab 安装失败，环境仍保持为当前已激活状态。"
            return 1
        fi
    else
        echo "已跳过重新 install。"
    fi

    activate_mode_env
}

main "$@"
stupid_setup_status=$?
# shellcheck disable=SC2317
return "$stupid_setup_status" 2>/dev/null || exit "$stupid_setup_status"
