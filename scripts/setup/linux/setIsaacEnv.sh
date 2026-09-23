#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -eo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/isaac_sim_build_utils.sh"

MODE="conda"
ENV_NAME="${CONDA_ENV_NAME:-env_isaaclab}"
REBUILD=0

_normalize_env_name() {
    local name="$1"
    while :; do
        case "$name" in
            *。|*.) name="${name%?}" ;;
            *) break ;;
        esac
    done
    printf '%s' "$name"
}

usage() {
    cat <<'EOF'
用法:
  source scripts/setup/linux/setIsaacEnv.sh [--conda [环境名]] [--uv [环境名]] [--rb|--rebuild]

说明:
  --conda      使用 Conda 创建/配置环境；默认环境名为 env_isaaclab。
  --uv         使用 uv 创建/配置环境；默认环境名为 env_isaaclab。
  --rb         删除 Isaac Sim build 目录后重新构建。
  --rebuild    同 --rb。
  --help, -h   显示帮助。
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda)
            MODE="conda"
            if [[ -n "${2-}" && "${2-}" != --* ]]; then
                ENV_NAME="$2"
                shift 2
            else
                shift
            fi
            ;;
        --uv)
            MODE="uv"
            if [[ -n "${2-}" && "${2-}" != --* ]]; then
                ENV_NAME="$2"
                shift 2
            else
                shift
            fi
            ;;
        --rb|--rebuild)
            REBUILD=1
            shift
            ;;
        --help|-h)
            usage
            return 0 2>/dev/null || exit 0
            ;;
        *)
            echo "错误: 未知参数: $1"
            usage
            return 1 2>/dev/null || exit 1
            ;;
    esac
done

ENV_NAME="$(_normalize_env_name "$ENV_NAME")"

if [[ -z "${ISAAC_SIM_DIR:-}" ]]; then
    echo "错误: ISAAC_SIM_DIR 未设置。"
    echo "请先从仓库根目录运行 stupidSetup.sh，或手动导出 ISAAC_SIM_DIR。"
    return 1 2>/dev/null || exit 1
fi

ISAACLAB_ROOT="${ISAAC_LIB_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Isaac Sim root directory.
export ISAAC_BUILD_DIR="_build/linux-x86_64/release"
export ISAACSIM_PATH="${ISAAC_SIM_DIR}/${ISAAC_BUILD_DIR}"

run_isaacsim_build() {
    echo "Building Isaac Sim..."
    bash "${ISAAC_SIM_DIR}/build.sh"
}

format_missing_isaacsim_files() {
    local sim_path="$1"
    local missing_files=()
    local missing_file
    local joined=""

    mapfile -t missing_files < <(isaacsim_missing_runtime_files "$sim_path")
    for missing_file in "${missing_files[@]}"; do
        joined="${joined:+$joined, }${missing_file}"
    done
    printf '%s' "$joined"
}

ensure_isaacsim_build() {
    local missing

    if [[ "$REBUILD" -eq 1 ]]; then
        rm -rf "$ISAACSIM_PATH"
        echo "Isaac Sim build directory removed."
    fi

    if [[ ! -d "$ISAACSIM_PATH" ]]; then
        echo "Isaac Sim build directory not found at $ISAACSIM_PATH"
        run_isaacsim_build
    elif ! isaacsim_build_is_complete "$ISAACSIM_PATH"; then
        missing="$(format_missing_isaacsim_files "$ISAACSIM_PATH")"
        echo "Isaac Sim build directory is incomplete at $ISAACSIM_PATH; missing: $missing"
        run_isaacsim_build
    fi

    if ! isaacsim_build_is_complete "$ISAACSIM_PATH"; then
        missing="$(format_missing_isaacsim_files "$ISAACSIM_PATH")"
        echo "错误: Isaac Sim build did not produce required runtime files at $ISAACSIM_PATH: $missing"
        return 1
    fi
}

link_isaacsim_build() {
    local link_path="${ISAACLAB_ROOT}/_isaac_sim"
    local backup_path
    local missing

    if [[ -L "$link_path" || ! -e "$link_path" ]]; then
        ln -sfn "$ISAACSIM_PATH" "$link_path"
        return
    fi

    if ! isaacsim_build_is_complete "$link_path"; then
        missing="$(format_missing_isaacsim_files "$link_path")"
        backup_path="${link_path}.incomplete.$(date +%Y%m%d%H%M%S)"
        echo "警告: _isaac_sim 已存在但不完整，缺失: $missing"
        echo "移动到: $backup_path"
        mv "$link_path" "$backup_path"
        ln -s "$ISAACSIM_PATH" "$link_path"
        return
    fi

    echo "警告: _isaac_sim 已存在但不是符号链接，且看起来是完整 Isaac Sim，跳过链接创建。"
}

echo "Beginning setup..."

# Check if Ubuntu 24.04 and install GCC 11.
if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    if [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]]; then
        # Skip sudo when GCC 11 is already installed and selected (keeps non-interactive reruns working).
        if command -v gcc-11 >/dev/null 2>&1 && command -v g++-11 >/dev/null 2>&1 \
            && [[ "$(update-alternatives --query gcc 2>/dev/null | awk '/^Value:/{print $2}')" == "/usr/bin/gcc-11" ]]; then
            echo "GCC 11 already installed and selected, skipping."
        else
            echo "Detected Ubuntu 24.04, installing GCC 11..."
            sudo apt-get install -y gcc-11 g++-11
            sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
            sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200
        fi
    fi
fi

ensure_isaacsim_build

# Isaac Sim python executable.
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

link_isaacsim_build

if [[ "${ISAACLAB_SKIP_ENV_CREATE:-0}" != "1" ]]; then
    if [[ "$MODE" == "uv" ]]; then
        echo "创建/使用 uv 环境: $ENV_NAME"
        ./isaaclab.sh --uv "$ENV_NAME"
    else
        export CONDA_ENV_NAME="$ENV_NAME"
        echo "创建/使用 Conda 环境: $CONDA_ENV_NAME"
        ./isaaclab.sh --conda "$CONDA_ENV_NAME"
    fi
fi

if [[ "$MODE" == "uv" ]]; then
    UV_ENV_PATH="${VIRTUAL_ENV:-${ISAAC_LIB_DIR:-$(pwd)}/${ENV_NAME}}"
    if [[ -f "${UV_ENV_PATH}/bin/activate" ]]; then
        echo "激活 uv 环境: ${UV_ENV_PATH}"
        # shellcheck disable=SC1091
        source "${UV_ENV_PATH}/bin/activate"
    else
        echo "错误: uv 环境激活脚本不存在: ${UV_ENV_PATH}/bin/activate"
        return 1 2>/dev/null || exit 1
    fi
else
    export CONDA_ENV_NAME="${CONDA_ENV_NAME:-$ENV_NAME}"

    # This script is often run as a plain `bash` subprocess by stupidSetup.sh
    # *after* it activated the target env, so it inherits a PYTHONPATH already
    # polluted with IsaacSim's cp312 stdlib (via the env's activate.d). Strip it
    # up front so every `conda` invocation below (info/activate, all cp313
    # binaries) runs clean; the `conda activate` further down re-adds the
    # correct env paths. Safe: this is the subprocess's own shell.
    strip_isaac_sim_pollution

    # 本脚本常被 stupidSetup.sh 以纯 `bash` 派生子进程运行（不带 BASH_ENV=conda.sh），
    # 子 shell 里 `conda` 只是 PATH 上的二进制、并非 shell 函数；`command -v conda` 会
    # 误判为已就绪，导致后面 `conda activate` 报 "Run 'conda init' before 'conda activate'"。
    # 这里显式加载 conda 的 shell hook，使 `conda activate` 函数在子进程中可用。
    if [ "$(type -t conda 2>/dev/null || true)" != "function" ]; then
        _conda_base=""
        if [ -n "${CONDA_EXE:-}" ]; then
            _conda_base="$(dirname "$(dirname "$CONDA_EXE")")"
        elif command -v conda >/dev/null 2>&1; then
            _conda_base="$(conda info --base 2>/dev/null || true)"
        fi
        if [ -n "$_conda_base" ] && [ -f "$_conda_base/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$_conda_base/etc/profile.d/conda.sh"
        fi
    fi

    if [ "$(type -t conda 2>/dev/null || true)" != "function" ]; then
        echo "错误: conda shell 函数不可用，无法激活环境（请确认已安装 Conda 且存在 etc/profile.d/conda.sh）。"
        return 1 2>/dev/null || exit 1
    fi
    echo "conda initialized"

    echo "激活 Conda 环境: $CONDA_ENV_NAME"
    # 本脚本被 stupidSetup.sh 以子进程派生，会继承父 shell 在其 `conda activate`
    # 时由 Isaac env 钩子（activate.d/setenv.sh -> setup_conda_env.sh ->
    # setup_python_env.sh）注入的 cp312 `_isaac_sim` PYTHONPATH/LD_LIBRARY_PATH。
    # 下面的 `conda activate` 走 base 的 cp313 conda 二进制，import 到那份继承来的
    # cp312 `json`/`re` stdlib 即 `AssertionError: SRE module mismatch` 崩。
    # 故激活前就地剥离 `_isaac_sim` 段（无法依赖 utils/shell_common.sh——这里是
    # 不 source 它的子进程）；激活成功后钩子会按 cp312 env 重新注入正确路径。
    _strip_isaac_pp() {
        local cur="$1" cleaned="" entry IFS_OLD="$IFS"
        IFS=':'
        for entry in $cur; do
            case "$entry" in
                *"/_isaac_sim/"* | *"/_isaac_sim") ;;
                "") ;;
                *) cleaned="${cleaned:+$cleaned:}$entry" ;;
            esac
        done
        IFS="$IFS_OLD"
        printf '%s' "$cleaned"
    }
    _clean_pp="$(_strip_isaac_pp "${PYTHONPATH:-}")"
    if [ -z "$_clean_pp" ]; then unset PYTHONPATH; else export PYTHONPATH="$_clean_pp"; fi
    _clean_ld="$(_strip_isaac_pp "${LD_LIBRARY_PATH:-}")"
    if [ -z "$_clean_ld" ]; then unset LD_LIBRARY_PATH; else export LD_LIBRARY_PATH="$_clean_ld"; fi
    unset _clean_pp _clean_ld
    # conda 的 shell 函数依赖 bash builtin `pop_var_context`；在 `set -eo pipefail`
    # 下、且从父进程继承了 CONDA_SHLVL>=2 的嵌套环境里（本脚本被 BASH_ENV 注入式
    # 子 shell 调用即如此），`conda activate` 会触发
    # "pop_var_context: head of shell_variables not a function context" 并返回非0，
    # 导致 errexit 直接中止安装。这是 conda 与 set -e 的已知不兼容，官方亦建议不要
    # 在 set -e 下调 conda activate。故临时关闭 errexit/pipefail 包住激活，再显式
    # 检查返回码——真正的激活失败不会被吞掉。
    set +e +o pipefail
    strip_isaac_sim_pollution
    conda activate "$CONDA_ENV_NAME"
    _activate_rc=$?
    set -eo pipefail
    if [ "$_activate_rc" -ne 0 ]; then
        echo "错误: conda activate $CONDA_ENV_NAME 失败 (rc=$_activate_rc)。"
        return 1 2>/dev/null || exit 1
    fi
fi

# Install system and Python dependencies (skip sudo when already present, for non-interactive reruns).
if dpkg -s cmake build-essential >/dev/null 2>&1; then
    echo "cmake and build-essential already installed, skipping."
else
    sudo apt-get install -y cmake build-essential
fi
./isaaclab.sh --install
