#!/usr/bin/env bash

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Exit on error.
set -e

# Get repo directory.
export ISAACLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

_is_python_312() {
    "$1" - <<'PY' >/dev/null 2>&1
import sys

raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)
PY
}

# Find python to run CLI.
if [ -n "$VIRTUAL_ENV" ]; then
    python_exe="$VIRTUAL_ENV/bin/python"
elif [ -n "$CONDA_PREFIX" ]; then
    conda_python_exe="$CONDA_PREFIX/bin/python"
    if _is_python_312 "$conda_python_exe"; then
        python_exe="$conda_python_exe"
    elif [ -f "$ISAACLAB_PATH/env_isaaclab/bin/python" ]; then
        echo "[WARNING] Active conda Python is not 3.12; using ./env_isaaclab/bin/python for Isaac Sim compatibility." >&2
        python_exe="$ISAACLAB_PATH/env_isaaclab/bin/python"
    else
        python_exe="$conda_python_exe"
    fi
elif [ -f "$ISAACLAB_PATH/env_isaaclab/bin/python" ]; then
    python_exe="$ISAACLAB_PATH/env_isaaclab/bin/python"
elif [ -f "$ISAACLAB_PATH/_isaac_sim/python.sh" ]; then
    python_exe="$ISAACLAB_PATH/_isaac_sim/python.sh"
else
    # Fallback to system python
    python_exe="python3"
fi

# Add source/isaaclab to PYTHONPATH so we can import isaaclab.cli.
export PYTHONPATH="$ISAACLAB_PATH/source/isaaclab:$PYTHONPATH"

# Let Kit associate direct wrapper launches with the Isaac Sim desktop icon.
export RESOURCE_NAME="${RESOURCE_NAME:-IsaacSim}"

# If a local Isaac Sim binary is present, source its env setup so that
# PYTHONPATH/PATH/EXP_PATH are correct without depending on a conda
# activate.d hook (those don't fire reliably under e.g. `conda run`).
if [ -d "$ISAACLAB_PATH/_isaac_sim" ]; then
    if [ -f "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" ]; then
        # shellcheck disable=SC1091
        . "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" >/dev/null 2>&1 || true
    elif [ -f "$ISAACLAB_PATH/_isaac_sim/setup_python_env.sh" ]; then
        export ISAAC_PATH="$ISAACLAB_PATH/_isaac_sim"
        export CARB_APP_PATH="$ISAAC_PATH/kit"
        export EXP_PATH="$ISAAC_PATH/apps"
        # shellcheck disable=SC1091
        . "$ISAACLAB_PATH/_isaac_sim/setup_python_env.sh" >/dev/null 2>&1 || true
        # Unlike setup_conda_env.sh, setup_python_env.sh prepends Kit's
        # pip_prebundle directories to PYTHONPATH. Those ship vendored copies of
        # common libraries (e.g. an older typing_extensions lacking Sentinel)
        # that then shadow the active venv/conda environment. Put the active
        # environment's site-packages first so it always wins.
        if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_PREFIX" ]; then
            env_site_packages="$("$python_exe" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
            if [ -n "$env_site_packages" ]; then
                export PYTHONPATH="$env_site_packages:$PYTHONPATH"
            fi
        fi
    else
        echo "[WARNING] _isaac_sim is present but _isaac_sim/setup_conda_env.sh or _isaac_sim/setup_python_env.sh is missing; Isaac Sim env vars not exported." >&2
        echo "[WARNING] Re-extract the Isaac Sim binary zip if you intend to use the bundled binary." >&2
    fi

    # `setup_python_env.sh` prepends the bundled Kit cp312 *stdlib* dirs
    # (`_isaac_sim/kit/python/lib/python3.12[/site-packages]`) onto PYTHONPATH.
    # Those exist for Kit's own bundled `kit/python/bin/python3`; under a conda/
    # venv python (which ships its OWN complete, matching cp312 stdlib) they
    # SHADOW it. Kit/omni modules then `import platform` from the bundled stdlib
    # whose `_sys_version()` regex cannot parse conda-forge's version string
    # ('3.12.13 | packaged by conda-forge | ...') -> `ValueError: failed to parse
    # CPython sys.version`, which cascades into omni.kit.test / omni.usd failing
    # to start and finally `AttributeError: module 'omni.usd' has no attribute
    # 'get_context'` at SimulationApp stage creation. Strip only the bare stdlib
    # dirs (keep bindings-python / kernel/py / *pip_prebundle, which Kit needs)
    # when NOT running under Kit's own python.sh interpreter.
    case "$python_exe" in
        *"/_isaac_sim/python.sh") ;;  # Kit's own python: its stdlib is correct, leave PYTHONPATH alone
        *)
            if [ -n "${PYTHONPATH:-}" ]; then
                _kit_stdlib="$ISAACLAB_PATH/_isaac_sim/kit/python/lib/python3.12"
                _cleaned_pp=""
                _OLD_IFS="$IFS"
                IFS=':'
                for _pp_entry in $PYTHONPATH; do
                    case "$_pp_entry" in
                        "$_kit_stdlib" | "$_kit_stdlib/site-packages" | "") ;;  # drop bundled stdlib + empty
                        *) _cleaned_pp="${_cleaned_pp:+$_cleaned_pp:}$_pp_entry" ;;
                    esac
                done
                IFS="$_OLD_IFS"
                export PYTHONPATH="$_cleaned_pp"
                unset _kit_stdlib _cleaned_pp _pp_entry _OLD_IFS
            fi
            ;;
    esac

    # `setup_python_env.sh` only exports LD_LIBRARY_PATH/PYTHONPATH; the launcher
    # vars (EXP_PATH/CARB_APP_PATH/ISAAC_PATH) are set by `python.sh` around the
    # source, which we bypass when running under the venv/conda python. Derive
    # them from the _isaac_sim dir when unset so AppLauncher can resolve the
    # experience (.kit) files. Older `setup_conda_env.sh` already exports these,
    # so the `:=` defaults leave any existing value untouched.
    _isaac_sim_dir="$ISAACLAB_PATH/_isaac_sim"
    : "${EXP_PATH:=${_isaac_sim_dir}/apps}"
    : "${CARB_APP_PATH:=${_isaac_sim_dir}/kit}"
    : "${ISAAC_PATH:=${_isaac_sim_dir}}"
    export EXP_PATH CARB_APP_PATH ISAAC_PATH

    # `python.sh` also preloads libcarb.so (a workaround for a missing-symbol
    # issue) before launching; without it Kit resolves an older carb logging
    # plugin and fails with "ILogging v1.7 requested but ... v1.6". Mirror that
    # preload here since we run under the venv/conda python, not python.sh.
    if [ -f "${_isaac_sim_dir}/kit/libcarb.so" ]; then
        case ":${LD_PRELOAD:-}:" in
            *":${_isaac_sim_dir}/kit/libcarb.so:"*) ;;  # already present
            *) export LD_PRELOAD="${_isaac_sim_dir}/kit/libcarb.so${LD_PRELOAD:+:$LD_PRELOAD}" ;;
        esac
    fi
fi

# Execute CLI.
exec "$python_exe" -c "from isaaclab.cli import cli; cli()" "$@"
