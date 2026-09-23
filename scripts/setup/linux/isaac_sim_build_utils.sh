#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Utility functions shared by Linux setup entry points. This file is intended
# to be sourced and must not execute setup work on its own.

isaacsim_required_runtime_files() {
    printf '%s\n' "isaac-sim.sh" "python.sh" "setup_python_env.sh"
}

isaacsim_missing_runtime_files() {
    local sim_path="$1"
    local rel_path
    local missing=()

    while IFS= read -r rel_path; do
        if [[ ! -f "${sim_path}/${rel_path}" ]]; then
            missing+=("$rel_path")
        fi
    done < <(isaacsim_required_runtime_files)

    if [[ "${#missing[@]}" -gt 0 ]]; then
        printf '%s\n' "${missing[@]}"
    fi
}

isaacsim_build_is_complete() {
    local sim_path="$1"

    if [[ ! -d "$sim_path" ]]; then
        return 1
    fi

    [[ -z "$(isaacsim_missing_runtime_files "$sim_path")" ]]
}

isaacsim_setup_needs_run() {
    local build_path="$1"
    local link_path="$2"

    if ! isaacsim_build_is_complete "$build_path"; then
        return 0
    fi

    if [[ -L "$link_path" ]]; then
        if ! isaacsim_build_is_complete "$link_path"; then
            return 0
        fi
        return 1
    fi

    if [[ ! -e "$link_path" ]]; then
        return 0
    fi

    if ! isaacsim_build_is_complete "$link_path"; then
        return 0
    fi

    return 1
}

# Pure helper: echo $1 (a PATH-style string) with all `_isaac_sim` and empty
# segments removed. Does not touch the environment.
strip_isaac_sim_from_pathvar() {
    local cur="$1" cleaned="" entry IFS_OLD="$IFS"
    IFS=':'
    for entry in $cur; do
        case "$entry" in
            *"/_isaac_sim/"* | *"/_isaac_sim") ;; # drop Isaac entries
            "") ;;                                # drop empty segments
            *) cleaned="${cleaned:+$cleaned:}$entry" ;;
        esac
    done
    IFS="$IFS_OLD"
    printf '%s' "$cleaned"
}

# Strip `_isaac_sim` cp312 pollution from the CURRENT shell's PYTHONPATH /
# LD_LIBRARY_PATH. Once any Isaac conda env is activated, its activate.d hook
# sources `_isaac_sim/setup_conda_env.sh`, which prepends the cp312 IsaacSim
# stdlib (.../_isaac_sim/kit/python/lib/python3.12) onto PYTHONPATH. A later
# bare `conda` (system base, cp313) then imports cp312's `re`/`_sre` and dies
# with `AssertionError: SRE module mismatch`. Call this immediately before any
# bare `conda activate` so the activation itself runs from a clean PYTHONPATH;
# the env's own activate.d re-adds the correct paths afterward.
strip_isaac_sim_pollution() {
    local _clean
    _clean="$(strip_isaac_sim_from_pathvar "${PYTHONPATH:-}")"
    if [ -z "$_clean" ]; then unset PYTHONPATH; else export PYTHONPATH="$_clean"; fi
    _clean="$(strip_isaac_sim_from_pathvar "${LD_LIBRARY_PATH:-}")"
    if [ -z "$_clean" ]; then unset LD_LIBRARY_PATH; else export LD_LIBRARY_PATH="$_clean"; fi
}
