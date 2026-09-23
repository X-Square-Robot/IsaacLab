# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for loading optional OvPhysX runtime modules."""

from __future__ import annotations

import importlib
from types import ModuleType

_OVPHYSX_INSTALL_MESSAGE = (
    "The OvPhysX backend requires the optional 'ovphysx' runtime wheel, which is not installed. "
    "Run your command with: uv run --extra ovphysx <command> "
    "(or, manually: python -m pip install --extra-index-url https://pypi.nvidia.com ovphysx)."
)
_OVSTAGE_INSTALL_MESSAGE = (
    "The installed OvPhysX runtime requires its 'ovstage' companion package, which is not installed. "
    "Reinstall the pinned runtime with: ./isaaclab.sh -i 'ov[ovphysx]'"
)


def import_ovphysx(module_name: str = "ovphysx") -> ModuleType:
    """Import an optional ``ovphysx`` runtime module with an actionable install error.

    Args:
        module_name: Name of the ``ovphysx`` module to import.

    Returns:
        The imported runtime module.

    Raises:
        ModuleNotFoundError: If the optional ``ovphysx`` runtime wheel is not installed.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != "ovphysx":
            raise
        raise ModuleNotFoundError(_OVPHYSX_INSTALL_MESSAGE, name="ovphysx") from exc


def import_ovstage() -> ModuleType:
    """Import the optional ``ovstage`` companion with an actionable install error.

    Returns:
        The imported ``ovstage`` runtime module.

    Raises:
        ModuleNotFoundError: If ``ovstage`` is not installed.
    """
    try:
        return importlib.import_module("ovstage")
    except ModuleNotFoundError as exc:
        if exc.name != "ovstage":
            raise
        raise ModuleNotFoundError(_OVSTAGE_INSTALL_MESSAGE, name="ovstage") from exc
