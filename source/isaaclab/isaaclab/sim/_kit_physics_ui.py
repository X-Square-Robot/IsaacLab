# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for reflecting Isaac Lab's active backend in Kit physics UI."""

from __future__ import annotations

import importlib
import logging
from types import FunctionType
from typing import Any

logger = logging.getLogger(__name__)

_SIMULATION_CONFIG_MODULE = "omni.physics.ui.scripts.simulation_config"
_MENUBAR_MODEL_MODULE = "omni.kit.viewport.menubar.core"
_PATCHED_FLAG = "_isaaclab_backend_label_patched"
_ORIGINAL_BUILD_NAME = "_isaaclab_original_build_menu"
_APPLY_TEXT_NAME = "_isaaclab_apply_backend_label"
_APPLY_SIMULATOR_NAMES_NAME = "_isaaclab_apply_backend_simulator_names"
_RESTORE_SIMULATOR_NAMES_NAME = "_isaaclab_restore_backend_simulator_names"
_UNSET: Any = object()

_menu_text_override: str | None = None
_extension_enable_hook: Any = None


def sync_kit_physics_ui_label(physics_cfg: object | None = _UNSET) -> None:
    """Mirror the current Isaac Lab physics backend in Kit's existing physics menu.

    Isaac Sim's ``omni.physics.ui`` viewport menu reports the active simulator
    registered in ``omni.physics.core``. Isaac Lab's Newton backend does not use
    that simulator registry, so the menu can keep showing ``PhysX`` while Isaac
    Lab is stepping Newton. This helper patches the existing menu text in place
    when the optional Kit UI extension is present; it does not create a new UI
    label and does not switch Isaac Sim's physics engine.

    Args:
        physics_cfg: Active Isaac Lab physics configuration. When omitted, the
            current :class:`~isaaclab.physics.PhysicsManager` configuration is
            inspected.
    """
    global _menu_text_override

    if physics_cfg is _UNSET:
        physics_cfg = _current_physics_cfg()
    _menu_text_override = _physics_cfg_menu_label(physics_cfg)

    if _patch_simulation_config_menu():
        _refresh_existing_simulation_menu()
    else:
        _subscribe_to_physics_ui_enable()


def _current_physics_cfg() -> object | None:
    """Return the active physics config if Isaac Lab has initialized one."""
    try:
        from isaaclab.physics import PhysicsManager
    except Exception:
        return None
    return getattr(PhysicsManager, "_cfg", None)


def _physics_cfg_menu_label(physics_cfg: object | None) -> str | None:
    """Return the Kit menu label override for an Isaac Lab physics config."""
    if physics_cfg is None:
        return None

    cfg_type = type(physics_cfg)
    if cfg_type.__name__ == "NewtonCfg" or cfg_type.__module__.startswith("isaaclab_newton."):
        return "Newton"

    # For PhysX and all other backends, preserve omni.physics.ui's own active
    # simulator label. This restores ``PhysX`` without hiding ``None`` or
    # ``Multiple`` states when the user changes the generic simulator menu.
    return None


def _resolve_menu_text(current_text: str | None) -> str | None:
    """Return the text that should be shown on the existing Kit physics menu."""
    return _menu_text_override or current_text


def _patch_simulation_config_menu() -> bool:
    """Patch ``omni.physics.ui`` menu builds to apply Isaac Lab's label override."""
    try:
        simulation_config = importlib.import_module(_SIMULATION_CONFIG_MODULE)
    except Exception as exc:
        logger.debug("Kit physics UI is not available for Isaac Lab backend label sync: %s", exc)
        return False

    menu_cls = getattr(simulation_config, "SimulationConfigViewportMenu", None)
    build_menu = getattr(menu_cls, "_build_menu", None)
    if menu_cls is None or build_menu is None:
        return False

    simulation_config.__dict__[_APPLY_TEXT_NAME] = _apply_menu_text
    simulation_config.__dict__[_APPLY_SIMULATOR_NAMES_NAME] = _apply_simulator_names
    simulation_config.__dict__[_RESTORE_SIMULATOR_NAMES_NAME] = _restore_simulator_names

    if getattr(build_menu, _PATCHED_FLAG, False):
        return True

    original_build = _copy_function(build_menu)
    simulation_config.__dict__[_ORIGINAL_BUILD_NAME] = original_build

    exec(
        (
            f"def _isaaclab_build_menu(self, args):\n"
            f"    _isaaclab_simulator_name_restore = {_APPLY_SIMULATOR_NAMES_NAME}(self)\n"
            f"    try:\n"
            f"        {_ORIGINAL_BUILD_NAME}(self, args)\n"
            f"    finally:\n"
            f"        {_RESTORE_SIMULATOR_NAMES_NAME}(_isaaclab_simulator_name_restore)\n"
            f"    {_APPLY_TEXT_NAME}(self)\n"
        ),
        simulation_config.__dict__,
    )
    replacement = simulation_config.__dict__["_isaaclab_build_menu"]

    if isinstance(build_menu, FunctionType):
        # Existing ui.Menu callbacks can hold a bound method created before this
        # patch. Mutating the original function object keeps those callbacks on
        # the patched code path, whereas replacing the class attribute would not.
        build_menu.__code__ = replacement.__code__
        build_menu.__defaults__ = replacement.__defaults__
        build_menu.__kwdefaults__ = replacement.__kwdefaults__
        setattr(build_menu, _PATCHED_FLAG, True)
    else:
        setattr(replacement, _PATCHED_FLAG, True)
        setattr(menu_cls, "_build_menu", replacement)
    return True


def _copy_function(function: FunctionType) -> FunctionType:
    """Copy a function before mutating the original function object."""
    copied = FunctionType(
        function.__code__,
        function.__globals__,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    copied.__kwdefaults__ = function.__kwdefaults__
    copied.__doc__ = function.__doc__
    copied.__module__ = function.__module__
    copied.__annotations__ = dict(getattr(function, "__annotations__", {}))
    return copied


def _refresh_existing_simulation_menu() -> None:
    """Apply the backend label to an already-created viewport simulation menu."""
    try:
        menubar = importlib.import_module(_MENUBAR_MODEL_MODULE)
        get_menu_item = getattr(menubar, "get_menu_item", None) or getattr(menubar, "get_item")
        menu = get_menu_item("Simulation")
    except Exception as exc:
        logger.debug("Could not find Kit simulation menu for Isaac Lab backend label sync: %s", exc)
        return

    if menu is None:
        return

    _apply_menu_text(menu)
    invalidate = getattr(menu, "invalidate", None)
    if callable(invalidate):
        try:
            invalidate()
        except Exception as exc:
            logger.debug("Could not invalidate Kit simulation menu after backend label sync: %s", exc)


def _apply_simulator_names(menu: object) -> list[tuple[object, str]] | None:
    """Temporarily show the active PhysX simulator row as the Isaac Lab backend.

    ``omni.physics.ui`` builds the dropdown simulator rows directly from each
    simulator's ``name`` field. When Isaac Lab runs Newton, Kit still has an
    active PhysX simulator registered for USD/viewport integration, so the row
    would otherwise keep saying ``PhysX`` even though the Isaac Lab backend is
    Newton. Temporarily replacing the active PhysX row name before the original
    build keeps the existing dropdown item in sync without creating any new UI.
    """
    if _menu_text_override is None:
        return None

    simulators = getattr(menu, "simulators", getattr(menu, "_simulators", []))
    renamed: list[tuple[object, str]] = []
    for simulator in simulators:
        if not getattr(simulator, "active", False):
            continue
        current_name = getattr(simulator, "name", None)
        if current_name != "PhysX":
            continue
        try:
            simulator.name = _menu_text_override
        except Exception as exc:
            logger.debug("Could not update Kit simulator row label: %s", exc)
        else:
            renamed.append((simulator, current_name))
    return renamed


def _restore_simulator_names(renamed: list[tuple[object, str]] | None) -> None:
    """Restore simulator names after the existing Kit menu has been built."""
    if renamed is None:
        return

    for simulator, original_name in renamed:
        try:
            simulator.name = original_name
        except Exception as exc:
            logger.debug("Could not restore Kit simulator row label: %s", exc)


def _apply_menu_text(menu: object) -> None:
    """Apply the resolved backend label to an ``omni.physics.ui`` menu object."""
    ui_menu = getattr(menu, "_ui_menu", None)
    if ui_menu is None:
        return

    current_text = _active_simulator_text(getattr(menu, "simulators", getattr(menu, "_simulators", [])))
    try:
        ui_menu.text = _resolve_menu_text(current_text)
    except Exception as exc:
        logger.debug("Could not update Kit simulation menu text: %s", exc)


def _active_simulator_text(simulators: list[object]) -> str | None:
    """Return the label ``omni.physics.ui`` would show for active simulators."""
    current = "None"
    for simulator in simulators:
        if not getattr(simulator, "active", False):
            continue
        if current != "None":
            return "Multiple"
        current = getattr(simulator, "name", None)
    return current


def _subscribe_to_physics_ui_enable() -> None:
    """Patch later if ``omni.physics.ui`` is enabled after the sim context."""
    global _extension_enable_hook

    if _extension_enable_hook is not None:
        return

    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        _extension_enable_hook = manager.subscribe_to_extension_enable(
            on_enable_fn=lambda *_args, **_kwargs: sync_kit_physics_ui_label(),
            on_disable_fn=lambda *_args, **_kwargs: None,
            ext_name="omni.physics.ui",
            hook_name="isaaclab physics ui backend label listener",
        )
    except Exception as exc:
        logger.debug("Could not subscribe to omni.physics.ui enable events: %s", exc)
