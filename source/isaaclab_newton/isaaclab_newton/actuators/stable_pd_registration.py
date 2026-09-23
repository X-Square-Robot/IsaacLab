# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ensure the Tan 2011 stable-PD controller is registered with Newton's USD parser.

Current Newton builds register the ``NewtonStablePDControlAPI`` →
:class:`newton.actuators.ControllerStablePD` mapping themselves, as a built-in
entry of :data:`newton._src.actuators.usd_parser._SCHEMA_REGISTRY` (alongside
``NewtonPDControlAPI`` / ``NewtonPIDControlAPI`` / ``NewtonNeuralControlAPI``).
On such builds :func:`register_stable_pd_schema` is a no-op: it finds the entry
already present and returns without touching the registry (so it never triggers
the "already registered; overwriting" warning).

The function is retained as a backward-compat shim for older Newton builds that
ship :class:`ControllerStablePD` but predate the built-in registry entry: there
it adds the mapping via the public
:func:`newton.actuators.register_actuator_component` API. Either way,
:class:`~isaaclab.actuators.StablePDActuatorCfg` configs authored by
:func:`isaaclab.sim.schemas.define_actuator_properties` resolve to the implicit
controller on both the Newton and PhysX backends.

The registration is idempotent and is invoked as an import-time side effect from
:mod:`isaaclab_newton.actuators`. It no-ops gracefully when the installed Newton
build predates ``ControllerStablePD`` or the public registration API.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_STABLE_PD_SCHEMA_NAME = "NewtonStablePDControlAPI"

_registered = False


def register_stable_pd_schema() -> bool:
    """Register ``NewtonStablePDControlAPI`` → ``ControllerStablePD`` (idempotent).

    Returns:
        ``True`` when the schema is registered (or was already), ``False``
        when the installed Newton build lacks the required symbols.
    """
    global _registered
    if _registered:
        return True

    try:
        from newton.actuators import (  # noqa: PLC0415
            ComponentKind,
            ControllerStablePD,
            register_actuator_component,
        )
    except ImportError:
        logger.debug(
            "Newton build lacks ControllerStablePD / register_actuator_component;"
            " skipping NewtonStablePDControlAPI registration."
        )
        return False

    # Already registered by a prior import or by Newton itself: treat as success.
    from newton._src.actuators.usd_parser import _SCHEMA_REGISTRY  # noqa: PLC0415

    if _STABLE_PD_SCHEMA_NAME not in _SCHEMA_REGISTRY:
        register_actuator_component(
            _STABLE_PD_SCHEMA_NAME,
            ControllerStablePD,
            ComponentKind.CONTROLLER,
        )
    _registered = True
    return True
