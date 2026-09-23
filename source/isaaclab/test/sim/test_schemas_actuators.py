# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton actuator USD authoring selection."""

from pxr import Usd

from isaaclab.sim.schemas.schemas_actuators import define_actuator_properties


def test_opted_out_articulation_deactivates_existing_newton_actuators() -> None:
    """An explicit per-articulation opt-out must block referenced native prims."""
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Robot", "Xform")
    native_prim = stage.DefinePrim("/World/Robot/should_not_run", "NewtonActuator")

    define_actuator_properties(
        "/World/Robot",
        actuator_cfgs={},
        stage=stage,
        use_newton_actuators=False,
    )

    assert native_prim.IsValid()
    assert native_prim.IsActive() is False


def test_omitted_selection_without_simulation_context_is_a_noop(monkeypatch) -> None:
    """The backward-compatible omitted-selection path must not mutate a stage."""
    import isaaclab.sim as sim_utils

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Robot", "Xform")
    native_prim = stage.DefinePrim("/World/Robot/native", "NewtonActuator")
    monkeypatch.setattr(
        sim_utils,
        "SimulationContext",
        type("SimulationContextStub", (), {"instance": classmethod(lambda cls: None)}),
    )

    define_actuator_properties("/World/Robot", actuator_cfgs={}, stage=stage)

    assert native_prim.IsActive() is True
