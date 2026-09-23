# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for per-articulation Newton actuator selection."""

from types import SimpleNamespace

from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg


def _sim_cfg(*, enabled: bool, device: str | None = "cuda:0", graph: bool | None = True) -> SimpleNamespace:
    return SimpleNamespace(
        use_newton_actuators=enabled,
        newton_actuator_device=device,
        newton_actuator_cuda_graph=graph,
    )


def test_native_actuator_settings_inherit_simulation_defaults() -> None:
    cfg = ArticulationCfg(prim_path="/World/Robot", actuators={})

    assert cfg.native_actuators is None
    assert cfg.resolve_newton_actuator_settings(_sim_cfg(enabled=True)) == (True, "cuda:0", True)
    assert cfg.resolve_newton_actuator_settings(_sim_cfg(enabled=False)) == (False, None, None)


def test_native_actuator_settings_can_opt_out_per_articulation() -> None:
    cfg = ArticulationCfg(
        prim_path="/World/Drawer",
        actuators={},
        native_actuators=False,
    )

    assert cfg.resolve_newton_actuator_settings(_sim_cfg(enabled=True)) == (False, None, None)


def test_native_actuator_settings_can_override_device_and_graph() -> None:
    cfg = ArticulationCfg(
        prim_path="/World/Robot",
        actuators={},
        native_actuators=True,
        newton_actuator_device="cuda:1",
        newton_actuator_cuda_graph=False,
    )

    assert cfg.resolve_newton_actuator_settings(_sim_cfg(enabled=True)) == (True, "cuda:1", False)


def test_post_spawn_forwards_per_articulation_selection(monkeypatch) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.schemas import schemas_actuators

    calls = []
    monkeypatch.setattr(
        schemas_actuators,
        "define_actuator_properties",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        sim_utils,
        "SimulationContext",
        type(
            "SimulationContextStub",
            (),
            {"instance": classmethod(lambda cls: SimpleNamespace(cfg=_sim_cfg(enabled=True)))},
        ),
    )

    stage = object()
    cfg = ArticulationCfg(
        prim_path="/World/Drawer",
        actuators={},
        native_actuators=False,
    )
    cfg._post_spawn(stage=stage)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("/World/Drawer", {})
    assert kwargs["stage"] is stage
    assert kwargs["use_newton_actuators"] is False
