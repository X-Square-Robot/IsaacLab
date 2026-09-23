# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.utils.configclass import configclass

from ..asset_base_cfg import AssetBaseCfg
from .ordering import ArticulationOrderingConvention

if TYPE_CHECKING:
    from .articulation import Articulation


@configclass
class ArticulationCfg(AssetBaseCfg):
    """Configuration parameters for an articulation."""

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial state of the articulation."""

        # root velocity
        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

        # joint state
        joint_pos: dict[str, float] = {".*": 0.0}
        """Joint positions of the joints. Defaults to 0.0 for all joints."""
        joint_vel: dict[str, float] = {".*": 0.0}
        """Joint velocities of the joints. Defaults to 0.0 for all joints."""

    ##
    # Initialize configurations.
    ##

    class_type: type[Articulation] | str = "{DIR}.articulation:Articulation"

    articulation_root_prim_path: str | None = None
    """Path to the articulation root prim under the :attr:`prim_path`. Defaults to None, in which case the class
    will search for a prim with the USD ArticulationRootAPI on it.

    This path should be relative to the :attr:`prim_path` of the asset. If the asset is loaded from a USD file,
    this path should be relative to the root of the USD stage. For instance, if the loaded USD file at :attr:`prim_path`
    contains two articulations, one at `/robot1` and another at `/robot2`, and you want to use `robot2`,
    then you should set this to `/robot2`.

    The path must start with a slash (`/`).
    """

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the articulated object. Defaults to identity pose with zero velocity and zero joint state."""

    soft_joint_pos_limit_factor: float = 1.0
    """Fraction specifying the range of joint position limits (parsed from the asset) to use. Defaults to 1.0.

    The soft joint position limits are scaled by this factor to specify a safety region within the simulated
    joint position limits. This isn't used by the simulation, but is useful for learning agents to prevent the joint
    positions from violating the limits, such as for termination conditions.

    The soft joint position limits are accessible through the :attr:`ArticulationData.soft_joint_pos_limits` attribute.
    """

    joint_ordering: list[str] | tuple[str, ...] | str | ArticulationOrderingConvention | None = None
    """Public joint-name ordering convention or complete explicit permutation.

    Accepts ``"physx"``, ``"mjwarp"``, and ``"robot_schema"`` aliases, the
    corresponding :class:`ArticulationOrderingConvention` members, or a list or
    tuple (normalized to a tuple at initialization) containing every backend joint name exactly once.

    ``None`` is the default: public joint order follows active backend solver-view
    order and no ordering map is installed. An order that resolves to backend
    order is normalized to ``None`` as well, so an installed map always denotes
    an actual permutation. Symbolic resolution and map construction occur during
    articulation initialization only, not each step.
    """

    body_ordering: list[str] | tuple[str, ...] | str | ArticulationOrderingConvention | None = None
    """Public body-name ordering convention or complete explicit permutation.

    Accepts ``"physx"``, ``"mjwarp"``, and ``"robot_schema"`` aliases, the
    corresponding :class:`ArticulationOrderingConvention` members, or a list or
    tuple (normalized to a tuple at initialization) containing every backend body name exactly once.

    ``None`` is the default: public body order follows active backend solver-view
    order and no ordering map is installed. An order that resolves to backend
    order is normalized to ``None`` as well, so an installed map always denotes
    an actual permutation. Symbolic resolution and map construction occur during
    articulation initialization only, not each step.

    For fixed-base articulations, the backend root body must remain at public index
    zero; all remaining bodies may be permuted. Floating-base orders may relocate
    the root body.
    """

    actuators: dict[str, ActuatorBaseCfg] = MISSING
    """Actuators for the robot with corresponding joint names."""

    native_actuators: bool | None = None
    """Whether this articulation opts into Newton-native actuator execution.

    ``None`` inherits :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators`,
    preserving the simulation-wide behavior used by existing configurations.
    ``True`` opts this articulation in when the simulation has enabled the
    Newton actuator capability; ``False`` keeps it on the standard Isaac Lab
    actuator path. The capability flag remains global because the physics
    backend and USD schema extension are initialized once per simulation.
    """

    newton_actuator_device: str | None = None
    """Device for this articulation's PhysX-hosted Newton actuator work.

    ``None`` inherits :attr:`~isaaclab.sim.SimulationCfg.newton_actuator_device`.
    The setting is ignored when :attr:`native_actuators` resolves to ``False``
    and by Newton physics backends whose actuator work runs in the solver.
    """

    newton_actuator_cuda_graph: bool | None = None
    """Whether to capture this articulation's PhysX Newton actuator work in CUDA Graphs.

    ``None`` inherits :attr:`~isaaclab.sim.SimulationCfg.newton_actuator_cuda_graph`.
    The setting is ignored when :attr:`native_actuators` resolves to ``False``.
    """

    actuator_value_resolution_debug_print = False
    """Print the resolution of actuator final value when input cfg is different from USD value, Defaults to False
    """

    def _post_spawn(self, stage: Any) -> None:
        """Author ``NewtonActuator`` USD prims from :attr:`actuators` after spawn.

        Invoked by :class:`~isaaclab.assets.AssetBase` once the articulation's prims
        exist on the stage. Delegates to
        :func:`~isaaclab.sim.schemas.define_actuator_properties` with this
        articulation's resolved native-actuator selection.
        """
        if self.actuators is MISSING:
            return
        from isaaclab.sim.schemas.schemas_actuators import define_actuator_properties  # noqa: PLC0415

        # In InteractiveScene, articulated assets are often spawned first under
        # a template path (for example ``/World/template/Robot``) and cloned
        # into ``{ENV_REGEX_NS}`` later. Author NewtonActuator prims on the
        # actual spawned source prim so clones inherit them.
        author_prim_path = (
            self.spawn.spawn_path if self.spawn is not None and self.spawn.spawn_path is not None else self.prim_path
        )
        use_newton_actuators, _, _ = self.resolve_newton_actuator_settings()
        define_actuator_properties(
            author_prim_path,
            self.actuators,
            stage=stage,
            use_newton_actuators=use_newton_actuators,
        )

    def resolve_newton_actuator_settings(self, sim_cfg: Any | None = None) -> tuple[bool, str | None, bool | None]:
        """Resolve simulation capability and per-articulation Newton settings.

        Args:
            sim_cfg: Optional resolved simulation configuration. When omitted,
                the active :class:`~isaaclab.sim.SimulationContext` is queried.

        Returns:
            A tuple ``(enabled, device, cuda_graph)``. ``enabled`` is false
            when the simulation-level Newton actuator capability is off; the
            device and graph values inherit from the simulation configuration
            unless overridden on this articulation.
        """
        if sim_cfg is None:
            from isaaclab.sim import SimulationContext  # noqa: PLC0415

            sim_context = SimulationContext.instance()
            sim_cfg = sim_context.cfg if sim_context is not None else None

        capability_enabled = bool(getattr(sim_cfg, "use_newton_actuators", False))
        enabled = capability_enabled and self.native_actuators is not False
        if not enabled:
            return False, None, None

        device = self.newton_actuator_device
        if device is None:
            device = getattr(sim_cfg, "newton_actuator_device", None)
        cuda_graph = self.newton_actuator_cuda_graph
        if cuda_graph is None:
            cuda_graph = getattr(sim_cfg, "newton_actuator_cuda_graph", None)
        return True, device, cuda_graph
