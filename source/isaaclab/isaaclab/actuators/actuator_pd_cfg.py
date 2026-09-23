# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.utils.configclass import configclass

from .actuator_base_cfg import ActuatorBaseCfg

if TYPE_CHECKING:
    from .actuator_pd import (
        DCMotor,
        DelayedPDActuator,
        IdealPDActuator,
        ImplicitActuator,
        RemotizedPDActuator,
        StablePDActuator,
    )

"""
Implicit Actuator Models.
"""


@configclass
class ImplicitActuatorCfg(ActuatorBaseCfg):
    """Configuration for an implicit actuator.

    Note:
        The PD control is handled implicitly by the simulation.
    """

    class_type: type["ImplicitActuator"] | str = "{DIR}.actuator_pd:ImplicitActuator"


"""
Explicit Actuator Models.
"""


@configclass
class IdealPDActuatorCfg(ActuatorBaseCfg):
    """Configuration for an ideal PD actuator."""

    class_type: type["IdealPDActuator"] | str = "{DIR}.actuator_pd:IdealPDActuator"


@configclass
class StablePDActuatorCfg(IdealPDActuatorCfg):
    r"""Configuration for a Stable PD actuator (Tan et al. 2011).

    The actuator uses the semi-implicit Stable PD law:

    .. math::

        \tau = - k_p \, (q + \Delta t \, \dot q - q_{des})
               - k_d \, (\dot q - \dot q_{des})
               + \tau_{ff}

    The one-step look-ahead term :math:`k_p \, \Delta t \, \dot q` predicts the joint position
    one physics step into the future, which suppresses the high-frequency oscillations that
    appear in plain explicit PD when :attr:`stiffness` is large relative to the physics time
    step.

    Reference:
        J. Tan, K. Liu, G. Turk. *Stable Proportional-Derivative Controllers*.
        IEEE Computer Graphics and Applications, 2011.
    """

    class_type: type["StablePDActuator"] | str = "{DIR}.actuator_pd:StablePDActuator"

    gravity_compensation: Literal["bias", "feedforward", "none"] = "feedforward"
    """Gravity-compensation scheme for this actuator's joints. Defaults to ``"feedforward"``.

    - ``"feedforward"`` (default): add the full-weight gravity torque to the actuator effort
      through the controller's constant-effort channel, outside the implicit solve — zero
      steady-state droop, so :attr:`stiffness` can be lowered for compliant/force-control tasks
      without the joints sagging. The compensation torque counts toward :attr:`effort_limit` like
      any other effort. The predictor's bias then carries the Coriolis term only.
    - ``"bias"``: feed :math:`g(q)` into the implicit predictor's bias (RHS) together with the
      Coriolis term. The implicit solve scales it by :math:`k_d \\Delta t A^{-1} < 1`, so the
      compensation is partial and a steady-state droop of roughly the residual gravity over
      :attr:`stiffness` remains. This is the native :class:`~newton.actuators.ControllerStablePD`
      convention; suits high-:attr:`stiffness` tracking / RL policies where the residual droop is
      negligible and a gravity feedforward would only add model error.
    - ``"none"``: no gravity term anywhere; the bias carries the Coriolis term only.

    On the PhysX backend a robot spawned with ``rigid_props.disable_gravity=True`` forces
    ``"none"``: PhysX computes :math:`g(q)` from scene gravity regardless of the per-body flag,
    so feeding it would compensate a load the plant never feels. The Newton backend needs no such
    guard — its RNEA respects the model's gravity.

    Joints merged into one in-graph controller (actuator groups with identical gains) must agree
    on this setting; conflicting values raise an error at articulation initialization.
    """

    sim_dt: float | None = None
    """Physics sub-step time used in the Stable PD predictor [s].

    .. deprecated::
        This field is no longer used and has no effect. The Tan 2011 predictor is
        evaluated by Newton's in-graph :class:`~newton.actuators.ControllerStablePD`,
        which uses the physics sub-step dt directly; there is nothing to override
        here. The field is retained for backwards compatibility and will be removed
        in a future release.
    """


@configclass
class DCMotorCfg(IdealPDActuatorCfg):
    """Configuration for direct control (DC) motor actuator model."""

    class_type: type["DCMotor"] | str = "{DIR}.actuator_pd:DCMotor"

    saturation_effort: float = MISSING
    """Peak motor force/torque of the electric DC motor (in N-m)."""


@configclass
class DelayedPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type["DelayedPDActuator"] | str = "{DIR}.actuator_pd:DelayedPDActuator"

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""


@configclass
class RemotizedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for a remotized PD actuator.

    Note:
        The torque output limits for this actuator is derived from a linear interpolation of a lookup table
        in :attr:`joint_parameter_lookup`. This table describes the relationship between joint angles and
        the output torques.
    """

    class_type: type["RemotizedPDActuator"] | str = "{DIR}.actuator_pd:RemotizedPDActuator"

    joint_parameter_lookup: list[list[float]] = MISSING
    """Joint parameter lookup table. Shape is (num_lookup_points, 3).

    This tensor describes the relationship between the joint angle (rad), the transmission ratio (in/out),
    and the output torque (N*m). The table is used to interpolate the output torque based on the joint angle.
    """
