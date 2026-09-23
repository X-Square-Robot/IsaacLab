# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for X2Robot EX001 series robots.

The following configuration parameters are available:

* :obj:`EX001_CFG`: The base EX001 arm (v00.04.04), loaded from
  :obj:`~isaaclab.utils.assets.X2ROBOT_NUCLEUS_DIR`.
* :obj:`EX001_6R_CFG`: The EX001 6-axis variant (v00.04.04), loaded from
  :obj:`~isaaclab.utils.assets.X2ROBOT_NUCLEUS_DIR`.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, StablePDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import X2ROBOT_NUCLEUS_DIR

##
# Configuration
##

EX001_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{X2ROBOT_NUCLEUS_DIR}/robots/ex001/v00.04.04/ex001.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Lift the base off the ground plane [m] — the EX001 USD origin sits at
        # the robot base so without a Z offset the first links clip into the floor.
        pos=(0.0, 0.0, 0.01),
        # Empty dict falls back to joint defaults defined in the USD.
        # TODO: populate with real joint names and home pose after first demo run.
        joint_pos={},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=400.0,
            damping=40.0,
        ),
    },
)
"""Configuration of the base EX001 arm with placeholder implicit actuators.

.. note::
    Joint names, limits, and per-joint actuator tuning are pending inspection.
    Run a demo script once, then refine
    :attr:`~isaaclab.assets.ArticulationCfg.InitialStateCfg.joint_pos` and
    :attr:`~isaaclab.assets.ArticulationCfg.actuators` using the printed
    articulation metadata.
"""


EX001_6R_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{X2ROBOT_NUCLEUS_DIR}/robots/ex001/v00.04.04/ex001_6r_meshcollision.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        # Manipulator: anchor the base link to the world so the arm doesn't
        # topple/drift under gravity (StablePD feeds a gravity bias).
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.01),
        joint_pos={},
    ),
    actuators={
        "arm": StablePDActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=400.0,
            damping=40.0,
        ),
    },
)
"""Configuration of the EX001 6-axis variant with placeholder Stable PD actuators.

Shares the same init state and PD gains as :obj:`EX001_CFG`, but drives all
joints with a Stable PD actuator (Newton in-graph) instead of the implicit
actuator; the USD asset also differs.

.. note::
    Joint names, limits, and per-joint actuator tuning are pending inspection.
    Run ``scripts/demos/test_ex001_6r.py`` once, then refine
    :attr:`~isaaclab.assets.ArticulationCfg.InitialStateCfg.joint_pos` and
    :attr:`~isaaclab.assets.ArticulationCfg.actuators` using the printed
    articulation metadata.
"""
