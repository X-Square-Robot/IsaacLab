# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for X2Robot CX002 series robots.

The following configuration parameters are available:

* :obj:`CX002_CFG`: The CX002 arm (v00.02.04), loaded from
  :obj:`~isaaclab.utils.assets.X2ROBOT_NUCLEUS_DIR`.
* :obj:`CX002_FIXED_BASE_CFG`: Fixed-root variant for position-control smoke
  tests that should not spend solver energy on floating-base drift.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import X2ROBOT_NUCLEUS_DIR

##
# Configuration
##

CX002_GRIPPER_HOME_POSITION = 0.1
"""Conservative CX002 gripper motor home position [rad] for MJWarp smoke tests."""

CX002_GRIPPER_MIMIC_OFFSET = 0.07300188936622275
"""CX002 gripper finger mimic offset [rad].

The USD authors the PhysX angular mimic offset as 4.182700157165527 degrees.
Newton and Isaac Lab articulation buffers store revolute joint coordinates in
radians, so the default joint pose below uses the converted value.
"""

CX002_GRIPPER_RIGHT_FINGER_HOME_POSITION = CX002_GRIPPER_MIMIC_OFFSET - 0.5 * CX002_GRIPPER_HOME_POSITION
"""CX002 right finger home position [rad] satisfying the PhysX mimic equation."""

CX002_GRIPPER_LEFT_FINGER_HOME_POSITION = -CX002_GRIPPER_RIGHT_FINGER_HOME_POSITION
"""CX002 left finger home position [rad] satisfying the PhysX mimic equation."""

CX002_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{X2ROBOT_NUCLEUS_DIR}/robots/cx002/v00.02.04/cx002.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Keep CX002 self-contact semantics enabled by default. Use
            # scripts/demos/test_cx002.py --disable_robot_collision only when
            # isolating MJWarp dense-contact instability.
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Lift the base off the ground plane [m] — the CX002 USD origin sits at
        # the robot base so without a Z offset the first links clip into the floor.
        pos=(0.0, 0.0, 0.1),
        # Stable MJWarp smoke-test home pose [rad] for revolute joints and [m]
        # for any prismatic joints. The USD-authored defaults are all zero,
        # which folds fingers and neighboring visual-mesh colliders into each
        # other. These small offsets are a conservative non-self-colliding demo
        # pose seed; production control should replace them with robot-ID data.
        joint_pos={
            # Bow joints
            "bow_pitch_joint_01": -0.000078,
            "bow_pitch_joint_02": -0.000145,
            "bow_pitch_joint_03": 0.000028,
            "bow_yaw_joint": 0.000051,
            # Head joints
            "head_yaw_joint": -0.000001,
            "head_pitch_joint": -0.000104,
            # Left arm joints
            "left_shoulder_pitch_joint": -0.000531,
            "left_shoulder_roll_joint": 0.000141,
            "left_shoulder_yaw_joint": -0.000106,
            "left_elbow_roll_joint": 0.000082,
            "left_elbow_yaw_joint": -0.000013,
            "left_wrist_roll_joint": 0.000034,
            "left_wrist_pitch_joint": -0.000559,
            # Left gripper joints
            "left_gripper": CX002_GRIPPER_HOME_POSITION,
            "left_gripper_left_joint": CX002_GRIPPER_LEFT_FINGER_HOME_POSITION,
            "left_gripper_right_joint": CX002_GRIPPER_RIGHT_FINGER_HOME_POSITION,
            # Right arm joints
            "right_shoulder_pitch_joint": -0.000887,
            "right_shoulder_roll_joint": 0.000117,
            "right_shoulder_yaw_joint": 0.000033,
            "right_elbow_roll_joint": -0.000157,
            "right_elbow_yaw_joint": -0.000077,
            "right_wrist_roll_joint": -0.000043,
            "right_wrist_pitch_joint": 0.001548,
            # Right gripper joints
            "right_gripper": CX002_GRIPPER_HOME_POSITION,
            "right_gripper_left_joint": CX002_GRIPPER_LEFT_FINGER_HOME_POSITION,
            "right_gripper_right_joint": CX002_GRIPPER_RIGHT_FINGER_HOME_POSITION,
            # Wheel joints
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
    ),
    actuators={
        "bow": ImplicitActuatorCfg(
            joint_names_expr=["bow_pitch_joint_0[1-3]", "bow_yaw_joint"],
            # Full CX002 collision in MJWarp applies sustained contact/gravity
            # loads to the bow chain during startup. Use stronger derivative
            # damping so the bow drive can support the home pose quietly.
            effort_limit_sim=600.0,
            stiffness=600.0,
            damping=240.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim=40.0,
            stiffness=80.0,
            damping=8.0,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_(?:shoulder|elbow|wrist)_.*_joint"],
            effort_limit_sim=120.0,
            stiffness=160.0,
            damping=16.0,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_(?:shoulder|elbow|wrist)_.*_joint"],
            effort_limit_sim=120.0,
            stiffness=160.0,
            damping=16.0,
        ),
        "gripper_motors": ImplicitActuatorCfg(
            joint_names_expr=["(?:left|right)_gripper"],
            # The CX002 USD has zero effort on several finger joints. Give the
            # MJWarp demo a small non-zero solver limit so position targets are
            # actually enforced at the real gripper motor joints without
            # turning gripper contacts into hard high-gain constraints.
            effort_limit_sim=10.0,
            stiffness=10.0,
            damping=2.0,
        ),
        "gripper_passive_fingers": ImplicitActuatorCfg(
            joint_names_expr=["(?:left|right)_gripper_(?:left|right)_joint"],
            # These four joints are constrained by PhysxMimicJointAPI in the
            # USD.  Giving them independent PD drives fights the mimic
            # constraints in MJWarp, so keep them passive while still covering
            # every articulation joint with an actuator group.
            effort_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel_joint"],
            effort_limit_sim=20.0,
            stiffness=20.0,
            damping=2.0,
        ),
    },
)
"""Configuration of the CX002 arm with conservative implicit actuators.

.. note::
    The joint home pose and actuator groups are intentionally explicit for
    MJWarp smoke tests. Production control should still tune gains from robot
    identification data instead of treating these conservative demo gains as
    hardware parameters.
"""

CX002_FIXED_BASE_CFG = CX002_CFG.replace(
    spawn=CX002_CFG.spawn.replace(
        articulation_props=CX002_CFG.spawn.articulation_props.replace(
            # MJWarp full-collision position-hold smoke tests should measure
            # joint tracking, not floating-base contact drift. Keep the base
            # CX002_CFG mobile while exposing a fixed-root variant for demos
            # and tests that require stable upper-body position control.
            fix_root_link=True,
        )
    )
)
"""Fixed-base CX002 configuration for MJWarp position-control smoke tests.

This variant preserves the same USD, colliders, self-collision setting, home
pose, and actuator gains as :obj:`CX002_CFG`; it only adds a fixed root link so
dense full-collision MJWarp tests can validate joint-space position holding
without floating-base energy drift.
"""
