# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX-only stepping helper for :class:`~newton.actuators.Actuator`.

Newton's :meth:`Actuator.step` requires a ``sim_state`` / ``sim_control``
pair that exposes flat 1-D Warp arrays (``joint_q``, ``joint_qd``,
``joint_target_q``, ``joint_f``, …).  On the **Newton backend** these
are the ``State`` and ``Control`` objects that the solver already owns —
no wrapper is needed because:

* The solver manages double-buffered ``State`` objects for CUDA-graph
  capture, and actuators are stepped inside the solver's own simulation
  loop where states are already available.
* Wrapping them would add indirection with no benefit; the Newton
  articulation code that calls :meth:`Actuator.step` lives in
  ``newton_manager.py`` and has direct access to the model's state.

On the **PhysX backend**, no Newton solver exists — the actuators are
stepped manually from the Lab articulation's ``write_data_to_sim``
path.  Isaac Lab stores joint data as 2-D tensors (``num_envs ×
num_joints``), so :class:`PhysxActuatorWrapper` provides zero-copy flat
views that satisfy the protocol without allocating new memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp


@dataclass
class PhysxActuatorWrapper:
    """Flat-array wrapper serving as ``sim_state`` / ``sim_control`` for
    :meth:`Actuator.step` on the PhysX backend.

    Most attributes are bound once at articulation init to zero-copy flat
    views of Isaac Lab's 2-D buffers. ``joint_f_2d`` is the only persistent
    allocation, sized via :meth:`create`; ``joint_f`` is its flat alias
    consumed by the Newton actuator step. The legacy target field names remain
    dataclass fields for constructor compatibility, while the canonical Newton
    1.5 names are properties backed by the same arrays.
    """

    joint_q: wp.array | None = None
    joint_qd: wp.array | None = None
    joint_target_pos: wp.array | None = None
    joint_target_vel: wp.array | None = None
    joint_act: wp.array | None = None
    joint_f: wp.array | None = None
    joint_f_2d: wp.array | None = None
    joint_q_2d: wp.array | None = None
    joint_qd_2d: wp.array | None = None
    joint_target_pos_2d: wp.array | None = None
    joint_target_vel_2d: wp.array | None = None
    joint_act_2d: wp.array | None = None

    @property
    def joint_target_q(self) -> wp.array | None:
        """Canonical position targets [m or rad, depending on joint type]."""
        return self.joint_target_pos

    @joint_target_q.setter
    def joint_target_q(self, value: wp.array | None) -> None:
        self.joint_target_pos = value

    @property
    def joint_target_qd(self) -> wp.array | None:
        """Canonical velocity targets [m/s or rad/s, depending on joint type]."""
        return self.joint_target_vel

    @joint_target_qd.setter
    def joint_target_qd(self, value: wp.array | None) -> None:
        self.joint_target_vel = value

    @property
    def joint_target_q_2d(self) -> wp.array | None:
        """Canonical 2-D position targets [m or rad, depending on joint type]."""
        return self.joint_target_pos_2d

    @joint_target_q_2d.setter
    def joint_target_q_2d(self, value: wp.array | None) -> None:
        self.joint_target_pos_2d = value

    @property
    def joint_target_qd_2d(self) -> wp.array | None:
        """Canonical 2-D velocity targets [m/s or rad/s, depending on joint type]."""
        return self.joint_target_vel_2d

    @joint_target_qd_2d.setter
    def joint_target_qd_2d(self, value: wp.array | None) -> None:
        self.joint_target_vel_2d = value

    @classmethod
    def create(
        cls,
        num_envs: int,
        num_joints: int,
        device: str,
        *,
        allocate_inputs: bool = False,
    ) -> PhysxActuatorWrapper:
        """Allocate persistent buffers for the given articulation shape.

        ``allocate_inputs`` is used by the PhysX split-device bridge. The
        default retains the zero-copy aliases bound by the caller.
        """
        w = cls()
        w.joint_f_2d = wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device)
        w.joint_f = w.joint_f_2d.reshape(-1)
        if allocate_inputs:
            w.joint_q_2d = wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device)
            w.joint_qd_2d = wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device)
            w.joint_target_q_2d = wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device)
            w.joint_target_qd_2d = wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device)
            w.joint_act_2d = wp.zeros((num_envs, num_joints), dtype=wp.float32, device=device)
            w.joint_q = w.joint_q_2d.reshape(-1)
            w.joint_qd = w.joint_qd_2d.reshape(-1)
            w.joint_target_q = w.joint_target_q_2d.reshape(-1)
            w.joint_target_qd = w.joint_target_qd_2d.reshape(-1)
            w.joint_act = w.joint_act_2d.reshape(-1)
        return w
