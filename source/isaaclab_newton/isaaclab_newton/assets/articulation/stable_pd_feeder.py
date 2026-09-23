# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env StablePD mass-matrix/bias feeding from ``newton.eval_inverse_dynamics_passive``."""

from __future__ import annotations

import newton
import numpy as np
import warp as wp

from . import kernels


class StablePDFeeder:
    """Feeds per-env ``ControllerStablePD`` inputs from a single inverse-dynamics eval.

    Precomputes device-resident index buffers at construction (topology is static):
    per-env articulation indices, per-actuator global DOF indices, and
    articulation-local rows (``global_dof - joint_qd_start[articulation_start[art]]``,
    which lands actuated rows at >= 6 on a floating base without special-casing).
    :meth:`feed` then issues only ``newton.eval_*`` / ``wp.launch`` into
    pre-allocated buffers, so it is CUDA-graph capturable.

    On a floating base the actuated block is Schur-reduced against the 6 base DOFs
    (``M_eff = M_aa - M_ab M_bb^-1 M_ba``, ``bias_eff = C_a - M_ab M_bb^-1 C_b``),
    which is the exact elimination of the unactuated base accelerations from the
    Tan 2011 implicit solve.
    """

    def __init__(
        self,
        model: newton.Model,
        articulation_indices: np.ndarray,
        actuator_dof_indices: list[np.ndarray],
        num_base_dofs: int,
        device: str,
        gravity_modes: list[str] | None = None,
        const_effort_arrays: list[wp.array | None] | None = None,
    ):
        """Precompute index buffers and scratch for one articulation asset.

        Args:
            model: Finalized Newton model.
            articulation_indices: Global articulation index per env, shape (num_envs,), int.
            actuator_dof_indices: One array per StablePD actuator, each of shape
                (num_envs, n_actuator) with global (env-major) DOF indices.
            num_base_dofs: ``0`` for a fixed base, ``6`` for a floating base.
            device: Warp device string.
            gravity_modes: Per-actuator gravity-compensation mode (``"bias"``, ``"feedforward"``
                or ``"none"``), aligned with ``actuator_dof_indices``. Defaults to all ``"bias"``.
                ``"bias"`` folds g(q) into the implicit bias; ``"feedforward"`` gathers it into
                the controller's ``const_effort`` instead (full weight, effort-clamped, outside
                the implicit solve); ``"none"`` drops it. On a floating base the 6 base rows keep
                their gravity term in every mode — the base is unactuated and uncompensated, so
                the Schur elimination must see its true load.
            const_effort_arrays: Per-actuator ``ControllerStablePD.const_effort`` buffer, env-major
                flat shape (num_envs · n_actuator,) [N or N·m]. Required (non-``None``) for every
                ``"feedforward"`` entry; ignored otherwise.
        """
        if num_base_dofs not in (0, 6):
            raise ValueError(f"num_base_dofs must be 0 or 6, got {num_base_dofs}")
        self._model = model
        self._device = device
        self._num_base_dofs = num_base_dofs
        num_envs = int(articulation_indices.shape[0])

        art_start = model.articulation_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        art = np.asarray(articulation_indices, dtype=np.int64)
        dof_block_start = qd_start[art_start[art]].astype(np.int64)

        self._art_indices = wp.array(art.astype(np.int32), dtype=wp.int32, device=device)
        self._base_dof_start = wp.array(dof_block_start.astype(np.int32), dtype=wp.int32, device=device)

        self._gdofs: list[wp.array] = []
        self._rows: list[wp.array] = []
        self._schur_L: list[wp.array | None] = []
        self._schur_Y: list[wp.array | None] = []
        for gdofs_np in actuator_dof_indices:
            gdofs64 = np.asarray(gdofs_np, dtype=np.int64)
            if gdofs64.shape[0] != num_envs:
                raise ValueError(f"actuator dof indices have {gdofs64.shape[0]} envs, expected {num_envs}")
            rows = gdofs64 - dof_block_start[:, None]
            if rows.min() < num_base_dofs:
                raise ValueError(
                    "actuator DOF overlaps the floating-base block: local row "
                    f"{int(rows.min())} < num_base_dofs {num_base_dofs}"
                )
            if rows.max() >= model.max_dofs_per_articulation:
                raise ValueError(
                    f"actuator DOF row {int(rows.max())} exceeds articulation width {model.max_dofs_per_articulation}"
                )
            self._gdofs.append(wp.array(gdofs64.astype(np.int32), dtype=wp.int32, device=device))
            self._rows.append(wp.array(rows.astype(np.int32), dtype=wp.int32, device=device))
            if num_base_dofs:
                n = int(gdofs64.shape[1])
                self._schur_L.append(wp.zeros((num_envs, 6, 6), dtype=wp.float32, device=device))
                self._schur_Y.append(wp.zeros((num_envs, 6, n + 1), dtype=wp.float32, device=device))
            else:
                self._schur_L.append(None)
                self._schur_Y.append(None)

        num_actuators = len(actuator_dof_indices)
        self._gravity_modes = list(gravity_modes) if gravity_modes is not None else ["bias"] * num_actuators
        if len(self._gravity_modes) != num_actuators:
            raise ValueError(f"gravity_modes has {len(self._gravity_modes)} entries, expected {num_actuators}")
        self._const_effort = list(const_effort_arrays) if const_effort_arrays is not None else [None] * num_actuators
        if len(self._const_effort) != num_actuators:
            raise ValueError(f"const_effort_arrays has {len(self._const_effort)} entries, expected {num_actuators}")
        for i, mode in enumerate(self._gravity_modes):
            if mode not in ("bias", "feedforward", "none"):
                raise ValueError(f"gravity_modes[{i}] must be 'bias', 'feedforward' or 'none', got {mode!r}")
            if mode == "feedforward":
                arr = self._const_effort[i]
                expected = num_envs * int(self._rows[i].shape[1])
                if arr is None:
                    raise ValueError(f"gravity_modes[{i}] is 'feedforward' but const_effort_arrays[{i}] is None")
                if arr.shape != (expected,):
                    raise ValueError(f"const_effort_arrays[{i}] shape {arr.shape} != ({expected},)")

        # Pre-allocated outputs for eval_inverse_dynamics_passive (the caller
        # allocates since newton 95a1cb9b removed the Model.inverse_dynamics container).
        dofs = model.max_dofs_per_articulation
        self._mass_matrix = wp.zeros((model.articulation_count, dofs, dofs), dtype=wp.float32, device=device)
        self._gravity_force = wp.zeros((model.joint_dof_count,), dtype=wp.float32, device=device)
        self._coriolis_force = wp.zeros((model.joint_dof_count,), dtype=wp.float32, device=device)

    def feed(self, state: newton.State, controller_states: list) -> None:
        """Populate each controller state's ``mass_matrix``/``bias_forces`` for this substep.

        Args:
            state: Newton state; ``body_q`` must be FK-consistent with ``joint_q``.
            controller_states: One ``ControllerStablePD.State`` (or ``None``) per
                actuator, in the order of ``actuator_dof_indices`` at construction.
                Re-fetch each step — the adapter double-buffers.
        """
        newton.eval_inverse_dynamics_passive(
            self._model,
            state,
            mass_matrix=self._mass_matrix,
            gravity_force=self._gravity_force,
            coriolis_force=self._coriolis_force,
        )
        for i, ctrl_state in enumerate(controller_states):
            if ctrl_state is None or ctrl_state.mass_matrix is None:
                continue
            rows = self._rows[i]
            num_envs, n = rows.shape
            wp.launch(
                kernels.stable_pd_feed_mass_matrix,
                dim=(num_envs, n, n),
                inputs=[
                    self._mass_matrix,
                    self._model.joint_armature,
                    self._art_indices,
                    rows,
                    self._gdofs[i],
                ],
                outputs=[ctrl_state.mass_matrix],
                device=self._device,
            )
            mode = self._gravity_modes[i]
            has_bias = ctrl_state.bias_forces is not None
            if has_bias:
                wp.launch(
                    kernels.stable_pd_gather_bias,
                    dim=(num_envs, n),
                    inputs=[
                        self._gravity_force,
                        self._coriolis_force,
                        self._gdofs[i],
                        1.0 if mode == "bias" else 0.0,
                    ],
                    outputs=[ctrl_state.bias_forces],
                    device=self._device,
                )
            if mode == "feedforward":
                wp.launch(
                    kernels.stable_pd_gather_gravity,
                    dim=(num_envs, n),
                    inputs=[self._gravity_force, self._gdofs[i], n],
                    outputs=[self._const_effort[i]],
                    device=self._device,
                )
            if self._num_base_dofs:
                wp.launch(
                    kernels.stable_pd_schur_factor_base,
                    dim=num_envs,
                    inputs=[self._mass_matrix, self._art_indices],
                    outputs=[self._schur_L[i]],
                    device=self._device,
                )
                wp.launch(
                    kernels.stable_pd_schur_solve_base,
                    dim=(num_envs, n + 1),
                    inputs=[
                        self._mass_matrix,
                        self._gravity_force,
                        self._coriolis_force,
                        self._art_indices,
                        rows,
                        self._base_dof_start,
                        self._schur_L[i],
                    ],
                    outputs=[self._schur_Y[i]],
                    device=self._device,
                )
                wp.launch(
                    kernels.stable_pd_schur_reduce_mass,
                    dim=(num_envs, n, n),
                    inputs=[self._mass_matrix, self._art_indices, rows, self._schur_Y[i]],
                    outputs=[ctrl_state.mass_matrix],
                    device=self._device,
                )
                if has_bias:
                    wp.launch(
                        kernels.stable_pd_schur_reduce_bias,
                        dim=(num_envs, n),
                        inputs=[self._mass_matrix, self._art_indices, rows, self._schur_Y[i]],
                        outputs=[ctrl_state.bias_forces],
                        device=self._device,
                    )
