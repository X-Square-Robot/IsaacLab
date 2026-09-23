# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp Newton manager."""

from __future__ import annotations

import logging

import numpy as np
import warp as wp
from newton import Contacts, Model
from newton.solvers import SolverMuJoCo

from isaaclab.physics import PhysicsManager

from .mjwarp_manager_cfg import MJWarpSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)


class NewtonMJWarpManager(NewtonManager):
    """:class:`NewtonManager` specialization for the MuJoCo Warp solver.

    Owns construction of :class:`SolverMuJoCo`, contact-buffer allocation in
    both internal-MuJoCo and Newton-pipeline contact modes, and the debug
    convergence logging emitted from :meth:`_log_solver_debug` when
    :attr:`NewtonCfg.debug_mode` is enabled.
    """

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> SolverMuJoCo:
        """Construct the configured MuJoCo Warp solver."""
        kwargs = cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg)
        # ls_parallel is deprecated in newton; forwarding it (even as False) emits a warning.
        kwargs.pop("ls_parallel", None)
        return SolverMuJoCo(model, **kwargs)

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> None:
        """Construct :class:`SolverMuJoCo` and populate the base-class slots.

        Filters cfg fields against the solver's ``__init__`` signature so
        non-constructor metadata (``solver_type``, ``class_type``) and the
        ignored deprecated ``ls_parallel`` field are not forwarded. Sets
        :attr:`NewtonManager._needs_collision_pipeline` to
        ``True`` only when ``use_mujoco_contacts=False``.
        """
        # Clamp sliding friction to avoid NaN in MuJoCo's constraint solver
        # when condim >= 3 (division by friction coefficient).
        min_mu = solver_cfg.min_friction
        if min_mu > 0.0 and model.shape_material_mu is not None:
            mu = model.shape_material_mu.numpy()
            below = mu < min_mu
            if below.any():
                count = int(below.sum())
                logger.warning(
                    "Clamping %d geom(s) with friction < %.1e to min_friction=%.1e to prevent MJWarp NaN.",
                    count,
                    min_mu,
                    min_mu,
                )
                mu[below] = min_mu
                model.shape_material_mu.assign(mu)

        if solver_cfg.contact_ke is not None:
            logger.warning(
                "Overriding MJWarp contact stiffness for %d geom(s): contact_ke=%g.",
                model.shape_count,
                solver_cfg.contact_ke,
            )
            cls._assign_numeric_array(model.shape_material_ke, float(solver_cfg.contact_ke))
        if solver_cfg.contact_kd is not None:
            logger.warning(
                "Overriding MJWarp contact damping for %d geom(s): contact_kd=%g.",
                model.shape_count,
                solver_cfg.contact_kd,
            )
            cls._assign_numeric_array(model.shape_material_kd, float(solver_cfg.contact_kd))

        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = not solver_cfg.use_mujoco_contacts

        if solver_cfg.contact_condim is not None:
            cls._override_contact_condim(solver_cfg.contact_condim)

        cfg = PhysicsManager._cfg
        # Cross-config validation that needs both halves.
        if solver_cfg.use_mujoco_contacts and cfg.collision_cfg is not None:
            raise ValueError(
                "NewtonCfg: collision_cfg cannot be set when "
                "solver_cfg.use_mujoco_contacts=True. Either set "
                "use_mujoco_contacts=False or remove collision_cfg."
            )

    @staticmethod
    def _assign_numeric_array(array, value: int | float) -> None:
        """Assign a scalar into a Newton/MuJoCo array regardless of backend."""
        if array is None:
            return
        if hasattr(array, "fill_"):
            array.fill_(value)
            return
        if hasattr(array, "numpy") and hasattr(array, "assign"):
            data = array.numpy()
            data[...] = value
            array.assign(data)
            return
        array[...] = value

    @classmethod
    def _override_contact_condim(cls, condim: int) -> None:
        """Force MuJoCo contact dimensionality after solver construction."""
        if condim < 1 or condim > 6:
            raise ValueError(f"MJWarp contact_condim must be in [1, 6], got {condim}.")

        solver = NewtonManager._solver
        geom_count = getattr(getattr(solver, "mj_model", None), "ngeom", 0)
        logger.warning("Overriding MJWarp contact dimensionality for %d geom(s): condim=%d.", geom_count, condim)

        # MuJoCo CPU and MuJoCo Warp keep separate model buffers. Update both
        # so initial construction and later property refreshes see the same
        # normal-only contact setting.
        cls._assign_numeric_array(getattr(getattr(solver, "mj_model", None), "geom_condim", None), condim)
        cls._assign_numeric_array(getattr(getattr(solver, "mjw_model", None), "geom_condim", None), condim)

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Allocate contact buffers.

        Delegates to the base implementation when Newton's
        :class:`CollisionPipeline` is active.  When ``use_mujoco_contacts=True``
        the solver runs MuJoCo's internal collision detection, so this method
        instead pre-allocates a :class:`Contacts` buffer sized to the solver's
        maximum contact count; ``solver.update_contacts`` later populates it
        from MuJoCo data for contact-sensor reporting.
        """
        if cls._needs_collision_pipeline:
            super()._initialize_contacts()
            return
        if cls._solver is not None:
            NewtonManager._contacts = Contacts(
                rigid_contact_max=cls._solver.get_max_contact_count(),
                soft_contact_max=0,
                device=PhysicsManager._device,
                requested_attributes=cls._model.get_requested_contact_attributes(),
            )

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
        """Clear MuJoCo Warp solver-internal state for flagged worlds.

        Specializes the base hook, whose :meth:`SolverBase.reset` call resolves
        to :meth:`SolverMuJoCo.reset` here: with ``flags=0`` it zeroes only the
        solver-owned buffers persisting across steps (``qacc_warmstart``,
        ``qfrc_applied``, ``xfrc_applied``, ``ctrl``, ``act``) for the flagged
        worlds, while the joint state IsaacLab authored during the env reset is
        left untouched.  Without this, a NaN produced in one solve persists
        across :meth:`isaaclab.envs.ManagerBasedEnv.reset` because the next
        solver substep warm-starts from the NaN — the world is then permanently
        dead.  See https://github.com/newton-physics/newton/issues/1266.

        With ``use_mujoco_cpu=True`` the solver owns a single global ``MjData``
        and its reset path is not mask-aware — it clears the buffers for every
        world.  Since this hook fires on every step/forward boundary (usually
        with an all-``False`` mask), the CPU path is gated on at least one
        world actually being flagged so warm-starting is not defeated on every
        step.

        Args:
            world_mask: Per-world bool mask of shape ``(world_count,)``;
                ``True`` for worlds that need their MJWarp internals cleared.
                ``None`` is treated as a no-op.
        """
        if world_mask is None:
            return
        if cls._solver.use_mujoco_cpu and not world_mask.numpy().any():
            return
        # flags=0 skips the joint-state reset to model defaults: IsaacLab owns
        # joint_q/joint_qd and has already written the authored reset pose.
        cls._solver.reset(cls._state_0, world_mask=world_mask, flags=0)

    @classmethod
    def _log_solver_debug(cls) -> None:
        """Optionally log MuJoCo solver convergence at the end of step."""
        cfg = PhysicsManager._cfg
        if cfg is not None and cfg.debug_mode:  # type: ignore[union-attr]
            data = cls._get_solver_convergence_steps()
            logger.info(f"Solver convergence data: {data}")
            if data["max"] == cls._solver.mjw_model.opt.iterations:
                logger.warning(f"Solver didn't converge! max_iter={data['max']}")

    @classmethod
    def _get_solver_convergence_steps(cls) -> dict[str, float | int]:
        """Return MuJoCo Warp solver convergence statistics.

        Reads ``mjw_data.solver_niter`` (only available on
        :class:`SolverMuJoCo`) and summarizes per-environment iteration counts.
        """
        niter = cls._solver.mjw_data.solver_niter.numpy()
        return {
            "max": np.max(niter),
            "mean": np.mean(niter),
            "min": np.min(niter),
            "std": np.std(niter),
        }
