# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility helpers for Newton articulations used by task-space controllers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import warp as wp


class OSCCompatArticulationView:
    """Proxy a Newton articulation view with PhysX-style dynamics methods.

    Args:
        view: Newton articulation view to wrap.
        state_accessor: Callable returning the current Newton state used for
            dynamics queries.
    """

    def __init__(self, view: Any, state_accessor: Callable[[], Any]):
        self._view = view
        self._state_accessor = state_accessor

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped Newton articulation view."""
        return getattr(self._view, name)

    def get_jacobians(self) -> wp.array | None:
        """Return spatial Jacobians in PhysX TensorAPI-compatible layout."""
        jacobians = self._view.eval_jacobian(self._state_accessor())
        if jacobians is None or len(jacobians.shape) == 4:
            return jacobians
        return jacobians.reshape((self._view.count, self._view.link_count, 6, self._view.joint_dof_count))

    def get_generalized_mass_matrices(self) -> wp.array | None:
        """Return generalized mass matrices for all articulations."""
        return self._view.eval_mass_matrix(self._state_accessor())

    def get_gravity_compensation_forces(self) -> wp.array:
        """Return generalized gravity compensation forces.

        Newton does not expose analytic gravity compensation through
        :class:`newton.selection.ArticulationView` yet, so this compatibility
        layer returns zeros with the PhysX TensorAPI-compatible shape.
        """
        return wp.zeros((self._view.count, self._view.joint_dof_count), dtype=wp.float32, device=self._view.device)
