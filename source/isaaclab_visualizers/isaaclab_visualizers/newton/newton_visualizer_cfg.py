# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton OpenGL Visualizer."""

from isaaclab.utils.configclass import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg


@configclass
class NewtonVisualizerCfg(VisualizerCfg):
    """Configuration for Newton OpenGL visualizer."""

    visualizer_type: str = "newton"
    """Type identifier for Newton visualizer."""

    renderer: str = "gl"
    """Newton viewer rendering backend: ``"gl"`` (OpenGL rasterizer, default) or ``"rtx"``.

    ``"gl"`` uses :class:`newton.viewer.ViewerGL` and exposes the full Isaac Lab control
    panel (joint control, joint limits, inertia boxes, sky/light editing). ``"rtx"`` uses
    :class:`newton.viewer.ViewerRTX` (Omniverse OVRTX path-traced renderer, kitless) for
    higher visual fidelity; it keeps the Isaac Lab simulation/rendering pause controls but
    relies on the RTX viewer's own rendering UI for the rest. The RTX path requires the
    ``newton[rtx]`` extra (``ovrtx``); on environments where the OVRTX native libraries fail
    to load it falls back to the OpenGL renderer.
    """

    rtx_environment: str = "default"
    """RTX viewer scene environment preset: ``"default"``, ``"studio"`` or ``"none"``.

    Only consulted when :attr:`renderer` is ``"rtx"``. Mirrors
    :paramref:`newton.viewer.ViewerRTX.environment`.
    """

    window_width: int = 1920
    """Window width in pixels."""

    window_height: int = 1080
    """Window height in pixels."""

    headless: bool = False
    """Run the Newton viewer without requiring a display server."""

    update_frequency: int = 1
    """Visualizer update frequency (updates every N frames)."""

    world_spacing: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Visual spacing between simulation worlds along each axis [m].

    Non-zero axes arrange visible worlds in a compact grid without changing their simulated poses.
    """

    show_joints: bool = False
    """Show joint visualization."""

    show_joint_limits: bool = False
    """Show joint position-limit arcs (revolute) and segments (prismatic)."""

    joint_limit_radius: float = 0.12
    """Radius [m] of the revolute limit arc (also reused as the limit-arc display scale)."""

    show_joint_zero_ref_lines: bool = False
    """Show Kit-style zero-angle reference dashed line for each revolute joint.

    Emulates the Kit/PhysX UI ``_target0_rotation_baseline_transform`` dashed marker: the
    zero-angle direction is aligned to the child body AABB side-center farthest from the
    joint anchor, projected onto the plane perpendicular to the joint axis.
    """

    show_contacts: bool = False
    """Show contact visualization."""

    show_collision: bool = False
    """Show collision visualization."""

    show_springs: bool = False
    """Show spring visualization."""

    show_inertia_boxes: bool = False
    """Show inertia box visualization."""

    show_com: bool = False
    """Show center of mass visualization."""

    show_particles: bool = False
    """Show particle visualization."""

    particle_color: tuple[float, float, float] | None = None
    """Optional particle color RGB [0, 1]. If None, use Newton viewer defaults.

    Values are passed through to the Newton viewer unchanged.
    """

    enable_shadows: bool = True
    """Enable shadow rendering."""

    enable_sky: bool = True
    """Enable sky rendering."""

    enable_wireframe: bool = False
    """Enable wireframe rendering."""

    sky_upper_color: tuple[float, float, float] = (0.2, 0.4, 0.6)
    """Sky upper color RGB [0,1]."""

    sky_lower_color: tuple[float, float, float] = (0.5, 0.6, 0.7)
    """Sky lower color RGB [0,1]."""

    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Light color RGB [0,1]."""
