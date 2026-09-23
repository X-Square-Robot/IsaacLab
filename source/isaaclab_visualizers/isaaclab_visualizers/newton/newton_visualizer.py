# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OpenGL Visualizer implementation."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from contextlib import suppress
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

# pyglet.options["headless"] must be set before the first import of pyglet.window.
# newton.viewer.ViewerGL triggers that import at class-definition time, so we must
# set the option here — before the ViewerGL import below — not in initialize().
# Without this, importing NewtonViewerGL in a headless environment (e.g. a pytest
# session on a CI runner with no DISPLAY) causes ViewerGL to open an X11 connection
# that blocks indefinitely.
if sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY"):
    import pyglet as _pyglet_headless_init

    _pyglet_headless_init.options["headless"] = True
    del _pyglet_headless_init

from newton.viewer import ViewerGL
from pyglet.math import Vec3 as PygletVec3

from isaaclab.envs.utils.camera_view import (
    VISUALIZER_TILED_CAMERA_MAX_TILES,
    apply_camera_target_positions,
    camera_rgb_batch,
    compute_tile_resolution,
    create_visualizer_camera,
    find_camera_by_prim_path,
    prim_world_positions,
    remove_generated_prims,
    resolve_tiled_env_indices,
)
from isaaclab.visualizers.base_visualizer import BaseVisualizer

from isaaclab_visualizers.newton.newton_visualization_markers import render_newton_visualization_markers
from isaaclab_visualizers.newton_adapter import apply_viewer_visible_worlds, resolve_visible_env_indices

from ._joint_limit_kernels import LINES_PER_JOINT, compute_joint_limit_lines
from .newton_visualizer_cfg import NewtonVisualizerCfg

logger = logging.getLogger(__name__)


def _newton_scalar_base_name(name: str) -> str:
    """Strip a trailing ``[N]`` component index from a scalar name to get the term base name."""
    if name.endswith("]") and "[" in name:
        bracket = name.rfind("[")
        if name[bracket + 1 : -1].isdigit():
            return name[:bracket]
    return name


_BACKEND_DISPLAY_NAMES = {
    "physx": "PhysX",
    "ovphysx": "OVPhysX",
    "newton": "Newton MJWarp",
}

CONTACT_ARROW_PATH = "/contacts"
"""Viewer path used for native and synthesized contact arrows."""

CONTACT_ARROW_COLOR = (0.0, 1.0, 0.0)
"""Color used by Newton's native contact visualization."""

CONTACT_ARROW_LENGTH = 0.1
"""Length of synthesized contact arrows in meters."""

if TYPE_CHECKING:
    from isaaclab.scene_data import SceneDataProvider


class NewtonViewerGL(ViewerGL):
    """Wrapper around Newton's ViewerGL with training/rendering pause controls."""

    def __init__(
        self,
        *args,
        metadata: dict | None = None,
        update_frequency: int = 1,
        **kwargs,
    ):
        """Initialize Newton viewer wrapper state.

        Args:
            *args: Positional arguments forwarded to ``ViewerGL``.
            metadata: Optional metadata shown in viewer panels.
            update_frequency: Viewer refresh cadence in simulation frames.
            **kwargs: Keyword arguments forwarded to ``ViewerGL``.
        """
        super().__init__(*args, **kwargs)
        self._paused_training = True
        self._paused_rendering = False
        self._pending_single_step = False
        """One-shot latch: unblock the Isaac Lab pause gate for a single tick (Newton ``Step``)."""
        # Seed Newton's native pause flag to match the authoritative state so the first panel
        # frame does not mistake the default ``_paused=False`` for a user "Resume" and drop the
        # intended initial pause. The panel wrapper keeps the two in sync from here on.
        self._paused = self._paused_training
        self._reset_requested = False
        self._metadata = metadata or {}
        self._update_frequency = update_frequency

        # Joint limit visualization state (disabled by default; toggled via cfg or ImGui checkbox).
        self.show_joint_limits: bool = False
        self.joint_limit_radius: float = 0.12
        self._joint_limit_points0: wp.array | None = None
        self._joint_limit_points1: wp.array | None = None
        self._joint_limit_colors: wp.array | None = None

        # Kit-style zero-angle reference dashed line — one per revolute joint, rigidly attached
        # to the child body pointing to the farthest AABB side center in the plane perpendicular
        # to the joint axis. Offsets are precomputed on the CPU and cached per model.
        self.show_joint_zero_ref_lines: bool = False
        self._joint_zero_angle_offset: wp.array | None = None
        self._joint_zero_angle_model_key: int | None = None

        # Joint control panel state — per-joint sliders that optionally override state.joint_q.
        # Lazy-populated the first time we see a model (via _collect_joint_metadata).
        self.joint_control_enabled: bool = False
        """If True, slider values are written into state.joint_q and FK is re-evaluated each frame."""
        self.joint_control_use_pd_target: bool = False
        """Joint Control mode selector.

        ``False`` (default): kinematic override — overwrite ``state.joint_q`` and re-evaluate FK,
        teleporting the robot every frame (works while paused, moves all 1-DoF joints, no physics).

        ``True``: PD-target mode — write slider values into ``control.joint_target_q``, the setpoint
        read by the solver's position drive and by composed actuators like
        :class:`~newton.actuators.ControllerStablePD`. Only moves while stepping; pure-effort/velocity
        DOFs (wheels) ignore it; an active policy overwrites the target each control step.

        Newton sim backend only: on PhysX the shadow Newton control is never read by the
        simulation, so the app loop must bridge :meth:`get_joint_override_targets` into its
        articulation instead (both panel modes then behave as PD targets).
        """
        self._joint_control_meta: list[dict] | None = None
        """Per 1-DoF joint metadata: {id, q_idx, name, is_revolute, lo, hi, slider_lo, slider_hi}."""
        self._joint_control_values: list[float] = []
        """Current slider values (in same order as _joint_control_meta)."""
        self._joint_control_values_external: bool = False
        """True once the app seeds slider values (:meth:`seed_joint_override_values`); disables
        the shadow-state slider sync, which is stale on backends that only refresh body_q."""
        self._joint_control_model_key: int | None = None
        """id() of the model metadata was built for; rebuild when model changes."""
        self.particle_color: tuple[float, float, float] | None = None
        self._particle_color_buffer: wp.array | None = None
        self._particle_color_buffer_count = 0
        self._particle_color_buffer_value: tuple[float, float, float] | None = None
        self._mpm_particle_flags_cache_key: tuple[int, int, int] | None = None
        self._mpm_particles_all_active = False
        self._live_plots_callback = None

        from isaaclab.utils.backend_utils import FactoryBase

        backend = FactoryBase._get_backend()
        self._backend_display = _BACKEND_DISPLAY_NAMES.get(backend, backend)

        with contextlib.suppress(AttributeError):
            self._patch_scalar_plot_width()
            self._patch_viewer_panel()

        try:
            self.register_ui_callback(self._render_training_controls, position="side")
            # Joint Control keeps its own collapsing header via the supported "panel" hook,
            # which the Isaac Lab panel draws before the main sections.
            self.register_ui_callback(self._render_joint_control_panel, position="panel")
        except AttributeError:
            pass

    def _patch_scalar_plot_width(self) -> None:
        """Set up ImPlot and suppress Newton's built-in floating Plots window.

        Plots are rendered inline in the left panel by
        :meth:`~NewtonVisualizer._live_plots_panel_imgui` instead.
        """
        gui = self.gui

        # Initialise ImPlot context once.  Newton does not use ImPlot itself, so we create
        # and own the context here.  set_imgui_context links it to the active imgui context.
        try:
            from imgui_bundle import implot as _implot

            self._implot_ctx = _implot.create_context()
            _implot.set_imgui_context(gui.ui.imgui.get_current_context())
            self._implot = _implot
        except Exception:
            self._implot = None
            self._implot_ctx = None

        # Replace Newton's floating plots window with a no-op; rendering is in the panel.
        gui._render_scalar_plots = lambda: None

    def is_training_paused(self) -> bool:
        """Return whether the Isaac Lab step loop should block this tick.

        :attr:`_paused_training` is the single source of truth for the simulation-pause state,
        driven by the Isaac Lab panel and keyboard shortcuts. A pending single-step request
        unblocks the gate for one full tick so the physics gate and render gate both fall through
        and the stepped state is logged and drawn. The latch is cleared by
        :meth:`NewtonVisualizer.step` once a real (``dt > 0``) frame is logged, so querying this
        several times within a tick stays consistent.
        """
        if self._pending_single_step:
            return False
        return self._paused_training

    def request_single_step(self) -> None:
        """Queue exactly one physics step while paused (Newton ``Step`` button semantics)."""
        self._pending_single_step = True

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused by viewer controls."""
        return self._paused_rendering

    def get_joint_override_targets(self) -> list[tuple[str, float]]:
        """Return the Joint Control sliders as ``(USD joint prim path, target value)`` pairs.

        Empty unless the panel override is enabled. Sim backends whose actuators do not read
        the shadow Newton control (e.g. PhysX) cannot be driven from :meth:`apply_joint_overrides`;
        the app loop bridges these pairs into its live articulation instead — see
        :meth:`NewtonVisualizer.get_joint_override_targets`.
        """
        meta = self._joint_control_meta
        if not self.joint_control_enabled or not meta:
            return []
        return [(m["full_path"], float(v)) for m, v in zip(meta, self._joint_control_values)]

    def seed_joint_override_values(self, values: dict[str, float]) -> None:
        """Seed the Joint Control sliders from live joint positions supplied by the app.

        Keys are USD joint prim paths or their trailing prim names. Intended for sim backends
        whose live joint state never reaches the shadow Newton state (e.g. PhysX, which only
        refreshes ``body_q``): the app seeds every step while the override is off, so enabling
        the override starts from the robot's true pose instead of stale shadow values. Seeding
        also disables the internal shadow-state slider sync; no-op while the override is active
        (the user owns the sliders then).
        """
        meta = self._joint_control_meta
        if not meta or not values:
            return
        self._joint_control_values_external = True
        if self.joint_control_enabled:
            return
        for i, m in enumerate(meta):
            value = values.get(m["full_path"])
            if value is None:
                value = values.get(m["name"])
            if value is not None:
                self._joint_control_values[i] = float(value)

    def is_reset_requested(self) -> bool:
        """Return whether an episode reset was requested without clearing the flag."""
        return self._reset_requested

    def consume_reset_request(self) -> bool:
        """Return whether an episode reset was requested and clear the flag."""
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def _patch_viewer_panel(self) -> None:
        """Replace Newton's left panel with an IsaacLab-oriented layout.

        New section order:

        1. **Isaac Lab** (open) — physics backend, model info, training controls.
        2. **Live Plots** (closed) — injected when :meth:`~NewtonVisualizer.add_live_plots`
           is called.
        3. **Visualization Markers** (open) — Newton's debug overlays, renamed.
        4. **Rendering Options** (open) — VSync and renderer-specific options.
        5. **Wind** (closed) — only shown when ``viewer.wind`` is set.
        6. **Controls** (closed) — camera keyboard reference.
        7. **Selection API** (closed) — Newton's selection panel.

        The top-level Newton ``Pause / Step / Reset`` row keeps Newton's native layout while
        bridging each action to Isaac Lab's simulation pause gate and episode lifecycle.
        """
        import newton as nt

        gui = self.gui

        def _render_left_panel(_g=gui):
            if not _g.is_available:
                return

            viewer = _g._viewer
            imgui = _g.ui.imgui
            io = _g.ui.io
            s = _g.ui.dpi_scale
            nav_highlight_color = _g.ui.get_theme_color(imgui.Col_.nav_cursor, (1.0, 1.0, 1.0, 1.0))

            imgui.set_next_window_pos(imgui.ImVec2(10 * s, 10 * s), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size(
                imgui.ImVec2(363 * s, io.display_size[1] - 20 * s),
                imgui.Cond_.first_use_ever,
            )
            panel_h = io.display_size[1] - 20 * s
            imgui.set_next_window_size_constraints(
                imgui.ImVec2(160 * s, panel_h),
                imgui.ImVec2(io.display_size[0], panel_h),
            )

            if not imgui.begin(f"Newton Viewer v{nt.__version__}"):
                imgui.end()
                return

            imgui.separator()

            if viewer.model is not None:
                viewer._render_simulation_controls(imgui)
                imgui.separator()

            # Layers panel callback (ViewerGL built-in, only shown with >1 layer).
            for callback in _g._ui_callbacks.get("panel", []):
                callback(imgui)

            # --- Isaac Lab --------------------------------------------------
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Isaac Lab"):
                imgui.separator()
                imgui.text(f"Physics: {getattr(viewer, '_backend_display', 'Unknown')}")
                if viewer.model is not None:
                    axis_names = ["X", "Y", "Z"]
                    imgui.text(f"Up Axis: {axis_names[viewer.model.up_axis]}")
                    gravity = viewer.model.gravity.numpy()[0]
                    imgui.text(f"Gravity: ({gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f})")
                imgui.separator()
                for callback in _g._ui_callbacks.get("side", []):
                    callback(imgui)

            # --- Live Plots -------------------------------------------------
            live_plots_cb = getattr(viewer, "_live_plots_callback", None)
            if live_plots_cb is not None:
                live_plots_cb(imgui)

            # --- Visualization Markers --------------------------------------
            if viewer.model is not None:
                imgui.set_next_item_open(False, imgui.Cond_.appearing)
                if imgui.collapsing_header("Visualization Markers"):
                    imgui.separator()
                    renderer = getattr(viewer, "renderer", None)
                    _c, viewer.show_joints = imgui.checkbox("Show Joints", viewer.show_joints)
                    if viewer.show_joints and renderer is not None and hasattr(renderer, "joint_scale"):
                        _, renderer.joint_scale = imgui.slider_float("Joint Scale", renderer.joint_scale, 0.25, 5.0)
                    _c, viewer.show_contacts = imgui.checkbox("Show Contacts", viewer.show_contacts)
                    if viewer.show_contacts and renderer is not None:
                        if hasattr(renderer, "arrow_length_scale"):
                            _, renderer.arrow_length_scale = imgui.slider_float(
                                "Contact Length", renderer.arrow_length_scale, 0.25, 5.0
                            )
                        if hasattr(renderer, "arrow_scale"):
                            _, renderer.arrow_scale = imgui.slider_float(
                                "Contact Width", renderer.arrow_scale, 0.25, 5.0
                            )
                    _c, viewer.show_particles = imgui.checkbox("Show Particles", viewer.show_particles)
                    _c, viewer.show_springs = imgui.checkbox("Show Springs", viewer.show_springs)
                    _c, viewer.show_com = imgui.checkbox("Show Center of Mass", viewer.show_com)
                    if viewer.show_com and renderer is not None and hasattr(renderer, "com_scale"):
                        _, renderer.com_scale = imgui.slider_float("COM Scale", renderer.com_scale, 0.25, 5.0)
                    _c, viewer.show_triangles = imgui.checkbox("Show Cloth", viewer.show_triangles)
                    _c, viewer.show_collision = imgui.checkbox("Show Collision", viewer.show_collision)
                    if renderer is not None and hasattr(renderer, "draw_edges"):
                        _c, renderer.draw_edges = imgui.checkbox("Show Edges", renderer.draw_edges)
                    sdf_margin_mode = getattr(viewer, "sdf_margin_mode", None)
                    SDFMarginMode = getattr(type(viewer), "SDFMarginMode", None)
                    if sdf_margin_mode is not None and SDFMarginMode is not None:
                        _sdf_labels = ["Off", "Margin", "Margin + Gap"]
                        _, new_sdf_idx = imgui.combo("Gap + Margin", int(sdf_margin_mode), _sdf_labels)
                        viewer.sdf_margin_mode = SDFMarginMode(new_sdf_idx)
                        if viewer.sdf_margin_mode != SDFMarginMode.OFF and renderer is not None:
                            _, renderer.wireframe_line_width = imgui.slider_float(
                                "Wireframe Width (px)", renderer.wireframe_line_width, 0.5, 5.0
                            )
                    _c, viewer.show_visual = imgui.checkbox("Show Visual", viewer.show_visual)
                    _c, viewer.show_inertia_boxes = imgui.checkbox("Show Inertia Boxes", viewer.show_inertia_boxes)
                    viewer._draw_isaaclab_joint_toggles(imgui)

            # --- Rendering Options ------------------------------------------
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Rendering Options"):
                imgui.separator()
                _c, viewer.vsync = imgui.checkbox("VSync", viewer.vsync)
                for callback in _g._ui_callbacks.get("rendering", []):
                    callback(imgui)

            # --- Wind -------------------------------------------------------
            wind = getattr(viewer, "wind", None)
            if wind is not None:
                imgui.set_next_item_open(False, imgui.Cond_.once)
                if imgui.collapsing_header("Wind"):
                    imgui.separator()
                    changed, wind.amplitude = imgui.slider_float("Wind Amplitude", wind.amplitude, -2.0, 2.0, "%.2f")
                    changed, wind.period = imgui.slider_float("Wind Period", wind.period, 1.0, 30.0, "%.2f")
                    changed, wind.frequency = imgui.slider_float("Wind Frequency", wind.frequency, 0.1, 5.0, "%.2f")
                    direction = [wind.direction[0], wind.direction[1], wind.direction[2]]
                    changed, direction = imgui.slider_float3("Wind Direction", direction, -1.0, 1.0, "%.2f")
                    if changed:
                        wind.direction = direction

            # --- Controls ---------------------------------------------------
            imgui.set_next_item_open(False, imgui.Cond_.appearing)
            if imgui.collapsing_header("Controls"):
                imgui.separator()
                _g._render_camera_info()
                imgui.separator()
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*nav_highlight_color))
                imgui.text("Controls:")
                imgui.pop_style_color()
                imgui.text("WASD - Move camera")
                imgui.text("QE - Pan up/down")
                imgui.text("Left Click - Look around")
                imgui.text("Right Click - Pick objects")
                imgui.text("Middle Click - Orbit")
                imgui.text("Shift + Middle Click - Pan")
                imgui.text("Ctrl + Middle Click - Dolly")
                imgui.text("Scroll - Dolly")
                imgui.text("Ctrl + Scroll - FOV zoom")
                imgui.text("Space - Pause/Resume")
                imgui.text(". - Step one frame (when paused)")
                imgui.text("H - Toggle UI")
                imgui.text("F - Frame camera around model")

            # --- Selection API ----------------------------------------------
            _g._render_selection_panel()

            imgui.end()

        gui._render_left_panel = _render_left_panel

    def _render_simulation_controls(self, imgui) -> None:
        """Render Newton's native control row using Isaac Lab lifecycle semantics."""
        changed, paused = imgui.checkbox("Pause", self._paused_training)
        if changed:
            self._paused_training = paused
            self._paused = paused
            self._pending_single_step = False

        imgui.same_line()
        imgui.begin_disabled(not self._paused_training)
        if imgui.button("Step"):
            self._paused_training = True
            self._paused = True
            self.request_single_step()
        imgui.end_disabled()

        imgui.same_line()
        if imgui.button("Reset"):
            self._reset_requested = True

    def _render_training_controls(self, imgui):
        """Render Isaac Lab-specific rendering controls inside the Isaac Lab panel."""
        rendering_label = "Resume Rendering" if self._paused_rendering else "Pause Rendering"
        if imgui.button(rendering_label):
            self._paused_rendering = not self._paused_rendering

        imgui.text("Visualizer Update Frequency")
        current_frequency = self._update_frequency
        changed, new_frequency = imgui.slider_int(
            "##VisualizerUpdateFreq", current_frequency, 1, 20, f"Every {current_frequency} frames"
        )
        if changed:
            self._update_frequency = new_frequency

        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Controls visualizer update frequency\nlower values -> more responsive visualizer but slower"
                " training\nhigher values -> less responsive visualizer but faster training"
            )

    def _draw_isaaclab_joint_toggles(self, imgui) -> None:
        """Draw the Isaac Lab joint-limit / zero-reference checkboxes.

        Emitted inside the "Visualization Markers" section immediately after
        "Show Inertia Boxes" by :meth:`_patch_viewer_panel`.
        """
        _, self.show_joint_limits = imgui.checkbox("Show Joint Limits", self.show_joint_limits)

        _, self.show_joint_zero_ref_lines = imgui.checkbox("Show Joint Zero-Ref Lines", self.show_joint_zero_ref_lines)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Kit-style dashed zero-angle reference line per revolute joint,\n"
                "pointing to the child body AABB side-center farthest from the\n"
                "joint anchor (projected onto the plane perpendicular to the axis)."
            )

    def _render_joint_control_panel(self, imgui):
        """Render joint control sliders in the left panel."""
        if self.model is None:
            return
        self._collect_joint_metadata()
        if not self._joint_control_meta:
            return
        if imgui.collapsing_header("Joint Control"):
            _, self.joint_control_enabled = imgui.checkbox("Enable Joint Override", self.joint_control_enabled)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "When enabled, the sliders drive the joints using the mode selected below.\n"
                    "Pause training/rendering for stable manual control; otherwise the simulator\n"
                    "keeps overwriting the joints each step."
                )

            imgui.same_line()
            if imgui.button("Reset to Current"):
                self._sync_slider_values_from_state(getattr(self, "_cached_state", None))

            # Mode selector: kinematic teleport (state.joint_q + FK) vs PD position target
            # (control.joint_target_q). See ``joint_control_use_pd_target`` for the trade-offs.
            _, self.joint_control_use_pd_target = imgui.checkbox(
                "Drive via PD target (else teleport joint_q)", self.joint_control_use_pd_target
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Off (default): kinematic override — overwrite state.joint_q and re-evaluate FK.\n"
                    "  Instant, works while paused, moves every 1-DoF joint, does not affect physics.\n"
                    "On: PD target — write slider values into control.joint_target_q.\n"
                    "  Only moves while the simulator is stepping. Followed by the solver's position\n"
                    "  drive and by StablePD/composed actuators; pure-effort joints (wheels) ignore\n"
                    "  it; an active policy overwrites the target each control step."
                )

            imgui.separator()
            rad_to_deg = 180.0 / float(np.pi)
            imgui.push_item_width(120)
            for i, m in enumerate(self._joint_control_meta):
                name = m["name"]
                display_name = (name[:16] + "…") if len(name) > 17 else name
                slider_id = f"##joint_ctrl_{m['id']}"

                is_unlimited = not np.isfinite(m["lo"]) or not np.isfinite(m["hi"]) or m["lo"] > m["hi"]

                if m["is_revolute"]:
                    ui_scale = rad_to_deg
                    fmt = "%.1f°"
                    if is_unlimited:
                        range_text = "[unlimited]"
                    else:
                        range_text = f"[{m['lo'] * ui_scale:+.1f}°, {m['hi'] * ui_scale:+.1f}°]"
                    unit_for_tooltip = "° (stored as rad)"
                else:
                    ui_scale = 1.0
                    fmt = "%.3f m"
                    if is_unlimited:
                        range_text = "[unlimited]"
                    else:
                        range_text = f"[{m['lo']:+.3f}, {m['hi']:+.3f}] m"
                    unit_for_tooltip = "m"

                imgui.text(f"{display_name:<18}")
                imgui.same_line()
                _, new_val_ui = imgui.slider_float(
                    slider_id,
                    float(self._joint_control_values[i]) * ui_scale,
                    float(m["slider_lo"]) * ui_scale,
                    float(m["slider_hi"]) * ui_scale,
                    fmt,
                )
                self._joint_control_values[i] = float(new_val_ui) / ui_scale
                if imgui.is_item_hovered():
                    slider_span = f"{m['slider_lo'] * ui_scale:+.1f} .. {m['slider_hi'] * ui_scale:+.1f}"
                    tooltip_lines = [
                        m["full_path"],
                        f"limit = {range_text}",
                        f"slider range = [{slider_span}]",
                        f"q_idx = {m['q_idx']}   unit = {unit_for_tooltip}",
                    ]
                    imgui.set_tooltip("\n".join(tooltip_lines))
                imgui.same_line()
                imgui.text(range_text)
            imgui.pop_item_width()

    def on_key_press(self, symbol, modifiers):
        """Forward key presses unless UI is currently capturing input.

        Intercepts the simulation run-control keys so they drive Isaac Lab's pause gate directly:
        ``Space`` toggles :attr:`_paused_training`, and ``.`` requests a single step while paused.
        All other keys fall through to Newton's default handling.
        """
        if self.ui.is_capturing():
            return

        try:
            import pyglet  # noqa: PLC0415 — viewer-time import, only needed for key symbols.

            if symbol == pyglet.window.key.SPACE:
                self._paused_training = not self._paused_training
                self._paused = self._paused_training
                return
            if symbol == pyglet.window.key.PERIOD and self._paused_training:
                self.request_single_step()
                return
        except Exception:
            logger.debug("[NewtonVisualizer] run-control key handling failed.", exc_info=True)

        super().on_key_press(symbol, modifiers)

    def _coerce_color3(self, color) -> tuple[float, float, float]:
        """Normalize color values from imgui/renderer into an RGB tuple."""
        if hasattr(color, "x") and hasattr(color, "y") and hasattr(color, "z"):
            return (float(color.x), float(color.y), float(color.z))
        return (float(color[0]), float(color[1]), float(color[2]))

    def _particle_color_array(self, count: int) -> wp.array:
        """Return a cached Warp color array for Newton's particle point batch."""
        color = self._coerce_color3(self.particle_color)
        if (
            self._particle_color_buffer is None
            or self._particle_color_buffer_count != count
            or self._particle_color_buffer_value != color
        ):
            self._particle_color_buffer = wp.full(
                shape=count,
                value=wp.vec3(*color),
                dtype=wp.vec3,
                device=self.device,
            )
            self._particle_color_buffer_count = count
            self._particle_color_buffer_value = color
        return self._particle_color_buffer

    def _particle_color_update_array(self, name: str, count: int) -> wp.array | None:
        """Return particle colors only when Newton needs the GL color buffer refreshed."""
        obj = self.objects.get(name)
        capacity = obj.num_instances if obj is not None else 0
        if (
            obj is None
            or count > capacity
            or self._particle_color_buffer_value != self._coerce_color3(self.particle_color)
        ):
            return self._particle_color_array(max(count, capacity))
        return None

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        """Apply configured model-particle appearance while preserving Newton's point logging.

        The configured particle color only applies to Newton's canonical
        ``/model/particles`` point batch. User-defined point clouds retain the
        colors provided by their own ``log_points`` calls.
        """
        if name != "/model/particles" or points is None or self.particle_color is None:
            return super().log_points(name, points, radii, colors, hidden)

        colors = self._particle_color_update_array(name, len(points))
        return super().log_points(name, points, radii, colors, hidden)

    def _all_mpm_particles_active(self) -> bool:
        """Return whether an MPM model's static particle flags are all active."""
        model = self.model
        if model is None or getattr(model, "mpm", None) is None or not model.particle_count:
            return False
        if model.particle_flags is None:
            return False

        cache_key = (id(model), id(model.particle_flags), int(model.particle_count))
        if self._mpm_particle_flags_cache_key != cache_key:
            import newton as nt

            flags = model.particle_flags.numpy()[: model.particle_count]
            self._mpm_particles_all_active = bool(((flags & int(nt.ParticleFlags.ACTIVE)) != 0).all())
            self._mpm_particle_flags_cache_key = cache_key
        return self._mpm_particles_all_active

    def _log_particles(self, state):
        """Log MPM particles without per-frame active-flag compaction when all particles are active.

        Newton's base implementation stream-compacts active particles every
        frame, which costs two device-to-host reads per render. MPM particle
        flags are static, so when they are all active the compaction is skipped
        and ``state.particle_q`` is logged directly.
        """
        if not self._all_mpm_particles_active():
            super()._log_particles(state)
            return

        colors = None
        if self.model_changed and self.particle_color is None:
            colors = wp.full(shape=len(state.particle_q), value=wp.vec3(0.7, 0.6, 0.4), device=self.device)

        self.log_points(
            name="/model/particles",
            points=state.particle_q,
            radii=self.model.particle_radius,
            colors=colors,
            hidden=not self.show_particles,
        )

    def _log_joints(self, state) -> None:
        """Log joint basis lines (parent class) and the joint position-limit overlay.

        Delegates frame/axis drawing to :meth:`newton.viewer.ViewerBase._log_joints`, then
        adds an arc/segment rendering of each 1-DoF joint's [lower, upper] range when
        :attr:`show_joint_limits` is True, plus a Kit-style zero-angle reference dashed line
        when :attr:`show_joint_zero_ref_lines` is True. Other joint types are NaN-hidden.
        """
        super()._log_joints(state)

        show_limits = bool(self.show_joint_limits)
        show_zero_ref = bool(self.show_joint_zero_ref_lines)
        if not show_limits and not show_zero_ref:
            self.log_lines("/model/joint_limits", None, None, None)
            return

        model = self.model
        if model is None:
            return
        num_joints = int(len(model.joint_type))
        if num_joints == 0:
            return

        # Skip quietly if any required attribute is missing (e.g. pure point-cloud models).
        required = (
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_qd_start,
            model.joint_axis,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_limit_lower,
            model.joint_limit_upper,
        )
        if any(attr is None for attr in required):
            self.log_lines("/model/joint_limits", None, None, None)
            return

        max_lines = num_joints * LINES_PER_JOINT
        if self._joint_limit_points0 is None or len(self._joint_limit_points0) < max_lines:
            self._joint_limit_points0 = wp.zeros(max_lines, dtype=wp.vec3, device=self.device)
            self._joint_limit_points1 = wp.zeros(max_lines, dtype=wp.vec3, device=self.device)
            self._joint_limit_colors = wp.zeros(max_lines, dtype=wp.vec3, device=self.device)

        # Lazily rebuild zero-ref offsets when the model changes (or initially). If we only
        # need limits, still pass a zero-filled buffer so the kernel signature is consistent.
        self._ensure_joint_zero_angle_offsets(num_joints)
        if self._joint_zero_angle_offset is None:
            # Allocation failure path — zero-fill so the kernel dereferences a valid array.
            self._joint_zero_angle_offset = wp.zeros(num_joints, dtype=float, device=self.device)

        wp.launch(
            kernel=compute_joint_limit_lines,
            dim=max_lines,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_qd_start,
                model.joint_axis,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_limit_lower,
                model.joint_limit_upper,
                self._joint_zero_angle_offset,
                state.body_q,
                model.body_world,
                self.world_offsets,
                self._visible_worlds_mask,
                float(self.joint_limit_radius),
                int(show_limits),
                int(show_zero_ref),
            ],
            outputs=[
                self._joint_limit_points0,
                self._joint_limit_points1,
                self._joint_limit_colors,
            ],
            device=self.device,
        )

        self.log_lines(
            "/model/joint_limits",
            self._joint_limit_points0,
            self._joint_limit_points1,
            self._joint_limit_colors,
        )

    # ------------------------------------------------------------------
    # Zero-angle reference line precomputation (Kit ``_target0_rotation_offset_transform``).
    # ------------------------------------------------------------------
    def _ensure_joint_zero_angle_offsets(self, num_joints: int) -> None:
        """Compute and cache per-joint zero-angle offsets (radians) for the current model.

        For each revolute joint, the offset is the angle (measured in the ``(e0, e1)`` basis
        perpendicular to ``axis_local``) pointing from the joint's child anchor origin toward
        the farthest of the child body's AABB side centers projected onto that plane.
        Non-revolute joints get ``0.0``. Rebuilds only when the underlying model changes.
        """
        model = self.model
        model_key = id(model) if model is not None else None
        if model_key == self._joint_zero_angle_model_key and self._joint_zero_angle_offset is not None:
            return

        self._joint_zero_angle_model_key = model_key
        self._joint_zero_angle_offset = None
        if model is None or num_joints <= 0:
            return

        required = (
            getattr(model, "joint_type", None),
            getattr(model, "joint_child", None),
            getattr(model, "joint_qd_start", None),
            getattr(model, "joint_axis", None),
            getattr(model, "joint_X_c", None),
        )
        if any(attr is None for attr in required):
            return

        try:
            joint_types = model.joint_type.numpy()
            joint_children = model.joint_child.numpy()
            joint_qd_starts = model.joint_qd_start.numpy()
            joint_axes = model.joint_axis.numpy()
            joint_X_c = np.asarray(model.joint_X_c.numpy()).reshape(-1, 7)
        except Exception as exc:
            logger.debug("[NewtonVisualizer] could not read joint arrays for zero-ref offsets: %s", exc)
            return

        body_aabb = self._compute_body_local_aabbs()
        offsets = np.zeros(num_joints, dtype=np.float32)

        if body_aabb is not None:
            for j in range(num_joints):
                if int(joint_types[j]) != 1:  # only REVOLUTE (newton.JointType.REVOLUTE == 1)
                    continue
                child = int(joint_children[j])
                if child < 0 or child not in body_aabb:
                    continue
                qd_idx = int(joint_qd_starts[j])
                if qd_idx < 0 or qd_idx >= len(joint_axes):
                    continue

                axis_local = np.asarray(joint_axes[qd_idx], dtype=np.float64)
                axis_norm = float(np.linalg.norm(axis_local))
                if axis_norm < 1e-9:
                    continue
                axis_local = axis_local / axis_norm

                # Gram–Schmidt basis: mirrors the cyclic X→Y→Z→X pick done in the kernel so
                # the offset angle is measured in the same (e0, e1) frame the kernel uses.
                ax_abs = abs(axis_local[0])
                ay_abs = abs(axis_local[1])
                az_abs = abs(axis_local[2])
                if ax_abs >= ay_abs and ax_abs >= az_abs:
                    ref_basis = np.array([0.0, 1.0, 0.0])
                elif ay_abs >= ax_abs and ay_abs >= az_abs:
                    ref_basis = np.array([0.0, 0.0, 1.0])
                else:
                    ref_basis = np.array([1.0, 0.0, 0.0])
                e0 = ref_basis - float(np.dot(ref_basis, axis_local)) * axis_local
                e0_norm = float(np.linalg.norm(e0))
                if e0_norm < 1e-12:
                    continue
                e0 = e0 / e0_norm
                e1 = np.cross(axis_local, e0)

                aabb_min, aabb_max = body_aabb[child]
                corners_body = self._aabb_corners(aabb_min, aabb_max)

                # Child anchor in child-body frame: joint_X_c = (p_c, q_c), anchor-frame
                # coordinate = q_c⁻¹ · (p_body - p_c). Quaternion layout is (qx, qy, qz, qw).
                xc = joint_X_c[j]
                p_c = xc[0:3].astype(np.float64)
                q_c = xc[3:7].astype(np.float64)
                q_c_inv = np.array([-q_c[0], -q_c[1], -q_c[2], q_c[3]])
                delta = corners_body - p_c
                corners_anchor = self._quat_rotate_vectors(q_c_inv, delta)

                # Six face centers (midpoints of opposite-corner diagonals).
                side_midpoints = np.stack(
                    [
                        (corners_anchor[0] + corners_anchor[3]) * 0.5,
                        (corners_anchor[0] + corners_anchor[5]) * 0.5,
                        (corners_anchor[0] + corners_anchor[6]) * 0.5,
                        (corners_anchor[1] + corners_anchor[7]) * 0.5,
                        (corners_anchor[2] + corners_anchor[7]) * 0.5,
                        (corners_anchor[4] + corners_anchor[7]) * 0.5,
                    ]
                )

                # Project each face center onto the plane perpendicular to axis_local, pick
                # the one with the largest projected length — that direction becomes 0°.
                axial = side_midpoints @ axis_local
                projections = side_midpoints - axial[:, None] * axis_local
                lengths = np.linalg.norm(projections, axis=1)
                best = int(np.argmax(lengths))
                if lengths[best] < 1e-9:
                    continue
                v = projections[best]
                offsets[j] = float(np.arctan2(float(v @ e1), float(v @ e0)))

        self._joint_zero_angle_offset = wp.array(offsets, dtype=float, device=self.device)

    def _compute_body_local_aabbs(self) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
        """Build each rigid body's local-frame AABB by merging its shapes' collision AABBs.

        Returns ``None`` when required Newton model attributes are missing (e.g. pure
        point-cloud models). Shapes are transformed from shape-local to body-local space
        using ``shape_transform`` before the per-body AABB is accumulated.
        """
        model = self.model
        if model is None:
            return None
        required = (
            getattr(model, "body_shapes", None),
            getattr(model, "shape_body", None),
            getattr(model, "shape_transform", None),
            getattr(model, "shape_collision_aabb_lower", None),
            getattr(model, "shape_collision_aabb_upper", None),
        )
        if any(attr is None for attr in required):
            return None

        try:
            shape_transform = np.asarray(model.shape_transform.numpy()).reshape(-1, 7)
            aabb_lower = np.asarray(model.shape_collision_aabb_lower.numpy()).reshape(-1, 3)
            aabb_upper = np.asarray(model.shape_collision_aabb_upper.numpy()).reshape(-1, 3)
        except Exception as exc:
            logger.debug("[NewtonVisualizer] could not read shape arrays for body AABBs: %s", exc)
            return None

        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for body_id, shape_ids in model.body_shapes.items():
            if body_id is None or int(body_id) < 0 or not shape_ids:
                continue
            lo = np.full(3, np.inf, dtype=np.float64)
            hi = np.full(3, -np.inf, dtype=np.float64)
            for sid in shape_ids:
                sid_i = int(sid)
                if sid_i < 0 or sid_i >= len(aabb_lower):
                    continue
                s_lo = aabb_lower[sid_i]
                s_hi = aabb_upper[sid_i]
                if not (np.all(np.isfinite(s_lo)) and np.all(np.isfinite(s_hi))):
                    continue
                corners_shape = self._aabb_corners(s_lo, s_hi)
                xf = shape_transform[sid_i]
                p = xf[0:3].astype(np.float64)
                q = xf[3:7].astype(np.float64)
                corners_body = self._quat_rotate_vectors(q, corners_shape) + p
                lo = np.minimum(lo, corners_body.min(axis=0))
                hi = np.maximum(hi, corners_body.max(axis=0))
            if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                out[int(body_id)] = (lo, hi)
        return out

    @staticmethod
    def _aabb_corners(aabb_min: np.ndarray, aabb_max: np.ndarray) -> np.ndarray:
        """Return 8 AABB corners in USD corner-index order (LDB/RDB/LUB/RUB/LDF/RDF/LUF/RUF)."""
        lo = np.asarray(aabb_min, dtype=np.float64)
        hi = np.asarray(aabb_max, dtype=np.float64)
        return np.array(
            [
                [lo[0], lo[1], lo[2]],
                [hi[0], lo[1], lo[2]],
                [lo[0], hi[1], lo[2]],
                [hi[0], hi[1], lo[2]],
                [lo[0], lo[1], hi[2]],
                [hi[0], lo[1], hi[2]],
                [lo[0], hi[1], hi[2]],
                [hi[0], hi[1], hi[2]],
            ]
        )

    @staticmethod
    def _quat_rotate_vectors(q: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """Rotate each row of ``vs`` (N×3) by unit quaternion ``q`` in (qx, qy, qz, qw) order."""
        qv = np.asarray(q[:3], dtype=np.float64)
        qw = float(q[3])
        vs = np.asarray(vs, dtype=np.float64)
        if vs.ndim == 1:
            vs = vs[None, :]
        # v' = v + 2·qv × (qv × v + qw·v), numerically stable for unit quaternions.
        t = 2.0 * np.cross(np.broadcast_to(qv, vs.shape), vs)
        return vs + qw * t + np.cross(np.broadcast_to(qv, t.shape), t)

    # ------------------------------------------------------------------
    # Joint control panel (per-joint slider + optional state.joint_q override)
    # ------------------------------------------------------------------
    def _collect_joint_metadata(self) -> None:
        """Cache per-1-DoF-joint metadata used by the Joint Control panel.

        Rebuilds the cache whenever the underlying model object changes. Silently clears
        the cache if required attributes are unavailable (e.g. point-cloud-only models).
        """
        model = self.model
        model_key = id(model) if model is not None else None
        if model_key == self._joint_control_model_key and self._joint_control_meta is not None:
            return

        self._joint_control_model_key = model_key
        self._joint_control_meta = []
        self._joint_control_values = []
        if model is None:
            return

        required = (
            getattr(model, "joint_type", None),
            getattr(model, "joint_q_start", None),
            getattr(model, "joint_qd_start", None),
            getattr(model, "joint_limit_lower", None),
            getattr(model, "joint_limit_upper", None),
        )
        if any(attr is None for attr in required):
            return

        try:
            joint_types = model.joint_type.numpy()
            q_starts = model.joint_q_start.numpy()
            qd_starts = model.joint_qd_start.numpy()
            limits_lo = model.joint_limit_lower.numpy()
            limits_hi = model.joint_limit_upper.numpy()
        except Exception as exc:
            logger.debug("[NewtonVisualizer] could not read joint arrays for control panel: %s", exc)
            return

        labels = getattr(model, "joint_label", None) or []
        two_pi = 2.0 * float(np.pi)

        for j, jt in enumerate(joint_types):
            # 0 = PRISMATIC, 1 = REVOLUTE (see newton.JointType).
            if int(jt) not in (0, 1):
                continue
            # joint_q_start is a COORDINATE index (used to read/write state.joint_q).
            # joint_qd_start is a VELOCITY-DOF index (used to read joint_limit_*, joint_axis).
            # For REVOLUTE/PRISMATIC coord_count == dof_count == 1, so the two happen to match;
            # keep them separate anyway to stay correct if multi-DoF joints are ever mixed in.
            q_idx = int(q_starts[j])
            qd_idx = int(qd_starts[j])
            if qd_idx < 0 or qd_idx >= len(limits_lo):
                continue
            lo = float(limits_lo[qd_idx])
            hi = float(limits_hi[qd_idx])
            is_revolute = int(jt) == 1
            # Slider bounds: use limits when finite and valid; otherwise fall back.
            if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
                slider_lo = -two_pi if is_revolute else -1.0
                slider_hi = +two_pi if is_revolute else +1.0
            else:
                slider_lo, slider_hi = lo, hi

            full_label = labels[j] if j < len(labels) else f"joint_{j}"
            name = full_label.rsplit("/", 1)[-1] if "/" in full_label else full_label
            self._joint_control_meta.append(
                {
                    "id": j,
                    "q_idx": q_idx,
                    "qd_idx": qd_idx,
                    "name": name,
                    "full_path": full_label,
                    "is_revolute": is_revolute,
                    "lo": lo,
                    "hi": hi,
                    "slider_lo": slider_lo,
                    "slider_hi": slider_hi,
                    "debug_lo_deg": (lo * 180.0 / float(np.pi)) if is_revolute and np.isfinite(lo) else lo,
                    "debug_hi_deg": (hi * 180.0 / float(np.pi)) if is_revolute and np.isfinite(hi) else hi,
                    # Followers driven by this leader as ``(follower_q_idx, coef0, coef1)`` with
                    # ``follower = coef0 + coef1 * leader``. Filled below; empty for non-leaders.
                    "mimic_followers": [],
                }
            )
            self._joint_control_values.append(0.0)

        # Fold mimic constraints in: drop follower sliders and attach them to their leader, so one
        # leader slider (e.g. the gripper closure joint) drives the coupled fingers.
        self._fold_mimic_followers(model)
        self._log_coupling_diagnostics(model)

        # One-shot diagnostic print so the exact values the viewer reads from
        # the Newton model are visible in stdout. Compare against USD raw degrees.
        logger.info(
            "[NewtonVisualizer] Joint Control panel registered %d 1-DoF joints from Newton model:",
            len(self._joint_control_meta),
        )
        for m in self._joint_control_meta:
            kind = "REV" if m["is_revolute"] else "PRI"
            if m["is_revolute"]:
                lo_s = f"{m['debug_lo_deg']:+8.3f}°" if np.isfinite(m["debug_lo_deg"]) else f"{m['debug_lo_deg']!s:>9}"
                hi_s = f"{m['debug_hi_deg']:+8.3f}°" if np.isfinite(m["debug_hi_deg"]) else f"{m['debug_hi_deg']!s:>9}"
            else:
                lo_s = f"{m['lo']:+8.4f} m" if np.isfinite(m["lo"]) else f"{m['lo']!s:>10}"
                hi_s = f"{m['hi']:+8.4f} m" if np.isfinite(m["hi"]) else f"{m['hi']!s:>10}"
            logger.info(
                "  [%-3s id=%2d  q=%-3d qd=%-3d]  %-32s  lo=%s  hi=%s",
                kind,
                m["id"],
                m["q_idx"],
                m["qd_idx"],
                m["name"],
                lo_s,
                hi_s,
            )

    def _fold_mimic_followers(self, model) -> None:
        """Remove mimic-follower joints from the sliders and attach them to their leader.

        Reads the model's mimic constraints (``constraint_mimic_joint0`` = follower,
        ``constraint_mimic_joint1`` = leader, ``follower = coef0 + coef1 * leader``), records each
        follower on the leader's ``mimic_followers``, and drops the follower's own slider. No-op when
        the model has no mimic constraints (followers then remain as ordinary sliders).
        """
        meta = self._joint_control_meta
        if not meta:
            return
        count = int(getattr(model, "constraint_mimic_count", 0) or 0)
        if count <= 0:
            return
        j0 = getattr(model, "constraint_mimic_joint0", None)
        j1 = getattr(model, "constraint_mimic_joint1", None)
        if j0 is None or j1 is None:
            return
        c0 = getattr(model, "constraint_mimic_coef0", None)
        c1 = getattr(model, "constraint_mimic_coef1", None)
        enabled = getattr(model, "constraint_mimic_enabled", None)
        try:
            follower_ids = j0.numpy()
            leader_ids = j1.numpy()
            coef0 = c0.numpy() if c0 is not None else np.zeros(count)
            coef1 = c1.numpy() if c1 is not None else np.ones(count)
            enabled_np = enabled.numpy() if enabled is not None else None
        except Exception as exc:
            logger.debug("[NewtonVisualizer] could not read mimic constraints: %s", exc)
            return

        # Direct follower -> (leader, coef0, coef1), honoring the enabled flag.
        direct: dict[int, tuple[int, float, float]] = {}
        for k in range(min(count, len(follower_ids), len(leader_ids))):
            if enabled_np is not None and k < len(enabled_np) and not bool(enabled_np[k]):
                continue
            f_id = int(follower_ids[k])
            l_id = int(leader_ids[k])
            if f_id != l_id:
                direct[f_id] = (l_id, float(coef0[k]), float(coef1[k]))

        def _resolve_root(f_id: int) -> tuple[int, float, float] | None:
            """Flatten a (possibly chained) follower to its root leader: ``follower = C0 + C1 * root``.

            Mimic constraints can chain (``a = f(b)``, ``b = g(c)``); composing the affine
            coefficients lets one root-leader slider drive the whole chain. Returns ``None`` on a cycle.
            """
            l_id, cc0, cc1 = direct[f_id]
            seen = {f_id}
            while l_id in direct:
                if l_id in seen:
                    return None
                seen.add(l_id)
                pl_id, pc0, pc1 = direct[l_id]
                # l = pc0 + pc1 * pl  =>  f = (cc0 + cc1*pc0) + (cc1*pc1) * pl
                cc0 = cc0 + cc1 * pc0
                cc1 = cc1 * pc1
                l_id = pl_id
            return l_id, cc0, cc1

        by_joint_id = {m["id"]: m for m in meta}
        dropped_follower_ids: set[int] = set()
        # Per root leader: the slider range that keeps every follower within its own limits.
        leader_range: dict[int, tuple[float, float]] = {}
        for f_id in direct:
            resolved = _resolve_root(f_id)
            if resolved is None:
                continue
            root_id, cc0, cc1 = resolved
            follower_meta = by_joint_id.get(f_id)
            root_meta = by_joint_id.get(root_id)
            if follower_meta is None or root_meta is None:
                continue
            root_meta["mimic_followers"].append((follower_meta["q_idx"], cc0, cc1))
            dropped_follower_ids.add(f_id)
            # Map the follower's finite limits back onto the leader axis to size the leader slider.
            # follower = cc0 + cc1 * leader  =>  leader = (follower_limit - cc0) / cc1.
            f_lo, f_hi = follower_meta["lo"], follower_meta["hi"]
            if abs(cc1) > 1e-12 and np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo <= f_hi:
                a, b = (f_lo - cc0) / cc1, (f_hi - cc0) / cc1
                r_lo, r_hi = (a, b) if a <= b else (b, a)
                cur = leader_range.get(root_id)
                # Intersect across followers so none exceeds its travel.
                leader_range[root_id] = (r_lo, r_hi) if cur is None else (max(cur[0], r_lo), min(cur[1], r_hi))

        if not dropped_follower_ids:
            return

        # Size each leader slider to the follower-derived range, clamped to the leader's own joint
        # limit (the solver clamps there anyway, so a wider slider would only set unreachable targets).
        for root_id, (r_lo, r_hi) in leader_range.items():
            lm = by_joint_id.get(root_id)
            if lm is None:
                continue
            if np.isfinite(lm["lo"]) and np.isfinite(lm["hi"]) and lm["lo"] <= lm["hi"]:
                r_lo, r_hi = max(r_lo, lm["lo"]), min(r_hi, lm["hi"])
            if r_lo < r_hi:
                lm["slider_lo"], lm["slider_hi"] = r_lo, r_hi

        # Rebuild the slider list/values without the followers, preserving order and alignment.
        kept = [(i, m) for i, m in enumerate(meta) if m["id"] not in dropped_follower_ids]
        self._joint_control_meta = [m for _, m in kept]
        self._joint_control_values = [self._joint_control_values[i] for i, _ in kept]
        logger.info(
            "[NewtonVisualizer] Folded %d mimic-follower joint(s) into their leader(s); %d slider(s) remain.",
            len(dropped_follower_ids),
            len(self._joint_control_meta),
        )

    def _log_coupling_diagnostics(self, model) -> None:
        """One-shot log of how joints are coupled (mimic vs MuJoCo equality).

        Reports Newton *mimic* constraints (which the panel folds into the leader slider) and, when a
        MuJoCo solver is active, the equality count from its compiled ``mj_model`` (post-conversion, so
        it includes equalities derived from mimic). Best-effort: never raises.
        """
        labels = getattr(model, "joint_label", None) or []

        def _joint_name(idx: int) -> str:
            if 0 <= idx < len(labels):
                return labels[idx].rsplit("/", 1)[-1]
            return f"joint_{idx}"

        # Mimic constraints (joint0 = coef0 + coef1 * joint1).
        mimic_count = int(getattr(model, "constraint_mimic_count", 0) or 0)
        logger.info("[NewtonVisualizer] joint coupling — mimic constraints: %d", mimic_count)
        if mimic_count:
            try:
                f_ids = model.constraint_mimic_joint0.numpy()
                l_ids = model.constraint_mimic_joint1.numpy()
                c0 = model.constraint_mimic_coef0.numpy()
                c1 = model.constraint_mimic_coef1.numpy()
                for k in range(mimic_count):
                    logger.info(
                        "    [mimic] %s = %.4f + %.4f * %s",
                        _joint_name(int(f_ids[k])),
                        float(c0[k]),
                        float(c1[k]),
                        _joint_name(int(l_ids[k])),
                    )
            except Exception as exc:
                logger.debug("[NewtonVisualizer] mimic dump failed: %s", exc)

        # MuJoCo equality constraints from the solver's *compiled* ``mj_model`` (post-conversion),
        # not ``model.mujoco`` (which counts only Newton-authored equalities and is 0 for mimic
        # grippers). Mimic constraints become ``mjEQ_JOINT`` rows in the solver, so ``mj_model.neq``
        # reflects what it actually enforces. Skipped on backends with no MuJoCo solver (e.g. PhysX
        # shadow-visualization). Best-effort: never raises.
        try:
            from isaaclab_newton.physics import NewtonManager

            solver = NewtonManager.get_solver()
            mj_model = getattr(solver, "mj_model", None)
            neq = getattr(mj_model, "neq", None)
            if neq is not None:
                from_mimic = 0
                eq_map = getattr(solver, "mjc_eq_to_newton_mimic", None)
                if eq_map is not None:
                    try:
                        rows = eq_map.numpy()
                        if rows.size:
                            from_mimic = int((rows[0] >= 0).sum())
                    except Exception:
                        from_mimic = 0
                logger.info(
                    "[NewtonVisualizer] joint coupling — MuJoCo equality constraints "
                    "(compiled, incl. mimic): %d (%d from mimic)",
                    int(neq),
                    from_mimic,
                )
        except Exception as exc:
            logger.debug("[NewtonVisualizer] compiled equality probe failed: %s", exc)

    def _sync_slider_values_from_state(self, state) -> None:
        """Refresh slider values from the current state.joint_q (used when override is OFF)."""
        meta = self._joint_control_meta
        if not meta or state is None or getattr(state, "joint_q", None) is None:
            return
        try:
            qs = state.joint_q.numpy()
        except Exception:
            return
        for i, m in enumerate(meta):
            q_idx = m["q_idx"]
            if 0 <= q_idx < len(qs):
                self._joint_control_values[i] = float(qs[q_idx])

    def apply_joint_overrides(self, state, control=None) -> None:
        """Drive the joints from the Joint Control sliders.

        Called each frame before ``log_state``. Depending on :attr:`joint_control_use_pd_target`,
        either overwrites ``state.joint_q`` and re-evaluates FK (kinematic teleport) or writes the
        slider values into ``control.joint_target_q`` (PD position target). No-op when the panel is
        disabled, the model has no 1-DoF joints, or the required Newton entry point is unavailable.

        Args:
            state: Newton state read for slider sync and overwritten in kinematic mode.
            control: Newton control whose ``joint_target_q`` receives the targets in PD-target mode;
                ignored in kinematic mode.
        """
        if not self.joint_control_enabled:
            # Keep sliders tracking live state for next time the user toggles override on.
            # Skipped once the app seeds values externally (seed_joint_override_values):
            # on body_q-only backends the shadow joint_q here is stale.
            if not self._joint_control_values_external:
                self._sync_slider_values_from_state(state)
            return

        meta = self._joint_control_meta
        if not meta or self.model is None:
            return

        if self.joint_control_use_pd_target:
            self._apply_pd_targets(control)
            return

        if state is None or getattr(state, "joint_q", None) is None:
            return

        try:
            import newton  # noqa: PLC0415 — viewer-time import, guarded below.
        except Exception:
            return

        try:
            joint_q_np = state.joint_q.numpy()
        except Exception:
            return

        # Write all slider values into joint_q each frame. FK is cheap for typical robots
        # and a float32 round-trip through numpy would otherwise make any equality check
        # spuriously trigger every frame anyway.
        n = joint_q_np.shape[0]
        for i, m in enumerate(meta):
            q_idx = m["q_idx"]
            if 0 <= q_idx < n:
                val = float(self._joint_control_values[i])
                joint_q_np[q_idx] = val
                # Propagate to mimic followers (e.g. gripper fingers) so coupled joints track the
                # leader even while paused — FK alone does not enforce the mimic constraint.
                for f_q_idx, coef0, coef1 in m["mimic_followers"]:
                    if 0 <= f_q_idx < n:
                        joint_q_np[f_q_idx] = coef0 + coef1 * val

        state.joint_q.assign(joint_q_np)
        try:
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        except Exception as exc:
            logger.debug("[NewtonVisualizer] eval_fk after joint override failed: %s", exc)

    def _apply_pd_targets(self, control) -> None:
        """Write slider values into ``control.joint_target_q`` (the joint position setpoint).

        Both consumers read this setpoint: the solver's position drive (``POSITION`` DOFs) and
        composed actuators like ``ControllerStablePD`` (``EFFORT`` DOFs that still read
        ``joint_target_q``). We write every panel joint rather than gating on ``POSITION``, which
        would skip StablePD joints; pure-effort/velocity DOFs (wheels) harmlessly ignore it.

        No-op when ``control`` or its ``joint_target_q`` is unavailable (e.g. before the first step).
        """
        meta = self._joint_control_meta
        if not meta or control is None:
            return
        target = getattr(control, "joint_target_q", None)
        if target is None:
            return

        try:
            target_np = target.numpy()
        except Exception:
            return

        # ``joint_target_q`` is coordinate-indexed, matching the panel's ``q_idx``.
        n = target_np.shape[0]
        wrote = False
        for i, m in enumerate(meta):
            q_idx = m["q_idx"]
            if 0 <= q_idx < n:
                target_np[q_idx] = float(self._joint_control_values[i])
                wrote = True

        if wrote:
            target.assign(target_np)


class NewtonViewerRTX:  # noqa: D101 — defined dynamically below to keep the RTX import optional.
    pass


def _build_newton_viewer_rtx_class():
    """Build the :class:`NewtonViewerRTX` class, importing ``ViewerRTX`` lazily.

    The OVRTX renderer (``newton[rtx]`` extra) is optional, so the RTX viewer subclass is
    only constructed when the RTX renderer is actually requested. Returns ``None`` if
    :class:`newton.viewer.ViewerRTX` cannot be imported.
    """
    try:
        from newton.viewer import ViewerRTX
    except Exception as exc:  # noqa: BLE001 — optional dependency probe.
        logger.warning("[NewtonVisualizer] ViewerRTX unavailable (%s); RTX renderer disabled.", exc)
        return None

    class _NewtonViewerRTX(ViewerRTX):
        """Wrapper around Newton's ViewerRTX with Isaac Lab pause controls.

        Unlike :class:`NewtonViewerGL`, the RTX viewer keeps Newton's native rendering UI
        and only adds the Isaac Lab simulation/rendering pause buttons. Joint control and
        the extra visualization toggles are GL-only (they depend on :class:`NewtonViewerGL`
        wrapping ``ViewerGui._render_left_panel``) and are intentionally not reproduced here.
        """

        def __init__(self, *args, metadata: dict | None = None, update_frequency: int = 1, **kwargs):
            super().__init__(*args, **kwargs)
            self._paused_training = True
            self._paused_rendering = False
            self._pending_single_step = False
            # Seed the native pause flag to match the authoritative state (see NewtonViewerGL).
            self._paused = self._paused_training
            self._metadata = metadata or {}
            self._update_frequency = update_frequency
            with suppress(Exception):
                self.register_ui_callback(self._render_training_controls, position="side")
            # Bridge Newton's native top-panel Pause/Step into Isaac Lab's pause gate, same as
            # the GL viewer. RTX has no joint panel, so the wrapper only syncs the run controls.
            gui = getattr(self, "gui", None)
            if gui is not None and not getattr(gui, "_isaaclab_panel_wrapped", False):
                self._newton_render_left_panel = gui._render_left_panel
                gui._render_left_panel = self._render_left_panel_with_isaaclab
                gui._isaaclab_panel_wrapped = True

        def is_training_paused(self) -> bool:
            """Return whether the Isaac Lab step loop should block this tick.

            See :meth:`NewtonViewerGL.is_training_paused`. ``_paused_training`` is authoritative;
            a pending single-step request releases one tick.
            """
            if self._pending_single_step:
                return False
            return self._paused_training

        def request_single_step(self) -> None:
            """Queue exactly one physics step while paused (Newton ``Step`` button semantics)."""
            self._pending_single_step = True

        def _render_left_panel_with_isaaclab(self) -> None:
            """Sync Newton's native top-panel Pause/Step (and ``Space`` / ``.`` keys) with the gate.

            Adopts an external edit to Newton's :attr:`_paused` (e.g. the ``Space`` key) made before
            this frame, mirrors the authoritative pause state back onto it for the native panel,
            then syncs a native ``Pause`` toggle back and forwards a ``Step`` click to
            :meth:`request_single_step`. See :meth:`NewtonViewerGL._render_left_panel_with_isaaclab`.
            """
            original = getattr(self, "_newton_render_left_panel", None)
            if original is None:
                return
            if self._paused != self._paused_training:
                self._paused_training = self._paused
            self._paused = self._paused_training
            original()
            if self._paused != self._paused_training:
                self._paused_training = self._paused
            if getattr(self, "_step_requested", False):
                self._step_requested = False
                self.request_single_step()

        def is_rendering_paused(self) -> bool:
            """Return whether rendering is paused by viewer controls."""
            return self._paused_rendering

        def _render_training_controls(self, imgui):
            """Render Isaac Lab simulation and rendering controls in the RTX viewer UI."""
            imgui.separator()
            imgui.text("IsaacLab Controls")

            imgui.text("Simulation Controls")
            run_stop_label = "Run" if self._paused_training else "Stop"
            if imgui.button(run_stop_label):
                self._paused_training = not self._paused_training
                self._paused = self._paused_training
                self._pending_single_step = False
            imgui.same_line()
            imgui.begin_disabled(not self._paused_training)
            if imgui.button("Step"):
                self._paused_training = True
                self._paused = True
                self.request_single_step()
            imgui.end_disabled()

            rendering_label = "Resume Rendering" if self._paused_rendering else "Pause Rendering"
            if imgui.button(rendering_label):
                self._paused_rendering = not self._paused_rendering

            imgui.text("Visualizer Update Frequency")
            changed, new_frequency = imgui.slider_int(
                "##VisualizerUpdateFreq", self._update_frequency, 1, 20, f"Every {self._update_frequency} frames"
            )
            if changed:
                self._update_frequency = new_frequency

    return _NewtonViewerRTX


class NewtonVisualizer(BaseVisualizer):
    """Newton OpenGL visualizer for Isaac Lab."""

    def __init__(self, cfg: NewtonVisualizerCfg):
        """Initialize Newton visualizer state.

        Args:
            cfg: Newton visualizer configuration.
        """
        super().__init__(cfg)
        self.cfg: NewtonVisualizerCfg = cfg
        self._viewer: NewtonViewerGL | None = None
        self._sim_time = 0.0
        self._step_counter = 0
        # Cleared until the first full ``log_state`` frame is drawn. The Newton viewer starts
        # paused, and the paused branch of :meth:`step` only redraws the existing instance buffers
        # via ``_update()`` — it never logs state. Without a first logged frame those buffers hold
        # no body transforms, so a paused-on-startup window renders empty until the user resumes.
        # This latch forces exactly one full ``log_state`` on the first tick regardless of pause
        # state, so the initial frame shows the scene.
        self._has_logged_first_frame = False
        self._model = None
        self._state = None
        self._update_frequency = cfg.update_frequency
        self._last_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self._headless_no_viewer = False
        self._resolved_visible_env_ids: list[int] | None = None
        self._camera_sensor = None
        self._camera_sensor_indices: list[int] = []
        self._camera_env_indices: list[int] = []
        self._camera_is_owned = False
        self._generated_camera_prim_paths: list[str] = []
        self._renderer = str(getattr(cfg, "renderer", "gl")).lower()
        self._live_plots_manager_visible: dict[str, bool] = {}

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        """Initialize viewer resources and bind scene data provider.

        Args:
            scene_data_provider: Scene data provider used to fetch model/state data.
        """
        from isaaclab_newton.physics import NewtonManager

        if self._is_initialized:
            logger.debug("[NewtonVisualizer] initialize() called while already initialized.")
            return

        scene_data_provider = self._set_scene_data_provider(scene_data_provider)
        num_envs = scene_data_provider.num_envs
        metadata = {"num_envs": num_envs}
        self._env_ids = self._compute_visualized_env_ids()
        self._resolved_visible_env_ids = resolve_visible_env_indices(self._env_ids, self.cfg.max_visible_envs, num_envs)
        self._model = NewtonManager.get_model()
        self._state = NewtonManager.get_state(self._scene_data_provider)

        runtime_headless = self.cfg.headless or (
            sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY")
        )
        if runtime_headless and not self.cfg.headless:
            # print() instead of logger.warning(): the kitless launch path does not
            # install a logging handler, so this user-facing notice would be swallowed.
            print(
                "[WARNING] [NewtonVisualizer] No display found (DISPLAY is unset); the Newton viewer runs"
                " headless via EGL and no window will open. Run from a session with a display (or set"
                " DISPLAY, e.g. 'export DISPLAY=:0') to see the viewer."
            )
        self._runtime_headless = runtime_headless

        # Use pyglet's EGL headless backend when requested or when no Linux X display is available.
        # This must run before the first ``pyglet.window`` import so ``Window`` resolves to
        # :class:`~pyglet.window.headless.HeadlessWindow`.
        if runtime_headless:
            import pyglet

            pyglet.options["headless"] = True

        # Resolve the requested renderer backend. The RTX path is optional (``newton[rtx]``);
        # if its native libraries are unavailable we fall back to the OpenGL viewer.
        self._renderer = str(getattr(self.cfg, "renderer", "gl")).lower()
        if self._renderer == "rtx":
            rtx_cls = _build_newton_viewer_rtx_class()
            if rtx_cls is None:
                logger.warning("[NewtonVisualizer] RTX renderer requested but unavailable; falling back to OpenGL.")
                self._renderer = "gl"
            else:
                self._viewer = rtx_cls(
                    width=self.cfg.window_width,
                    height=self.cfg.window_height,
                    headless=runtime_headless,
                    up_axis="Z",
                    environment=str(getattr(self.cfg, "rtx_environment", "default")),
                    metadata=metadata,
                    update_frequency=self.cfg.update_frequency,
                )

        if self._renderer != "rtx":
            self._viewer = NewtonViewerGL(
                width=self.cfg.window_width,
                height=self.cfg.window_height,
                headless=runtime_headless,
                metadata=metadata,
                update_frequency=self.cfg.update_frequency,
            )

        if self._viewer is not None:
            self._viewer.set_model(self._model)
            apply_viewer_visible_worlds(
                self._viewer,
                env_ids=self._env_ids,
                max_visible_envs=self.cfg.max_visible_envs,
                num_envs=num_envs,
            )
            # Re-lay parallel worlds onto a compact viewer grid (no-op when spacing is 0).
            self._apply_compact_world_offsets(self.cfg.world_spacing)
            self._apply_camera_focal_length()
            initial_pose = self._resolve_initial_camera_pose()
            self._apply_camera_pose(initial_pose)
            self._viewer.up_axis = 2  # Z-up
            self._viewer.scaling = 1.0

            # Display toggles + sky/shadow/light, routed through the guarded helper so the RTX viewer
            # (no ``.renderer`` / GL-only ``show_*`` attributes) is a safe no-op instead of raising.
            self._apply_viewer_display_options()

        self._setup_camera_sensor_view(num_envs)
        num_visualized_envs = (
            len(self._resolved_visible_env_ids) if self._resolved_visible_env_ids is not None else num_envs
        )
        self._log_initialization_table(
            logger=logger,
            title="NewtonVisualizer Configuration",
            rows=[
                (
                    "eye",
                    tuple(float(x) for x in self._viewer.camera.pos) if self._viewer is not None else self.cfg.eye,
                ),
                ("lookat", self._last_camera_pose[1] if self._last_camera_pose else self.cfg.lookat),
                ("focal_length", self.cfg.focal_length),
                ("tiled_cam_view", self.cfg.tiled_cam_view),
                ("tiled_cam_num", self.cfg.tiled_cam_num),
                ("num_visualized_envs", num_visualized_envs),
                ("headless", self.cfg.headless),
                ("show_particles", self.cfg.show_particles),
                ("particle_color", self.cfg.particle_color),
            ],
        )
        self._is_initialized = True

    def _apply_viewer_display_options(self) -> None:
        """Apply Isaac Lab display toggles to the active viewer.

        The joint/visualization toggles and the ``renderer`` (sky/shadow/wireframe/colors)
        block are specific to :class:`NewtonViewerGL`; the RTX viewer drives those through its
        own native rendering UI. Guard each access so the RTX path is a no-op for GL-only state.
        """
        viewer = self._viewer
        if viewer is None:
            return

        # Joint / debug visualization toggles (GL-only attributes).
        if hasattr(viewer, "show_joints"):
            viewer.show_joints = self.cfg.show_joints
            viewer.show_joint_limits = self.cfg.show_joint_limits
            viewer.show_joint_zero_ref_lines = self.cfg.show_joint_zero_ref_lines
            viewer.joint_limit_radius = float(self.cfg.joint_limit_radius)
            viewer.show_contacts = self.cfg.show_contacts
            viewer.show_collision = self.cfg.show_collision
            viewer.show_springs = self.cfg.show_springs
            viewer.show_inertia_boxes = self.cfg.show_inertia_boxes
            viewer.show_com = self.cfg.show_com
            viewer.show_particles = self.cfg.show_particles
            viewer.particle_color = self.cfg.particle_color

        # Sky/shadow/light editing lives on ``ViewerGL.renderer``; ViewerRTX has no such attribute.
        renderer = getattr(viewer, "renderer", None)
        if renderer is not None:
            renderer.draw_shadows = self.cfg.enable_shadows
            renderer.draw_sky = self.cfg.enable_sky
            renderer.draw_wireframe = self.cfg.enable_wireframe
            # Accept list/tuple/array-like config colors and provide a stable tuple for nanobind conversion.
            renderer.sky_upper = viewer._coerce_color3(self.cfg.sky_upper_color)
            renderer.sky_lower = viewer._coerce_color3(self.cfg.sky_lower_color)
            renderer._light_color = viewer._coerce_color3(self.cfg.light_color)

    def step(self, dt: float) -> None:
        """Advance visualization by one simulation step.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized or self._is_closed:
            return

        self._sim_time += dt
        self._step_counter += 1

        from isaaclab_newton.physics import NewtonManager

        if self._viewer is None:
            self._state = NewtonManager.get_state(self._scene_data_provider)
            return

        # ``dt > 0`` marks a real physics frame (the Isaac Lab pause gate released a tick). It is
        # used to (a) throttle only advancing frames via the update-frequency divisor and (b)
        # consume a pending single-step request, so the Newton ``Step`` button advances exactly
        # one gate release. Paused ticks (``dt == 0``) bypass the throttle and never consume a step.
        advancing = dt > 0.0

        update_frequency = self._viewer._update_frequency if self._viewer else self._update_frequency
        if advancing and self._step_counter % update_frequency != 0:
            # Frame skipped: still clear any pending single-step request so the Newton ``Step``
            # button advances exactly one *gate* release rather than latching across skips.
            if hasattr(self._viewer, "_pending_single_step"):
                self._viewer._pending_single_step = False
            return

        num_envs = NewtonManager.get_num_envs()

        try:
            # Draw a full frame when not paused, or when the first frame has not been logged yet.
            # The viewer starts paused, and the paused branch below only redraws existing instance
            # buffers; forcing the first frame through the full ``log_state`` path here ensures a
            # paused-on-startup window shows the scene instead of an empty viewport.
            if not self._viewer.is_paused() or not self._has_logged_first_frame:
                self._state = NewtonManager.get_state(self._scene_data_provider)
                self._viewer.begin_frame(self._sim_time)
                try:
                    if self._state is not None:
                        body_q = getattr(self._state, "body_q", None)
                        if hasattr(body_q, "shape") and body_q.shape[0] == 0:
                            return
                        # Expose the live state so the Joint Control panel's "Reset to Current" button
                        # can snap sliders back to the simulator's values even when override is off.
                        # The Joint Control panel is a ViewerGL feature; the RTX viewer has neither
                        # ``_cached_state`` nor ``apply_joint_overrides``.
                        if hasattr(self._viewer, "apply_joint_overrides"):
                            self._viewer._cached_state = self._state
                            # Apply any per-joint slider overrides before the state is logged, so the
                            # overridden body_q is what ends up in the viewer's instance buffers. Pass
                            # the live control so PD-target mode can write control.joint_target_q.
                            self._viewer.apply_joint_overrides(self._state, NewtonManager.get_control())
                        self._viewer.log_state(self._state)
                        self._has_logged_first_frame = True
                        contacts = NewtonManager.get_contacts()
                        if contacts is not None:
                            self._viewer.log_contacts(contacts, self._state)
                        else:
                            self._log_scene_contact_sensor_arrows(num_envs)
                        if self.cfg.enable_markers:
                            render_newton_visualization_markers(
                                self._viewer, self._resolved_visible_env_ids, num_envs=num_envs
                            )
                        self._log_camera_sensor_image()
                        self._render_live_plots()
                finally:
                    self._viewer.end_frame()
            elif hasattr(self._viewer, "_update"):
                # Rendering is paused: keep the window responsive without refreshing the scene.
                # ViewerGL redraws the existing buffers via ``_update()``. The RTX viewer pumps
                # window events inside ``begin_frame``, so redraw a frame instead.
                self._viewer._update()
            else:
                self._viewer.begin_frame(self._sim_time)
                self._viewer.end_frame()
            # A real physics frame was processed — consume any pending single-step request so the
            # pause gate re-engages on the next tick (Newton ``Step`` = advance one frame). This is
            # independent of rendering: a single step still counts even if rendering is paused.
            if advancing and hasattr(self._viewer, "_pending_single_step"):
                self._viewer._pending_single_step = False
        except Exception:
            logger.exception("[NewtonVisualizer] Viewer update failed.")

    def close(self) -> None:
        """Release viewer resources."""
        if self._is_closed:
            return
        if self._viewer is not None:
            # The RTX viewer holds OVRTX render bindings that must be released explicitly;
            # otherwise the native renderer warns about active bindings on teardown. The GL
            # viewer's close() is a safe no-op-equivalent, so call it for both backends.
            if hasattr(self._viewer, "close"):
                with suppress(Exception):
                    self._viewer.close()
            self._viewer = None
        if self._camera_sensor is not None and self._camera_is_owned:
            remove_generated_prims(self._generated_camera_prim_paths)
        self._camera_sensor = None
        self._is_closed = True

    def _log_scene_contact_sensor_arrows(self, num_envs: int) -> None:
        """Render contact sensor data as Newton-style arrows when native contacts are unavailable."""
        if self._viewer is None:
            return
        if not self._viewer.show_contacts:
            self._viewer.log_arrows(CONTACT_ARROW_PATH, None, None, None)
            return
        contact_sensors = (
            self._scene_data_provider.get_contact_sensors() if self._scene_data_provider is not None else {}
        )
        if not contact_sensors:
            self._viewer.log_arrows(CONTACT_ARROW_PATH, None, None, None)
            return

        starts: list[torch.Tensor] = []
        ends: list[torch.Tensor] = []
        for sensor in contact_sensors.values():
            sensor_starts, sensor_ends = self._contact_sensor_arrow_tensors(sensor, num_envs)
            if sensor_starts is not None and sensor_ends is not None:
                starts.append(sensor_starts)
                ends.append(sensor_ends)

        if not starts:
            self._viewer.log_arrows(CONTACT_ARROW_PATH, None, None, None)
            return

        starts_t = torch.cat(starts, dim=0).detach().to(dtype=torch.float32, device="cpu").contiguous()
        ends_t = torch.cat(ends, dim=0).detach().to(dtype=torch.float32, device="cpu").contiguous()
        self._viewer.log_arrows(
            CONTACT_ARROW_PATH,
            wp.array(starts_t.numpy(), dtype=wp.vec3, device=self._viewer.device),
            wp.array(ends_t.numpy(), dtype=wp.vec3, device=self._viewer.device),
            CONTACT_ARROW_COLOR,
        )

    def _contact_sensor_arrow_tensors(self, sensor, num_envs: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Build Newton-style arrow starts/ends from an Isaac Lab contact sensor."""
        try:
            data = sensor.data
            net_forces_proxy = data.net_forces_w
            net_forces = net_forces_proxy.torch if net_forces_proxy is not None else None
        except (AttributeError, NotImplementedError, RuntimeError):
            return None, None

        if net_forces is None or net_forces.numel() == 0:
            return None, None
        net_forces = self._filter_visible_env_tensor(net_forces, num_envs)

        force_threshold = getattr(getattr(sensor, "cfg", None), "force_threshold", None)
        if force_threshold is None:
            force_threshold = 0.0

        try:
            contact_pos = getattr(data, "contact_pos_w", None)
            force_matrix = getattr(data, "force_matrix_w", None)
        except NotImplementedError:
            contact_pos = None
            force_matrix = None
        if contact_pos is not None and force_matrix is not None:
            contact_pos_t = self._filter_visible_env_tensor(contact_pos.torch, num_envs)
            force_matrix_t = self._filter_visible_env_tensor(force_matrix.torch, num_envs)
            if contact_pos_t.numel() != 0 and force_matrix_t.numel() != 0:
                force_norm = torch.linalg.norm(force_matrix_t, dim=-1)
                finite_pos = torch.isfinite(contact_pos_t).all(dim=-1)
                active = (force_norm > force_threshold) & finite_pos
                if torch.any(active):
                    starts = contact_pos_t[active]
                    directions = torch.nn.functional.normalize(force_matrix_t[active], dim=-1)
                    return starts, starts + directions * CONTACT_ARROW_LENGTH

        origins = self._contact_sensor_origin_positions(sensor, data, net_forces)
        if origins is None:
            return None, None
        origins = self._filter_visible_env_tensor(origins, num_envs)

        force_norm = torch.linalg.norm(net_forces, dim=-1)
        active = force_norm > force_threshold
        if not torch.any(active):
            return None, None

        starts = origins[active]
        directions = torch.nn.functional.normalize(net_forces[active], dim=-1)
        return starts, starts + directions * CONTACT_ARROW_LENGTH

    def _contact_sensor_origin_positions(self, sensor, data, net_forces: torch.Tensor) -> torch.Tensor | None:
        """Return per-sensor origins for contact arrow starts."""
        try:
            pos_w = getattr(data, "pos_w", None)
        except NotImplementedError:
            pos_w = None
        if pos_w is not None:
            return pos_w.torch

        body_physx_view = getattr(sensor, "body_physx_view", None)
        if body_physx_view is None:
            return None
        try:
            pose = body_physx_view.get_transforms()
        except RuntimeError:
            return None
        # the flat view data is body-major (one view pattern per body); net_forces is (N, B, 3)
        num_envs, num_bodies = net_forces.shape[0], net_forces.shape[1]
        return wp.to_torch(pose).view(num_bodies, num_envs, 7).transpose(0, 1)[..., :3]

    def _filter_visible_env_tensor(self, tensor: torch.Tensor, num_envs: int) -> torch.Tensor:
        """Apply Newton visualizer visible-world filtering to a sensor tensor."""
        if self._resolved_visible_env_ids is None or tensor.ndim == 0 or tensor.shape[0] != num_envs:
            return tensor
        ids = torch.as_tensor(self._resolved_visible_env_ids, dtype=torch.long, device=tensor.device)
        return tensor.index_select(0, ids)

    def is_running(self) -> bool:
        """Return whether the visualizer should continue stepping.

        Returns:
            ``True`` while the visualizer is active, otherwise ``False``.
        """
        if not self._is_initialized or self._is_closed:
            return False
        if self._headless_no_viewer and self._viewer is None:
            return True
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    def _resolve_initial_camera_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Resolve initial camera pose from config or USD camera path.

        Returns:
            Camera eye and target tuples.
        """
        return self._resolve_cfg_camera_pose("NewtonVisualizer")

    def _uses_camera_sensor_view(self) -> bool:
        """Return whether the visualizer displays camera sensor images instead of interactive camera controls."""
        return bool(self.cfg.tiled_cam_view)

    def _setup_camera_sensor_view(self, num_envs: int) -> None:
        """Resolve or create the camera sensor used by non-interactive image views."""
        if not self._uses_camera_sensor_view():
            return
        env_ids = resolve_tiled_env_indices(
            num_envs,
            self.cfg.tiled_cam_num,
            self.cfg.tiled_cam_env_indices,
            max_tiles=VISUALIZER_TILED_CAMERA_MAX_TILES,
            sample_from=self._resolved_visible_env_ids,
        )
        self._camera_env_indices = env_ids
        if self.cfg.tiled_cam_prim_path is not None:
            logger.debug(
                "[NewtonVisualizer] tiled_cam_prim_path uses existing camera sensor output; "
                "generated tiled camera pose fields are ignored."
            )
            cameras = self._scene_data_provider.get_camera_sensors()
            self._camera_sensor = find_camera_by_prim_path(cameras, self.cfg.tiled_cam_prim_path, env_ids)
            self._camera_sensor_indices = env_ids
            return

        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        count = max(1, len(env_ids))
        tile_w, tile_h = compute_tile_resolution(self.cfg.window_width, self.cfg.window_height, count)
        self._camera_sensor, self._generated_camera_prim_paths = create_visualizer_camera(
            num_envs=num_envs,
            width=tile_w,
            height=tile_h,
            renderer_cfg=NewtonWarpRendererCfg(),
        )
        self._camera_sensor_indices = env_ids
        self._camera_is_owned = True
        self._update_owned_camera_poses()

    def _update_owned_camera_poses(self) -> None:
        """Update generated camera poses from env origins or follow prims."""
        if self._camera_sensor is None or not self._camera_is_owned:
            return
        target_positions = prim_world_positions(
            self._scene_data_provider.get_usd_stage(),
            self.cfg.tiled_cam_target_prim_path,
            self._camera_env_indices,
            scene=self._scene_data_provider.get_interactive_scene(),
        )
        eyes, targets = apply_camera_target_positions(
            self._camera_sensor, target_positions, self.cfg.tiled_cam_eye, self._camera_env_indices
        )

    def _log_camera_sensor_image(self) -> None:
        """Log the selected camera sensor RGB output into Newton's image panel."""
        if self._viewer is None or self._camera_sensor is None:
            return
        if self._camera_is_owned:
            self._update_owned_camera_poses()
        if self._camera_is_owned:
            self._camera_sensor.update(dt=0.0, force_recompute=True)
        rgb = camera_rgb_batch(self._camera_sensor, self._camera_sensor_indices).contiguous()
        self._viewer.log_image("Visualizer Tiled Camera", wp.from_torch(rgb))

    def _apply_compact_world_offsets(self, spacing: tuple[float, float, float]) -> None:
        """Re-lay parallel worlds on a compact grid centered at the origin.

        ``set_world_offsets`` adds its grid offset on top of each body's physics
        position (env_spacing), so it can't pull far-apart envs together. Instead
        set ``world_offsets[w] = -env_center_xy + compact_grid`` from live ``body_q``.
        Zero spacing is a no-op; on any failure falls back to ``set_world_offsets``.
        """
        try:
            if not any(float(s) != 0.0 for s in spacing):
                self._viewer.set_world_offsets((0.0, 0.0, 0.0))
                return
            model = self._model
            state = self._state
            world_count = int(getattr(model, "world_count", 0) or 0)
            body_q = getattr(state, "body_q", None)
            body_world = getattr(model, "body_world", None)
            if world_count <= 0 or body_q is None or body_world is None:
                self._viewer.set_world_offsets(spacing)
                return
            bq = body_q.numpy()
            bw = body_world.numpy()
            # Per-world mean body xy = that world's physics center (env origin +
            # local layout). Subtract it to cancel the env_spacing spread.
            centers = np.zeros((world_count, 3), dtype=np.float32)
            for w in range(world_count):
                mask = bw == w
                if mask.any():
                    centers[w, :2] = bq[mask, :2].mean(axis=0)
            from newton._src.utils import compute_world_offsets

            grid = compute_world_offsets(world_count, spacing, getattr(model, "up_axis", None))
            offsets = np.zeros((world_count, 3), dtype=np.float32)
            offsets[:, 0] = grid[:, 0] - centers[:, 0]
            offsets[:, 1] = grid[:, 1] - centers[:, 1]
            self._viewer.world_offsets = wp.array(offsets, dtype=wp.vec3, device=self._viewer.device)
            if getattr(self._viewer, "picking", None) is not None:
                self._viewer.picking.world_offsets = self._viewer.world_offsets
        except Exception as exc:  # pragma: no cover - viewer-only safety net
            logger.warning("Compact world offsets failed (%r); falling back to stock spacing.", exc)
            with suppress(Exception):
                self._viewer.set_world_offsets(spacing)

    def _apply_camera_pose(self, pose: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
        """Apply camera eye/target pose to the Newton viewer.

        Args:
            pose: Camera eye and target tuples.
        """
        if self._viewer is None:
            return
        cam_pos, cam_target = pose
        # Match Newton's Camera native pos type: PyVec3, not wp.vec3.
        self._viewer.camera.pos = PygletVec3(*cam_pos)
        self._viewer.camera.look_at(cam_target)
        self._last_camera_pose = (cam_pos, cam_target)

    def _apply_camera_focal_length(self) -> None:
        """Apply cfg focal length to Newton's vertical-FOV camera."""
        if self._viewer is None:
            return
        self._viewer.camera.fov = self._focal_length_to_vertical_fov_degrees()

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        """Set active viewer camera eye/target.

        Args:
            eye: Camera eye position.
            target: Camera look-at target.
        """
        eye_t = (float(eye[0]), float(eye[1]), float(eye[2]))
        target_t = (float(target[0]), float(target[1]), float(target[2]))
        self.cfg.eye = eye_t
        self.cfg.lookat = target_t
        self._apply_camera_pose((eye_t, target_t))

    def render_rgb_array(self) -> np.ndarray:
        """Return the latest RGB frame rendered by the Newton viewer.

        Returns:
            The latest viewer framebuffer as a uint8 array with shape ``(height, width, 3)``.

        Raises:
            RuntimeError: If the visualizer has not been initialized.
        """
        if self._viewer is None:
            raise RuntimeError("NewtonVisualizer must be initialized before capturing an RGB frame.")
        return self._viewer.get_frame().numpy()

    def supports_markers(self) -> bool:
        """Newton OpenGL viewer supports Isaac Lab markers through viewer-side meshes and lines."""
        return bool(self.cfg.enable_markers)

    def supports_live_plots(self) -> bool:
        """Newton OpenGL viewer supports live plots via :meth:`newton.Viewer.log_scalar`."""
        return True

    def add_live_plots(
        self,
        managers: dict,
        scalars: dict | None = None,
        term_names: dict[str, list[str]] | None = None,
        env_idx: int = 0,
    ) -> None:
        """Register managers for live plotting and add per-manager sidebar toggles.

        Calls the base implementation to populate :attr:`_live_plot_sources`, then registers
        one checkbox per manager in the Newton viewer sidebar under a ``Live Plots`` heading.
        Each manager's plots are shown by default and can be hidden by unchecking the
        corresponding box.

        Args:
            managers: Mapping of manager name to manager instance.
            scalars: Optional mapping of group name to a dict of ``{term_name: callable}``.
                Each callable must take no arguments and return a numeric value.
            term_names: Optional per-manager allowlists of term names to include.
            env_idx: Environment index to sample each step.  Defaults to ``0``.
        """
        super().add_live_plots(managers, scalars=scalars, term_names=term_names, env_idx=env_idx)
        if not self._live_plot_sources or self._viewer is None:
            return
        self._live_plots_manager_visible = {source.manager_name: True for source in self._live_plot_sources}
        self._viewer._live_plots_callback = self._live_plots_panel_imgui

    def _live_plots_panel_imgui(self, imgui) -> None:
        """Render a Live Plots collapsing section at the bottom of the Newton panel.

        The top-level section header starts open; individual per-term plot headers start
        closed and can be expanded on demand.
        """
        if not self._live_plot_sources or self._viewer is None:
            return
        viewer = self._viewer
        scalar_buffers = getattr(viewer, "_scalar_buffers", None)
        array_buffers = getattr(viewer, "_array_buffers", None)
        if not scalar_buffers and not array_buffers:
            return

        _ip = getattr(viewer, "_implot", None)
        if not hasattr(viewer, "_scalar_arrays"):
            viewer._scalar_arrays = {}
        scalar_arrays = viewer._scalar_arrays
        n = getattr(viewer, "_plot_history_size", 250)
        s = viewer.gui.ui.dpi_scale
        plot_h = 180 * s

        # Group scalar names by base term name (strip trailing [N] index).
        groups: dict[str, list[str]] = {}
        for name in scalar_buffers or {}:
            base = _newton_scalar_base_name(name)
            groups.setdefault(base, []).append(name)

        # Promote episode metrics (mean_reward, episode_length) to the top.
        episode_keys = [k for k in groups if k.startswith("episode/")]
        other_keys = [k for k in groups if not k.startswith("episode/")]
        groups = {k: groups[k] for k in episode_keys + other_keys}

        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Live Plots"):
            return
        imgui.separator()

        for base_name, names in groups.items():
            term_label = base_name.rsplit("/", 1)[-1]
            if not imgui.collapsing_header(term_label):
                continue
            for name in names:
                buf = scalar_buffers.get(name, [])
                arr = scalar_arrays.get(name)
                if arr is None:
                    arr = np.full(n, np.nan, dtype=np.float32)
                    arr[n - len(buf) :] = np.array(buf, dtype=np.float32)
                    scalar_arrays[name] = arr
            if _ip is not None and _ip.begin_plot(f"##{base_name}", imgui.ImVec2(-1, plot_h)):
                _auto = _ip.AxisFlags_.auto_fit.value
                _ip.setup_axes("", "", _auto, _auto)
                _ip.setup_finish()
                for name in names:
                    arr = scalar_arrays.get(name)
                    if arr is not None:
                        suffix = name[len(base_name) :]
                        label = suffix if suffix else term_label
                        _ip.plot_line(label, arr)
                _ip.end_plot()
            else:
                # Fallback: stacked imgui.plot_lines if ImPlot unavailable.
                graph_size = imgui.ImVec2(-1, 80 * s)
                for name in names:
                    arr = scalar_arrays.get(name)
                    if arr is not None:
                        buf = scalar_buffers.get(name, [])
                        overlay = f"{buf[-1]:.4g}" if buf else ""
                        imgui.plot_lines(f"##{name}", arr, graph_size=graph_size, overlay_text=overlay)

        render_heatmap = getattr(viewer, "_render_array_heatmap", None)
        if render_heatmap is not None:
            panel_width = imgui.get_content_region_avail().x
            for name, array in (array_buffers or {}).items():
                if imgui.collapsing_header(name):
                    render_heatmap(name, array, panel_width - 20.0 * s, dpi_scale=s)

    def _render_live_plots(self) -> None:
        """Push manager-term scalars to the Newton viewer's built-in plot panel."""
        if self._viewer is None or not self._live_plot_sources:
            return
        # In headless mode the panel is never visible — skip collection entirely.
        if getattr(self, "_runtime_headless", False):
            return
        self._live_plots_step_counter += 1
        if self._live_plots_step_counter % max(1, getattr(self.cfg, "live_plots_update_interval", 10)) != 0:
            return
        for source in self._live_plot_sources:
            if not self._live_plots_manager_visible.get(source.manager_name, True):
                continue
            for term_name, values in source.collect(self._live_plot_env_idx).items():
                if len(values) == 1:
                    self._viewer.log_scalar(f"{source.manager_name}/{term_name}", values[0])
                else:
                    for i, v in enumerate(values):
                        self._viewer.log_scalar(f"{source.manager_name}/{term_name}[{i}]", v)

    def is_training_paused(self) -> bool:
        """Return whether training is paused from viewer controls."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_training_paused()

    def is_joint_override_active(self) -> bool:
        """Return whether the Joint Control panel is actively overriding the joints.

        When ``True``, the app loop should stop commanding the robot (e.g. skip its per-step
        ``set_joint_position_target`` / ``write_data_to_sim``) so the panel's slider values are not
        overwritten each step. GL viewer only — always ``False`` for the RTX viewer (no panel).
        """
        if not self._is_initialized or self._viewer is None:
            return False
        return bool(getattr(self._viewer, "joint_control_enabled", False))

    def get_joint_override_targets(self) -> list[tuple[str, float]]:
        """Return the Joint Control panel's ``(USD joint prim path, target value)`` pairs.

        Empty when the panel is not overriding (and always for the RTX viewer, which has no
        panel). On the Newton sim backend the panel drives the solver directly through the
        Newton control, so callers need not consume this. On the PhysX backend the shadow
        Newton control is never read by the simulation and the composed actuators only run
        inside ``write_data_to_sim``: the app loop must bridge these targets into its
        articulation (:meth:`~isaaclab.assets.Articulation.set_joint_position_target` +
        ``write_data_to_sim``) every step while :meth:`is_joint_override_active` is ``True``.
        """
        if not self._is_initialized or self._viewer is None:
            return []
        getter = getattr(self._viewer, "get_joint_override_targets", None)
        return getter() if getter is not None else []

    def seed_joint_override_values(self, values: dict[str, float]) -> None:
        """Seed the Joint Control sliders from live joint positions supplied by the app.

        See :meth:`NewtonViewerGL.seed_joint_override_values`. Call every step while
        :meth:`is_joint_override_active` is ``False`` on backends whose live joint state
        never reaches the shadow Newton state (e.g. PhysX). No panel (RTX viewer) → no-op.
        """
        if not self._is_initialized or self._viewer is None:
            return
        seeder = getattr(self._viewer, "seed_joint_override_values", None)
        if seeder is not None:
            seeder(values)

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused from viewer controls."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_rendering_paused()

    def is_reset_requested(self) -> bool:
        """Return whether an episode reset was requested from viewer controls without clearing the flag."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_reset_requested()

    def consume_reset_request(self) -> bool:
        """Return whether an episode reset was requested from viewer controls and clear the flag."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.consume_reset_request()
