# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for joint limit visualization in the Newton OpenGL viewer.

Newton stores each joint with two anchor frames (see ``newton/_src/sim/builder.py``):

* ``joint_X_p`` maps the parent body frame to the **parent anchor frame**.
* ``joint_X_c`` maps the child body frame to the **child anchor frame**.
* ``joint_axis`` is defined in the parent anchor frame and, by construction, equals
  the axis expressed in the child anchor frame as well (rotation about the axis preserves it).

At zero coordinate both anchor frames coincide at the physical pivot; when the joint moves,
the child anchor frame rotates by ``Rot(axis, q)`` relative to the parent anchor frame while
sharing the same origin (up to solver drift). Using the child anchor as the rendering anchor
keeps the arc glued to the articulation's moving pivot point, which is the viewer intuition
when looking at a robot link.
"""

from __future__ import annotations

import warp as wp

# Line-slot layout per joint (fixed so threads map by tid = joint_id * LINES_PER_JOINT + slot).
# Revolute uses: N_ARC arc segments + 2 parent-anchor end radials + 1 pointer
#              + N_DASH zero-reference dash segments + 2 zero-reference lo/hi end radials
#              + N_DASH_ARC zero-reference arc segments (child-anchor frame).
# Prismatic uses slot 0 (limit segment) and slot 1 (current pointer); remaining slots are NaN-hidden.
N_ARC: int = 32
"""Tessellation of the revolute limit arc (polyline segments)."""

N_DASH: int = 10
"""Number of visible dash segments for the Kit-style zero-angle reference line (child body frame)."""

N_DASH_END: int = 2
"""Lower and upper limit end-ray markers anchored at the child anchor (zero-reference frame)."""

N_DASH_ARC: int = 32
"""Tessellation of the zero-reference arc drawn in the child-anchor frame."""

LINES_PER_JOINT: int = N_ARC + 3 + N_DASH + N_DASH_END + N_DASH_ARC
"""Total line-segment slots reserved per joint: parent-anchor arc + parent-anchor end radials
+ pointer + zero-reference dashes + zero-reference lo/hi end radials + zero-reference arc."""

# Slot offsets within each per-joint block.
_SLOT_LOWER_RAY = N_ARC
_SLOT_UPPER_RAY = N_ARC + 1
_SLOT_CURRENT = N_ARC + 2
_SLOT_DASH_START = N_ARC + 3
_SLOT_DASH_LO_END = _SLOT_DASH_START + N_DASH
_SLOT_DASH_HI_END = _SLOT_DASH_LO_END + 1
_SLOT_DASH_ARC_START = _SLOT_DASH_HI_END + 1

# Colors (RGB in [0, 1]).
_COLOR_ARC = wp.vec3(1.0, 0.8, 0.0)
"""Yellow arc along the limit range."""
_COLOR_END = wp.vec3(1.0, 0.4, 0.0)
"""Orange endpoint radials marking lower and upper bounds."""
_COLOR_POINTER = wp.vec3(0.0, 1.0, 1.0)
"""Cyan pointer — ``anchor + R · (X_wc_rot · e0_local)``. This is mathematically identical
to the Kit / PhysX native dashed body1 indicator: body1's anchor rotation applied to the
same in-plane reference direction used by the arc. Drift-free, it lands exactly on the arc;
with solver drift the pointer faithfully reflects body1's true orientation (may leave the
arc plane — that visual offset *is* the drift signal)."""
_COLOR_DASH = wp.vec3(0.4, 0.933, 0.4)
"""Bright-green zero-angle reference dashed line (Kit ``_target0_rotation_baseline_transform``).
Matches Kit ``Style.color_base = #66EE66``, the body1-frame colour — keeps the dashed line
visually distinct from the yellow limit arc, orange endpoint radials, and cyan pointer."""
_COLOR_DASH_END = wp.vec3(0.267, 0.733, 0.267)
"""Darker green lower/upper end radials anchored at the child anchor (Kit ``color_base_shaded`` =
``#44BB44``). Pairs with :data:`_COLOR_DASH` to form the complete Kit-style limit diagram in
the child body's zero-reference frame."""
_COLOR_DASH_ARC = wp.vec3(0.4, 0.933, 0.4)
"""Zero-reference limit arc drawn in the parent-anchor frame (same bright green as
:data:`_COLOR_DASH` so the dashed 0° line and the sector arc read as one connected glyph).
Parenting the whole zero-reference cluster to body0 matches Kit's ``_attachments[0]._ui_transform_world``
placement, which keeps the diagram static under joint_q changes."""


@wp.kernel
def compute_joint_limit_lines(
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_zero_angle_offset: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_world: wp.array(dtype=int),
    world_offsets: wp.array(dtype=wp.vec3),
    visible_worlds_mask: wp.array(dtype=int),
    radius: float,
    show_limits: int,
    show_zero_ref: int,
    # outputs
    line_starts: wp.array(dtype=wp.vec3),
    line_ends: wp.array(dtype=wp.vec3),
    line_colors: wp.array(dtype=wp.vec3),
):
    """Produce line segments rendering the position limit range for each supported joint.

    Each joint reserves ``LINES_PER_JOINT`` consecutive line slots. Unused slots and joints
    without valid limits are NaN-hidden so the GL renderer skips them.

    Geometry conventions (PhysX ``PxConstraintVisualizer``–style):

    * **Anchor origin, arc plane, endpoint radials** all live in the parent anchor frame
      (``body_q[parent] ⊗ joint_X_p``). This keeps the entire limit diagram static with
      respect to the parent body, which is the correct semantics for a limit constraint.
    * **Current-angle pointer** is the only element that moves with the child. Its angle
      is extracted by a quaternion twist decomposition about ``axis_local`` — this
      discards any non-axial swing drift and guarantees the pointer stays coplanar with
      the arc, so pointer at ``joint_q = lo`` coincides exactly with the arc's lower end.
    * In drift-free cases parent and child anchors are geometrically identical, so this
      convention matches the intuitive "anchor is the joint pivot" reading as well.
    """
    tid = wp.tid()
    nan_line = wp.vec3(wp.nan, wp.nan, wp.nan)
    zero_color = wp.vec3(0.0, 0.0, 0.0)

    joint_id = tid // LINES_PER_JOINT
    slot = tid % LINES_PER_JOINT

    # Out-of-range guard (buffer may be oversized).
    if joint_id >= joint_type.shape[0]:
        line_starts[tid] = nan_line
        line_ends[tid] = nan_line
        line_colors[tid] = zero_color
        return

    jt = joint_type[joint_id]
    # Only REVOLUTE (1) and PRISMATIC (0) have meaningful 1-D position limits here.
    if jt != 1 and jt != 0:
        line_starts[tid] = nan_line
        line_ends[tid] = nan_line
        line_colors[tid] = zero_color
        return

    parent_body = joint_parent[joint_id]
    child_body = joint_child[joint_id]

    # Visibility gating — prefer the child body for ground-attached joints.
    filter_body = child_body
    if filter_body < 0:
        filter_body = parent_body
    if visible_worlds_mask:
        if filter_body >= 0:
            world_idx = body_world[filter_body]
            if world_idx >= 0:
                if visible_worlds_mask[world_idx] == 0:
                    line_starts[tid] = nan_line
                    line_ends[tid] = nan_line
                    line_colors[tid] = zero_color
                    return

    # Parent anchor frame in world: body_q[parent] ⊗ joint_X_p (used to orient the limit plane).
    X_p = joint_X_p[joint_id]
    X_p_pos = wp.transform_get_translation(X_p)
    X_p_rot = wp.transform_get_rotation(X_p)
    if parent_body >= 0:
        parent_tf = body_q[parent_body]
        X_wp_pos = wp.transform_point(parent_tf, X_p_pos)
        X_wp_rot = wp.mul(wp.transform_get_rotation(parent_tf), X_p_rot)
    else:
        X_wp_pos = X_p_pos
        X_wp_rot = X_p_rot

    # Child anchor frame in world: body_q[child] ⊗ joint_X_c (anchor origin and pointer direction).
    X_c = joint_X_c[joint_id]
    X_c_pos = wp.transform_get_translation(X_c)
    X_c_rot = wp.transform_get_rotation(X_c)
    if child_body >= 0:
        child_tf = body_q[child_body]
        X_wc_pos = wp.transform_point(child_tf, X_c_pos)
        X_wc_rot = wp.mul(wp.transform_get_rotation(child_tf), X_c_rot)
    else:
        X_wc_pos = X_c_pos
        X_wc_rot = X_c_rot

    # Apply world-offset based on the child body's world index (matches basis-line convention).
    if world_offsets:
        if child_body >= 0:
            cw = body_world[child_body]
            if cw >= 0:
                X_wc_pos = X_wc_pos + world_offsets[cw]
                X_wp_pos = X_wp_pos + world_offsets[cw]
        elif parent_body >= 0:
            pw = body_world[parent_body]
            if pw >= 0:
                X_wc_pos = X_wc_pos + world_offsets[pw]
                X_wp_pos = X_wp_pos + world_offsets[pw]

    # Two anchor origins are kept separate so the arc can stay relative to the parent
    # (``X_wp_pos``, PhysX "static limit frame") while the pointer starts at the child
    # (``X_wc_pos``, matching Kit's dashed body1-frame indicator). In drift-free
    # articulations they coincide, so the usual visual is a single connected figure.

    # Newton stores joint_axis / joint_limit_* keyed by the velocity DOF index
    # (see solvers/*/kernels.py which all use qd_start); joint_q_start is the
    # position-coordinate index and is only correct here for 1-DoF joints.
    qd_idx = joint_qd_start[joint_id]
    lo = joint_limit_lower[qd_idx]
    hi = joint_limit_upper[qd_idx]
    # USD/Newton convention: lo > hi means "unlimited" — no clamp range. Limits get hidden
    # below, but the zero-reference dash (if any) still renders, matching Kit's behaviour.
    limits_valid = lo <= hi

    axis_local = joint_axis[qd_idx]

    # Axis world direction: using the parent anchor frame. Parent and child give the same
    # answer in the absence of solver drift; parent stays stable even under large drift.
    axis_world = wp.quat_rotate(X_wp_rot, axis_local)

    if jt == 1:  # REVOLUTE — draw arc in the plane perpendicular to axis_world.
        # Forward cyclic convention X→Y→Z→X: pick e0 as the basis vector that follows
        # the axis in the standard (X, Y, Z) order. With axis = local X this makes
        # e0 = local Y (so the pointer aligns with Kit's *green* dashed body1-frame Y axis),
        # and ``e1 = cross(axis, e0)`` lands on the next basis again (e.g. axis=X → e1 = Z).
        ax = wp.abs(axis_local[0])
        ay = wp.abs(axis_local[1])
        az = wp.abs(axis_local[2])
        if ax >= ay and ax >= az:
            ref_basis = wp.vec3(0.0, 1.0, 0.0)  # axis ≈ X → e0 = local Y
        elif ay >= ax and ay >= az:
            ref_basis = wp.vec3(0.0, 0.0, 1.0)  # axis ≈ Y → e0 = local Z
        else:
            ref_basis = wp.vec3(1.0, 0.0, 0.0)  # axis ≈ Z → e0 = local X
        # Gram–Schmidt: remove the axis-parallel component (handles non-axis-aligned axes).
        e0_local = ref_basis - wp.dot(ref_basis, axis_local) * axis_local
        e0_local = wp.normalize(e0_local)
        e1_local = wp.cross(axis_local, e0_local)

        # Arc geometry (arc + endpoint radials) is anchored at the PARENT anchor origin
        # (X_wp_pos, X_wp_rot) — this is the classical "static limit" frame used by PhysX.
        e0_world = wp.quat_rotate(X_wp_rot, e0_local)
        e1_world = wp.quat_rotate(X_wp_rot, e1_local)

        # Limits sub-slots (arc + endpoint radials + pointer) are gated by ``show_limits``
        # and ``limits_valid``; when hidden, NaN the slot so nothing is drawn.
        draw_limits = (show_limits != 0) and limits_valid

        if slot < N_ARC:
            if draw_limits:
                # Arc polyline: segment k spans θ(k) → θ(k+1) of the [lo, hi] range.
                t0 = float(slot) / float(N_ARC)
                t1 = float(slot + 1) / float(N_ARC)
                th0 = lo + (hi - lo) * t0
                th1 = lo + (hi - lo) * t1
                p0 = X_wp_pos + radius * (wp.cos(th0) * e0_world + wp.sin(th0) * e1_world)
                p1 = X_wp_pos + radius * (wp.cos(th1) * e0_world + wp.sin(th1) * e1_world)
                line_starts[tid] = p0
                line_ends[tid] = p1
                line_colors[tid] = _COLOR_ARC
            else:
                line_starts[tid] = nan_line
                line_ends[tid] = nan_line
                line_colors[tid] = zero_color
        elif slot == _SLOT_LOWER_RAY:
            if draw_limits:
                p_lo = X_wp_pos + radius * (wp.cos(lo) * e0_world + wp.sin(lo) * e1_world)
                line_starts[tid] = X_wp_pos
                line_ends[tid] = p_lo
                line_colors[tid] = _COLOR_END
            else:
                line_starts[tid] = nan_line
                line_ends[tid] = nan_line
                line_colors[tid] = zero_color
        elif slot == _SLOT_UPPER_RAY:
            if draw_limits:
                p_hi = X_wp_pos + radius * (wp.cos(hi) * e0_world + wp.sin(hi) * e1_world)
                line_starts[tid] = X_wp_pos
                line_ends[tid] = p_hi
                line_colors[tid] = _COLOR_END
            else:
                line_starts[tid] = nan_line
                line_ends[tid] = nan_line
                line_colors[tid] = zero_color
        elif slot == _SLOT_CURRENT:
            if draw_limits:
                # Kit-aligned pointer: anchored at X_wc (body1 anchor origin).
                # Matches Kit's dashed body1-frame line verbatim: start at the child anchor
                # position, direction = X_wc_rot · e0_local (= body1 local Z for revolute axis X).
                # In drift-free articulations it coincides with the arc's current-θ position; any
                # visible gap between pointer and arc is genuine solver drift.
                p_now_end = X_wc_pos + radius * wp.quat_rotate(X_wc_rot, e0_local)
                line_starts[tid] = X_wc_pos
                line_ends[tid] = p_now_end
                line_colors[tid] = _COLOR_POINTER
            else:
                line_starts[tid] = nan_line
                line_ends[tid] = nan_line
                line_colors[tid] = zero_color
        elif slot < _SLOT_DASH_LO_END:
            # Kit-style zero-angle reference dashed line (one of N_DASH visible sub-segments).
            # Anchored at the PARENT anchor (``X_wp_*``) — same parenting choice as Kit's
            # ``_attachments[0]._ui_transform_world``, which keeps the whole zero-reference
            # glyph static with respect to body0 and *not* rotating with joint_q.
            # Direction = parent-anchor rotation applied to ``e0_local`` rotated by
            # ``zero_angle_offset`` around ``axis_local`` — i.e. pointing to the child body's
            # AABB side-center farthest from the anchor, projected onto the plane ⟂ axis.
            if show_zero_ref == 0:
                line_starts[tid] = nan_line
                line_ends[tid] = nan_line
                line_colors[tid] = zero_color
            else:
                dash_idx = slot - _SLOT_DASH_START
                off = joint_zero_angle_offset[joint_id]
                cos_off = wp.cos(off)
                sin_off = wp.sin(off)
                e0_local_ref = cos_off * e0_local + sin_off * e1_local
                dir_world = wp.quat_rotate(X_wp_rot, e0_local_ref)
                inv_n = 1.0 / float(N_DASH)
                # Dash k draws [2k, 2k+1] of 2*N_DASH equal sub-intervals, so gaps equal segments.
                t0 = float(dash_idx) * inv_n
                t1 = t0 + 0.5 * inv_n
                line_starts[tid] = X_wp_pos + radius * t0 * dir_world
                line_ends[tid] = X_wp_pos + radius * t1 * dir_world
                line_colors[tid] = _COLOR_DASH
        elif slot == _SLOT_DASH_LO_END or slot == _SLOT_DASH_HI_END:
            # Zero-reference lo/hi end radials — anchored at the parent anchor and pointing
            # toward the limit bound, measured in the same zero-reference frame as the dashed
            # line. Rotating ``e0_local_ref`` by ``th`` (= lo or hi) around ``axis_local`` gives
            # the world direction; with Rodrigues on a unit axis and (e0, e1) already orthonormal
            # to axis, this reduces to cos(off + th) * e0_local + sin(off + th) * e1_local.
            if show_zero_ref == 0 or not limits_valid:
                line_starts[tid] = nan_line
                line_ends[tid] = nan_line
                line_colors[tid] = zero_color
            else:
                off = joint_zero_angle_offset[joint_id]
                if slot == _SLOT_DASH_LO_END:
                    th = off + lo
                else:  # _SLOT_DASH_HI_END
                    th = off + hi
                c = wp.cos(th)
                s = wp.sin(th)
                e_local = c * e0_local + s * e1_local
                dir_world = wp.quat_rotate(X_wp_rot, e_local)
                line_starts[tid] = X_wp_pos
                line_ends[tid] = X_wp_pos + radius * dir_world
                line_colors[tid] = _COLOR_DASH_END
        else:
            # Zero-reference limit arc — polyline connecting the lo/hi end radials. Segment
            # k spans θ(k)→θ(k+1) with θ(t) = off + lo + t*(hi - lo), so t=0 coincides with
            # the lo end ray and t=1 with the hi end ray. Anchored at ``X_wp_pos`` (parent
            # anchor) so the entire Kit-style glyph stays static under joint_q changes and
            # only follows rigid-body motion of body0.
            if show_zero_ref == 0 or not limits_valid:
                line_starts[tid] = nan_line
                line_ends[tid] = nan_line
                line_colors[tid] = zero_color
            else:
                arc_idx = slot - _SLOT_DASH_ARC_START
                off = joint_zero_angle_offset[joint_id]
                inv_n = 1.0 / float(N_DASH_ARC)
                t0 = float(arc_idx) * inv_n
                t1 = float(arc_idx + 1) * inv_n
                th0 = off + lo + (hi - lo) * t0
                th1 = off + lo + (hi - lo) * t1
                e0_0 = wp.cos(th0) * e0_local + wp.sin(th0) * e1_local
                e0_1 = wp.cos(th1) * e0_local + wp.sin(th1) * e1_local
                p0 = X_wp_pos + radius * wp.quat_rotate(X_wp_rot, e0_0)
                p1 = X_wp_pos + radius * wp.quat_rotate(X_wp_rot, e0_1)
                line_starts[tid] = p0
                line_ends[tid] = p1
                line_colors[tid] = _COLOR_DASH_ARC
    else:  # PRISMATIC — axis-aligned segment from lo to hi, plus current-position pointer.
        # For a prismatic joint the "current position" is encoded in the child anchor's translation
        # relative to the parent anchor, projected onto the axis: gives drift-accurate display.
        delta_world = X_wc_pos - X_wp_pos
        q_now = wp.dot(delta_world, axis_world)
        draw_limits = (show_limits != 0) and limits_valid

        if slot == 0 and draw_limits:
            p_lo = X_wp_pos + axis_world * lo
            p_hi = X_wp_pos + axis_world * hi
            line_starts[tid] = p_lo
            line_ends[tid] = p_hi
            line_colors[tid] = _COLOR_ARC
        elif slot == 1 and draw_limits:
            p_now = X_wp_pos + axis_world * q_now
            line_starts[tid] = X_wp_pos
            line_ends[tid] = p_now
            line_colors[tid] = _COLOR_POINTER
        else:
            # No zero-reference dash for prismatic — Kit also only renders it on revolute.
            line_starts[tid] = nan_line
            line_ends[tid] = nan_line
            line_colors[tid] = zero_color
