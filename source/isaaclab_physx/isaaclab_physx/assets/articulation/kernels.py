# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher

"""
Articulation-specific warp functions.
"""


@wp.func
def compute_soft_joint_pos_limits_func(
    joint_pos_limits: wp.vec2f,
    soft_limit_factor: wp.float32,
):
    """Compute the soft joint position limits.

    Args:
        joint_pos_limits: The joint position limits.
        soft_limit_factor: The soft limit factor.

    Returns:
        The soft joint position limits.
    """
    joint_pos_mean = (joint_pos_limits[0] + joint_pos_limits[1]) / 2.0
    joint_pos_range = joint_pos_limits[1] - joint_pos_limits[0]
    return wp.vec2f(
        joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,
        joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,
    )


"""
Articulation-specific warp kernels.
"""


@wp.kernel
def write_joint_position_with_sim_ids(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    full_data: bool,
    user_pos: wp.array2d(dtype=wp.float32),
    backend_pos: wp.array2d(dtype=wp.float32),
    sim_env_ids: wp.array(dtype=wp.int32),
) -> None:
    """Write joint positions and materialize simulator indices."""
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    value = in_data[env_id, joint_id] if full_data else in_data[i, j]
    user_pos[env_id, joint_id] = value
    if has_ordering:
        backend_pos[env_id, user_to_backend[joint_id]] = value
    if j == 0:
        sim_env_ids[i] = env_id


_WRITE_JOINT_POSITION_WITH_SIM_IDS = IndexKernelDispatcher(write_joint_position_with_sim_ids, ("env_ids", "joint_ids"))


def write_joint_position_with_sim_ids_kernel(
    env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the joint-position writer for the selector dtypes."""
    return _WRITE_JOINT_POSITION_WITH_SIM_IDS.select(env_ids, joint_ids)


@wp.kernel
def write_joint_velocity_with_sim_ids(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    full_data: bool,
    user_vel: wp.array2d(dtype=wp.float32),
    user_prev_vel: wp.array2d(dtype=wp.float32),
    user_acc: wp.array2d(dtype=wp.float32),
    backend_vel: wp.array2d(dtype=wp.float32),
    sim_env_ids: wp.array(dtype=wp.int32),
) -> None:
    """Write joint velocities and materialize simulator indices."""
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    value = in_data[env_id, joint_id] if full_data else in_data[i, j]
    user_vel[env_id, joint_id] = value
    user_prev_vel[env_id, joint_id] = value
    user_acc[env_id, joint_id] = 0.0
    if has_ordering:
        backend_vel[env_id, user_to_backend[joint_id]] = value
    if j == 0:
        sim_env_ids[i] = env_id


_WRITE_JOINT_VELOCITY_WITH_SIM_IDS = IndexKernelDispatcher(write_joint_velocity_with_sim_ids, ("env_ids", "joint_ids"))


def write_joint_velocity_with_sim_ids_kernel(
    env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the joint-velocity writer for the selector dtypes."""
    return _WRITE_JOINT_VELOCITY_WITH_SIM_IDS.select(env_ids, joint_ids)


@wp.kernel
def write_joint_state_with_sim_ids(
    position: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    full_data: bool,
    user_pos: wp.array2d(dtype=wp.float32),
    user_vel: wp.array2d(dtype=wp.float32),
    user_prev_vel: wp.array2d(dtype=wp.float32),
    user_acc: wp.array2d(dtype=wp.float32),
    backend_pos: wp.array2d(dtype=wp.float32),
    backend_vel: wp.array2d(dtype=wp.float32),
    sim_env_ids: wp.array(dtype=wp.int32),
) -> None:
    """Write joint state and materialize simulator indices."""
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    pos_value = position[env_id, joint_id] if full_data else position[i, j]
    vel_value = velocity[env_id, joint_id] if full_data else velocity[i, j]
    user_pos[env_id, joint_id] = pos_value
    user_vel[env_id, joint_id] = vel_value
    user_prev_vel[env_id, joint_id] = vel_value
    user_acc[env_id, joint_id] = 0.0
    if has_ordering:
        backend_id = user_to_backend[joint_id]
        backend_pos[env_id, backend_id] = pos_value
        backend_vel[env_id, backend_id] = vel_value
    if j == 0:
        sim_env_ids[i] = env_id


_WRITE_JOINT_STATE_WITH_SIM_IDS = IndexKernelDispatcher(write_joint_state_with_sim_ids, ("env_ids", "joint_ids"))


def write_joint_state_with_sim_ids_kernel(
    env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the joint-state writer for the selector dtypes."""
    return _WRITE_JOINT_STATE_WITH_SIM_IDS.select(env_ids, joint_ids)


@wp.kernel
def get_joint_acc_from_joint_vel(
    joint_vel: wp.array2d(dtype=wp.float32),
    prev_joint_vel: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    joint_acc: wp.array2d(dtype=wp.float32),
):
    """Compute the joint acceleration from the joint velocity using finite differencing.

    This kernel computes the joint acceleration by taking the difference between the current
    and previous joint velocities, divided by the time step. It also updates the previous
    joint velocity buffer with the current values.

    Args:
        joint_vel: Input array of current joint velocities. Shape is (num_envs, num_joints).
        prev_joint_vel: Input/output array of previous joint velocities. Shape is (num_envs, num_joints).
            This buffer is updated with the current joint velocities after computing acceleration.
        dt: Input time step (scalar) used for finite differencing.
        joint_acc: Output array where joint accelerations are written. Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    joint_acc[i, j] = (joint_vel[i, j] - prev_joint_vel[i, j]) / dt
    prev_joint_vel[i, j] = joint_vel[i, j]


@wp.kernel
def write_joint_limit_data_to_buffer(
    in_data: wp.array2d(dtype=wp.vec2f),
    soft_limit_factor: wp.float32,
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    from_mask: bool,
    joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    clamped_defaults: wp.array(dtype=wp.int32),
    sim_env_ids: wp.array(dtype=wp.int32),
):
    """Write joint limit data to the output buffers and compute soft limits.

    This kernel writes joint position limits from the input array to the output buffer,
    computes soft joint position limits, and clamps default joint positions if they
    fall outside the limits.

    Args:
        in_data: Input array containing joint position limits as vec2f (lower, upper).
            Shape is (num_envs, num_joints) or (num_selected_envs, num_selected_joints)
            depending on from_mask.
        soft_limit_factor: Input scalar factor for computing soft limits (typically 0.0-1.0).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        joint_ids: Input array of joint indices to write to. Shape is (num_selected_joints,).
        from_mask: Input flag indicating whether to use masked indexing. If True, indices from
            env_ids and joint_ids are used to index into in_data. If False, in_data is indexed
            directly using the thread indices.
        joint_pos_limits: Output array where joint position limits are written. Shape is
            (num_envs, num_joints).
        soft_joint_pos_limits: Output array where soft joint position limits are written.
            Shape is (num_envs, num_joints).
        default_joint_pos: Input/output array of default joint positions. If any values fall
            outside the limits, they are clamped. Shape is (num_envs, num_joints).
        clamped_defaults: Output 1-element array flag indicating whether any default joint
            positions were clamped. Non-zero if any clamping occurred. Shape is (1,).
    """
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    if j == 0:
        sim_env_ids[i] = env_id
    if from_mask:
        joint_pos_limits[env_id, joint_id] = in_data[env_id, joint_id]
    else:
        joint_pos_limits[env_id, joint_id] = in_data[i, j]
    if (default_joint_pos[env_id, joint_id] < joint_pos_limits[env_id, joint_id][0]) or default_joint_pos[
        env_id, joint_id
    ] > joint_pos_limits[env_id, joint_id][1]:
        wp.atomic_add(clamped_defaults, 0, 1)
        default_joint_pos[env_id, joint_id] = wp.clamp(
            default_joint_pos[env_id, joint_id],
            joint_pos_limits[env_id, joint_id][0],
            joint_pos_limits[env_id, joint_id][1],
        )
    soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(
        joint_pos_limits[env_id, joint_id], soft_limit_factor
    )


@wp.kernel
def write_joint_friction_data_to_buffer(
    in_friction: wp.array2d(dtype=wp.float32),
    in_dynamic_friction: wp.array2d(dtype=wp.float32),
    in_viscous_friction: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    from_mask: bool,
    out_friction: wp.array2d(dtype=wp.float32),
    out_dynamic_friction: wp.array2d(dtype=wp.float32),
    out_viscous_friction: wp.array2d(dtype=wp.float32),
    friction_props: wp.array3d(dtype=wp.float32),
    sim_env_ids: wp.array(dtype=wp.int32),
):
    """Write joint friction data to the output buffers.

    This kernel writes joint friction coefficients from input arrays to output buffers
    and updates the friction properties array used by the physics simulation.

    Args:
        in_friction: Input array containing joint friction coefficients. Shape is
            (num_envs, num_joints) or (num_selected_envs, num_selected_joints) depending
            on from_mask. Can be None if not provided.
        in_dynamic_friction: Input array containing joint dynamic friction coefficients.
            Shape is (num_envs, num_joints) or (num_selected_envs, num_selected_joints).
            Can be None if not provided.
        in_viscous_friction: Input array containing joint viscous friction coefficients.
            Shape is (num_envs, num_joints) or (num_selected_envs, num_selected_joints).
            Can be None if not provided.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        joint_ids: Input array of joint indices to write to. Shape is (num_selected_joints,).
        from_mask: Input flag indicating whether to use masked indexing. If True, indices from
            env_ids and joint_ids are used to index into input arrays. If False, input arrays
            are indexed directly using the thread indices.
        out_friction: Output array where joint friction coefficients are written. Shape is
            (num_envs, num_joints).
        out_dynamic_friction: Output array where joint dynamic friction coefficients are written.
            Shape is (num_envs, num_joints).
        out_viscous_friction: Output array where joint viscous friction coefficients are written.
            Shape is (num_envs, num_joints).
        friction_props: Output array where friction properties are written for the physics
            simulation. Shape is (num_envs, num_joints, 3) where the last dimension contains
            [friction, dynamic_friction, viscous_friction].
    """
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    if j == 0:
        sim_env_ids[i] = env_id
    # First update the output buffers
    if from_mask:
        out_friction[env_id, joint_id] = in_friction[env_id, joint_id]
        if in_dynamic_friction:
            out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[env_id, joint_id]
        if in_viscous_friction:
            out_viscous_friction[env_id, joint_id] = in_viscous_friction[env_id, joint_id]
    else:
        out_friction[env_id, joint_id] = in_friction[i, j]
        if in_dynamic_friction:
            out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[i, j]
        if in_viscous_friction:
            out_viscous_friction[env_id, joint_id] = in_viscous_friction[i, j]
    # Then update the friction properties
    friction_props[env_id, joint_id, 0] = out_friction[env_id, joint_id]
    if in_dynamic_friction:
        friction_props[env_id, joint_id, 1] = out_dynamic_friction[env_id, joint_id]
    if in_viscous_friction:
        friction_props[env_id, joint_id, 2] = out_viscous_friction[env_id, joint_id]


@wp.kernel
def write_joint_friction_param_to_buffer(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    buffer_index: wp.int32,
    from_mask: bool,
    out_data: wp.array2d(dtype=wp.float32),
    out_buffer: wp.array3d(dtype=wp.float32),
    sim_env_ids: wp.array(dtype=wp.int32),
):
    """Write a joint friction parameter to the output buffers.

    This kernel writes a single joint friction parameter (e.g., dynamic or viscous friction)
    from the input array to both a 2D output array and a specific slice of a 3D buffer array.

    Args:
        in_data: Input array containing joint friction parameter values. Shape is
            (num_envs, num_joints) or (num_selected_envs, num_selected_joints) depending
            on from_mask.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        joint_ids: Input array of joint indices to write to. Shape is (num_selected_joints,).
        buffer_index: Input scalar index specifying which slice of the 3D buffer to write to.
            Typically 0 for friction, 1 for dynamic friction, or 2 for viscous friction.
        from_mask: Input flag indicating whether to use masked indexing. If True, indices from
            env_ids and joint_ids are used to index into in_data. If False, in_data is indexed
            directly using the thread indices.
        out_data: Output array where friction parameter values are written. Shape is
            (num_envs, num_joints).
        out_buffer: Output 3D array where friction parameter values are written to the specified
            slice. Shape is (num_envs, num_joints, num_friction_params).
    """
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    if j == 0:
        sim_env_ids[i] = env_id
    if from_mask:
        out_data[env_id, joint_id] = in_data[env_id, joint_id]
        out_buffer[env_id, joint_id, buffer_index] = in_data[env_id, joint_id]
    else:
        out_data[env_id, joint_id] = in_data[i, j]
        out_buffer[env_id, joint_id, buffer_index] = in_data[i, j]


@wp.kernel
def float_data_to_buffer_with_indices(
    in_data: wp.float32,
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    out_data: wp.array2d(dtype=wp.float32),
):
    """Write a scalar float value to a 2D buffer at specified indices.

    This kernel broadcasts a single scalar float value to all specified (env_id, joint_id)
    locations in the output buffer.

    Args:
        in_data: Input scalar float value to broadcast.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        joint_ids: Input array of joint indices to write to. Shape is (num_selected_joints,).
        out_data: Output array where the scalar value is written. Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    out_data[env_id, joint_id] = in_data


@wp.kernel
def _float_data_to_buffer_with_indices_and_sim_ids(
    in_data: wp.float32,
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    out_data: wp.array2d(dtype=wp.float32),
    sim_env_ids: wp.array(dtype=wp.int32),
):
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    if j == 0:
        sim_env_ids[i] = env_id
    out_data[env_id, wp.int32(joint_ids[j])] = in_data


_WRITE_JOINT_LIMIT_DATA_TO_BUFFER_DISPATCHER = IndexKernelDispatcher(
    write_joint_limit_data_to_buffer, ("env_ids", "joint_ids")
)
_WRITE_JOINT_FRICTION_DATA_TO_BUFFER_DISPATCHER = IndexKernelDispatcher(
    write_joint_friction_data_to_buffer, ("env_ids", "joint_ids")
)
_WRITE_JOINT_FRICTION_PARAM_TO_BUFFER_DISPATCHER = IndexKernelDispatcher(
    write_joint_friction_param_to_buffer, ("env_ids", "joint_ids")
)
_FLOAT_DATA_TO_BUFFER_WITH_INDICES_DISPATCHER = IndexKernelDispatcher(
    float_data_to_buffer_with_indices, ("env_ids", "joint_ids")
)
_FLOAT_DATA_TO_BUFFER_WITH_INDICES_AND_SIM_IDS_DISPATCHER = IndexKernelDispatcher(
    _float_data_to_buffer_with_indices_and_sim_ids, ("env_ids", "joint_ids")
)


def write_joint_limit_data_to_buffer_kernel(
    env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the joint-limit writer for the selector dtypes."""
    return _WRITE_JOINT_LIMIT_DATA_TO_BUFFER_DISPATCHER.select(env_ids, joint_ids)


def write_joint_friction_data_to_buffer_kernel(
    env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the joint-friction writer for the selector dtypes."""
    return _WRITE_JOINT_FRICTION_DATA_TO_BUFFER_DISPATCHER.select(env_ids, joint_ids)


def write_joint_friction_param_to_buffer_kernel(
    env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the joint-friction parameter writer for the selector dtypes."""
    return _WRITE_JOINT_FRICTION_PARAM_TO_BUFFER_DISPATCHER.select(env_ids, joint_ids)


def float_data_to_buffer_with_indices_kernel(
    env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the scalar buffer writer for the selector dtypes."""
    return _FLOAT_DATA_TO_BUFFER_WITH_INDICES_DISPATCHER.select(env_ids, joint_ids)


def float_data_to_buffer_with_indices_and_sim_ids_kernel(
    env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select a scalar writer that also emits int32 environment indices."""
    return _FLOAT_DATA_TO_BUFFER_WITH_INDICES_AND_SIM_IDS_DISPATCHER.select(env_ids, joint_ids)


@wp.kernel
def update_soft_joint_pos_limits(
    joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_limit_factor: wp.float32,
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
):
    """Update soft joint position limits based on hard limits and a soft limit factor.

    This kernel computes soft joint position limits from hard joint position limits using
    a soft limit factor. Soft limits are typically used to provide a safety margin before
    reaching the hard limits.

    Args:
        joint_pos_limits: Input array of hard joint position limits as vec2f (lower, upper).
            Shape is (num_envs, num_joints).
        soft_limit_factor: Input scalar factor for computing soft limits (typically 0.0-1.0).
            A value of 1.0 means soft limits equal hard limits, while smaller values create
            a tighter range.
        soft_joint_pos_limits: Output array where soft joint position limits are written.
            Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(joint_pos_limits[i, j], soft_limit_factor)


@wp.kernel
def update_default_joint_values(
    source: wp.array(dtype=wp.float32),
    ids: wp.array(dtype=wp.int32),
    target: wp.array2d(dtype=wp.float32),
):
    """Update default joint values from a source array using joint indices.

    This kernel writes values from a 1D source array to specific joint indices in a 2D
    target array for all environments.

    Args:
        source: Input array containing joint values to write. Shape is (num_joints,).
        ids: Input array of joint indices specifying which joints to update. Shape is
            (num_selected_joints,).
        target: Output array where joint values are written. Shape is (num_envs, num_joints).
            Values are written to target[i, ids[j]] for all environments i.
    """
    i, j = wp.tid()
    target[i, ids[j]] = source[j]


@wp.kernel
def update_targets(
    source_joint_positions: wp.array2d(dtype=wp.float32),
    source_joint_velocities: wp.array2d(dtype=wp.float32),
    source_joint_efforts: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    target_joint_positions: wp.array2d(dtype=wp.float32),
    target_joint_velocities: wp.array2d(dtype=wp.float32),
    target_joint_efforts: wp.array2d(dtype=wp.float32),
):
    """Update joint target values from source arrays using joint indices.

    This kernel copies joint positions, velocities, and efforts from source arrays to
    target arrays, remapping joint indices using the provided joint_indices array.
    Only non-None source arrays are processed.

    Args:
        source_joint_positions: Input array of source joint positions. Shape is
            (num_envs, num_selected_joints). Can be None if not provided.
        source_joint_velocities: Input array of source joint velocities. Shape is
            (num_envs, num_selected_joints). Can be None if not provided.
        source_joint_efforts: Input array of source joint efforts. Shape is
            (num_envs, num_selected_joints). Can be None if not provided.
        joint_indices: Input array of joint indices for remapping. Shape is
            (num_selected_joints,). Specifies which joints in the target arrays to update.
        target_joint_positions: Output array where joint positions are written. Shape is
            (num_envs, num_joints).
        target_joint_velocities: Output array where joint velocities are written. Shape is
            (num_envs, num_joints).
        target_joint_efforts: Output array where joint efforts are written. Shape is
            (num_envs, num_joints).
    """
    i, j = wp.tid()
    if source_joint_positions:
        target_joint_positions[i, joint_indices[j]] = source_joint_positions[i, j]
    if source_joint_velocities:
        target_joint_velocities[i, joint_indices[j]] = source_joint_velocities[i, j]
    if source_joint_efforts:
        target_joint_efforts[i, joint_indices[j]] = source_joint_efforts[i, j]


@wp.kernel
def update_actuator_state_model(
    source_computed_effort: wp.array2d(dtype=wp.float32),
    source_applied_effort: wp.array2d(dtype=wp.float32),
    source_gear_ratio: wp.array2d(dtype=wp.float32),
    source_vel_limits: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    target_computed_effort: wp.array2d(dtype=wp.float32),
    target_applied_effort: wp.array2d(dtype=wp.float32),
    target_gear_ratio: wp.array2d(dtype=wp.float32),
    target_soft_joint_vel_limits: wp.array2d(dtype=wp.float32),
):
    """Update actuator state model parameters from source arrays using joint indices.

    This kernel copies actuator state model parameters (computed effort, applied effort,
    gear ratio, and velocity limits) from source arrays to target arrays, remapping
    joint indices using the provided joint_indices array.

    Args:
        source_computed_effort: Input array of source computed effort values. Shape is
            (num_envs, num_selected_joints).
        source_applied_effort: Input array of source applied effort values. Shape is
            (num_envs, num_selected_joints).
        source_gear_ratio: Input array of source gear ratio values. Shape is
            (num_envs, num_selected_joints). Can be None if not provided.
        source_vel_limits: Input array of source velocity limit values. Shape is
            (num_envs, num_selected_joints).
        joint_indices: Input array of joint indices for remapping. Shape is
            (num_selected_joints,). Specifies which joints in the target arrays to update.
        target_computed_effort: Output array where computed effort values are written.
            Shape is (num_envs, num_joints).
        target_applied_effort: Output array where applied effort values are written.
            Shape is (num_envs, num_joints).
        target_gear_ratio: Output array where gear ratio values are written. Shape is
            (num_envs, num_joints).
        target_soft_joint_vel_limits: Output array where soft joint velocity limits are
            written. Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    target_computed_effort[i, joint_indices[j]] = source_computed_effort[i, j]
    target_applied_effort[i, joint_indices[j]] = source_applied_effort[i, j]
    target_soft_joint_vel_limits[i, joint_indices[j]] = source_vel_limits[i, j]
    if source_gear_ratio:
        target_gear_ratio[i, joint_indices[j]] = source_gear_ratio[i, j]


@wp.kernel
def extract_friction_properties(
    friction_props: wp.array3d(dtype=wp.float32),
    out_friction: wp.array2d(dtype=wp.float32),
    out_dynamic_friction: wp.array2d(dtype=wp.float32),
    out_viscous_friction: wp.array2d(dtype=wp.float32),
):
    """Extract friction properties from a 3D array into separate 2D arrays.

    This kernel extracts the three friction components (static friction, dynamic friction,
    and viscous friction) from a 3D friction properties array into three separate 2D arrays.

    Args:
        friction_props: Input 3D array containing friction properties. Shape is
            (num_envs, num_joints, 3) where the last dimension contains
            [friction, dynamic_friction, viscous_friction].
        out_friction: Output array where static friction coefficients are written.
            Shape is (num_envs, num_joints).
        out_dynamic_friction: Output array where dynamic friction coefficients are written.
            Shape is (num_envs, num_joints).
        out_viscous_friction: Output array where viscous friction coefficients are written.
            Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    out_friction[i, j] = friction_props[i, j, 0]
    out_dynamic_friction[i, j] = friction_props[i, j, 1]
    out_viscous_friction[i, j] = friction_props[i, j, 2]


@wp.kernel
def shift_jacobian_com_to_origin(
    body_link_pose: wp.array2d(dtype=wp.transformf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    link_offset: wp.int32,
    src: wp.array4d(dtype=wp.float32),
    dst: wp.array4d(dtype=wp.float32),
):
    """Shift the linear-velocity rows of the Jacobian from COM to link origin.

    PhysX's ``ArticulationView.get_jacobians()`` returns ``J · q_dot = [v_com_world, omega_world]``
    per link — the linear rows are the velocity at the link's center of mass, expressed in
    world frame. The :attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w` contract
    requires the linear rows to be the velocity at the link **origin** (USD prim transform) so
    that ``J · q_dot[body_idx]`` matches
    :attr:`~isaaclab.assets.BaseArticulationData.body_link_lin_vel_w` /
    :attr:`~isaaclab.assets.BaseArticulationData.body_link_ang_vel_w`.

    The shift identity is the same one applied per-body by
    :func:`~isaaclab_physx.assets.kernels.get_link_vel_from_root_com_vel_func`, but layered onto
    every Jacobian column: each column represents the spatial velocity contribution of one DoF,
    and shifting a spatial velocity from COM to link origin uses ``v_origin = v_com - omega x
    (R · body_com_pos_b)``.

    Notes on layout:
        * Jacobian rows ``[0:3]`` are linear velocity, ``[3:6]`` are angular.
        * ``body_link_pose`` and ``body_com_pos_b`` are indexed by the articulation's full body
          count. PhysX's Jacobian rows are also indexed by the full body count for floating-base
          and exclude only the root for fixed-base, so ``link_offset = 1`` for fixed-base and
          ``link_offset = 0`` for floating-base, matching Newton's convention.

    Args:
        body_link_pose: Per-body link pose in world frame. Shape is (num_instances, num_bodies).
        body_com_pos_b: Per-body center-of-mass offset expressed in the body's link frame. Shape
            is (num_instances, num_bodies).
        link_offset: Offset added to the jacobian-row body index to reach the full body index.
            ``1`` for fixed-base, ``0`` for floating-base.
        src: COM-referenced Jacobian (read-only). Shape is (num_instances, num_jacobi_bodies, 6,
            num_joints + num_base_dofs).
        dst: Output buffer for the link-origin Jacobian. Same shape as ``src``. Linear rows
            ``[0:3]`` are written with the shifted velocity; angular rows ``[3:6]`` are copied
            unchanged (angular velocity is reference-point invariant).
    """
    n, b, dof = wp.tid()
    full_body_idx = b + link_offset

    R = wp.transform_get_rotation(body_link_pose[n, full_body_idx])
    c_world = wp.quat_rotate(R, body_com_pos_b[n, full_body_idx])

    v_com = wp.vec3(src[n, b, 0, dof], src[n, b, 1, dof], src[n, b, 2, dof])
    omega = wp.vec3(src[n, b, 3, dof], src[n, b, 4, dof], src[n, b, 5, dof])

    v_origin = v_com - wp.cross(omega, c_world)

    dst[n, b, 0, dof] = v_origin[0]
    dst[n, b, 1, dof] = v_origin[1]
    dst[n, b, 2, dof] = v_origin[2]
    dst[n, b, 3, dof] = omega[0]
    dst[n, b, 4, dof] = omega[1]
    dst[n, b, 5, dof] = omega[2]


vec6f = wp.types.vector(length=6, dtype=wp.float32)


@wp.kernel
def gather_stable_pd_mass_block(
    mass_matrix: wp.array3d(dtype=wp.float32),
    armature: wp.array2d(dtype=wp.float32),
    joints: wp.array(dtype=wp.int32),
    base_offset: wp.int32,
    out_block: wp.array3d(dtype=wp.float32),
):
    """Gather one StablePD actuator's symmetric mass sub-block, adding armature on the diagonal.

    Kernel counterpart of the torch slicing formerly in
    ``Articulation._feed_stable_pd_mass_matrix``: pure in-place launches so the feed can be
    captured into a CUDA graph. On a floating base the Schur reduction kernels below must run
    afterwards to eliminate the unactuated base block.

    Args:
        mass_matrix: PhysX generalized mass matrix. Shape is
            (num_instances, base_offset + num_joints, base_offset + num_joints).
        armature: Joint armature (reflected motor inertia) added to the diagonal. Shape is
            (num_instances, num_joints).
        joints: Articulation-local joint index of each actuator DOF, in the actuator's DOF order.
            Shape is (num_actuator_dofs,).
        base_offset: Root DOFs prepended by PhysX to the generalized rows/columns — ``6`` on a
            floating base, ``0`` otherwise.
        out_block: The controller state's mass matrix. Shape is
            (num_instances, num_actuator_dofs, num_actuator_dofs).
    """
    e, i, j = wp.tid()
    v = mass_matrix[e, base_offset + joints[i], base_offset + joints[j]]
    if i == j:
        v += armature[e, joints[i]]
    out_block[e, i, j] = v


@wp.kernel
def gather_stable_pd_bias(
    coriolis: wp.array2d(dtype=wp.float32),
    gravity: wp.array2d(dtype=wp.float32),
    add_gravity: wp.int32,
    joints: wp.array(dtype=wp.int32),
    base_offset: wp.int32,
    out_bias: wp.array2d(dtype=wp.float32),
):
    """Gather one StablePD actuator's RNEA bias rows: ``C(q, q̇) q̇`` plus optionally ``g(q)``.

    Args:
        coriolis: Coriolis/centrifugal compensation forces. Shape is
            (num_instances, base_offset + num_joints).
        gravity: Gravity compensation forces. Pass ``coriolis`` again when unused. Shape is
            (num_instances, base_offset + num_joints).
        add_gravity: ``1`` to fold gravity into the bias (``"bias"`` mode), ``0`` otherwise.
        joints: Articulation-local joint index of each actuator DOF. Shape is (num_actuator_dofs,).
        base_offset: Root DOFs prepended by PhysX — ``6`` on a floating base, ``0`` otherwise.
        out_bias: The controller state's bias forces. Shape is (num_instances, num_actuator_dofs).
    """
    e, i = wp.tid()
    v = coriolis[e, base_offset + joints[i]]
    if add_gravity != 0:
        v += gravity[e, base_offset + joints[i]]
    out_bias[e, i] = v


@wp.kernel
def gather_stable_pd_const_effort(
    gravity: wp.array2d(dtype=wp.float32),
    joints: wp.array(dtype=wp.int32),
    base_offset: wp.int32,
    out_flat: wp.array(dtype=wp.float32),
):
    """Gather full-weight gravity compensation into a controller's flat ``const_effort`` channel.

    Args:
        gravity: Gravity compensation forces. Shape is (num_instances, base_offset + num_joints).
        joints: Articulation-local joint index of each actuator DOF. Shape is (num_actuator_dofs,).
        base_offset: Root DOFs prepended by PhysX — ``6`` on a floating base, ``0`` otherwise.
        out_flat: The controller's ``const_effort``, env-major flat. Shape is
            (num_instances * num_actuator_dofs,).
    """
    e, i = wp.tid()
    out_flat[e * joints.shape[0] + i] = gravity[e, base_offset + joints[i]]


@wp.kernel
def stable_pd_schur_factor_base(
    mass_matrix: wp.array3d(dtype=wp.float32),
    L_bb: wp.array3d(dtype=wp.float32),
):
    """Cholesky-factorize each env's 6x6 floating-base block ``M_bb`` (rows/cols 0..5).

    The free-joint spatial inertia block is symmetric positive definite for any
    articulation with mass, so the factorization needs no pivoting or regularization.

    Args:
        mass_matrix: PhysX generalized mass matrix. Shape is
            (num_instances, 6 + num_joints, 6 + num_joints).
        L_bb: Output lower-triangular factors. Shape is (num_instances, 6, 6). Upper
            triangle is never read or written.
    """
    e = wp.tid()
    for j in range(6):
        s = mass_matrix[e, j, j]
        for k in range(j):
            s = s - L_bb[e, j, k] * L_bb[e, j, k]
        d = wp.sqrt(s)
        L_bb[e, j, j] = d
        for i in range(j + 1, 6):
            t = mass_matrix[e, i, j]
            for k in range(j):
                t = t - L_bb[e, i, k] * L_bb[e, j, k]
            L_bb[e, i, j] = t / d


@wp.kernel
def stable_pd_schur_solve_base(
    mass_matrix: wp.array3d(dtype=wp.float32),
    coriolis: wp.array2d(dtype=wp.float32),
    gravity: wp.array2d(dtype=wp.float32),
    base_has_gravity: wp.int32,
    joints: wp.array(dtype=wp.int32),
    L_bb: wp.array3d(dtype=wp.float32),
    Y: wp.array3d(dtype=wp.float32),
):
    """Solve ``M_bb @ y = rhs`` per (env, column) via the pre-factored ``L_bb``.

    Columns ``0..n-1`` use ``rhs = M[base_row, 6 + joints[c]]`` (base-to-actuated
    coupling, giving ``Y[:, :, :n] = M_bb^{-1} M_ba``); column ``n`` uses the base
    bias ``rhs = C_b (+ g_b)`` (giving ``Y[:, :, n] = M_bb^{-1} C_b``). The base rows
    keep their gravity term in every gravity mode — the base is unactuated and
    uncompensated, so the elimination must see its true load.

    Args:
        mass_matrix: PhysX generalized mass matrix. Shape is
            (num_instances, 6 + num_joints, 6 + num_joints).
        coriolis: Coriolis/centrifugal compensation forces. Shape is (num_instances, 6 + num_joints).
        gravity: Gravity compensation forces. Pass ``coriolis`` again when unused. Shape is
            (num_instances, 6 + num_joints).
        base_has_gravity: ``1`` to add gravity to the base bias rows, ``0`` when gravity is
            disabled on the robot.
        joints: Articulation-local joint index of each actuator DOF. Shape is (num_actuator_dofs,).
        L_bb: Lower-triangular Cholesky factors of ``M_bb``. Shape is (num_instances, 6, 6).
        Y: Output solutions. Shape is (num_instances, 6, num_actuator_dofs + 1).
    """
    e, c = wp.tid()
    n = joints.shape[0]
    rhs = vec6f()
    for i in range(6):
        if c < n:
            rhs[i] = mass_matrix[e, i, 6 + joints[c]]
        else:
            b = coriolis[e, i]
            if base_has_gravity != 0:
                b += gravity[e, i]
            rhs[i] = b
    y = vec6f()
    for i in range(6):
        s = rhs[i]
        for k in range(i):
            s = s - L_bb[e, i, k] * y[k]
        y[i] = s / L_bb[e, i, i]
    x = vec6f()
    for ii in range(6):
        i = 5 - ii
        s = y[i]
        for k in range(i + 1, 6):
            s = s - L_bb[e, k, i] * x[k]
        x[i] = s / L_bb[e, i, i]
    for i in range(6):
        Y[e, i, c] = x[i]


@wp.kernel
def stable_pd_schur_reduce_mass(
    mass_matrix: wp.array3d(dtype=wp.float32),
    joints: wp.array(dtype=wp.int32),
    Y: wp.array3d(dtype=wp.float32),
    out_block: wp.array3d(dtype=wp.float32),
):
    """In-place Schur update ``M_eff -= M_ab @ (M_bb^{-1} M_ba)`` on the gathered block.

    ``M_ab[i, b] = mass_matrix[e, 6 + joints[i], b]`` reads the symmetric base coupling
    directly; must run after :func:`gather_stable_pd_mass_block` and
    :func:`stable_pd_schur_solve_base`.

    Args:
        mass_matrix: PhysX generalized mass matrix. Shape is
            (num_instances, 6 + num_joints, 6 + num_joints).
        joints: Articulation-local joint index of each actuator DOF. Shape is (num_actuator_dofs,).
        Y: Base solves from :func:`stable_pd_schur_solve_base`. Shape is
            (num_instances, 6, num_actuator_dofs + 1).
        out_block: The controller state's mass matrix, updated in place. Shape is
            (num_instances, num_actuator_dofs, num_actuator_dofs).
    """
    e, i, j = wp.tid()
    s = float(0.0)
    for b in range(6):
        s = s + mass_matrix[e, 6 + joints[i], b] * Y[e, b, j]
    out_block[e, i, j] = out_block[e, i, j] - s


@wp.kernel
def stable_pd_schur_reduce_bias(
    mass_matrix: wp.array3d(dtype=wp.float32),
    joints: wp.array(dtype=wp.int32),
    Y: wp.array3d(dtype=wp.float32),
    out_bias: wp.array2d(dtype=wp.float32),
):
    """In-place Schur update ``bias_eff -= M_ab @ (M_bb^{-1} C_b)`` on the gathered bias.

    Args:
        mass_matrix: PhysX generalized mass matrix. Shape is
            (num_instances, 6 + num_joints, 6 + num_joints).
        joints: Articulation-local joint index of each actuator DOF. Shape is (num_actuator_dofs,).
        Y: Base solves; column ``num_actuator_dofs`` holds ``M_bb^{-1} C_b``. Shape is
            (num_instances, 6, num_actuator_dofs + 1).
        out_bias: The controller state's bias forces, updated in place. Shape is
            (num_instances, num_actuator_dofs).
    """
    e, k = wp.tid()
    n = joints.shape[0]
    s = float(0.0)
    for b in range(6):
        s = s + mass_matrix[e, 6 + joints[k], b] * Y[e, b, n]
    out_bias[e, k] = out_bias[e, k] - s


"""
Deprecated kernels.

These kernels are retained for backward compatibility and will be removed in a future release. Prefer the
public-order asset write APIs (:meth:`~isaaclab.assets.Articulation.write_joint_position_to_sim_index` and its
siblings), which apply the ordering conversion internally, or the public-order reorder kernels from
``isaaclab.assets.articulation.ordering_kernels``. Because Warp kernels cannot emit a runtime deprecation warning
without breaking :func:`warp.launch`, the deprecation is documented here and in the changelog rather than raised at
call time.
"""


@wp.kernel
def write_joint_vel_data(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    from_mask: bool,
    joint_vel: wp.array2d(dtype=wp.float32),
    prev_joint_vel: wp.array2d(dtype=wp.float32),
    joint_acc: wp.array2d(dtype=wp.float32),
):
    """Write joint velocity data to the output buffers.

    Deprecated. Prefer :meth:`~isaaclab.assets.Articulation.write_joint_velocity_to_sim_index` (or the
    ``ordering_kernels`` reorder family), which apply the public-to-backend ordering conversion internally.

    This kernel writes joint velocity data from the input array to the output buffers.
    It also updates the previous joint velocity buffer and resets the joint acceleration to 0.0.

    Args:
        in_data: Input array containing joint velocity data. Shape is (num_envs, num_joints) or
            (num_selected_envs, num_selected_joints) depending on from_mask.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        joint_ids: Input array of joint indices to write to. Shape is (num_selected_joints,).
        from_mask: Input flag indicating whether to use masked indexing. If True, indices from
            env_ids and joint_ids are used to index into in_data. If False, in_data is indexed
            directly using the thread indices.
        joint_vel: Output array where joint velocities are written. Shape is (num_envs, num_joints).
        prev_joint_vel: Output array where previous joint velocities are written. Shape is
            (num_envs, num_joints).
        joint_acc: Output array where joint accelerations are reset to 0.0. Shape is
            (num_envs, num_joints).
    """
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    if from_mask:
        joint_vel[env_id, joint_id] = in_data[env_id, joint_id]
        prev_joint_vel[env_id, joint_id] = in_data[env_id, joint_id]
    else:
        joint_vel[env_id, joint_id] = in_data[i, j]
        prev_joint_vel[env_id, joint_id] = in_data[i, j]
    joint_acc[env_id, joint_id] = 0.0


@wp.kernel
def write_joint_state_data(
    pos_data: wp.array2d(dtype=wp.float32),
    vel_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    full_data: bool,
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    prev_joint_vel: wp.array2d(dtype=wp.float32),
    joint_acc: wp.array2d(dtype=wp.float32),
):
    """Write joint position and velocity data in a single kernel launch.

    Deprecated. Prefer :meth:`~isaaclab.assets.Articulation.write_joint_position_to_sim_index` and its siblings
    (or the ``ordering_kernels`` reorder family), which apply the public-to-backend ordering conversion internally.

    Args:
        pos_data: Input joint positions. Shape is (num_envs, num_joints) if full_data,
            otherwise (num_selected_envs, num_selected_joints).
        vel_data: Input joint velocities. Shape is (num_envs, num_joints) if full_data,
            otherwise (num_selected_envs, num_selected_joints).
        env_ids: Environment indices. Shape is (num_selected_envs,).
        joint_ids: Joint indices. Shape is (num_selected_joints,).
        full_data: If True, data has full (num_envs, num_joints) shape and env_ids/joint_ids
            index into it. If False, data is pre-sliced and indexed by thread position.
        joint_pos: Output joint positions. Shape is (num_envs, num_joints).
        joint_vel: Output joint velocities. Shape is (num_envs, num_joints).
        prev_joint_vel: Output previous joint velocities. Shape is (num_envs, num_joints).
        joint_acc: Output joint accelerations (reset to 0). Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    if full_data:
        p = pos_data[env_id, joint_id]
        v = vel_data[env_id, joint_id]
    else:
        p = pos_data[i, j]
        v = vel_data[i, j]
    joint_pos[env_id, joint_id] = p
    joint_vel[env_id, joint_id] = v
    prev_joint_vel[env_id, joint_id] = v
    joint_acc[env_id, joint_id] = 0.0


_WRITE_JOINT_VEL_DATA_DISPATCHER = IndexKernelDispatcher(write_joint_vel_data, ("env_ids", "joint_ids"))
_WRITE_JOINT_STATE_DATA_DISPATCHER = IndexKernelDispatcher(write_joint_state_data, ("env_ids", "joint_ids"))


def write_joint_vel_data_kernel(env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the deprecated joint-velocity writer for the selector dtypes."""
    return _WRITE_JOINT_VEL_DATA_DISPATCHER.select(env_ids, joint_ids)


def write_joint_state_data_kernel(env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the deprecated joint-state writer for the selector dtypes."""
    return _WRITE_JOINT_STATE_DATA_DISPATCHER.select(env_ids, joint_ids)
