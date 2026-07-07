# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AUV velocity control and target motion."""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from .geometry import quat_apply_inverse, yaw_toward_beacon


def apply_velocity_action(
    robot: RigidObject,
    actions: torch.Tensor,
    *,
    max_lin_vel: float,
    max_heave_vel: float,
    max_ang_vel: float,
    lateral_1d_only: bool = False,
) -> None:
    if lateral_1d_only:
        vx = torch.zeros(actions.shape[0], device=actions.device)
        vy = actions[:, 1] * max_lin_vel
        vz = torch.zeros(actions.shape[0], device=actions.device)
        wz = torch.zeros(actions.shape[0], device=actions.device)
    else:
        vx = actions[:, 0] * max_lin_vel
        vy = actions[:, 1] * max_lin_vel
        vz = actions[:, 2] * max_heave_vel
        wz = actions[:, 3] * max_ang_vel

    lin_vel_b = torch.stack([vx, vy, vz], dim=-1)
    lin_vel_w = quat_apply(robot.data.root_quat_w, lin_vel_b)
    ang_vel = torch.zeros(actions.shape[0], 3, device=actions.device)
    ang_vel[:, 2] = wz
    robot.write_root_velocity_to_sim(torch.cat([lin_vel_w, ang_vel], dim=-1))


def enforce_lateral_1d_kinematics(
    robot: RigidObject,
    station_origin_w: torch.Tensor,
    fixed_yaw_w: torch.Tensor,
    env_ids: torch.Tensor,
) -> None:
    """Keep surge/heave/yaw fixed; only body-frame lateral (sway) displacement is free."""
    n = len(env_ids)
    device = robot.device
    zeros = torch.zeros(n, device=device)
    yaw = fixed_yaw_w[env_ids]
    origin = station_origin_w[env_ids]
    pos = robot.data.root_pos_w[env_ids]
    delta = pos - origin
    lateral_w = torch.stack([-torch.sin(yaw), torch.cos(yaw), zeros], dim=-1)
    lateral_dist = (delta * lateral_w).sum(dim=-1, keepdim=True)
    pos_w = origin + lateral_dist * lateral_w
    quat = quat_from_euler_xyz(zeros, zeros, yaw)
    pose = torch.cat([pos_w, quat], dim=-1)
    robot.write_root_pose_to_sim(pose.clone(), env_ids)
    quat_wxyz = torch.cat([quat[:, 0:1], quat[:, 1:4]], dim=-1)
    vel_b = quat_apply_inverse(quat_wxyz, robot.data.root_lin_vel_w[env_ids])
    vel_b[:, 0] = 0.0
    vel_b[:, 2] = 0.0
    vel_w = quat_apply(quat_wxyz, vel_b)
    ang_vel = torch.zeros(n, 3, device=device)
    robot.write_root_velocity_to_sim(torch.cat([vel_w, ang_vel], dim=-1), env_ids)


def hold_robot_at_station(
    robot: RigidObject,
    station_origin_w: torch.Tensor,
    kinematic_offset_w: torch.Tensor,
    beacon_pos_w: torch.Tensor,
    env_ids: torch.Tensor,
    *,
    fixed_yaw_w: torch.Tensor | None = None,
) -> None:
    n = len(env_ids)
    device = robot.device
    zeros = torch.zeros(n, device=device)
    pos_w = station_origin_w[env_ids] + kinematic_offset_w[env_ids]
    if fixed_yaw_w is not None:
        yaw = fixed_yaw_w[env_ids]
    else:
        yaw = yaw_toward_beacon(beacon_pos_w[env_ids], pos_w)
    quat = quat_from_euler_xyz(zeros, zeros, yaw)
    pose = torch.cat([pos_w, quat], dim=-1)
    robot.write_root_pose_to_sim(pose.clone(), env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids)


hold_robot_at_hover = hold_robot_at_station


def apply_target_motion(
    beacon: RigidObject,
    beacon_pos_w: torch.Tensor,
    beacon_vel_w: torch.Tensor,
    *,
    target_speed_max: float,
    dt: float,
    all_env_ids: torch.Tensor,
) -> None:
    if target_speed_max <= 0.0:
        return
    beacon_pos_w.add_(beacon_vel_w * dt)
    n = beacon_pos_w.shape[0]
    pose = torch.zeros(n, 7, device=beacon_pos_w.device)
    pose[:, :3] = beacon_pos_w
    pose[:, 3] = 1.0
    beacon.write_root_pose_to_sim(pose.clone(), all_env_ids)
