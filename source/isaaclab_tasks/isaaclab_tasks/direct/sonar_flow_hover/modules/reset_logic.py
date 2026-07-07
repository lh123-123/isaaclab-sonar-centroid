# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Episode reset: beacon placement, robot spawn sampling."""

from __future__ import annotations

import math

import torch
from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_from_euler_xyz

from .geometry import quat_apply

from .reference_manager import ReferenceManager
from .sonar_pipeline import SonarPerceptionPipeline


def sample_spawn_offset(num_env_ids: int, radius: float, device: torch.device) -> torch.Tensor:
    if radius <= 0.0:
        return torch.zeros(num_env_ids, 3, device=device)
    xy = torch.randn(num_env_ids, 2, device=device)
    xy = xy / torch.norm(xy, dim=1, keepdim=True).clamp(min=1.0e-6)
    r = radius * torch.sqrt(torch.rand(num_env_ids, 1, device=device))
    offset = torch.zeros(num_env_ids, 3, device=device)
    offset[:, 0] = (xy[:, 0] * r).squeeze(-1)
    offset[:, 1] = (xy[:, 1] * r).squeeze(-1)
    z_jitter = float(radius) * 0.1
    offset[:, 2] = (torch.rand(num_env_ids, device=device) * 2 - 1) * z_jitter
    return offset


def sample_lateral_spawn_offset(
    num_env_ids: int,
    radius: float,
    yaw: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Sample spawn offset along body-Y only (lateral sway)."""
    if radius <= 0.0:
        return torch.zeros(num_env_ids, 3, device=device)
    dy = (torch.rand(num_env_ids, device=device) * 2.0 - 1.0) * float(radius)
    offset_b = torch.zeros(num_env_ids, 3, device=device)
    offset_b[:, 1] = dy
    zeros = torch.zeros(num_env_ids, device=device)
    quat = quat_from_euler_xyz(zeros, zeros, yaw)
    return quat_apply(quat, offset_b)


def write_beacon_pose(asset: RigidObject, pos_w: torch.Tensor, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    pose = torch.zeros(n, 7, device=pos_w.device)
    pose[:, :3] = pos_w
    pose[:, 3] = 1.0
    asset.write_root_pose_to_sim(pose.clone(), env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=pos_w.device), env_ids)


def reset_beacon_and_station(
    cfg,
    env_ids: torch.Tensor,
    scene,
    beacon: RigidObject,
    beacon2: RigidObject | None,
    *,
    station_origin_w: torch.Tensor,
    beacon_pos_w: torch.Tensor,
    beacon2_pos_w: torch.Tensor,
    beacon_vel_w: torch.Tensor,
    use_second_beacon: bool,
) -> None:
    origins = scene.env_origins[env_ids]
    origin_local = torch.tensor(cfg.station_origin_local, device=station_origin_w.device, dtype=torch.float32)
    beacon_local = torch.tensor(cfg.beacon_offset_local, device=station_origin_w.device, dtype=torch.float32)

    station_origin_w[env_ids] = origins + origin_local.unsqueeze(0)
    beacon_pos_w[env_ids] = station_origin_w[env_ids] + beacon_local.unsqueeze(0)
    write_beacon_pose(beacon, beacon_pos_w[env_ids], env_ids)

    if use_second_beacon and beacon2 is not None:
        beacon2_local = torch.tensor(
            getattr(cfg, "beacon2_offset_local", (3.0, 0.7, 0.0)),
            device=station_origin_w.device,
            dtype=torch.float32,
        )
        beacon2_pos_w[env_ids] = station_origin_w[env_ids] + beacon2_local.unsqueeze(0)
        write_beacon_pose(beacon2, beacon2_pos_w[env_ids], env_ids)

    target_speed = float(getattr(cfg, "target_speed_max", 0.0))
    lateral_1d = bool(getattr(cfg, "lateral_1d_mode", False))
    if target_speed > 0.0:
        if lateral_1d:
            beacon_vel_w[env_ids, 0] = 0.0
            beacon_vel_w[env_ids, 1] = (torch.rand(len(env_ids), device=station_origin_w.device) * 2.0 - 1.0) * target_speed
            beacon_vel_w[env_ids, 2] = 0.0
        else:
            angle = torch.rand(len(env_ids), device=station_origin_w.device) * 2.0 * math.pi
            beacon_vel_w[env_ids, 0] = target_speed * torch.cos(angle)
            beacon_vel_w[env_ids, 1] = target_speed * torch.sin(angle)
    else:
        beacon_vel_w[env_ids] = 0.0


reset_beacon_and_hover = reset_beacon_and_station


def reset_robot_episode(
    cfg,
    env_ids: torch.Tensor,
    robot: RigidObject,
    ref_mgr: ReferenceManager,
    pipeline: SonarPerceptionPipeline,
    term_checker,
    obs_builder,
    *,
    station_origin_w: torch.Tensor,
    beacon_pos_w: torch.Tensor,
    kinematic_offset_w: torch.Tensor,
    current_vel_w: torch.Tensor,
    actions: torch.Tensor,
    oceansim_sonar,
) -> None:
    kinematic_offset_w[env_ids] = 0.0

    n = len(env_ids)
    radius = float(getattr(cfg, "eval_spawn_radius", 0.25) if cfg.eval_mode else cfg.spawn_radius)
    lateral_1d = bool(getattr(cfg, "lateral_1d_mode", False))
    zeros = torch.zeros(n, device=robot.device)
    yaw = ref_mgr.spawn_yaw_for_reset(env_ids, station_origin_w[env_ids], station_origin_w, beacon_pos_w)
    if lateral_1d:
        offset = sample_lateral_spawn_offset(n, radius, yaw, robot.device)
    else:
        offset = sample_spawn_offset(n, radius, robot.device)
    spawn_pos = station_origin_w[env_ids] + offset

    quat = quat_from_euler_xyz(zeros, zeros, yaw)
    pose = torch.cat([spawn_pos, quat], dim=-1)

    robot.write_root_pose_to_sim(pose.clone(), env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=robot.device), env_ids)
    if hasattr(obs_builder, "reset_depth"):
        obs_builder.reset_depth(env_ids, spawn_pos[:, 2])

    if float(cfg.current_speed_max) > 0.0 and not lateral_1d:
        speed = float(cfg.current_speed_max)
        angle = torch.rand(n, device=robot.device) * 2.0 * math.pi
        current_vel_w[env_ids, 0] = speed * torch.cos(angle)
        current_vel_w[env_ids, 1] = speed * torch.sin(angle)
    else:
        current_vel_w[env_ids] = 0.0

    pipeline.reset(env_ids)
    term_checker.reset(env_ids)
    actions[env_ids] = 0.0

    if oceansim_sonar is not None:
        oceansim_sonar.reset_envs(env_ids)
