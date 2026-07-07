# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Beacon / sonar geometry helpers (pure torch, no sim dependency)."""

from __future__ import annotations

import math

import torch


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate 3-vectors by quaternions (w, x, y, z)."""
    q_w = quat[:, 0:1]
    q_vec = quat[:, 1:4]
    t = 2.0 * torch.cross(q_vec, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_vec, t, dim=-1)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Inverse rotation: q* v q."""
    q_w = quat[:, 0:1]
    q_vec = quat[:, 1:4]
    t = 2.0 * torch.cross(q_vec, vec, dim=-1)
    return vec - q_w * t + torch.cross(q_vec, t, dim=-1)


def sonar_pos_w(
    robot_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    sonar_mount_offset_b: torch.Tensor,
) -> torch.Tensor:
    offset_w = quat_apply(robot_quat_w, sonar_mount_offset_b)
    return robot_pos_w + offset_w


def beacon_bearing_range(
    beacon_pos_w: torch.Tensor,
    sonar_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rel_w = beacon_pos_w - sonar_pos_w
    rel_b = quat_apply_inverse(robot_quat_w, rel_w)
    bearing = torch.atan2(rel_b[:, 1], rel_b[:, 0])
    horiz_range = torch.sqrt(rel_b[:, 0] ** 2 + rel_b[:, 1] ** 2)
    return bearing, horiz_range


def yaw_toward_beacon(beacon_pos_w: torch.Tensor, from_pos_w: torch.Tensor) -> torch.Tensor:
    rel = beacon_pos_w - from_pos_w
    return torch.atan2(rel[:, 1], rel[:, 0])


def beacon_image_rois(
    bearing: torch.Tensor,
    horiz_range: torch.Tensor,
    env_ids: torch.Tensor,
    *,
    max_range: float,
    cart_image_size: int,
) -> list[tuple[float, float] | None]:
    """Project beacon bearing/range to Cartesian pixel coordinates."""
    rois: list[tuple[float, float] | None] = []
    size = float(cart_image_size)
    scale = size / (2.0 * max_range)
    for idx in env_ids.detach().cpu().tolist():
        b = float(bearing[idx].item())
        r = float(horiz_range[idx].item())
        x = r * math.sin(b)
        y = r * math.cos(b)
        px = (x + max_range) * scale
        py = (y + max_range) * scale
        if 0 <= px < size and 0 <= py < size:
            rois.append((px, py))
        else:
            rois.append(None)
    return rois


def beacon_polar_rois(
    bearing: torch.Tensor,
    horiz_range: torch.Tensor,
    env_ids: torch.Tensor,
    *,
    min_range: float,
    max_range: float,
    polar_height: int,
    polar_width: int,
    hori_fov_deg: float,
) -> list[tuple[float, float] | None]:
    """Project beacon bearing/range to polar sonar pixel (col=azimuth, row=range).

    Column mapping matches OceanSim / ``nonar2.modules.sonar_debug.peak_pixel_from_polar``:
    body +az (target left) → lower col; body −az (target right) → higher col.
    """
    rois: list[tuple[float, float] | None] = []
    fov_deg = max(float(hori_fov_deg), 1.0e-6)
    h_max = max(int(polar_height) - 1, 0)
    w_max = max(int(polar_width) - 1, 0)
    span = max(float(max_range) - float(min_range), 1.0e-6)
    for idx in env_ids.detach().cpu().tolist():
        az_deg = math.degrees(float(bearing[idx].item()))
        rng = float(horiz_range[idx].item())
        col = float(polar_width) / 2.0 - az_deg * float(polar_width) / fov_deg
        col = max(0.0, min(float(w_max), col))
        row = (rng - float(min_range)) / span * h_max
        row = max(0.0, min(float(h_max), row))
        if 0.0 <= col <= w_max and 0.0 <= row <= h_max:
            rois.append((col, row))
        else:
            rois.append(None)
    return rois


def geometric_beacon_col_px(
    *,
    beacon_pos_w: torch.Tensor,
    sonar_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    ref_col_px: float,
    min_range: float,
    max_range: float,
    polar_width: int,
    hori_fov_deg: float,
    env_id: int = 0,
) -> tuple[float, float]:
    """Project beacon to polar azimuth column (px) and return (col_px, delta_col_px vs teach)."""
    bearing, _rng = beacon_bearing_range(
        beacon_pos_w[env_id : env_id + 1],
        sonar_pos_w[env_id : env_id + 1],
        robot_quat_w[env_id : env_id + 1],
    )
    rois = beacon_polar_rois(
        bearing,
        _rng,
        torch.tensor([0], device=beacon_pos_w.device),
        min_range=min_range,
        max_range=max_range,
        polar_height=110,
        polar_width=polar_width,
        hori_fov_deg=hori_fov_deg,
    )
    col = float(rois[0][0]) if rois[0] is not None else float("nan")
    return col, col - float(ref_col_px)


def geometric_centroid_errors_px(
    feature_tracker,
    *,
    beacon_pos_w: torch.Tensor,
    robot_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    sonar_mount_offset_b: torch.Tensor,
    min_range: float,
    max_range: float,
    polar_height: int,
    polar_width: int,
    hori_fov_deg: float,
) -> torch.Tensor:
    """Teach vs geometry-projected beacon centroid offset in **pixels** (num_envs, 2)."""
    sonar_pw = sonar_pos_w(robot_pos_w, robot_quat_w, sonar_mount_offset_b)
    bearing, horiz_range = beacon_bearing_range(beacon_pos_w, sonar_pw, robot_quat_w)
    n = beacon_pos_w.shape[0]
    device = beacon_pos_w.device
    env_ids = torch.arange(n, device=device)
    rois = beacon_polar_rois(
        bearing,
        horiz_range,
        env_ids,
        min_range=min_range,
        max_range=max_range,
        polar_height=polar_height,
        polar_width=polar_width,
        hori_fov_deg=hori_fov_deg,
    )
    ref_c = feature_tracker.ref_centroid()
    out = torch.zeros(n, 2, device=device, dtype=ref_c.dtype)
    for i in range(n):
        if rois[i] is None:
            continue
        col, row = rois[i]
        out[i, 0] = float(col) - ref_c[i, 0]
        out[i, 1] = float(row) - ref_c[i, 1]
    return out


def geometric_centroid_errors_norm(
    feature_tracker,
    *,
    beacon_pos_w: torch.Tensor,
    robot_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    sonar_mount_offset_b: torch.Tensor,
    min_range: float,
    max_range: float,
    polar_height: int,
    polar_width: int,
    hori_fov_deg: float,
) -> torch.Tensor:
    """Normalized teach-to-beacon centroid error (legacy / debug)."""
    err_px = geometric_centroid_errors_px(
        feature_tracker,
        beacon_pos_w=beacon_pos_w,
        robot_pos_w=robot_pos_w,
        robot_quat_w=robot_quat_w,
        sonar_mount_offset_b=sonar_mount_offset_b,
        min_range=min_range,
        max_range=max_range,
        polar_height=polar_height,
        polar_width=polar_width,
        hori_fov_deg=hori_fov_deg,
    )
    lateral_norm = max(float(feature_tracker._lateral_norm), 1.0e-6)
    range_norm = max(float(feature_tracker._range_norm), 1.0e-6)
    scale = torch.tensor([lateral_norm, range_norm], device=err_px.device, dtype=err_px.dtype)
    return err_px / scale.unsqueeze(0)


def spawn_yaw(
    env_ids: torch.Tensor,
    spawn_pos: torch.Tensor,
    *,
    device: torch.device,
    use_file_reference: bool,
    spawn_use_teach_yaw: bool,
    file_ref_yaw_valid: bool,
    file_ref_yaw: torch.Tensor,
    spawn_yaw_range_deg: float,
    eval_mode: bool,
    station_origin_w: torch.Tensor,
    beacon_pos_w: torch.Tensor,
) -> torch.Tensor:
    if use_file_reference and spawn_use_teach_yaw:
        if file_ref_yaw_valid:
            yaw = file_ref_yaw[0].expand(len(env_ids)).clone()
            jitter = spawn_yaw_range_deg * math.pi / 180.0
            if jitter > 0 and not eval_mode:
                yaw = yaw + (torch.rand(len(env_ids), device=device) * 2 - 1) * jitter
            return yaw
        return yaw_toward_beacon(beacon_pos_w[env_ids], station_origin_w[env_ids])
    return yaw_toward_beacon(beacon_pos_w[env_ids], spawn_pos)


def station_pos_error_norm(station_origin_w: torch.Tensor, robot_pos_w: torch.Tensor) -> torch.Tensor:
    """Euclidean distance from robot to episode station origin (layout only)."""
    return torch.norm(station_origin_w - robot_pos_w, dim=1)


def robot_lateral_sway_signed(
    station_origin_w: torch.Tensor,
    robot_pos_w: torch.Tensor,
    yaw_w: torch.Tensor,
) -> torch.Tensor:
    """Signed body-Y sway relative to station origin (+ = robot left)."""
    delta = robot_pos_w - station_origin_w
    lateral_w = torch.stack([-torch.sin(yaw_w), torch.cos(yaw_w), torch.zeros_like(yaw_w)], dim=-1)
    return (delta * lateral_w).sum(dim=-1)


def robot_lateral_sway_abs(
    station_origin_w: torch.Tensor,
    robot_pos_w: torch.Tensor,
    yaw_w: torch.Tensor,
) -> torch.Tensor:
    return robot_lateral_sway_signed(station_origin_w, robot_pos_w, yaw_w).abs()


def target_lateral_offset_dy(target_kinematic_offset_b: torch.Tensor) -> torch.Tensor:
    """Target lateral offset in AUV body frame (m), from ``_target_kinematic_offset_b``."""
    return target_kinematic_offset_b[:, 1]


def sway_tracking_error_signed(
    station_origin_w: torch.Tensor,
    robot_pos_w: torch.Tensor,
    yaw_w: torch.Tensor,
    target_dy_b: torch.Tensor,
) -> torch.Tensor:
    """Robot sway minus target body-Y offset; 0 when lateral motion matches target."""
    return robot_lateral_sway_signed(station_origin_w, robot_pos_w, yaw_w) - target_dy_b


def sway_tracking_error_abs(
    station_origin_w: torch.Tensor,
    robot_pos_w: torch.Tensor,
    yaw_w: torch.Tensor,
    target_dy_b: torch.Tensor,
) -> torch.Tensor:
    return sway_tracking_error_signed(station_origin_w, robot_pos_w, yaw_w, target_dy_b).abs()


# Backward-compatible aliases (deprecated naming).
hover_pos_error_norm = station_pos_error_norm
hover_lateral_error_signed = robot_lateral_sway_signed
hover_lateral_error_abs = robot_lateral_sway_abs


def euler_xyz_from_quat(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """XYZ extrinsic Euler angles from quaternions (w, x, y, z). Pure torch."""
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.asin(torch.clamp(sin_pitch, -1.0, 1.0))
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)
    return roll, pitch, yaw
