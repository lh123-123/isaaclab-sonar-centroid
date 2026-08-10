# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sonar (r, α) measurement for AUV tracking KF (validated centroid + belief gate)."""

from __future__ import annotations

import math

import torch

from isaaclab_tasks.direct.sonar_centroid_track.modules.coordinates import (
    polar_range_m_to_row,
    polar_row_to_range_m,
)
from isaaclab_tasks.direct.sonar_centroid_track.modules.sonar_gt import SonarGtDetector


def body_polar_from_xy(
    target_xy: torch.Tensor,
    agent_xy: torch.Tensor,
    agent_yaw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World XY → body-frame (azimuth, range)."""
    diff = target_xy - agent_xy
    cos_yaw = torch.cos(agent_yaw)
    sin_yaw = torch.sin(agent_yaw)
    dx = cos_yaw * diff[:, 0] + sin_yaw * diff[:, 1]
    dy = -sin_yaw * diff[:, 0] + cos_yaw * diff[:, 1]
    r = torch.sqrt(dx * dx + dy * dy + 1.0e-10)
    alpha = torch.atan2(dy, dx)
    return alpha, r


def azimuth_to_col(az_rad: float, width: int, hori_fov_rad: float) -> float:
    """Body azimuth → polar column (matches ``beacon_polar_rois``: +az → lower col)."""
    fov_deg = max(math.degrees(float(hori_fov_rad)), 1.0e-6)
    az_deg = math.degrees(float(az_rad))
    w = float(width)
    col = w / 2.0 - az_deg * w / fov_deg
    return max(0.0, min(w - 1.0, col))


def col_to_azimuth_rad(col: float, width: int, hori_fov_rad: float) -> float:
    """Polar column → body azimuth (inverse of ``azimuth_to_col``)."""
    fov_deg = max(math.degrees(float(hori_fov_rad)), 1.0e-6)
    w = float(width)
    az_deg = (w / 2.0 - float(col)) * fov_deg / w
    return math.radians(az_deg)


def detect_centroid_polar(
    polar_image: torch.Tensor,
    detector: SonarGtDetector,
    *,
    prior_azimuth: torch.Tensor | None,
    prior_range: torch.Tensor | None,
    prior_gate_deg: float,
    hori_fov_deg: float,
    min_range: float,
    max_range: float,
    range_window_m: float = 0.6,
    detect_max_range: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intensity-weighted bright-spot on preprocessed polar → (r, α, valid).

    ``prior_*`` gate the search ROI (belief / KF mean). If either is None, use global detect.
    ``detect_max_range`` masks polar rows beyond the reliable measurement band
    (defaults to ``max_range``).
    """
    n = polar_image.shape[0]
    h = int(polar_image.shape[1])
    w = int(polar_image.shape[2])
    device = polar_image.device
    fov = math.radians(float(hori_fov_deg))
    det_max = float(max_range if detect_max_range is None else detect_max_range)
    det_max = min(det_max, float(max_range))

    r_out = torch.zeros(n, device=device)
    a_out = torch.zeros(n, device=device)
    valid_out = torch.zeros(n, device=device)

    fov_deg = max(float(hori_fov_deg), 1.0e-3)
    search_col = max(float(prior_gate_deg) / fov_deg * (w - 1), 4.0)
    span = max(float(max_range) - float(min_range), 1.0e-3)
    search_row = max(float(range_window_m) / span * (h - 1), 6.0)
    # Zero rows beyond reliable band so global/ROI peak cannot latch far clutter.
    row_cut = int(
        math.floor(
            polar_range_m_to_row(det_max, min_range=min_range, max_range=max_range, polar_height=h)
        )
    )
    row_cut = max(1, min(h, row_cut + 1))

    for i in range(n):
        polar = polar_image[i].detach().cpu().numpy().copy()
        if row_cut < h:
            polar[row_cut:, :] = 0.0
        if prior_azimuth is not None and prior_range is not None:
            az = float(prior_azimuth[i].item())
            rng = float(prior_range[i].item())
            col = azimuth_to_col(az, w, fov)
            row = polar_range_m_to_row(rng, min_range=min_range, max_range=max_range, polar_height=h)
            pt = detector.detect_abs_roi(
                polar,
                (col, row),
                search_radius_col=search_col,
                search_radius_row=search_row,
            )
        else:
            pt = detector.detect_abs(polar)
        if pt is None:
            continue
        r_m = polar_row_to_range_m(pt.row, min_range=min_range, max_range=max_range, polar_height=h)
        if r_m > det_max + 1.0e-3:
            continue
        a_out[i] = col_to_azimuth_rad(pt.col, w, fov)
        r_out[i] = r_m
        valid_out[i] = 1.0
    return r_out, a_out, valid_out
