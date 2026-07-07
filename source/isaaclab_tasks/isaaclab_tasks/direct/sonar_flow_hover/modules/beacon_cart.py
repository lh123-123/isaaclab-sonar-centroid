# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Detect beacon / target position in Cartesian sonar images (no pose ground truth)."""

from __future__ import annotations

import numpy as np
import torch

from ..sparse_feature_tracker import SparseFeatureTracker


def teach_beacon_centroid(feature_tracker: SparseFeatureTracker, env_id: int = 0) -> tuple[float, float]:
    """Mean teach feature position in track-image pixels (setpoint for beacon servo)."""
    c = feature_tracker.ref_centroid()[env_id]
    return float(c[0].item()), float(c[1].item())


def tracked_beacon_centroid(feature_tracker: SparseFeatureTracker, env_id: int = 0) -> tuple[float, float] | None:
    """Tracked target centroid; None only when no taught features exist."""
    if not bool(feature_tracker._taught_mask()[env_id].any().item()):
        return None
    if not bool(feature_tracker.valid[env_id].any().item()):
        return None
    c = feature_tracker.tracked_centroid()[env_id]
    return float(c[0].item()), float(c[1].item())


def detect_beacon_peak(
    cart: np.ndarray,
    *,
    y_frac_min: float = 0.15,
    y_frac_max: float = 0.90,
    threshold: float = 0.10,
    min_pixels: int = 4,
    center_xy: tuple[float, float] | None = None,
    search_radius_px: float = 28.0,
) -> tuple[float, float] | None:
    """Bright-spot centroid in cart image (lateral x, forward y), pixels."""
    if cart.ndim != 2:
        return None
    h, w = cart.shape
    y0 = int(max(0, min(h - 1, round(h * y_frac_min))))
    y1 = int(max(y0 + 1, min(h, round(h * y_frac_max))))
    roi = np.asarray(cart[y0:y1, :], dtype=np.float32)
    if roi.size == 0:
        return None

    mask_roi = np.ones_like(roi, dtype=bool)
    if center_xy is not None:
        cx0, cy0 = center_xy
        yy, xx = np.mgrid[y0:y1, 0:w]
        dist2 = (xx - cx0) ** 2 + (yy - cy0) ** 2
        mask_roi = dist2 <= float(search_radius_px) ** 2

    peak = float(roi[mask_roi].max()) if bool(mask_roi.any()) else float(roi.max())
    if peak < threshold:
        return None
    thr = max(threshold, 0.35 * peak)
    mask = (roi >= thr) & mask_roi
    if int(mask.sum()) < min_pixels:
        mask = roi >= thr
        if int(mask.sum()) < min_pixels:
            return None
    ys, xs = np.nonzero(mask)
    weights = roi[mask]
    cx = float(np.average(xs, weights=weights))
    cy = float(np.average(ys, weights=weights)) + float(y0)
    return cx, cy


def detect_beacon_peak_polar(
    polar: np.ndarray,
    *,
    center_col_row: tuple[float, float] | None = None,
    search_radius_col: float = 14.0,
    search_radius_row: float = 18.0,
    threshold_ratio: float = 0.35,
    min_pixels: int = 4,
    exclude_boresight_col_half_width: int = 0,
) -> tuple[float, float] | None:
    """Bright-spot centroid on polar sonar (col=azimuth, row=range), pixels."""
    if polar.ndim != 2:
        return None
    h, w = polar.shape
    roi = np.asarray(polar, dtype=np.float32)
    if roi.size == 0:
        return None

    mask_roi = np.ones_like(roi, dtype=bool)
    if exclude_boresight_col_half_width > 0:
        cx = int(round((w - 1) * 0.5))
        hw = int(exclude_boresight_col_half_width)
        c0 = max(0, cx - hw)
        c1 = min(w, cx + hw + 1)
        mask_roi[:, c0:c1] = False
    if center_col_row is not None:
        cc, cr = center_col_row
        yy, xx = np.mgrid[0:h, 0:w]
        dc = (xx - cc) / max(float(search_radius_col), 1.0e-3)
        dr = (yy - cr) / max(float(search_radius_row), 1.0e-3)
        mask_roi = (dc * dc + dr * dr) <= 1.0

    peak = float(roi[mask_roi].max()) if bool(mask_roi.any()) else float(roi.max())
    if peak <= 1.0e-6:
        return None
    thr = max(peak * float(threshold_ratio), peak * 0.10)
    mask = (roi >= thr) & mask_roi
    if int(mask.sum()) < min_pixels:
        mask = roi >= thr
        if exclude_boresight_col_half_width > 0:
            mask = mask & mask_roi
        if int(mask.sum()) < min_pixels:
            return None
    rows, cols = np.nonzero(mask)
    weights = roi[mask]
    col = float(np.average(cols, weights=weights))
    row = float(np.average(rows, weights=weights))
    return col, row


def detect_beacon_peak_polar_global(
    polar: np.ndarray,
    *,
    threshold_ratio: float = 0.35,
    min_pixels: int = 4,
    exclude_boresight_col_half_width: int = 0,
) -> tuple[float, float] | None:
    """Bright-spot centroid over full polar frame (no teach-centered ROI)."""
    return detect_beacon_peak_polar(
        polar,
        center_col_row=None,
        threshold_ratio=threshold_ratio,
        min_pixels=min_pixels,
        exclude_boresight_col_half_width=exclude_boresight_col_half_width,
    )


def beacon_lateral_error_norm(
    beacon_x: float,
    setpoint_x: float,
    diagonal: float,
) -> float:
    """Signed normalized lateral pixel error (beacon - setpoint), same sign as mean_ex."""
    return (float(beacon_x) - float(setpoint_x)) / max(float(diagonal), 1.0e-3)


def estimate_beacon_x(
    feature_tracker: SparseFeatureTracker,
    cart_image: torch.Tensor,
    env_id: int = 0,
    *,
    ref_beacon_xy: tuple[float, float] | None = None,
    y_frac_min: float = 0.15,
    y_frac_max: float = 0.90,
    peak_threshold: float = 0.10,
    search_radius_px: float = 40.0,
) -> tuple[float, float, str]:
    """Fuse tracked centroid + local peak. Returns (beacon_x, beacon_y, source)."""
    cart_np = cart_image.detach().cpu().numpy()
    peak = detect_beacon_peak(
        cart_np,
        y_frac_min=y_frac_min,
        y_frac_max=y_frac_max,
        threshold=peak_threshold,
        center_xy=ref_beacon_xy,
        search_radius_px=search_radius_px,
    )
    if peak is not None:
        return peak[0], peak[1], "peak"

    tracked = tracked_beacon_centroid(feature_tracker, env_id)
    if tracked is not None:
        return tracked[0], tracked[1], "track"

    tx, ty = teach_beacon_centroid(feature_tracker, env_id)
    return tx, ty, "lost"
