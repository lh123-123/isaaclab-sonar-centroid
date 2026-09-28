# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Camera (r, α) measurement for AUV tracking KF.

S1: teacher-lock on the first photo at the known init bearing, then track that
appearance (NCC of the locked RGB patch) — not a hardcoded hue. Azimuth is
pinhole u→α; range prefers Isaac ``distance_to_image_plane`` on the match
(optical-axis Z → polar r = Z/cos α). Bbox apparent size is the fallback when
depth is missing. If NCC drops (scale change), reacquire the nearest depth hit
inside the KF azimuth gate.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

_DBG_LOG = "/home/lh/IsaacLab/.cursor/debug-4d71f8.log"


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    if os.environ.get("AUV_CAM_MEAS_DEBUG", "") != "1":
        return
    try:
        with open(_DBG_LOG, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "4d71f8",
                        "runId": "appear",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data,
                        "timestamp": int(time.time() * 1000),
                        "id": f"log_{int(time.time() * 1000)}_{hypothesis_id}",
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion

from .sonar_meas import azimuth_to_col, body_polar_from_xy, col_to_azimuth_rad

__all__ = [
    "body_polar_from_xy",
    "azimuth_to_col",
    "col_to_azimuth_rad",
    "focal_px",
    "bbox_height_to_range",
    "range_from_depth_blob",
    "range_from_depth_window",
    "extract_patch",
    "ncc",
    "orange_affinity",
    "azimuth_to_col_cam",
    "col_to_azimuth_cam",
    "teacher_lock_patch",
    "teacher_lock_rgb",
    "detect_centroid_luma",
    "detect_s1_luma",
    "detect_s1_rgb",
]

_PATCH = 16
# Default matches AUVTrackingEnvCfgV4Cam.target_tint (beacon orange).
_DEFAULT_TINT = (0.85, 0.38, 0.12)


def focal_px(width: int, hori_fov_deg: float) -> float:
    """Pinhole fx in pixels for a centred principal point."""
    return float(width) * 0.5 / math.tan(math.radians(float(hori_fov_deg)) * 0.5)


def bbox_height_to_range(h_px: float, *, f_px: float, length_m: float) -> float:
    """Coarse range from vertical bbox extent (known object length)."""
    h = max(float(h_px), 1.0e-3)
    return float(f_px) * float(length_m) / h


def range_from_depth_blob(
    depth_hw: np.ndarray,
    aff: np.ndarray,
    *,
    u_c: float,
    v_c: float,
    h_px: float,
    w_px: float,
    hori_fov_deg: float,
    min_z: float = 0.05,
    max_z: float = 12.0,
) -> float | None:
    """Planar range from Isaac ``distance_to_image_plane`` on the orange blob.

    The annotator is optical-axis Z, not polar r. For a pinhole ray at azimuth α,
    ``r = Z / cos(α)``. Median over the blob rejects seafloor speckle.
    """
    d = np.asarray(depth_hw, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    h, w = aff.shape
    if d.shape[:2] != (h, w):
        return None
    half_h = max(float(h_px) * 0.5, 2.0)
    half_w = max(float(w_px) * 0.5, 2.0)
    r0 = max(0, int(math.floor(v_c - half_h)))
    r1 = min(h, int(math.ceil(v_c + half_h)) + 1)
    c0 = max(0, int(math.floor(u_c - half_w)))
    c1 = min(w, int(math.ceil(u_c + half_w)) + 1)
    if r1 <= r0 or c1 <= c0:
        return None
    patch_aff = aff[r0:r1, c0:c1]
    patch_d = d[r0:r1, c0:c1]
    cut = max(0.25 * float(patch_aff.max()), 1.0)
    ys, xs = np.nonzero(patch_aff >= cut)
    if ys.size < 4:
        return None
    z = patch_d[ys, xs]
    uu = xs.astype(np.float64) + float(c0)
    ok = np.isfinite(z) & (z > float(min_z)) & (z < float(max_z))
    if int(ok.sum()) < 4:
        return None
    z = z[ok].astype(np.float64)
    uu = uu[ok]
    fx = max(focal_px(w, hori_fov_deg), 1.0e-6)
    cx = 0.5 * (float(w) - 1.0)
    az = np.arctan((cx - uu) / fx)
    r = z / np.clip(np.cos(az), 0.2, None)
    return float(np.median(r))


def range_from_depth_window(
    depth_hw: np.ndarray,
    *,
    u_c: float,
    v_c: float,
    h_px: float,
    w_px: float,
    hori_fov_deg: float,
    min_z: float = 0.05,
    max_z: float = 12.0,
) -> float | None:
    """Planar range from optical-axis Z in a box around (u, v). Colour-agnostic."""
    d = np.asarray(depth_hw, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    h, w = d.shape[:2]
    half_h = max(float(h_px) * 0.5, 2.0)
    half_w = max(float(w_px) * 0.5, 2.0)
    r0 = max(0, int(math.floor(v_c - half_h)))
    r1 = min(h, int(math.ceil(v_c + half_h)) + 1)
    c0 = max(0, int(math.floor(u_c - half_w)))
    c1 = min(w, int(math.ceil(u_c + half_w)) + 1)
    if r1 <= r0 or c1 <= c0:
        return None
    patch_d = d[r0:r1, c0:c1]
    z = patch_d.reshape(-1)
    uu = np.tile(np.arange(c0, c1, dtype=np.float64), r1 - r0)
    ok = np.isfinite(z) & (z > float(min_z)) & (z < float(max_z))
    if int(ok.sum()) < 4:
        return None
    z = z[ok].astype(np.float64)
    uu = uu[ok]
    fx = max(focal_px(w, hori_fov_deg), 1.0e-6)
    cx = 0.5 * (float(w) - 1.0)
    az = np.arctan((cx - uu) / fx)
    r = z / np.clip(np.cos(az), 0.2, None)
    return float(np.median(r))


def _nearest_depth_uv(
    depth_hw: np.ndarray,
    *,
    u_center: float,
    col_half: float,
    min_z: float = 0.3,
    max_z: float = 8.0,
    v_center: float | None = None,
    row_half: float | None = None,
) -> tuple[float, float, float] | None:
    """Closest valid depth pixel in a column strip (optional row band)."""
    d = np.asarray(depth_hw, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    h, w = d.shape[:2]
    c0 = max(0, int(math.floor(u_center - col_half)))
    c1 = min(w, int(math.ceil(u_center + col_half)) + 1)
    r0, r1 = 0, h
    if v_center is not None and row_half is not None:
        r0 = max(0, int(math.floor(v_center - row_half)))
        r1 = min(h, int(math.ceil(v_center + row_half)) + 1)
    if c1 <= c0 or r1 <= r0:
        return None
    patch = d[r0:r1, c0:c1]
    ok = np.isfinite(patch) & (patch > float(min_z)) & (patch < float(max_z))
    if not bool(ok.any()):
        return None
    masked = np.where(ok, patch, np.inf)
    iy, ix = np.unravel_index(int(np.argmin(masked)), masked.shape)
    z = float(patch[iy, ix])
    if not math.isfinite(z):
        return None
    return float(ix + c0), float(iy + r0), z


def _scale_tmpl(tmpl: np.ndarray, scale: float) -> np.ndarray:
    t = np.asarray(tmpl)
    if abs(float(scale) - 1.0) < 0.05:
        return t
    h, w = t.shape[:2]
    nh = max(8, int(round(h * float(scale))))
    nw = max(8, int(round(w * float(scale))))
    if nh % 2:
        nh += 1
    if nw % 2:
        nw += 1
    try:
        import cv2

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(t, (nw, nh), interpolation=interp)
    except Exception:
        ys = np.linspace(0, h - 1, nh).round().astype(int)
        xs = np.linspace(0, w - 1, nw).round().astype(int)
        if t.ndim == 3:
            return t[ys][:, xs]
        return t[np.ix_(ys, xs)]


def _ncc_best_in_roi(
    image: np.ndarray,
    tmpl: np.ndarray,
    *,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> tuple[float, float, float]:
    """Best ZNCC of ``tmpl`` in ROI → (score, u_center, v_center). Fast path: OpenCV."""
    img = np.asarray(image)
    tmpl = np.asarray(tmpl)
    th, tw = int(tmpl.shape[0]), int(tmpl.shape[1])
    h, w = img.shape[:2]
    if th < 4 or tw < 4 or th > h or tw > w:
        return -1.0, 0.5 * (c0 + c1), 0.5 * (r0 + r1)
    hh, hw = th // 2, tw // 2
    # Expand ROI so matchTemplate peak maps back to image centre coords.
    rr0 = max(0, int(r0) - hh)
    rr1 = min(h, int(r1) + hh)
    cc0 = max(0, int(c0) - hw)
    cc1 = min(w, int(c1) + hw)
    if rr1 - rr0 < th or cc1 - cc0 < tw:
        return -1.0, 0.5 * (c0 + c1), 0.5 * (r0 + r1)
    roi = img[rr0:rr1, cc0:cc1]
    try:
        import cv2

        # uint8 keeps matchTemplate in the fast SIMD path.
        if img.dtype != np.uint8:
            roi_u = np.clip(roi, 0, 255).astype(np.uint8)
            tm_u = np.clip(tmpl, 0, 255).astype(np.uint8)
        else:
            roi_u = roi
            tm_u = tmpl if tmpl.dtype == np.uint8 else np.clip(tmpl, 0, 255).astype(np.uint8)
        res = cv2.matchTemplate(roi_u, tm_u, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        u = float(max_loc[0] + cc0 + hw)
        v = float(max_loc[1] + rr0 + hh)
        return float(max_val), u, v
    except Exception:
        # Slow fallback (should rarely hit).
        smap = _ncc_score_map_slow(img.astype(np.float32), tmpl.astype(np.float32), r0=r0, r1=r1, c0=c0, c1=c1)
        pk = float(smap.max()) if smap.size else -1.0
        vi, ui = np.unravel_index(int(np.argmax(smap)), smap.shape)
        return pk, float(ui), float(vi)


def _ncc_score_map_slow(
    image: np.ndarray,
    tmpl: np.ndarray,
    *,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    stride: int = 4,
) -> np.ndarray:
    """ZNCC fallback (Python loops). Prefer ``_ncc_best_in_roi``."""
    img = np.asarray(image, dtype=np.float32)
    tmpl = np.asarray(tmpl, dtype=np.float32)
    h, w = img.shape[:2]
    scores = np.zeros((h, w), dtype=np.float32)
    th, tw = int(tmpl.shape[0]), int(tmpl.shape[1])
    if th < 4 or tw < 4 or th > h or tw > w:
        return scores
    hh, hw = th // 2, tw // 2
    t = tmpl.reshape(-1).astype(np.float64)
    t = t - t.mean()
    dt = float(np.sqrt((t * t).sum()))
    if dt < 1.0e-6:
        return scores
    vr0 = max(int(r0), hh)
    vr1 = min(int(r1), h - hh)
    uc0 = max(int(c0), hw)
    uc1 = min(int(c1), w - hw)
    st = max(int(stride), 1)
    for v in range(vr0, vr1, st):
        for u in range(uc0, uc1, st):
            patch = img[v - hh : v - hh + th, u - hw : u - hw + tw]
            if patch.shape[:2] != (th, tw):
                continue
            a = patch.reshape(-1).astype(np.float64)
            a = a - a.mean()
            da = float(np.sqrt((a * a).sum()))
            if da < 1.0e-6:
                continue
            scores[v, u] = float((a * t).sum() / (da * dt))
    return scores


# Back-compat alias (debug / old call sites).
def _ncc_score_map(*args, **kwargs):
    return _ncc_score_map_slow(*args, **kwargs)


def _luma_u8(luma: np.ndarray) -> np.ndarray:
    x = np.asarray(luma)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.float32)
        if float(x.max()) <= 1.5:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def _rgb_u8(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb)
    if x.ndim != 3 or x.shape[-1] < 3:
        raise ValueError(f"expected HxWx3 RGB, got shape={getattr(x, 'shape', None)}")
    x = x[..., :3]
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        if float(x.max()) <= 1.5:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def orange_affinity(rgb: np.ndarray, tint: tuple[float, float, float] = _DEFAULT_TINT) -> np.ndarray:
    """Per-pixel score for orange tint vs blue-green water / grey clutter.

    Uses red-excess over G/B (illumination-tolerant) gated by alignment with ``tint``.
    Returns float32 HxW, higher = more target-like.
    """
    rgb = _rgb_u8(rgb).astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # Red excess: orange/red vs cyan-ish water and grey hulls.
    excess = np.maximum(r - np.maximum(g, b), 0.0)
    # Soft tint direction gate (tint is unit-ish in [0,1]).
    tr, tg, tb = (float(tint[0]), float(tint[1]), float(tint[2]))
    tn = math.sqrt(tr * tr + tg * tg + tb * tb) + 1.0e-6
    rn = np.sqrt(r * r + g * g + b * b) + 1.0e-6
    align = (r * tr + g * tg + b * tb) / (rn * tn)
    align = np.clip(align, 0.0, 1.0)
    return (excess * align).astype(np.float32)


def extract_patch(img: np.ndarray, u: float, v: float, size: int = _PATCH) -> np.ndarray | None:
    """Centered ``size×size`` patch (HxW or HxWxC). Clips the centre in-bounds."""
    x = np.asarray(img)
    h, w = x.shape[:2]
    half = int(size) // 2
    if h < int(size) or w < int(size):
        return None
    ui = int(np.clip(round(u), half, w - half - 1))
    vi = int(np.clip(round(v), half, h - half - 1))
    return x[vi - half : vi + half, ui - half : ui + half].astype(np.float32)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation in [-1, 1]."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size != b.size or a.size == 0:
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    da = float(np.sqrt((a * a).sum()))
    db = float(np.sqrt((b * b).sum()))
    if da < 1.0e-6 or db < 1.0e-6:
        return -1.0
    return float((a * b).sum() / (da * db))


def _score_blob(
    score: np.ndarray,
    *,
    min_area: int,
    center: tuple[float, float] | None,
    search_radius_col: float,
    search_radius_row: float,
    thr: float = 0.0,
) -> tuple[float, float, float, float, float] | None:
    """Positive score map → intensity-weighted centroid + bbox (O(HW))."""
    h, w = score.shape
    work = score
    u0 = v0 = 0
    if center is not None:
        cu, cv = float(center[0]), float(center[1])
        c0 = max(0, int(math.floor(cu - search_radius_col)))
        c1 = min(w, int(math.ceil(cu + search_radius_col)) + 1)
        r0 = max(0, int(math.floor(cv - search_radius_row)))
        r1 = min(h, int(math.ceil(cv + search_radius_row)) + 1)
        if c1 <= c0 or r1 <= r0:
            return None
        work = score[r0:r1, c0:c1]
        u0, v0 = c0, r0
        if work.size == 0:
            return None

    mass = np.maximum(work.astype(np.float64) - float(thr), 0.0)
    area = int((mass > 0.0).sum())
    if area < int(min_area):
        return None
    wsum = float(mass.sum())
    if wsum <= 1.0e-9:
        return None
    yy, xx = np.indices(work.shape, dtype=np.float64)
    v_c = float((yy * mass).sum() / wsum) + v0
    u_c = float((xx * mass).sum() / wsum) + u0
    cut = 0.25 * float(mass.max())
    ys, xs = np.nonzero(mass >= cut)
    if ys.size == 0:
        return None
    h_px = float(ys.max() - ys.min() + 1)
    w_px = float(xs.max() - xs.min() + 1)
    peak = float(wsum / max(area, 1))
    return u_c, v_c, h_px, w_px, peak


def _dark_blob(
    luma: np.ndarray,
    *,
    thr: int,
    min_area: int,
    center: tuple[float, float] | None,
    search_radius_col: float,
    search_radius_row: float,
) -> tuple[float, float, float, float, float] | None:
    """Dark pixels → intensity-weighted centroid + bbox (O(HW)). Legacy luma path."""
    dark = np.maximum(float(thr) - luma.astype(np.float64), 0.0).astype(np.float32)
    return _score_blob(
        dark,
        min_area=min_area,
        center=center,
        search_radius_col=search_radius_col,
        search_radius_row=search_radius_row,
        thr=0.0,
    )


def teacher_lock_patch(
    luma: np.ndarray,
    *,
    alpha_gt_rad: float,
    hori_fov_deg: float,
    dark_thr: int = 200,
    patch_size: int = _PATCH,
) -> tuple[np.ndarray, float, float] | None:
    """Sim teacher-lock (legacy luma): dark blob near GT azimuth → (template, u, v)."""
    luma = _luma_u8(luma)
    h, w = luma.shape
    fov = math.radians(float(hori_fov_deg))
    u_gt = azimuth_to_col(alpha_gt_rad, w, fov)
    v_gt = 0.5 * (h - 1)
    # Small window around GT bearing — avoid locking onto seafloor elsewhere.
    gate_col = max(0.08 * (w - 1), 4.0)
    gate_row = max(0.25 * (h - 1), 8.0)
    pt = _dark_blob(
        luma,
        thr=dark_thr,
        min_area=4,
        center=(u_gt, v_gt),
        search_radius_col=gate_col,
        search_radius_row=gate_row,
    )
    if pt is None:
        u, v = u_gt, v_gt
    else:
        u, v = pt[0], pt[1]
    patch = extract_patch(luma, u, v, patch_size)
    if patch is None:
        return None
    return patch, float(u), float(v)


def azimuth_to_col_cam(az_rad: float, width: int, hori_fov_deg: float) -> float:
    """Pinhole: +azimuth (left) → smaller column. Matches camera, not sonar linear polar."""
    w = float(width)
    fx = focal_px(int(width), hori_fov_deg)
    cx = 0.5 * (w - 1.0)
    col = cx - fx * math.tan(float(az_rad))
    return max(0.0, min(w - 1.0, col))


def col_to_azimuth_cam(col: float, width: int, hori_fov_deg: float) -> float:
    """Inverse of ``azimuth_to_col_cam``."""
    w = float(width)
    fx = max(focal_px(int(width), hori_fov_deg), 1.0e-6)
    cx = 0.5 * (w - 1.0)
    return math.atan((cx - float(col)) / fx)


def teacher_lock_rgb(
    rgb: np.ndarray,
    *,
    alpha_gt_rad: float,
    hori_fov_deg: float,
    tint: tuple[float, float, float] = _DEFAULT_TINT,
    patch_size: int = _PATCH,
    min_area: int = 4,
    min_aff: float = 8.0,
    depth: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float] | None:
    """Lock appearance from the first photo at the known init bearing.

    Init a2t already sits in the camera lobe, so GT azimuth is the correct track
    location. The near-object blob in that column (inverse depth, colour-agnostic)
    is the template. ``tint`` is unused. Returns None on a blank first render.
    """
    del tint, min_area, min_aff
    rgb = _rgb_u8(rgb)
    h, w = rgb.shape[:2]
    u_gt = azimuth_to_col_cam(alpha_gt_rad, w, hori_fov_deg)
    v_mid = 0.5 * (h - 1)
    gate_col = max(0.08 * (w - 1), 4.0)
    gate_row = max(0.40 * (h - 1), 8.0)
    if depth is None:
        return None
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if d.shape[:2] != (h, w):
        return None
    near = np.zeros((h, w), dtype=np.float32)
    ok = np.isfinite(d) & (d > 0.3) & (d < 6.0)
    if not bool(ok.any()):
        # #region agent log
        _agent_log(
            "H1",
            "cam_meas.py:teacher_lock_rgb",
            "lock_miss",
            {"u_gt": round(float(u_gt), 1), "reason": "no_depth_in_strip"},
        )
        # #endregion
        return None
    near[ok] = (6.0 - d[ok]).astype(np.float32)
    pt = _score_blob(
        near,
        min_area=4,
        center=(u_gt, v_mid),
        search_radius_col=gate_col,
        search_radius_row=gate_row,
        thr=0.5,
    )
    if pt is None:
        hit = _nearest_depth_uv(d, u_center=u_gt, col_half=gate_col, min_z=0.3, max_z=6.0)
        if hit is None:
            # #region agent log
            _agent_log(
                "H1",
                "cam_meas.py:teacher_lock_rgb",
                "lock_miss",
                {"u_gt": round(float(u_gt), 1), "reason": "no_near_blob"},
            )
            # #endregion
            return None
        u, v, z = hit
    else:
        u, v = float(pt[0]), float(pt[1])
        ui = int(np.clip(round(u), 0, w - 1))
        vi = int(np.clip(round(v), 0, h - 1))
        z = float(d[vi, ui]) if np.isfinite(d[vi, ui]) else -1.0
    patch = extract_patch(rgb, u, v, patch_size)
    if patch is None:
        return None
    # #region agent log
    _agent_log(
        "H1",
        "cam_meas.py:teacher_lock_rgb",
        "lock_ok",
        {
            "u": round(float(u), 1),
            "v": round(float(v), 1),
            "u_gt": round(float(u_gt), 1),
            "du_px": round(float(u) - float(u_gt), 1),
            "z": round(float(z), 3),
            "patch_shape": list(patch.shape),
            "patch_mean": round(float(patch.mean()), 2),
        },
    )
    # #endregion
    return patch, float(u), float(v)


def detect_centroid_luma(
    luma_image: torch.Tensor | np.ndarray,
    *,
    prior_azimuth: torch.Tensor | None,
    prior_range: torch.Tensor | None,
    prior_gate_deg: float,
    hori_fov_deg: float,
    length_m: float,
    min_range: float = 0.5,
    max_range: float = 4.0,
    range_window_m: float = 1.0,
    dark_thr: int = 200,
    min_area: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy dark-blob → (r, α, valid) with bbox range (no appearance). Prefer ``detect_s1_luma``."""
    if isinstance(luma_image, torch.Tensor):
        arr = luma_image.detach().cpu().numpy()
        device = luma_image.device
        n = int(luma_image.shape[0])
    else:
        arr = np.asarray(luma_image)
        if arr.ndim == 2:
            arr = arr[None, ...]
        device = torch.device("cpu")
        n = int(arr.shape[0])

    h = int(arr.shape[1])
    w = int(arr.shape[2])
    fov = math.radians(float(hori_fov_deg))
    f_px = focal_px(w, hori_fov_deg)
    mid_r = 0.5 * (float(min_range) + float(max_range))
    search_row = max(float(range_window_m) / max(mid_r, 1.0e-3) * f_px, 8.0)
    fov_deg = max(float(hori_fov_deg), 1.0e-3)
    search_col = max(float(prior_gate_deg) / fov_deg * (w - 1), 4.0)

    r_out = torch.zeros(n, device=device)
    a_out = torch.zeros(n, device=device)
    valid_out = torch.zeros(n, device=device)

    for i in range(n):
        luma = _luma_u8(arr[i])
        center = None
        if prior_azimuth is not None and prior_range is not None:
            az = float(prior_azimuth[i].item())
            col = azimuth_to_col(az, w, fov)
            center = (col, 0.5 * (h - 1))
            pt = _dark_blob(
                luma,
                thr=dark_thr,
                min_area=min_area,
                center=center,
                search_radius_col=search_col,
                search_radius_row=search_row,
            )
        else:
            pt = _dark_blob(
                luma,
                thr=dark_thr,
                min_area=min_area,
                center=None,
                search_radius_col=0.0,
                search_radius_row=0.0,
            )
        if pt is None:
            continue
        u_c, _v_c, h_px, _w_px, _score = pt
        r_m = bbox_height_to_range(h_px, f_px=f_px, length_m=length_m)
        if r_m < float(min_range) - 1.0e-3 or r_m > float(max_range) + 1.0e-3:
            continue
        a_out[i] = col_to_azimuth_rad(u_c, w, fov)
        r_out[i] = r_m
        valid_out[i] = 1.0
    return r_out, a_out, valid_out


def detect_s1_luma(
    luma_image: torch.Tensor | np.ndarray,
    *,
    templates: list[np.ndarray | None],
    prior_azimuth: torch.Tensor,
    prior_range: torch.Tensor,
    prior_gate_deg: float,
    hori_fov_deg: float,
    ncc_tau: float = 0.45,
    dark_thr: int = 200,
    min_area: int = 4,
    patch_size: int = _PATCH,
    alpha_only: bool = True,
    length_m: float = 0.55,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """S1: motion-gated dark blob + NCC vs teacher template → (r, α, valid, ncc).

    ``alpha_only=True`` (default): ``r`` is copied from ``prior_range`` (belief), only α is
    measured from the image — the safer replace-GT first step.
    """
    if isinstance(luma_image, torch.Tensor):
        arr = luma_image.detach().cpu().numpy()
        device = luma_image.device
        n = int(luma_image.shape[0])
    else:
        arr = np.asarray(luma_image)
        if arr.ndim == 2:
            arr = arr[None, ...]
        device = torch.device("cpu")
        n = int(arr.shape[0])

    h = int(arr.shape[1])
    w = int(arr.shape[2])
    fov = math.radians(float(hori_fov_deg))
    f_px = focal_px(w, hori_fov_deg)
    fov_deg = max(float(hori_fov_deg), 1.0e-3)
    search_col = max(float(prior_gate_deg) / fov_deg * (w - 1), 4.0)
    search_row = max(0.30 * (h - 1), 8.0)

    r_out = torch.zeros(n, device=device)
    a_out = torch.zeros(n, device=device)
    valid_out = torch.zeros(n, device=device)
    ncc_out = torch.full((n,), -1.0, device=device)

    for i in range(n):
        tmpl = templates[i] if i < len(templates) else None
        if tmpl is None:
            continue
        luma = _luma_u8(arr[i])
        az = float(prior_azimuth[i].item())
        rng = float(prior_range[i].item())
        col = azimuth_to_col(az, w, fov)
        pt = _dark_blob(
            luma,
            thr=dark_thr,
            min_area=min_area,
            center=(col, 0.5 * (h - 1)),
            search_radius_col=search_col,
            search_radius_row=search_row,
        )
        if pt is None:
            continue
        u_c, v_c, h_px, _w_px, _score = pt
        a_hat = col_to_azimuth_rad(u_c, w, fov)
        # Motion gate on azimuth (wrap).
        d_a = abs(math.atan2(math.sin(a_hat - az), math.cos(a_hat - az)))
        if d_a > math.radians(float(prior_gate_deg)):
            continue
        patch = extract_patch(luma, u_c, v_c, patch_size)
        if patch is None:
            continue
        sim = ncc(patch, tmpl)
        ncc_out[i] = sim
        if sim < float(ncc_tau):
            continue
        a_out[i] = a_hat
        if alpha_only:
            r_out[i] = max(rng, 1.0e-3)
        else:
            r_out[i] = bbox_height_to_range(h_px, f_px=f_px, length_m=length_m)
        valid_out[i] = 1.0
    return r_out, a_out, valid_out, ncc_out


def detect_s1_rgb(
    rgb_image: torch.Tensor | np.ndarray,
    *,
    templates: list[np.ndarray | None],
    prior_azimuth: torch.Tensor,
    prior_range: torch.Tensor,
    prior_gate_deg: float,
    hori_fov_deg: float,
    ncc_tau: float = 0.45,
    min_area: int = 4,
    patch_size: int = _PATCH,
    alpha_only: bool = True,
    length_m: float = 0.55,
    tint: tuple[float, float, float] = _DEFAULT_TINT,
    min_aff: float = 8.0,
    depth_image: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """S1 on tiled RGB: first-photo NCC in the KF gate + pinhole α.

    Range: Isaac ``distance_to_image_plane`` on the match when ``depth_image`` is
    set; otherwise bbox. If NCC < tau, reacquire nearest depth in the azimuth
    gate. ``tint`` is unused (kept for call-site compat).
    """
    del tint, min_aff, min_area, patch_size
    if isinstance(rgb_image, torch.Tensor):
        arr = rgb_image.detach().cpu().numpy()
        device = rgb_image.device
        n = int(rgb_image.shape[0])
    else:
        arr = np.asarray(rgb_image)
        if arr.ndim == 3:
            arr = arr[None, ...]
        device = torch.device("cpu")
        n = int(arr.shape[0])
    depth_arr = None
    if depth_image is not None:
        if isinstance(depth_image, torch.Tensor):
            depth_arr = depth_image.detach().cpu().numpy()
        else:
            depth_arr = np.asarray(depth_image)
        if depth_arr.ndim == 4:
            depth_arr = depth_arr[..., 0]
        if depth_arr.ndim == 2:
            depth_arr = depth_arr[None, ...]
        elif depth_arr.ndim == 3 and depth_arr.shape[-1] == 1:
            depth_arr = depth_arr[None, ..., 0]

    h = int(arr.shape[1])
    w = int(arr.shape[2])
    f_px = focal_px(w, hori_fov_deg)
    fov_deg = max(float(hori_fov_deg), 1.0e-3)
    search_col = max(float(prior_gate_deg) / fov_deg * (w - 1), 4.0)
    search_row = max(0.40 * (h - 1), 8.0)

    r_out = torch.zeros(n, device=device)
    a_out = torch.zeros(n, device=device)
    valid_out = torch.zeros(n, device=device)
    ncc_out = torch.full((n,), -1.0, device=device)
    u_out = torch.zeros(n, device=device)
    v_out = torch.zeros(n, device=device)
    # Speed-first: one nominal scale + mild shrink/grow. Quality not guaranteed.
    scales = (0.85, 1.0, 1.2)
    # Match on half-res RGB (4× fewer pixels); map peaks back to full-res.
    ds = 2

    for i in range(n):
        tmpl = templates[i] if i < len(templates) else None
        if tmpl is None:
            continue
        rgb = _rgb_u8(arr[i])
        tmpl_np = np.asarray(tmpl)
        if tmpl_np.dtype != np.uint8:
            tmpl_np = np.clip(tmpl_np, 0, 255).astype(np.uint8)
        az = float(prior_azimuth[i].item())
        rng = float(prior_range[i].item())
        col = azimuth_to_col_cam(az, w, hori_fov_deg)
        mid = 0.5 * (h - 1)
        c0 = max(0, int(math.floor(col - search_col)))
        c1 = min(w, int(math.ceil(col + search_col)) + 1)
        r0 = max(0, int(math.floor(mid - search_row)))
        r1 = min(h, int(math.ceil(mid + search_row)) + 1)

        # Downsample search image + template.
        try:
            import cv2

            rgb_s = cv2.resize(rgb, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
        except Exception:
            rgb_s = rgb[::ds, ::ds]
            ds = max(int(round(h / max(rgb_s.shape[0], 1))), 1)
        hs, ws = int(rgb_s.shape[0]), int(rgb_s.shape[1])
        r0s, r1s = max(0, r0 // ds), min(hs, (r1 + ds - 1) // ds)
        c0s, c1s = max(0, c0 // ds), min(ws, (c1 + ds - 1) // ds)

        best_pk = -1.0
        best_u = col
        best_v = mid
        best_hw = (float(tmpl_np.shape[0]), float(tmpl_np.shape[1]))
        for sc in scales:
            tm = _scale_tmpl(tmpl_np, sc)
            # Template also at match resolution.
            try:
                import cv2

                th_s = max(4, int(round(tm.shape[0] / ds)))
                tw_s = max(4, int(round(tm.shape[1] / ds)))
                if th_s % 2:
                    th_s += 1
                if tw_s % 2:
                    tw_s += 1
                tm_s = cv2.resize(tm, (tw_s, th_s), interpolation=cv2.INTER_AREA)
            except Exception:
                tm_s = tm[::ds, ::ds]
            if tm_s.shape[0] >= hs or tm_s.shape[1] >= ws:
                continue
            pk, us, vs = _ncc_best_in_roi(rgb_s, tm_s, r0=r0s, r1=r1s, c0=c0s, c1=c1s)
            if pk > best_pk:
                best_pk = pk
                best_u, best_v = float(us * ds), float(vs * ds)
                best_hw = (float(tm.shape[0]), float(tm.shape[1]))
        ncc_out[i] = best_pk
        u_c = v_c = None
        z_used = None
        src = "miss"
        v_ncc = best_v
        if best_pk >= float(ncc_tau):
            src = "ncc"
            u_c, v_c = best_u, best_v
        if u_c is not None and depth_arr is not None and i < int(depth_arr.shape[0]):
            # Range is nearest object on this bearing, not Z at the NCC peak.
            hit_r = _nearest_depth_uv(
                depth_arr[i],
                u_center=float(u_c),
                col_half=max(8.0, 0.03 * (w - 1)),
                min_z=0.3,
                max_z=6.0,
            )
            if hit_r is not None:
                u_c, v_c, z_used = hit_r
            else:
                src = "miss"
                u_c = v_c = None
        if u_c is None and depth_arr is not None and i < int(depth_arr.shape[0]):
            max_z = 6.0
            if 0.3 < rng < 6.0:
                max_z = float(min(6.0, max(2.0 * rng, 4.0)))
            hit = _nearest_depth_uv(
                depth_arr[i],
                u_center=col,
                col_half=search_col,
                min_z=0.3,
                max_z=max_z,
                v_center=mid,
                row_half=search_row,
            )
            if hit is not None:
                src = "depth_reacq"
                u_c, v_c, z_used = hit
        if u_c is None:
            if i == 0:
                # #region agent log
                _agent_log(
                    "H2",
                    "cam_meas.py:detect_s1_rgb",
                    "detect_miss",
                    {"ncc_pk": round(float(best_pk), 3), "ncc_tau": float(ncc_tau), "path": src},
                )
                # #endregion
            continue
        a_hat = col_to_azimuth_cam(u_c, w, hori_fov_deg)
        a_out[i] = a_hat
        u_out[i] = float(u_c)
        v_out[i] = float(v_c)
        if alpha_only:
            r_out[i] = max(rng, 1.0e-3)
            valid_out[i] = 1.0
            continue
        if z_used is not None:
            r_polar = float(z_used) / max(math.cos(a_hat), 0.2)
            r_out[i] = float(min(max(r_polar, 0.3), 6.0))
            valid_out[i] = 1.0
        else:
            extent_px = max(float(best_hw[0]), float(best_hw[1]), 1.0)
            r_bbox = bbox_height_to_range(extent_px, f_px=f_px, length_m=length_m)
            r_out[i] = float(min(max(r_bbox, 0.3), 6.0))
            valid_out[i] = 1.0
        if i == 0:
            # #region agent log
            _agent_log(
                "H3" if (src == "ncc" and abs(float(v_c) - float(v_ncc)) > 20) else ("H2" if src == "ncc" else "H4"),
                "cam_meas.py:detect_s1_rgb",
                "detect_ok",
                {
                    "path": src,
                    "ncc_pk": round(float(best_pk), 3),
                    "u": round(float(u_c), 1),
                    "v": round(float(v_c), 1),
                    "v_ncc": round(float(v_ncc), 1),
                    "z": None if z_used is None else round(float(z_used), 3),
                    "a_deg": round(math.degrees(a_hat), 2),
                    "r": round(float(r_out[i].item()), 3),
                    "used_orange": False,
                    "fast_ncc": True,
                    "ds": ds,
                },
            )
            # #endregion
    return r_out, a_out, valid_out, ncc_out, u_out, v_out
