# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sparse colour match of a live frame against one 2 m / 0 deg teach image.

Range comes from the scale of matched points (``r = r_teach / scale``). Bearing
comes from the pinhole azimuth of their centroid. No depth image is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

TEACH_DIM = 4
_POINTS = 40
_WIN = 11
_RADIUS = 8
_MIN_INLIERS = 4


@dataclass
class TeachRef:
    """RGB teach image ``(H, W, 3)`` uint8 and its ``(M, 2)`` points in pixel ``(x, y)``."""

    image: torch.Tensor
    points: torch.Tensor
    range_m: float = 2.0


def _luma_nchw(rgb: torch.Tensor) -> torch.Tensor:
    return (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])


def _shi_tomasi(gray: torch.Tensor) -> torch.Tensor:
    """``(N, 1, H, W)`` -> min-eigenvalue score ``(N, H, W)``."""
    kx = gray.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
    ky = gray.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0
    ix = F.conv2d(gray, kx, padding=1)
    iy = F.conv2d(gray, ky, padding=1)
    box = gray.new_ones(1, 1, 5, 5) / 25.0
    a = F.conv2d(ix * ix, box, padding=2)
    b = F.conv2d(ix * iy, box, padding=2)
    c = F.conv2d(iy * iy, box, padding=2)
    tr = a + c
    det = (a * c - b * b).clamp(min=0.0)
    disc = (tr * tr - 4.0 * det).clamp(min=0.0).sqrt()
    return (0.5 * (tr - disc)).squeeze(1)


def detect_points(rgb_nhwc: torch.Tensor, n: int = _POINTS) -> torch.Tensor:
    """Shi-Tomasi with greedy NMS. ``rgb`` is ``(N, H, W, 3)`` -> ``(N, n, 2)`` xy.

    Low-score slots are filled with ``-1`` so callers can drop them.
    """
    rgb = rgb_nhwc[..., :3].float()
    if float(rgb.max()) > 1.5:
        rgb = rgb / 255.0
    nchw = rgb.permute(0, 3, 1, 2).contiguous()
    score = _shi_tomasi(_luma_nchw(nchw))
    # The seafloor has corners too. Keep points on the orange target, dilated so the
    # silhouette corners survive. Frames with no orange pixels keep the unmasked score.
    r, g, bch = nchw[:, 0], nchw[:, 1], nchw[:, 2]
    orange = (r > g + 0.08) & (r > bch + 0.08) & (r > 0.25)
    dilated = F.max_pool2d(orange.float().unsqueeze(1), kernel_size=9, stride=1, padding=4).squeeze(1) > 0.5
    has_target = dilated.flatten(1).any(dim=-1)
    score = torch.where(has_target[:, None, None], score.masked_fill(~dilated, -1.0), score)
    batch, h, w = score.shape
    border = _WIN // 2 + 1
    score = score.clone()
    score[:, :border, :] = -1.0
    score[:, -border:, :] = -1.0
    score[:, :, :border] = -1.0
    score[:, :, -border:] = -1.0
    yy = torch.arange(h, device=score.device)
    xx = torch.arange(w, device=score.device)
    pts = []
    for _ in range(int(n)):
        flat = score.reshape(batch, -1)
        idx = flat.argmax(dim=-1)
        val = flat.gather(1, idx[:, None]).squeeze(1)
        ys = torch.div(idx, w, rounding_mode="floor")
        xs = idx - ys * w
        xy = torch.stack([xs.float(), ys.float()], dim=-1)
        xy = torch.where(val[:, None] > 1.0e-6, xy, torch.full_like(xy, -1.0))
        pts.append(xy)
        dist2 = (yy[None, :, None] - ys[:, None, None]) ** 2 + (xx[None, None, :] - xs[:, None, None]) ** 2
        score = score.masked_fill(dist2 <= float(_RADIUS * _RADIUS), -1.0)
    return torch.stack(pts, dim=1)


def _patches(rgb_nchw: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """``pts`` ``(N, M, 2)`` xy -> ``(N, M, 3, win, win)``."""
    b, _, h, w = rgb_nchw.shape
    m = int(pts.shape[1])
    half = _WIN // 2
    oy = torch.arange(-half, half + 1, device=rgb_nchw.device, dtype=rgb_nchw.dtype)
    gy, gx = torch.meshgrid(oy, oy, indexing="ij")
    x = pts[:, :, 0][:, :, None, None] + gx
    y = pts[:, :, 1][:, :, None, None] + gy
    valid = (pts[:, :, 0] >= 0.0)[:, :, None, None, None]
    x_n = (x / max(w - 1, 1)) * 2.0 - 1.0
    y_n = (y / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x_n, y_n], dim=-1).reshape(b, m * _WIN, _WIN, 2)
    sampled = F.grid_sample(rgb_nchw, grid, mode="bilinear", align_corners=True, padding_mode="border")
    patches = sampled.view(b, 3, m, _WIN, _WIN).permute(0, 2, 1, 3, 4).contiguous()
    return patches * valid.to(dtype=patches.dtype)


def _zncc(patches: torch.Tensor) -> torch.Tensor:
    flat = patches.flatten(2)
    flat = flat - flat.mean(dim=-1, keepdim=True)
    return flat / flat.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)


def orange_radius(rgb_nhwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-image orange-blob ``(cx, radius, count)``.

    Radius is the RMS distance of orange pixels from their centroid, so it scales
    with apparent size. A changing inlier subset does not move it.
    """
    rgb = rgb_nhwc[..., :3].float()
    if float(rgb.detach().max()) > 1.5:
        rgb = rgb / 255.0
    r, g, bch = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r > g + 0.08) & (r > bch + 0.08) & (r > 0.25)
    n, h, w = mask.shape
    ys = torch.arange(h, device=rgb.device, dtype=rgb.dtype).view(1, h, 1)
    xs = torch.arange(w, device=rgb.device, dtype=rgb.dtype).view(1, 1, w)
    m = mask.to(dtype=rgb.dtype)
    count = m.sum(dim=(1, 2))
    safe = count.clamp(min=1.0)
    cx = (m * xs).sum(dim=(1, 2)) / safe
    cy = (m * ys).sum(dim=(1, 2)) / safe
    var = (m * ((xs - cx[:, None, None]) ** 2 + (ys - cy[:, None, None]) ** 2)).sum(dim=(1, 2)) / safe
    return cx, var.clamp(min=0.0).sqrt(), count


def orange_core(rgb_nhwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(rms_radius, count, core_width, core_height)`` on the same orange mask.

    Core width and height are the 10th–90th percentile span, so a few fringe pixels
    from a turned face do not set the size.
    """
    _, radius, count = orange_radius(rgb_nhwc)
    rgb = rgb_nhwc[..., :3].float()
    if float(rgb.detach().max()) > 1.5:
        rgb = rgb / 255.0
    r, g, bch = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r > g + 0.08) & (r > bch + 0.08) & (r > 0.25)
    n, h, w = mask.shape
    ys = torch.arange(h, device=rgb.device, dtype=rgb.dtype).view(h, 1).expand(h, w)
    xs = torch.arange(w, device=rgb.device, dtype=rgb.dtype).view(1, w).expand(h, w)
    core_w = torch.zeros(n, device=rgb.device, dtype=rgb.dtype)
    core_h = torch.zeros(n, device=rgb.device, dtype=rgb.dtype)
    for i in range(n):
        if float(count[i]) < 20.0:
            continue
        sel = mask[i]
        core_w[i] = torch.quantile(xs[sel], 0.9) - torch.quantile(xs[sel], 0.1) + 1.0
        core_h[i] = torch.quantile(ys[sel], 0.9) - torch.quantile(ys[sel], 0.1) + 1.0
    return radius, count, core_w, core_h


def orange_extent(rgb_nhwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-image orange bbox ``(width_px, height_px, count)``.

    Same mask as ``orange_radius``. Empty frames return zeros.
    """
    rgb = rgb_nhwc[..., :3].float()
    if float(rgb.detach().max()) > 1.5:
        rgb = rgb / 255.0
    r, g, bch = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r > g + 0.08) & (r > bch + 0.08) & (r > 0.25)
    n, h, w = mask.shape
    ys = torch.arange(h, device=rgb.device, dtype=rgb.dtype).view(1, h, 1).expand(n, h, w)
    xs = torch.arange(w, device=rgb.device, dtype=rgb.dtype).view(1, 1, w).expand(n, h, w)
    big = torch.full((), 1.0e6, device=rgb.device, dtype=rgb.dtype)
    x_min = torch.where(mask, xs, big).amin(dim=(1, 2))
    x_max = torch.where(mask, xs, -big).amax(dim=(1, 2))
    y_min = torch.where(mask, ys, big).amin(dim=(1, 2))
    y_max = torch.where(mask, ys, -big).amax(dim=(1, 2))
    count = mask.flatten(1).sum(dim=-1).to(dtype=rgb.dtype)
    width = (x_max - x_min + 1.0).clamp(min=0.0)
    height = (y_max - y_min + 1.0).clamp(min=0.0)
    present = count >= 1.0
    width = torch.where(present, width, torch.zeros_like(width))
    height = torch.where(present, height, torch.zeros_like(height))
    return width, height, count


def _col_to_azimuth(col: torch.Tensor, width: int, hori_fov_deg: float) -> torch.Tensor:
    fx = float(width) * 0.5 / math.tan(math.radians(float(hori_fov_deg)) * 0.5)
    cx = 0.5 * (float(width) - 1.0)
    return torch.atan((cx - col) / max(fx, 1.0e-6))


def _match_detail(
    curr_nhwc: torch.Tensor,
    teach: TeachRef,
    *,
    hori_fov_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match ``curr`` and return the 4-vector plus current-frame target points."""
    curr = curr_nhwc[..., :3]
    if curr.dtype != torch.float32:
        curr_f = curr.float()
    else:
        curr_f = curr
    if float(curr_f.detach().max()) > 1.5:
        curr_f = curr_f / 255.0
    teach_img = teach.image
    if teach_img.device != curr_f.device:
        teach_img = teach_img.to(curr_f.device)
        teach_pts = teach.points.to(curr_f.device)
    else:
        teach_pts = teach.points
    if teach_img.dtype != torch.float32 or float(teach_img.detach().max()) > 1.5:
        teach_f = teach_img.float() / 255.0
    else:
        teach_f = teach_img.float()

    n, _, w, _ = curr_f.shape
    curr_pts = detect_points(curr_f, _POINTS)
    teach_nchw = teach_f.permute(2, 0, 1).unsqueeze(0).expand(n, -1, -1, -1).contiguous()
    curr_nchw = curr_f.permute(0, 3, 1, 2).contiguous()
    t_pts = teach_pts.unsqueeze(0).expand(n, -1, -1).contiguous()
    t_desc = _zncc(_patches(teach_nchw, t_pts))
    c_desc = _zncc(_patches(curr_nchw, curr_pts))
    sim = torch.matmul(t_desc, c_desc.transpose(1, 2))  # (N, Mt, Mc)
    curr_ok = curr_pts[:, :, 0] >= 0.0
    teach_ok = t_pts[:, :, 0] >= 0.0
    sim = sim.masked_fill(~curr_ok[:, None, :], -1.0)
    topv, topi = sim.topk(k=min(2, sim.shape[-1]), dim=-1)
    s0 = topv[:, :, 0]
    s1 = topv[:, :, 1] if topv.shape[-1] > 1 else torch.zeros_like(s0)
    accept = teach_ok & (s0 > 0.45) & (s0 > 1.15 * s1.clamp(min=0.0))
    matched = torch.gather(curr_pts, 1, topi[:, :, 0:1].expand(-1, -1, 2))

    curr_cx, curr_rad, curr_count = orange_radius(curr_f)
    _, teach_rad, _teach_count = orange_radius(teach_f.unsqueeze(0) if teach_f.ndim == 3 else teach_f)
    teach_rad0 = float(teach_rad.reshape(-1)[0].clamp(min=1.0e-3))

    out = curr_f.new_zeros(n, TEACH_DIM)
    pts = curr_f.new_full((n, int(curr_pts.shape[1]), 2), -1.0)
    for i in range(n):
        sel = accept[i]
        n_acc = int(sel.sum())
        score_sel = teach_ok[i] & (s0[i] > 0.45)
        # The orange hull is repetitive: best and second-best ZNCC both sit near 0.95, so the
        # ratio test keeps 1-3 points and the min-inlier gate then writes zeros while the target
        # is still filling the frame. Fall back to the orange blob, which does not need a unique match.
        fallback = n_acc < _MIN_INLIERS and int(score_sel.sum()) >= _MIN_INLIERS and float(curr_count[i]) >= 20.0
        if fallback:
            scale = float((curr_rad[i] / teach_rad0).clamp(0.15, 5.0))
            alpha = float(_col_to_azimuth(curr_cx[i], w, hori_fov_deg))
            out[i, 0] = float(teach.range_m) / scale
            out[i, 1] = alpha
            out[i, 2] = scale
            out[i, 3] = float(n_acc / max(float(teach_ok[i].sum()), 1.0))
            pts[i] = curr_pts[i]
            continue
        if n_acc < _MIN_INLIERS:
            continue
        tp = t_pts[i, sel]
        cp = matched[i, sel]
        cc = cp.mean(dim=0)
        if float(curr_count[i]) >= 20.0:
            scale = float((curr_rad[i] / teach_rad0).clamp(0.15, 5.0))
        else:
            ct = tp.mean(dim=0)
            st = (tp - ct).norm(dim=-1).mean().clamp(min=1.0e-3)
            sc = (cp - cc).norm(dim=-1).mean()
            scale = float((sc / st).clamp(0.15, 5.0))
        alpha = float(_col_to_azimuth(cc[0], w, hori_fov_deg))
        r_hat = float(teach.range_m) / scale
        out[i, 0] = r_hat
        out[i, 1] = alpha
        out[i, 2] = scale
        out[i, 3] = float(sel.float().sum() / max(float(teach_ok[i].sum()), 1.0))
        take = min(int(cp.shape[0]), int(pts.shape[1]))
        pts[i, :take] = cp[:take]
    return out, pts


def match_rgb(
    curr_nhwc: torch.Tensor,
    teach: TeachRef,
    *,
    hori_fov_deg: float,
) -> torch.Tensor:
    """``curr`` ``(N, H, W, 3)`` -> ``(N, 4)`` ``[r_hat, alpha_hat, scale, inlier_frac]``.

    A failed match is zeros.
    """
    out, _pts = _match_detail(curr_nhwc, teach, hori_fov_deg=hori_fov_deg)
    return out


def target_points(
    curr_nhwc: torch.Tensor,
    teach: TeachRef,
    *,
    hori_fov_deg: float,
) -> torch.Tensor:
    """Current-frame points ``(N, M, 2)`` that ``match_rgb`` treats as the target.

    Matched correspondences when the teach photo matches. Orange corners when the
    match falls back to the orange blob. ``-1`` when there is no target.
    """
    _out, pts = _match_detail(curr_nhwc, teach, hori_fov_deg=hori_fov_deg)
    return pts


def teach_from_numpy(image_hwc: np.ndarray, points_xy: np.ndarray, range_m: float = 2.0) -> TeachRef:
    image = torch.as_tensor(np.ascontiguousarray(image_hwc[..., :3]))
    points = torch.as_tensor(np.ascontiguousarray(points_xy), dtype=torch.float32)
    return TeachRef(image=image, points=points, range_m=float(range_m))


_DET_SCALES = (0.55, 0.75, 1.0, 1.35, 1.8)
_DET_SCORE = 0.40


def cut_teach_template(image_hwc: torch.Tensor) -> torch.Tensor:
    """Crop the object out of the stored teach photo.

    Colour is used only here, once, to cut the template. Live frames are matched
    by correlation against this crop.
    """
    img = image_hwc[..., :3].float()
    if float(img.detach().max()) > 1.5:
        img = img / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    core = (r > g + 0.08) & (r > b + 0.08) & (r > 0.25)
    h, w = int(img.shape[0]), int(img.shape[1])
    if bool(core.any()):
        ys, xs = torch.where(core)
        margin = 2
        y0 = max(int(ys.min()) - margin, 0)
        y1 = min(int(ys.max()) + margin + 1, h)
        x0 = max(int(xs.min()) - margin, 0)
        x1 = min(int(xs.max()) + margin + 1, w)
    else:
        y0, y1 = h // 2 - 16, h // 2 + 16
        x0, x1 = w // 2 - 24, w // 2 + 24
    crop = img[y0:y1, x0:x1]
    return crop.permute(2, 0, 1).unsqueeze(0).contiguous()


def _ncc_map(image: torch.Tensor, templ: torch.Tensor) -> torch.Tensor:
    """Normalized cross-correlation.

    ``image`` is ``(N, C, H, W)`` and ``templ`` is ``(1, C, th, tw)``. Colour channels
    are matched together, so a grey patch does not score like the orange teach block.
    """
    t = templ - templ.mean()
    tnorm = t.norm().clamp(min=1.0e-6)
    num = None
    sum_i = None
    sum_i2 = None
    ones = image.new_ones(1, 1, int(templ.shape[-2]), int(templ.shape[-1]))
    for c in range(int(templ.shape[1])):
        conv = F.conv2d(image[:, c : c + 1], t[:, c : c + 1])
        acc = F.conv2d(image[:, c : c + 1], ones)
        acc2 = F.conv2d(image[:, c : c + 1] * image[:, c : c + 1], ones)
        num = conv if num is None else num + conv
        sum_i = acc if sum_i is None else sum_i + acc
        sum_i2 = acc2 if sum_i2 is None else sum_i2 + acc2
    area = float(templ.shape[1] * templ.shape[-2] * templ.shape[-1])
    energy = (sum_i2 - (sum_i * sum_i) / area).clamp(min=0.0).sqrt()
    score = num / (energy * tnorm).clamp(min=1.0e-6)
    return score.masked_fill(energy < 1.0e-3, -1.0)


def detect_teach(
    curr_nhwc: torch.Tensor,
    template: torch.Tensor,
    prev_bw: torch.Tensor | None = None,
    prev_bh: torch.Tensor | None = None,
    prev_ok: torch.Tensor | None = None,
    scales: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Find the teach template in ``curr`` ``(N, H, W, 3)``.

    Returns ``(N, 5)`` ``[cx, cy, bw, bh, score]``. A miss is ``cx < 0``.
    No colour test is applied to the live frame. The previous box size is kept
    while its score still clears the detection threshold, so a nearly tied
    larger scale cannot pull the centre off the target.
    """
    curr = curr_nhwc[..., :3].float()
    if float(curr.detach().max()) > 1.5:
        curr = curr / 255.0
    templ = template.to(device=curr.device, dtype=curr.dtype)
    if int(templ.shape[1]) == 1:
        image = (0.299 * curr[..., 0] + 0.587 * curr[..., 1] + 0.114 * curr[..., 2]).unsqueeze(1)
    else:
        image = curr.permute(0, 3, 1, 2).contiguous()
    n, _, h, w = image.shape
    cxs: list[torch.Tensor] = []
    cys: list[torch.Tensor] = []
    bws: list[torch.Tensor] = []
    bhs: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    th0, tw0 = int(templ.shape[-2]), int(templ.shape[-1])
    for scale in _DET_SCALES if scales is None else scales:
        th = max(int(round(th0 * scale)), 1)
        tw = max(int(round(tw0 * scale)), 1)
        if th < 8 or tw < 8 or th >= h or tw >= w:
            continue
        sized = templ if (th, tw) == (th0, tw0) else F.interpolate(templ, size=(th, tw), mode="bilinear", align_corners=False)
        score_map = _ncc_map(image, sized)
        flat = score_map.reshape(n, -1)
        val, idx = flat.max(dim=-1)
        mw = int(score_map.shape[-1])
        y = torch.div(idx, mw, rounding_mode="floor")
        x = idx - y * mw
        cxs.append(x.float() + 0.5 * (tw - 1))
        cys.append(y.float() + 0.5 * (th - 1))
        bws.append(image.new_full((n,), float(tw)))
        bhs.append(image.new_full((n,), float(th)))
        scores.append(val)
    if not scores:
        miss = image.new_zeros(n, 5)
        miss[:, 0] = -1.0
        return miss
    cx_s = torch.stack(cxs)
    cy_s = torch.stack(cys)
    bw_s = torch.stack(bws)
    bh_s = torch.stack(bhs)
    score_s = torch.stack(scores)
    best_i = score_s.argmax(dim=0)
    pick = best_i
    if prev_bw is not None and prev_bh is not None and prev_ok is not None:
        dist = (bw_s - prev_bw.reshape(1, -1)).abs() + (bh_s - prev_bh.reshape(1, -1)).abs()
        prev_i = dist.argmin(dim=0)
        gather_i = torch.arange(n, device=image.device)
        prev_score = score_s[prev_i, gather_i]
        best_score = score_s[best_i, gather_i]
        stick = prev_ok.reshape(-1) & (prev_score >= _DET_SCORE)
        pick = torch.where(stick, prev_i, best_i)
    gather_i = torch.arange(n, device=image.device)
    cx = cx_s[pick, gather_i]
    cy = cy_s[pick, gather_i]
    bw = bw_s[pick, gather_i]
    bh = bh_s[pick, gather_i]
    score = score_s[pick, gather_i]
    hit = score >= _DET_SCORE
    cx = torch.where(hit, cx, torch.full_like(cx, -1.0))
    return torch.stack([cx, cy, bw, bh, score], dim=-1)
