# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sparse pyramidal Lucas-Kanade on the last two RGB frames of a channel stack.

The 12-channel observation is left unchanged. This returns a 3-vector
``(dx / width, dy / height, valid fraction)`` for the encoder trunk.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SPARSE_FLOW_DIM = 3
_POINTS = 32
_RADIUS = 2  # 5x5 window
_POINT_GAP = 6  # keep corners from collapsing onto one edge


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    """``(B, 3, H, W)`` -> ``(B, 1, H, W)`` Rec.601."""
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def last_two_luma(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Last two frames of a channel stack as luma.

    RGB stacks are grouped ``[R,G,B, ...]`` with the newest frame in the last 3
    channels. A grayscale stack uses the last two channels.
    """
    c = int(images.shape[1])
    if c >= 6 and c % 3 == 0:
        return _luma(images[:, -6:-3]), _luma(images[:, -3:])
    if c >= 2:
        return images[:, -2:-1], images[:, -1:]
    raise ValueError(f"need at least 2 channels to form a pair, got {c}")


def _grads(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kx = img.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
    ky = img.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0
    return F.conv2d(img, kx, padding=1), F.conv2d(img, ky, padding=1)


def _corner_score(img: torch.Tensor) -> torch.Tensor:
    """Shi-Tomasi min eigenvalue. A long edge scores near zero; a corner does not."""
    ix, iy = _grads(img)
    box = img.new_ones(1, 1, 5, 5) / 25.0
    a = F.conv2d(ix * ix, box, padding=2)
    b = F.conv2d(ix * iy, box, padding=2)
    c = F.conv2d(iy * iy, box, padding=2)
    tr = a + c
    det = (a * c - b * b).clamp(min=0.0)
    disc = (tr * tr - 4.0 * det).clamp(min=0.0).sqrt()
    return (0.5 * (tr - disc)).squeeze(1)


def _orange_mask(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dilated orange target on ``(B, 3, H, W)``. Returns ``(mask, has_target)``."""
    x = rgb
    if float(x.detach().max()) > 1.5:
        x = x / 255.0
    r, g, bch = x[:, 0], x[:, 1], x[:, 2]
    orange = (r > g + 0.08) & (r > bch + 0.08) & (r > 0.25)
    dilated = F.max_pool2d(orange.float().unsqueeze(1), kernel_size=9, stride=1, padding=4).squeeze(1) > 0.5
    return dilated, dilated.flatten(1).any(dim=-1)


def _prev_rgb(images: torch.Tensor) -> torch.Tensor | None:
    """RGB frame that the points are chosen on, or None for a grayscale stack."""
    c = int(images.shape[1])
    if c >= 6 and c % 3 == 0:
        return images[:, -6:-3]
    return None


def _topk_points(
    img: torch.Tensor,
    k: int,
    rgb: torch.Tensor | None = None,
    *,
    require_orange: bool = False,
    gap: int | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Spread-out corners. ``xs, ys`` are ``(B, K)``; ``ok`` is false when the image ran out of corners.

    When the frame shows the orange target, corners outside that silhouette are dropped.
    ``require_orange`` leaves every slot empty when the target is absent.
    """
    score = _corner_score(img)
    if mask is not None:
        score = score.masked_fill(~mask, -1.0)
    elif rgb is not None:
        mask, has_target = _orange_mask(rgb)
        if require_orange:
            score = score.masked_fill(~mask, -1.0)
            score = torch.where(has_target[:, None, None], score, torch.full_like(score, -1.0))
        else:
            score = torch.where(has_target[:, None, None], score.masked_fill(~mask, -1.0), score)
    b, h, w = score.shape
    border = _RADIUS + 1
    score = score.clone()
    score[:, :border, :] = -1.0
    score[:, -border:, :] = -1.0
    score[:, :, :border] = -1.0
    score[:, :, -border:] = -1.0
    yy = torch.arange(h, device=img.device)
    xx = torch.arange(w, device=img.device)
    xs_l: list[torch.Tensor] = []
    ys_l: list[torch.Tensor] = []
    ok_l: list[torch.Tensor] = []
    gap = max(_POINT_GAP, int(w) // 16) if gap is None else int(gap)
    gap2 = float(gap * gap)
    for _ in range(int(k)):
        flat = score.reshape(b, -1)
        idx = flat.argmax(dim=-1)
        val = flat.gather(1, idx[:, None]).squeeze(1)
        y = torch.div(idx, w, rounding_mode="floor")
        x = idx - y * w
        ok = val > 1.0e-6
        xs_l.append(x.to(dtype=img.dtype))
        ys_l.append(y.to(dtype=img.dtype))
        ok_l.append(ok)
        dist2 = (yy[None, :, None] - y[:, None, None]) ** 2 + (xx[None, None, :] - x[:, None, None]) ** 2
        score = score.masked_fill(dist2 <= gap2, -1.0)
    return torch.stack(xs_l, dim=1), torch.stack(ys_l, dim=1), torch.stack(ok_l, dim=1)


def _windows(img: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Sample a ``(2r+1)`` window at each point. Returns ``(B, K, win, win)``."""
    b, _, h, w = img.shape
    k = int(xs.shape[1])
    win = 2 * _RADIUS + 1
    oy = torch.arange(-_RADIUS, _RADIUS + 1, device=img.device, dtype=img.dtype)
    gy, gx = torch.meshgrid(oy, oy, indexing="ij")
    x = xs[:, :, None, None] + gx
    y = ys[:, :, None, None] + gy
    x_n = (x / max(w - 1, 1)) * 2.0 - 1.0
    y_n = (y / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x_n, y_n], dim=-1).reshape(b, k * win, win, 2)
    sampled = F.grid_sample(img, grid, mode="bilinear", align_corners=True, padding_mode="border")
    return sampled.view(b, 1, k, win, win).squeeze(1)


def _lk_windows(prev_w: torch.Tensor, curr_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve LK inside already-sampled windows. ``prev_w``/``curr_w`` are ``(B, K, win, win)``."""
    # Finite differences inside the window. The center pixel is the unknown.
    ix = 0.5 * (prev_w[:, :, :, 2:] - prev_w[:, :, :, :-2])
    iy = 0.5 * (prev_w[:, :, 2:, 1:-1] - prev_w[:, :, :-2, 1:-1])
    it = curr_w[:, :, 1:-1, 1:-1] - prev_w[:, :, 1:-1, 1:-1]
    ix = ix[:, :, 1:-1, :]
    a00 = (ix * ix).sum(dim=(-2, -1))
    a01 = (ix * iy).sum(dim=(-2, -1))
    a11 = (iy * iy).sum(dim=(-2, -1))
    b0 = -(ix * it).sum(dim=(-2, -1))
    b1 = -(iy * it).sum(dim=(-2, -1))
    det = a00 * a11 - a01 * a01
    valid = det > 1.0e-4
    det_s = torch.where(valid, det, torch.ones_like(det))
    du = (a11 * b0 - a01 * b1) / det_s
    dv = (a00 * b1 - a01 * b0) / det_s
    huge = (du.abs() > 8.0) | (dv.abs() > 8.0)
    valid = valid & ~huge
    du = torch.where(valid, du, torch.zeros_like(du))
    dv = torch.where(valid, dv, torch.zeros_like(dv))
    return du, dv, valid


def _masked_median(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    filled = values.masked_fill(~valid, float("nan"))
    med = torch.nanmedian(filled, dim=-1).values
    return torch.nan_to_num(med, nan=0.0)


def sparse_flow_vector(images: torch.Tensor) -> torch.Tensor:
    """``(B, C, H, W)`` float stack -> ``(B, 3)`` detached ``(dx/W, dy/H, valid_frac)``.

    Two pyramid levels. Points are spaced corners on the previous frame, kept on the orange target when it is visible.
    """
    out, _, _, _, _, _ = _sparse_lk(images)
    return out


def sparse_flow_points(
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-point LK on the same pair as ``sparse_flow_vector``.

    Returns detached ``xs, ys, du, dv, valid`` with point shape ``(B, K)``.
    ``du, dv`` are pixels on the luma frame. Invalid points have zero displacement.
    """
    _, xs, ys, du, dv, valid = _sparse_lk(images)
    return xs, ys, du, dv, valid


def lk_from_points(
    prev: torch.Tensor,
    curr: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    point_ok: torch.Tensor,
    init_du: torch.Tensor | None = None,
    init_dv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Track ``xs, ys`` from ``prev`` luma to ``curr`` luma. Both images are ``(B, 1, H, W)``."""
    prev_c = F.avg_pool2d(prev, 2, count_include_pad=False)
    curr_c = F.avg_pool2d(curr, 2, count_include_pad=False)
    # One linearisation under-shoots displacements larger than the 5x5 window.
    # A few iterations at each pyramid level are the usual Lucas-Kanade loop.
    if init_du is None:
        du_c = torch.zeros_like(xs)
        dv_c = torch.zeros_like(xs)
    else:
        du_c = init_du * 0.5
        dv_c = init_dv * 0.5
    valid_c = torch.zeros(xs.shape, dtype=torch.bool, device=xs.device)
    for _ in range(4):
        step_u, step_v, step_ok = _lk_windows(
            _windows(prev_c, xs * 0.5, ys * 0.5),
            _windows(curr_c, xs * 0.5 + du_c, ys * 0.5 + dv_c),
        )
        du_c = torch.where(step_ok, du_c + step_u, du_c)
        dv_c = torch.where(step_ok, dv_c + step_v, dv_c)
        valid_c = valid_c | step_ok
    du = du_c * 2.0
    dv = dv_c * 2.0
    valid_f = torch.zeros_like(valid_c)
    for _ in range(4):
        step_u, step_v, step_ok = _lk_windows(_windows(prev, xs, ys), _windows(curr, xs + du, ys + dv))
        du = torch.where(step_ok, du + step_u, du)
        dv = torch.where(step_ok, dv + step_v, dv)
        valid_f = valid_f | step_ok
    valid = (valid_f | valid_c) & point_ok
    du = torch.where(valid, du, torch.zeros_like(du))
    dv = torch.where(valid, dv, torch.zeros_like(dv))
    return du, dv, valid


def _flow_triplet(du: torch.Tensor, dv: torch.Tensor, valid: torch.Tensor, width: int, height: int) -> torch.Tensor:
    dx = _masked_median(du, valid)
    dy = _masked_median(dv, valid)
    frac = valid.float().mean(dim=-1)
    return torch.stack(
        [dx / max(float(width), 1.0), dy / max(float(height), 1.0), frac], dim=-1
    ).detach()


def _sparse_lk(
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prev, curr = last_two_luma(images)
    _, _, h, w = prev.shape
    xs, ys, point_ok = _topk_points(prev, _POINTS, _prev_rgb(images))
    du, dv, valid = lk_from_points(prev, curr, xs, ys, point_ok)
    out = _flow_triplet(du, dv, valid, w, h)
    return out, xs.detach(), ys.detach(), du.detach(), dv.detach(), valid.detach()


def _fill_empty(
    xs: torch.Tensor,
    ys: torch.Tensor,
    occupied: torch.Tensor,
    seed_x: torch.Tensor,
    seed_y: torch.Tensor,
    seed_ok: torch.Tensor,
    gap: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep tracked points and write new orange corners into empty slots."""
    out_x = torch.where(occupied, xs, torch.full_like(xs, -1.0))
    out_y = torch.where(occupied, ys, torch.full_like(ys, -1.0))
    out_ok = occupied.clone()
    gap2 = float(gap * gap)
    b = int(xs.shape[0])
    batch = torch.arange(b, device=xs.device)
    for i in range(int(seed_x.shape[1])):
        cand_ok = seed_ok[:, i]
        if not bool(cand_ok.any()):
            continue
        dx = seed_x[:, i : i + 1] - out_x
        dy = seed_y[:, i : i + 1] - out_y
        near = out_ok & (dx * dx + dy * dy <= gap2)
        empty = ~out_ok
        free = cand_ok & ~near.any(dim=-1) & empty.any(dim=-1)
        slot = empty.float().argmax(dim=-1)
        out_x[batch, slot] = torch.where(free, seed_x[:, i], out_x[batch, slot])
        out_y[batch, slot] = torch.where(free, seed_y[:, i], out_y[batch, slot])
        out_ok[batch, slot] = out_ok[batch, slot] | free
    return out_x, out_y, out_ok


def step_locked_flow(
    prev_rgb: torch.Tensor,
    curr_rgb: torch.Tensor,
    xy: torch.Tensor,
    alive: torch.Tensor,
    *,
    min_keep: int = 2,
    seed_xy: torch.Tensor | None = None,
    gate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continue a lock, or seed a new one inside ``gate_mask`` (else the orange target).

    ``prev_rgb`` and ``curr_rgb`` are ``(B, 3, H, W)``. ``xy`` is ``(B, K, 2)`` on the previous
    frame. Returns ``flow, draw_x, draw_y, du, dv, draw_ok, new_xy, new_alive``.
    A fresh lock has zero displacement; the next call tracks those points.
    """
    prev_f = prev_rgb.float()
    curr_f = curr_rgb.float()
    if float(torch.maximum(prev_f.detach().max(), curr_f.detach().max())) > 1.5:
        prev_f = prev_f / 255.0
        curr_f = curr_f / 255.0
    prev_l = _luma(prev_f)
    curr_l = _luma(curr_f)
    _, _, h, w = curr_l.shape
    xs = xy[..., 0]
    ys = xy[..., 1]
    du, dv, tracked = lk_from_points(prev_l, curr_l, xs, ys, alive)
    new_x = xs + du
    new_y = ys + dv
    border = _RADIUS + 1
    inside = (
        (new_x >= border)
        & (new_x <= (w - 1 - border))
        & (new_y >= border)
        & (new_y <= (h - 1 - border))
    )
    x_i = new_x.round().long().clamp(0, w - 1)
    y_i = new_y.round().long().clamp(0, h - 1)
    batch = torch.arange(curr_f.shape[0], device=curr_f.device)[:, None]
    if gate_mask is not None:
        on_target = gate_mask[batch, y_i, x_i]
    else:
        mask, has_target = _orange_mask(curr_f)
        on_target = mask[batch, y_i, x_i] & has_target[:, None]
    small = (du.abs() <= 8.0) & (dv.abs() <= 8.0)
    followed = alive & tracked & inside & on_target & small
    keep = followed.sum(dim=-1) >= min_keep
    if seed_xy is not None:
        seed_x = seed_xy[..., 0]
        seed_y = seed_xy[..., 1]
        seed_ok = seed_x >= 0.0
    elif gate_mask is not None:
        seed_x, seed_y, seed_ok = _topk_points(curr_l, int(xs.shape[1]), mask=gate_mask, gap=_POINT_GAP)
    elif bool((~followed).any()):
        seed_x, seed_y, seed_ok = _topk_points(
            curr_l, int(xs.shape[1]), curr_f, require_orange=True, gap=_POINT_GAP
        )
    else:
        seed_x = torch.zeros_like(xs)
        seed_y = torch.zeros_like(ys)
        seed_ok = torch.zeros_like(alive)
    filled_x, filled_y, filled_ok = _fill_empty(new_x, new_y, followed, seed_x, seed_y, seed_ok, _POINT_GAP)
    draw_x = torch.where(keep[:, None], xs, seed_x)
    draw_y = torch.where(keep[:, None], ys, seed_y)
    draw_du = torch.where(keep[:, None], du, torch.zeros_like(du))
    draw_dv = torch.where(keep[:, None], dv, torch.zeros_like(dv))
    draw_ok = torch.where(keep[:, None], followed, seed_ok)
    draw_x = torch.where(draw_ok, draw_x, torch.full_like(draw_x, -1.0))
    draw_y = torch.where(draw_ok, draw_y, torch.full_like(draw_y, -1.0))
    stored_x = torch.where(keep[:, None], filled_x, seed_x)
    stored_y = torch.where(keep[:, None], filled_y, seed_y)
    stored_ok = torch.where(keep[:, None], filled_ok, seed_ok)
    flow = _flow_triplet(draw_du, draw_dv, draw_ok & keep[:, None], w, h)
    # #region agent log
    try:
        import json as _json
        import time as _time

        _alive0 = alive[0]
        _n_alive = int(_alive0.sum())
        _du_alive = du[0][_alive0]
        _dv_alive = dv[0][_alive0]
        with open("/home/lh/IsaacLab/.cursor/debug-ffd96a.log", "a", encoding="utf-8") as _fh:
            _fh.write(
                _json.dumps(
                    {
                        "sessionId": "ffd96a",
                        "runId": "flow-follow",
                        "hypothesisId": "B-E",
                        "location": "sparse_flow.py:step_locked_flow",
                        "message": "lock gates",
                        "data": {
                            "alive": _n_alive,
                            "tracked": int((alive & tracked)[0].sum()),
                            "inside": int((alive & tracked & inside)[0].sum()),
                            "on_target": int((alive & tracked & inside & on_target)[0].sum()),
                            "small": int((alive & tracked & inside & on_target & small)[0].sum()),
                            "followed": int(followed[0].sum()),
                            "keep": bool(keep[0].item()),
                            "seed": int(seed_ok[0].sum()),
                            "gate_frac": None if gate_mask is None else round(float(gate_mask[0].float().mean()), 5),
                            "du_med": 0.0 if _n_alive == 0 else round(float(_du_alive.median()), 3),
                            "dv_med": 0.0 if _n_alive == 0 else round(float(_dv_alive.median()), 3),
                            "flow_dx": round(float(flow[0, 0]) * w, 3),
                            "flow_dy": round(float(flow[0, 1]) * h, 3),
                            "flow_valid": round(float(flow[0, 2]), 3),
                        },
                        "timestamp": int(_time.time() * 1000),
                    }
                )
                + "\n"
            )
    except OSError:
        pass
    # #endregion
    return (
        flow,
        draw_x.detach(),
        draw_y.detach(),
        draw_du.detach(),
        draw_dv.detach(),
        draw_ok.detach(),
        torch.stack([stored_x, stored_y], dim=-1).detach(),
        stored_ok.detach(),
    )
