# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""First-frame template match (TM_CCOEFF_NORMED).

Pulled out of the v7 hover env so the training path can keep calling it while a
position-field hover is tried alongside. The env still owns the buffers; this
function only updates them.
"""

from __future__ import annotations

import cv2
import torch
import torch.nn.functional as F


def update_template_anchor(
    curr_raw: torch.Tensor,
    *,
    texture_count: torch.Tensor,
    min_texture_px: int,
    anchor_valid: torch.Tensor,
    anchor_tmpl: torch.Tensor,
    anchor_off_px: torch.Tensor,
    anchor_ncc: torch.Tensor,
    anchor_disp_m: torch.Tensor,
    anchor_yaw: torch.Tensor,
    flow_h: int,
    flow_w: int,
    crop_h: int,
    crop_w: int,
    f_px: float | torch.Tensor,
    plane_distance: float | torch.Tensor,
    local_x: torch.Tensor,
    yaw: torch.Tensor,
) -> None:
    """Capture frame 1, then match every later frame against that crop.

    Outputs (written in place):
      anchor_off_px (N, 2)  scene displacement vs frame 1, flow-res px.
                            +u = scene moved right = ROV moved left.
                            Saturates at the match-window edge; the sign still
                            points home.
      anchor_ncc (N,)       TM_CCOEFF_NORMED in [-1, 1]. 1 = identical to frame 1.
      anchor_disp_m (N,)    |offset| in metres on the fronto-parallel plane.
      anchor_yaw (N,)       heading at capture.
    On the capture step offset = 0 and ncc = 1.

    ``f_px`` and ``plane_distance`` accept a scalar or a per-env (N,) tensor. They are the
    estimator's calibration PRIOR, not the true camera: two units in one batch may believe
    different things about their own intrinsics and standoff (see ``dr_visodom_*``).
    """
    fh, fw = int(flow_h), int(flow_w)
    ch, cw = int(crop_h), int(crop_w)
    y0, x0 = (fh - ch) // 2, (fw - cw) // 2
    tex_ok = texture_count >= int(min_texture_px)
    capture = (~anchor_valid) & tex_ok
    match = anchor_valid & (~capture)
    if not bool(capture.any() or match.any()):
        return
    rows = (capture | match).nonzero().flatten()
    down = (
        F.interpolate(
            curr_raw[rows].permute(0, 3, 1, 2).to(torch.float32), size=(fh, fw), mode="area"
        )
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    captured: list[int] = []
    for i, r in enumerate(rows.tolist()):
        gray = cv2.cvtColor(down[i], cv2.COLOR_RGB2GRAY)
        if bool(capture[r]):
            crop = gray[y0 : y0 + ch, x0 : x0 + cw]
            # A near-constant template (dark wall inside the crop) makes
            # TM_CCOEFF_NORMED degenerate (NaN peak). Refuse it and retry.
            if float(crop.std()) < 1.0:
                continue
            anchor_tmpl[r] = torch.from_numpy(crop.copy()).to(anchor_tmpl.device)
            captured.append(r)
        else:
            res = cv2.matchTemplate(gray, anchor_tmpl[r].cpu().numpy(), cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            anchor_off_px[r, 0] = float(max_loc[0]) - float(x0)
            anchor_off_px[r, 1] = float(max_loc[1]) - float(y0)
            anchor_ncc[r] = float(max_val)
    if captured:
        cap = torch.as_tensor(captured, device=anchor_valid.device, dtype=torch.long)
        anchor_valid[cap] = True
        anchor_off_px[cap] = 0.0
        anchor_ncc[cap] = 1.0
        anchor_yaw[cap] = yaw[cap]
    valid = anchor_valid
    if bool(valid.any()):
        # d = |px| · Z / f_px. Z is the env-local forward distance to the plane
        # (origins are spaced tens of metres apart, so world x would be wrong).
        # Both may be per-env: they carry the calibration the estimator BELIEVES, which
        # is how a wrong intrinsics / unknown standoff enters the metric output.
        dev = anchor_off_px.device
        pd = torch.as_tensor(plane_distance, device=dev, dtype=torch.float32)
        fp = torch.as_tensor(f_px, device=dev, dtype=torch.float32)
        z_plane = (pd - local_x).clamp(min=0.3)
        d_px = anchor_off_px[valid].norm(dim=-1)
        if z_plane.dim():
            z_plane = z_plane[valid]
        if fp.dim():
            fp = fp[valid]
        anchor_disp_m[valid] = d_px / fp * z_plane
