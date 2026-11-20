# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Displacement field of the current frame against the episode's first frame.

Farneback(frame_1, frame_now) is a position field, not a velocity. On the
fronto-parallel plane it fits

    du = a + k x,   dv = b + k y

with a, b the lateral shift in flow-resolution pixels and k the radial
expansion (ΔZ / Z). Consecutive-frame flow is a different signal and stays
where it is.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class FirstFramePositionField:
    """Stores frame 1 and measures the later frames against it."""

    def __init__(self, num_envs: int, flow_h: int, flow_w: int, out_h: int, out_w: int, device: str):
        self.fh = int(flow_h)
        self.fw = int(flow_w)
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.device = device
        self.first = torch.zeros((num_envs, self.fh, self.fw, 3), dtype=torch.uint8, device=device)
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.a_px = torch.zeros(num_envs, device=device)
        self.b_px = torch.zeros(num_envs, device=device)
        self.k = torch.zeros(num_envs, device=device)
        self.just_captured = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids: torch.Tensor | None = None) -> None:
        if ids is None:
            self.valid[:] = False
            self.a_px[:] = 0.0
            self.b_px[:] = 0.0
            self.k[:] = 0.0
            self.just_captured[:] = False
            return
        self.valid[ids] = False
        self.a_px[ids] = 0.0
        self.b_px[ids] = 0.0
        self.k[ids] = 0.0
        self.just_captured[ids] = False

    def observe(
        self,
        curr_raw: torch.Tensor,
        tex_ok: torch.Tensor,
        tex_mask: torch.Tensor,
    ) -> None:
        """Update a_px, b_px, k. tex_mask is (N, 1, out_h, out_w) or (N, out_h, out_w)."""
        self.just_captured[:] = False
        self.a_px[:] = 0.0
        self.b_px[:] = 0.0
        self.k[:] = 0.0
        down = self._downsample(curr_raw)
        capture = (~self.valid) & tex_ok
        if bool(capture.any()):
            self.first[capture] = down[capture]
            self.valid[capture] = True
            self.just_captured[capture] = True
        match = self.valid & (~self.just_captured)
        if not bool(match.any()):
            return
        est = _farneback_flow(self.first[match], down[match])
        small = F.interpolate(est, size=(self.out_h, self.out_w), mode="area")
        tex = tex_mask[match]
        if tex.ndim == 4:
            tex = tex.squeeze(1)
        a, b, k = _fit_translational_flow(small[:, 0], small[:, 1], tex, self.fw, self.fh)
        self.a_px[match] = a
        self.b_px[match] = b
        self.k[match] = k

    def _downsample(self, curr_raw: torch.Tensor) -> torch.Tensor:
        down = F.interpolate(
            curr_raw.permute(0, 3, 1, 2).to(torch.float32),
            size=(self.fh, self.fw),
            mode="area",
        )
        return down.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def _farneback_flow(prev_rgb: torch.Tensor, curr_rgb: torch.Tensor) -> torch.Tensor:
    """Farneback optical flow, (N, 2, H, W) pixels. Same settings as the v7 env."""
    use_ocl = False
    try:
        cv2.ocl.setUseOpenCL(True)
        use_ocl = bool(cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL())
    except Exception:
        use_ocl = False
    n = prev_rgb.shape[0]
    frames_prev = prev_rgb.detach().cpu().numpy()
    frames_curr = curr_rgb.detach().cpu().numpy()
    h, w = frames_prev.shape[1], frames_prev.shape[2]
    flows = torch.empty((n, 2, h, w), dtype=torch.float32)
    for i in range(n):
        if use_ocl:
            g_prev = cv2.cvtColor(cv2.UMat(frames_prev[i]), cv2.COLOR_RGB2GRAY)
            g_curr = cv2.cvtColor(cv2.UMat(frames_curr[i]), cv2.COLOR_RGB2GRAY)
            flow_hwc = cv2.calcOpticalFlowFarneback(
                g_prev, g_curr, None,
                pyr_scale=0.5, levels=3, winsize=21,
                iterations=2, poly_n=5, poly_sigma=1.2, flags=0,
            ).get()
        else:
            g_prev = cv2.cvtColor(frames_prev[i], cv2.COLOR_RGB2GRAY)
            g_curr = cv2.cvtColor(frames_curr[i], cv2.COLOR_RGB2GRAY)
            flow_hwc = cv2.calcOpticalFlowFarneback(
                g_prev, g_curr, None,
                pyr_scale=0.5, levels=3, winsize=21,
                iterations=2, poly_n=5, poly_sigma=1.2, flags=0,
            )
        flows[i] = torch.from_numpy(np.ascontiguousarray(flow_hwc.transpose(2, 0, 1)))
    return flows.to(prev_rgb.device)


def _fit_translational_flow(du_px, dv_px, tex, fw: int, fh: int):
    """Fit du = a + k x, dv = b + k y. a, b, k are in flow-resolution pixels."""
    m, h, w = du_px.shape
    device, dtype = du_px.device, du_px.dtype
    xs = (torch.arange(w, device=device, dtype=dtype) - (w - 1) * 0.5) * (float(fw) / float(w))
    ys = (torch.arange(h, device=device, dtype=dtype) - (h - 1) * 0.5) * (float(fh) / float(h))
    x = xs.view(1, 1, w).expand(m, h, w)
    y = ys.view(1, h, 1).expand(m, h, w)
    wt = tex.reshape(m, h, w).to(dtype)
    sw = wt.sum(dim=(1, 2)).clamp(min=1.0)
    sx = (wt * x).sum(dim=(1, 2))
    sy = (wt * y).sum(dim=(1, 2))
    sxx = (wt * x * x).sum(dim=(1, 2))
    syy = (wt * y * y).sum(dim=(1, 2))
    su = (wt * du_px).sum(dim=(1, 2))
    sv = (wt * dv_px).sum(dim=(1, 2))
    sux = (wt * du_px * x).sum(dim=(1, 2))
    svy = (wt * dv_px * y).sum(dim=(1, 2))
    k = (sux + svy) / (sxx + syy).clamp(min=1.0)
    a = (su - k * sx) / sw
    b = (sv - k * sy) / sw
    return a, b, k
