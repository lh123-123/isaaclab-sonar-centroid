# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Module M10: human-readable debug helpers."""

from __future__ import annotations

from typing import Any

import torch

from .centroid_bus import CentroidFrame
from .constants import POLICY_OBS_DIM, POLICY_OBS_DIM_IMAGE


def format_centroid_frame(frame: CentroidFrame) -> str:
    return frame.summary()


def decode_observation(obs: torch.Tensor) -> dict[str, Any]:
    """Decode discrete-PPO observation (4D image-only or 7D full)."""
    o = obs.detach().cpu().numpy().reshape(-1)
    n = o.shape[0]
    if n == POLICY_OBS_DIM_IMAGE:
        prev = "stay"
        if o[3] < -0.5:
            prev = "left"
        elif o[3] > 0.5:
            prev = "right"
        return {
            "u_norm": float(o[0]),
            "v_norm": float(o[1]),
            "valid_ratio": float(o[2]),
            "prev_action": prev,
        }
    if n != POLICY_OBS_DIM:
        raise ValueError(f"Expected obs dim {POLICY_OBS_DIM_IMAGE} or {POLICY_OBS_DIM}, got {n}")
    prev = "stay"
    if o[6] < -0.5:
        prev = "left"
    elif o[6] > 0.5:
        prev = "right"
    return {
        "u_norm": float(o[0]),
        "v_norm": float(o[1]),
        "valid_ratio": float(o[2]),
        "robot_lat": float(o[3]),
        "target_dy": float(o[4]),
        "target_dy_vel": float(o[5]),
        "prev_action": prev,
    }
