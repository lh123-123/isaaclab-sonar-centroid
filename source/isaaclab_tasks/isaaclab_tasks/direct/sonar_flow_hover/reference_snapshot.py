# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Save/load taught sonar reference + sparse feature points for IBVS-DRL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ReferenceSnapshot:
    """Reference captured at the desired station pose (teach phase)."""

    ref_points: np.ndarray
    sonar_image: np.ndarray
    cart_image: np.ndarray | None = None
    track_image: np.ndarray | None = None
    perception_frame: str = "cart"
    ref_yaw: float | None = None
    # Legacy fields for backward-compatible snapshots
    ref_theta: float | None = None
    ref_range: float | None = None
    ref_flow_obs: np.ndarray | None = None
    beam_data: np.ndarray | None = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "ref_points": np.asarray(self.ref_points, dtype=np.float32),
            "sonar_image": np.asarray(self.sonar_image, dtype=np.float32),
        }
        if self.cart_image is not None:
            payload["cart_image"] = np.asarray(self.cart_image, dtype=np.float32)
        if self.track_image is not None:
            payload["track_image"] = np.asarray(self.track_image, dtype=np.float32)
        if self.perception_frame:
            payload["perception_frame"] = np.asarray(self.perception_frame)
        if self.ref_yaw is not None:
            payload["ref_yaw"] = np.float32(self.ref_yaw)
        if self.ref_theta is not None:
            payload["ref_theta"] = np.float32(self.ref_theta)
        if self.ref_range is not None:
            payload["ref_range"] = np.float32(self.ref_range)
        if self.ref_flow_obs is not None:
            payload["ref_flow_obs"] = np.asarray(self.ref_flow_obs, dtype=np.float32)
        if self.beam_data is not None:
            payload["beam_data"] = np.asarray(self.beam_data, dtype=np.float32)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> ReferenceSnapshot:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Reference snapshot not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            if "ref_points" not in data:
                raise ValueError(
                    f"Snapshot {path} lacks ref_points; re-run teach with IBVS-DRL pipeline."
                )
            ref_yaw = float(data["ref_yaw"]) if "ref_yaw" in data else None
            ref_theta = float(data["ref_theta"]) if "ref_theta" in data else None
            ref_range = float(data["ref_range"]) if "ref_range" in data else None
            ref_flow = np.asarray(data["ref_flow_obs"], dtype=np.float32) if "ref_flow_obs" in data else None
            cart = np.asarray(data["cart_image"], dtype=np.float32) if "cart_image" in data else None
            track = np.asarray(data["track_image"], dtype=np.float32) if "track_image" in data else None
            frame = str(data["perception_frame"].item()) if "perception_frame" in data else "cart"
            if track is None and cart is not None:
                track = cart
            beam = np.asarray(data["beam_data"], dtype=np.float32) if "beam_data" in data else None
            return cls(
                ref_points=np.asarray(data["ref_points"], dtype=np.float32),
                sonar_image=np.asarray(data["sonar_image"], dtype=np.float32),
                cart_image=cart,
                track_image=track,
                perception_frame=frame,
                ref_yaw=ref_yaw,
                ref_theta=ref_theta,
                ref_range=ref_range,
                ref_flow_obs=ref_flow,
                beam_data=beam,
            )

    def to_tensors(self, device: str | torch.device) -> dict[str, torch.Tensor]:
        out = {
            "ref_points": torch.tensor(self.ref_points, device=device, dtype=torch.float32),
            "sonar_image": torch.tensor(self.sonar_image, device=device, dtype=torch.float32),
        }
        if self.cart_image is not None:
            out["cart_image"] = torch.tensor(self.cart_image, device=device, dtype=torch.float32)
        if self.track_image is not None:
            out["track_image"] = torch.tensor(self.track_image, device=device, dtype=torch.float32)
        return out
