# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-step ball centroid from polar sonar image (bright-spot detection)."""

from __future__ import annotations

import torch

from .coordinates import PolarPoint
from .fixed_point import FixedPointStore
from .sonar_gt import SonarGtDetector


class SonarCentroidTracker:
    """Image-based centroid for obs/reward (SonarGtDetector on current polar frame)."""

    def __init__(
        self,
        detector: SonarGtDetector,
        fixed_point: FixedPointStore,
        *,
        polar_width: int,
        polar_height: int,
    ):
        self.detector = detector
        self.fixed_point = fixed_point
        self.polar_width = int(polar_width)
        self.polar_height = int(polar_height)
        self._polar: torch.Tensor | None = None

    def bind_polar(self, polar_batch: torch.Tensor) -> None:
        self._polar = polar_batch

    def detected_abs(self, env_id: int = 0) -> PolarPoint | None:
        if self._polar is None:
            return None
        polar_np = self._polar[env_id].detach().cpu().numpy()
        p_fixed = self.fixed_point.p_fixed_abs_env(env_id)
        if p_fixed is not None:
            p = self.detector.detect_abs_roi(polar_np, (p_fixed.col, p_fixed.row))
            if p is not None:
                return p
        return self.detector.detect_abs(polar_np)

    def detected_rel(self, env_id: int = 0) -> PolarPoint | None:
        p_abs = self.detected_abs(env_id)
        if p_abs is None:
            return None
        return self.fixed_point.to_relative(p_abs, env_id)

    def valid_ratio(self, env_id: int = 0) -> float:
        return 1.0 if self.detected_abs(env_id) is not None else 0.0

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids
