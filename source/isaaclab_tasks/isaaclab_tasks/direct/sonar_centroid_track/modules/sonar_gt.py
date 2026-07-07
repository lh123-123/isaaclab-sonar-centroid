# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Module M2: sonar-image ground truth centroid (p_sonar) on polar frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from isaaclab_tasks.direct.sonar_flow_hover.modules.beacon_cart import (
    detect_beacon_peak_polar,
    detect_beacon_peak_polar_global,
)

from .coordinates import PolarPoint, mirror_azimuth_col, polar_point, to_fixed_frame

if TYPE_CHECKING:
    import torch


class SonarGtDetector:
    """Bright-spot centroid on preprocessed polar sonar (evaluation / fixed-point calib)."""

    def __init__(
        self,
        *,
        threshold_ratio: float = 0.35,
        min_pixels: int = 4,
        apply_azimuth_mirror: bool = False,
        polar_width: int = 128,
    ):
        self.threshold_ratio = float(threshold_ratio)
        self.min_pixels = int(min_pixels)
        self.apply_azimuth_mirror = bool(apply_azimuth_mirror)
        self.polar_width = int(polar_width)

    def detect_abs(self, polar: np.ndarray | "torch.Tensor") -> PolarPoint | None:
        """Detect ball centroid from polar image brightness (no pose ground truth)."""
        return self.detect_abs_roi(polar, center_col_row=None)

    def detect_abs_roi(
        self,
        polar: np.ndarray | "torch.Tensor",
        center_col_row: tuple[float, float] | None,
        *,
        search_radius_col: float = 14.0,
        search_radius_row: float = 18.0,
    ) -> PolarPoint | None:
        """Bright-spot centroid; optional elliptical ROI around ``center_col_row``."""
        arr = self._to_numpy(polar)
        if center_col_row is None:
            peak = detect_beacon_peak_polar_global(
                arr,
                threshold_ratio=self.threshold_ratio,
                min_pixels=self.min_pixels,
            )
        else:
            peak = detect_beacon_peak_polar(
                arr,
                center_col_row=center_col_row,
                search_radius_col=search_radius_col,
                search_radius_row=search_radius_row,
                threshold_ratio=self.threshold_ratio,
                min_pixels=self.min_pixels,
            )
        if peak is None:
            return None
        col, row = peak
        if self.apply_azimuth_mirror:
            col = mirror_azimuth_col(col, self.polar_width)
        return polar_point(col, row)

    def detect_rel(self, polar: np.ndarray | "torch.Tensor", p_fixed_abs: PolarPoint) -> PolarPoint | None:
        p_abs = self.detect_abs(polar)
        if p_abs is None:
            return None
        return to_fixed_frame(p_abs, p_fixed_abs)

    @staticmethod
    def _to_numpy(polar: np.ndarray | "torch.Tensor") -> np.ndarray:
        if hasattr(polar, "detach"):
            polar = polar.detach().cpu().numpy()
        return np.asarray(polar, dtype=np.float32)
