# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Module M4: fixed-point origin calibration and persistence (multi-env)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .coordinates import PolarPoint, polar_point, to_fixed_frame
from .sonar_gt import SonarGtDetector

CalibMode = Literal["sonar_gt", "manual"]


@dataclass
class FixedPointSnapshot:
    p_fixed_abs: PolarPoint
    polar_snapshot: np.ndarray | None = None
    ref_points: np.ndarray | None = None
    mode: str = "sonar_gt"

    def to_relative(self, p_abs: PolarPoint) -> PolarPoint:
        return to_fixed_frame(p_abs, self.p_fixed_abs)


class FixedPointStore:
    """Per-env p_fixed_abs; converts absolute detections to fixed-point frame."""

    def __init__(
        self,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        *,
        polar_width: int = 128,
        polar_height: int = 64,
        sonar_gt: SonarGtDetector | None = None,
    ):
        self.num_envs = int(num_envs)
        self.device = device
        self.polar_width = int(polar_width)
        self.polar_height = int(polar_height)
        self._gt = sonar_gt if sonar_gt is not None else SonarGtDetector(polar_width=self.polar_width)
        self._col = torch.full((self.num_envs,), float("nan"), device=device)
        self._row = torch.full((self.num_envs,), float("nan"), device=device)
        self._calibrated = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._snapshot: FixedPointSnapshot | None = None

    def is_calibrated_env(self, env_id: int = 0) -> bool:
        return bool(self._calibrated[env_id].item())

    @property
    def is_calibrated(self) -> bool:
        return bool(self._calibrated.all().item()) if self.num_envs > 1 else self.is_calibrated_env(0)

    def p_fixed_abs_env(self, env_id: int = 0) -> PolarPoint | None:
        if not self.is_calibrated_env(env_id):
            return None
        return polar_point(float(self._col[env_id].item()), float(self._row[env_id].item()))

    @property
    def p_fixed_abs(self) -> PolarPoint | None:
        return self.p_fixed_abs_env(0)

    def calibrate_from_polar(
        self,
        polar: np.ndarray,
        *,
        env_id: int = 0,
        mode: CalibMode = "sonar_gt",
        manual_col_row: tuple[float, float] | None = None,
        roi_center: tuple[float, float] | None = None,
        roi_radius_col: float = 14.0,
        roi_radius_row: float = 18.0,
        ref_points: np.ndarray | None = None,
    ) -> PolarPoint:
        if mode == "manual":
            if manual_col_row is None:
                raise ValueError("manual_col_row required for mode='manual'")
            p_fixed = polar_point(manual_col_row[0], manual_col_row[1])
        elif roi_center is not None:
            p_fixed = self._gt.detect_abs_roi(
                polar,
                roi_center,
                search_radius_col=roi_radius_col,
                search_radius_row=roi_radius_row,
            )
            if p_fixed is None:
                p_fixed = self._gt.detect_abs(polar)
            if p_fixed is None:
                raise RuntimeError(f"Sonar GT failed fixed-point calibration (env {env_id})")
        else:
            p_fixed = self._gt.detect_abs(polar)
            if p_fixed is None:
                raise RuntimeError(f"Sonar GT failed fixed-point calibration (env {env_id})")
        self._col[env_id] = p_fixed.col
        self._row[env_id] = p_fixed.row
        self._calibrated[env_id] = True
        if self.num_envs == 1:
            self._snapshot = FixedPointSnapshot(
                p_fixed_abs=p_fixed,
                polar_snapshot=np.asarray(polar, dtype=np.float32).copy(),
                ref_points=None if ref_points is None else np.asarray(ref_points, dtype=np.float32).copy(),
                mode=mode,
            )
        return p_fixed

    def to_relative(self, p_abs: PolarPoint | None, env_id: int = 0) -> PolarPoint | None:
        if p_abs is None or not self.is_calibrated_env(env_id):
            return None
        return to_fixed_frame(p_abs, self.p_fixed_abs_env(env_id))

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        self._calibrated[env_ids] = False
        self._col[env_ids] = float("nan")
        self._row[env_ids] = float("nan")
        if self.num_envs == 1:
            self._snapshot = None

    def reset(self) -> None:
        self.reset_envs(torch.arange(self.num_envs, device=self.device))

    def save(self, path: str | Path, *, env_id: int = 0) -> None:
        p = self.p_fixed_abs_env(env_id)
        if p is None:
            raise RuntimeError("Cannot save: fixed point not calibrated")
        payload = {
            "col0": np.float32(p.col),
            "row0": np.float32(p.row),
            "mode": "sonar_gt",
        }
        if self._snapshot is not None and self._snapshot.polar_snapshot is not None:
            payload["polar_snapshot"] = self._snapshot.polar_snapshot
        np.savez(path, **payload)

    def load(self, path: str | Path, *, env_id: int = 0) -> PolarPoint:
        data = np.load(path)
        p_fixed = polar_point(float(data["col0"]), float(data["row0"]))
        self._col[env_id] = p_fixed.col
        self._row[env_id] = p_fixed.row
        self._calibrated[env_id] = True
        polar_snap = data["polar_snapshot"] if "polar_snapshot" in data else None
        self._snapshot = FixedPointSnapshot(
            p_fixed_abs=p_fixed,
            polar_snapshot=polar_snap,
            mode=str(data["mode"]) if "mode" in data else "file",
        )
        return p_fixed
