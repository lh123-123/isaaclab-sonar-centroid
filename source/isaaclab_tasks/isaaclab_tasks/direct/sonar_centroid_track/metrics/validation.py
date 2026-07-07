# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 1/2 validation metrics and VERDICT."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..modules.constants import (
    PERC_SONAR_COL_MEAN_MAX,
    PERC_SONAR_COL_P90_MAX,
    PERC_SONAR_DIST_MEAN_MAX,
    PERC_SONAR_ROW_MEAN_MAX,
    TRACK_ORIGIN_DIST_MAX,
)
from ..modules.coordinates import PolarPoint


@dataclass
class ValidationRow:
    label: str = ""
    perc_sonar_dcol: float = float("nan")
    perc_sonar_drow: float = float("nan")
    perc_sonar_dist: float = float("nan")
    track_origin_dist: float = float("nan")
    valid_ratio: float = 0.0


@dataclass
class ValidationReport:
    rows: list[ValidationRow] = field(default_factory=list)
    phase: str = "perc"  # "perc" | "track"

    def add_perc_sample(self, p_perc_abs: PolarPoint, p_sonar_abs: PolarPoint, *, label: str = "") -> None:
        du = p_perc_abs.col - p_sonar_abs.col
        dv = p_perc_abs.row - p_sonar_abs.row
        self.rows.append(
            ValidationRow(
                label=label,
                perc_sonar_dcol=du,
                perc_sonar_drow=dv,
                perc_sonar_dist=float(np.hypot(du, dv)),
            )
        )

    def add_track_sample(self, track_origin_dist: float, *, valid_ratio: float = 0.0, label: str = "") -> None:
        self.rows.append(
            ValidationRow(
                label=label,
                track_origin_dist=track_origin_dist,
                valid_ratio=valid_ratio,
            )
        )

    def perc_verdict(self) -> tuple[bool, str]:
        if not self.rows:
            return False, "no samples"
        dcol = np.abs([r.perc_sonar_dcol for r in self.rows if not np.isnan(r.perc_sonar_dcol)])
        drow = np.abs([r.perc_sonar_drow for r in self.rows if not np.isnan(r.perc_sonar_drow)])
        dist = np.array([r.perc_sonar_dist for r in self.rows if not np.isnan(r.perc_sonar_dist)])
        if len(dcol) == 0:
            return False, "no valid perc samples"
        ok = (
            float(np.mean(dcol)) < PERC_SONAR_COL_MEAN_MAX
            and float(np.percentile(dcol, 90)) < PERC_SONAR_COL_P90_MAX
            and float(np.mean(drow)) < PERC_SONAR_ROW_MEAN_MAX
            and float(np.mean(dist)) < PERC_SONAR_DIST_MEAN_MAX
        )
        msg = (
            f"|dcol| mean={np.mean(dcol):.2f} p90={np.percentile(dcol, 90):.2f} "
            f"|drow| mean={np.mean(drow):.2f} dist mean={np.mean(dist):.2f}"
        )
        return ok, msg

    def track_verdict(self) -> tuple[bool, str]:
        dists = [r.track_origin_dist for r in self.rows if not np.isnan(r.track_origin_dist)]
        if not dists:
            return False, "no track samples"
        mean_d = float(np.mean(dists))
        ok = mean_d < TRACK_ORIGIN_DIST_MAX
        return ok, f"||p_sonar|| mean={mean_d:.2f} px (max {TRACK_ORIGIN_DIST_MAX})"
