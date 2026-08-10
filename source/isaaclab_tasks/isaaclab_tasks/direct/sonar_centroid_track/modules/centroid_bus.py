# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Module M6: single-frame bundle of sonar / geometry centroids (debug-friendly)."""

from __future__ import annotations

from dataclasses import dataclass

from .coordinates import PolarPoint


@dataclass
class CentroidFrame:
    """One timestep: image sonar + optional geometry centroids (abs and fixed-point)."""

    p_sonar_abs: PolarPoint | None = None
    p_perc_abs: PolarPoint | None = None
    p_sonar: PolarPoint | None = None
    p_perc: PolarPoint | None = None
    valid_ratio: float = 0.0

    @property
    def perc_sonar_gap_abs(self) -> PolarPoint | None:
        if self.p_perc_abs is None or self.p_sonar_abs is None:
            return None
        return self.p_perc_abs - self.p_sonar_abs

    @property
    def sonar_origin_dist(self) -> float | None:
        if self.p_sonar is None:
            return None
        return self.p_sonar.norm()

    @property
    def sonar_d_row_px(self) -> float | None:
        """Green minus yellow row (px); positive → detected farther than teach."""
        if self.p_sonar is None:
            return None
        return float(self.p_sonar.row)

    def sonar_range_abs_m(
        self,
        *,
        min_range: float,
        max_range: float,
        polar_height: int,
    ) -> float | None:
        if self.p_sonar_abs is None:
            return None
        from .coordinates import polar_row_to_range_m

        return polar_row_to_range_m(
            self.p_sonar_abs.row,
            min_range=min_range,
            max_range=max_range,
            polar_height=polar_height,
        )

    def sonar_d_range_m(
        self,
        *,
        range_ref_m: float,
        min_range: float,
        max_range: float,
        polar_height: int,
    ) -> float | None:
        rng = self.sonar_range_abs_m(
            min_range=min_range, max_range=max_range, polar_height=polar_height
        )
        if rng is None:
            return None
        return rng - float(range_ref_m)

    def summary(self) -> str:
        def _fmt(p: PolarPoint | None) -> str:
            return "None" if p is None else f"({p.col:.1f},{p.row:.1f})"

        gap = self.perc_sonar_gap_abs
        gap_s = "None" if gap is None else f"du={gap.col:+.2f} dv={gap.row:+.2f}"
        return (
            f"sonar_abs={_fmt(self.p_sonar_abs)} perc_abs={_fmt(self.p_perc_abs)} "
            f"sonar_rel={_fmt(self.p_sonar)} valid={self.valid_ratio:.2f} gap={gap_s} "
            f"||sonar||={self.sonar_origin_dist}"
        )


def make_centroid_frame(
    *,
    p_sonar_abs: PolarPoint | None,
    p_perc_abs: PolarPoint | None,
    p_fixed_abs: PolarPoint | None,
    valid_ratio: float = 0.0,
) -> CentroidFrame:
    from .coordinates import to_fixed_frame

    def _rel(p_abs: PolarPoint | None) -> PolarPoint | None:
        if p_abs is None or p_fixed_abs is None:
            return None
        return to_fixed_frame(p_abs, p_fixed_abs)

    return CentroidFrame(
        p_sonar_abs=p_sonar_abs,
        p_perc_abs=p_perc_abs,
        p_sonar=_rel(p_sonar_abs),
        p_perc=_rel(p_perc_abs),
        valid_ratio=valid_ratio,
    )
