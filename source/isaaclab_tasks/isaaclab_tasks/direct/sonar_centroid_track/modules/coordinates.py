# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Polar pixel coordinates: absolute frame and fixed-point origin frame."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PolarPoint:
    """Polar image pixel (col=azimuth, row=range)."""

    col: float
    row: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.col, self.row)

    def norm(self) -> float:
        return math.hypot(self.col, self.row)

    def __sub__(self, other: PolarPoint) -> PolarPoint:
        return PolarPoint(self.col - other.col, self.row - other.row)


def polar_point(col: float, row: float) -> PolarPoint:
    return PolarPoint(float(col), float(row))


def mirror_azimuth_col(col: float, width: int) -> float:
    """Map OceanSim column index to geometry-aligned azimuth column."""
    return float(width - 1) - float(col)


def to_fixed_frame(p_abs: PolarPoint | Sequence[float], p_fixed_abs: PolarPoint | Sequence[float]) -> PolarPoint:
    """Absolute polar pixel → fixed-point frame (origin = teach alignment)."""
    if not isinstance(p_abs, PolarPoint):
        p_abs = polar_point(p_abs[0], p_abs[1])
    if not isinstance(p_fixed_abs, PolarPoint):
        p_fixed_abs = polar_point(p_fixed_abs[0], p_fixed_abs[1])
    return p_abs - p_fixed_abs


def to_absolute_frame(p_rel: PolarPoint | Sequence[float], p_fixed_abs: PolarPoint | Sequence[float]) -> PolarPoint:
    if not isinstance(p_rel, PolarPoint):
        p_rel = polar_point(p_rel[0], p_rel[1])
    if not isinstance(p_fixed_abs, PolarPoint):
        p_fixed_abs = polar_point(p_fixed_abs[0], p_fixed_abs[1])
    return PolarPoint(p_rel.col + p_fixed_abs.col, p_rel.row + p_fixed_abs.row)


def normalize_uv(p_rel: PolarPoint, width: int, height: int) -> tuple[float, float]:
    """Fixed-point coords normalized for RL obs."""
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    return p_rel.col / w, p_rel.row / h
