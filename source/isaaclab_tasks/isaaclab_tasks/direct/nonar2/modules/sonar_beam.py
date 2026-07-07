"""Sonar beam index ↔ body-frame azimuth (linear FOV mapping)."""

from __future__ import annotations

import math

import torch


def beam_center(beam_count: int) -> float:
    """Boresight beam index (e.g. 30 for 61 beams)."""
    return (beam_count - 1) / 2.0


def azimuth_rad_to_beam(
    azimuth_rad: float,
    *,
    hori_fov_deg: float,
    beam_count: int,
) -> int:
    """Map body-frame azimuth (rad) to beam index [0, beam_count-1].

    Linear FOV mapping: beam 0 → -FOV/2, center beam → 0°, last → +FOV/2.
    For 61 beams / 120° FOV: ``beam = az_deg / 2 + 30`` (2° per beam).
    """
    fov_rad = math.radians(float(hori_fov_deg))
    center = beam_center(beam_count)
    beam = center + float(azimuth_rad) / fov_rad * (beam_count - 1)
    return int(round(max(0.0, min(float(beam_count - 1), beam))))


def beam_to_azimuth_rad(
    beam: int,
    *,
    hori_fov_deg: float,
    beam_count: int,
) -> float:
    """Inverse of ``azimuth_rad_to_beam``."""
    fov_rad = math.radians(float(hori_fov_deg))
    center = beam_center(beam_count)
    return (float(beam) - center) / max(beam_count - 1, 1) * fov_rad


def azimuth_rad_to_beam_tensor(
    azimuth_rad: torch.Tensor,
    *,
    hori_fov_deg: float,
    beam_count: int,
) -> torch.Tensor:
    """Batch azimuth (rad) → beam index, rounded and clamped."""
    fov_rad = math.radians(float(hori_fov_deg))
    center = (beam_count - 1) / 2.0
    beam = center + azimuth_rad / fov_rad * (beam_count - 1)
    return beam.round().long().clamp(0, beam_count - 1)
