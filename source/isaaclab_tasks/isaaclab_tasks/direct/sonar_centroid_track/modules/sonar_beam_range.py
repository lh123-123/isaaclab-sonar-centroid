# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sonar ranging from OceanSim native beam data [angle, range, intensity]."""

from __future__ import annotations

import math

import torch


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _pick_peak_idx(
    scores: torch.Tensor,
    angles: torch.Tensor,
    *,
    prior_azimuth: torch.Tensor | None,
    prior_gate_deg: float,
    peak_threshold: float,
) -> torch.Tensor:
    """Argmax with optional azimuth gate; fall back to global peak if gate is empty."""
    global_peak_idx = torch.argmax(scores, dim=1)
    if prior_azimuth is None or float(prior_gate_deg) <= 0.0:
        return global_peak_idx

    gate_rad = math.radians(float(prior_gate_deg))
    delta = _wrap_to_pi(angles - prior_azimuth.unsqueeze(1))
    in_az_gate = torch.abs(delta) <= gate_rad
    prior_scores = torch.where(in_az_gate, scores, torch.full_like(scores, -1.0e6))
    prior_peak_idx = torch.argmax(prior_scores, dim=1)
    env_idx = torch.arange(scores.shape[0], device=scores.device)
    prior_peak_val = scores[env_idx, prior_peak_idx]
    use_prior = prior_peak_val >= float(peak_threshold)
    return torch.where(use_prior, prior_peak_idx, global_peak_idx)


def peak_beam_range_from_beams(
    beam_data: torch.Tensor,
    *,
    peak_threshold: float = 0.08,
    prior_azimuth: torch.Tensor | None = None,
    prior_gate_deg: float = 15.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick peak-intensity beam (optionally azimuth-gated); return range (m) per env.

    Args:
        beam_data: ``[num_envs, beam_count, 3]`` with columns
            ``(azimuth_rad, range_m, intensity)``.
        peak_threshold: Minimum peak intensity to mark detection valid.
        prior_azimuth: Optional ``[num_envs]`` gate center (rad).
        prior_gate_deg: Half-width azimuth gate (deg); 0 disables gating.

    Returns:
        ``range_m, azimuth_rad, peak_intensity, valid`` each ``[num_envs]``.
        ``valid`` is float 0/1 for downstream masking.
    """
    if beam_data.ndim != 3 or beam_data.shape[-1] != 3:
        raise ValueError(f"beam_data must be [N, B, 3], got {tuple(beam_data.shape)}")

    angles = beam_data[:, :, 0]
    ranges = beam_data[:, :, 1]
    intensity = beam_data[:, :, 2]

    peak_idx = _pick_peak_idx(
        intensity,
        angles,
        prior_azimuth=prior_azimuth,
        prior_gate_deg=prior_gate_deg,
        peak_threshold=peak_threshold,
    )
    env_idx = torch.arange(beam_data.shape[0], device=beam_data.device)
    range_m = ranges[env_idx, peak_idx]
    azimuth = angles[env_idx, peak_idx]
    peak_val = intensity[env_idx, peak_idx]
    valid = (peak_val >= float(peak_threshold)).float()
    return range_m, azimuth, peak_val, valid
