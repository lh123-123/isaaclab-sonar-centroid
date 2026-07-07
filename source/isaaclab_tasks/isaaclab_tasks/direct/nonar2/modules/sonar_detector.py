"""Module B: lightweight sonar target detection from beam / image data."""

from __future__ import annotations

import math

import torch


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


class SonarDetector:
    """Peak-beam detector — placeholder for YOLOv8-nano fine-tuning."""

    def __init__(
        self,
        conf_threshold: float = 0.15,
        peak_threshold: float = 0.20,
        mean_threshold: float = 0.03,
    ):
        self.conf_threshold = conf_threshold
        self.peak_threshold = peak_threshold
        self.mean_threshold = mean_threshold

    def _pick_peak_idx(
        self,
        scores: torch.Tensor,
        angles: torch.Tensor,
        *,
        prior_azimuth: torch.Tensor | None,
        prior_gate_deg: float,
    ) -> torch.Tensor:
        """Argmax over score rows with optional azimuth gate."""
        global_peak_idx = torch.argmax(scores, dim=1)
        if prior_azimuth is None:
            return global_peak_idx

        gate_rad = math.radians(float(prior_gate_deg))
        delta = _wrap_to_pi(angles - prior_azimuth.unsqueeze(1))
        in_az_gate = torch.abs(delta) <= gate_rad
        prior_scores = torch.where(in_az_gate, scores, torch.full_like(scores, -1.0e6))
        prior_peak_idx = torch.argmax(prior_scores, dim=1)
        env_idx = torch.arange(scores.shape[0], device=scores.device)
        prior_peak_val = scores[env_idx, prior_peak_idx]
        use_prior = prior_peak_val >= self.peak_threshold
        return torch.where(use_prior, prior_peak_idx, global_peak_idx)

    def detect_from_beams(
        self,
        beam_data: torch.Tensor,
        *,
        intensity_base: torch.Tensor | None = None,
        range_min: float | None = None,
        range_max: float | None = None,
        prior_azimuth: torch.Tensor | None = None,
        prior_gate_deg: float = 35.0,
        col_contrast: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract polar detections from beam triplets [N, B, 3] (angle rad, range m, intensity).

        When ``prior_azimuth`` is set, pick the strongest in-band peak within ±``prior_gate_deg``
        of the prior instead of the global argmax (reduces off-target clutter peaks).

        Returns:
            azimuth [N], range [N], confidence [N], detected [N] bool
        """
        if beam_data.ndim != 3 or beam_data.shape[-1] != 3:
            raise ValueError(f"Expected [N,B,3], got {tuple(beam_data.shape)}")

        angles = beam_data[:, :, 0]
        ranges = beam_data[:, :, 1]
        raw_intensity = beam_data[:, :, 2] if intensity_base is None else intensity_base

        in_band = torch.ones_like(raw_intensity, dtype=torch.bool)
        if range_min is not None:
            in_band = in_band & (ranges >= float(range_min))
        if range_max is not None:
            in_band = in_band & (ranges <= float(range_max))

        if range_min is not None or range_max is not None:
            intensity = torch.where(in_band, raw_intensity, torch.full_like(raw_intensity, -1.0e6))
            band_count = in_band.float().sum(dim=1).clamp(min=1.0)
            mean_val = torch.where(in_band, raw_intensity, torch.zeros_like(raw_intensity)).sum(dim=1) / band_count
        else:
            intensity = raw_intensity
            mean_val = raw_intensity.mean(dim=1)

        scores = intensity
        if col_contrast:
            scores = intensity - intensity.median(dim=1, keepdim=True).values

        peak_idx = self._pick_peak_idx(
            scores,
            angles,
            prior_azimuth=prior_azimuth,
            prior_gate_deg=prior_gate_deg,
        )

        env_idx = torch.arange(beam_data.shape[0], device=beam_data.device)
        azimuth = angles[env_idx, peak_idx]
        target_range = ranges[env_idx, peak_idx]
        peak_val = raw_intensity[env_idx, peak_idx]

        conf = torch.clamp(peak_val, 0.0, 1.0)
        detected = (peak_val >= self.peak_threshold) & (mean_val >= self.mean_threshold)
        detected = detected & (conf >= self.conf_threshold)

        azimuth = torch.where(detected, azimuth, torch.zeros_like(azimuth))
        target_range = torch.where(
            detected,
            target_range,
            torch.full_like(target_range, float("inf")),
        )
        conf = torch.where(detected, conf, torch.zeros_like(conf))
        return azimuth, target_range, conf, detected

    def detect_from_image_roi(
        self,
        images: torch.Tensor,
        *,
        hori_fov_deg: float,
        min_range: float,
        max_range: float,
        range_min: float,
        range_max: float,
        prior_azimuth: torch.Tensor | None = None,
        prior_gate_deg: float = 35.0,
        col_contrast: bool = False,
        peak_relative_min: float = 0.35,
        range_weight_sigma: float = 0.0,
        azimuth_margin_cols: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect from raw sonar via per-column range peak inside the band.

        Each azimuth column contributes one peak (brightest pixel along range). This
        avoids global 2D argmax locking onto vertical multipath streaks below the
        true target blob (row 0 = near range at image top).
        """
        if images.ndim != 3:
            raise ValueError(f"Expected [N,H,W], got {tuple(images.shape)}")

        n_envs, height, width = images.shape
        span = max(float(max_range) - float(min_range), 1.0e-6)
        row_min = int(round((float(range_min) - float(min_range)) / span * max(height - 1, 0)))
        row_max = int(round((float(range_max) - float(min_range)) / span * max(height - 1, 0)))
        row_min = max(0, min(height - 1, row_min))
        row_max = max(row_min, min(height - 1, row_max))

        roi = images[:, row_min : row_max + 1, :]
        flat = roi.reshape(n_envs, -1)
        denom = float(max(height - 1, 1))

        col_peak_vals, col_peak_rows = roi.max(dim=1)

        scores = col_peak_vals
        if float(range_weight_sigma) > 0.0:
            r_center = 0.5 * (float(range_min) + float(range_max))
            range_m = float(min_range) + ((col_peak_rows.float() + float(row_min)) / denom) * span
            sigma = float(range_weight_sigma)
            range_w = torch.exp(-0.5 * ((range_m - r_center) / sigma) ** 2)
            scores = col_peak_vals * range_w

        fov_rad = math.radians(float(hori_fov_deg))
        cols = torch.arange(width, device=images.device, dtype=images.dtype)
        az_per_col = -((cols - (width - 1) / 2.0) / max(width - 1, 1) * fov_rad)
        az_grid = az_per_col.unsqueeze(0).expand(n_envs, -1)

        if col_contrast:
            scores = scores - scores.median(dim=1, keepdim=True).values

        margin = max(0, int(azimuth_margin_cols))
        if margin > 0 and scores.shape[1] > 2 * margin:
            scores = scores.clone()
            scores[:, :margin] = -1.0e6
            scores[:, -margin:] = -1.0e6

        peak_col = self._pick_peak_idx(
            scores,
            az_grid,
            prior_azimuth=prior_azimuth,
            prior_gate_deg=prior_gate_deg,
        )

        env_idx = torch.arange(n_envs, device=images.device)
        peak_val = col_peak_vals[env_idx, peak_col]
        peak_row_local = col_peak_rows[env_idx, peak_col]
        actual_row = peak_row_local.float() + float(row_min)
        target_range = float(min_range) + (actual_row / denom) * span
        azimuth = az_grid[env_idx, peak_col]

        roi_max = flat.max(dim=1).values
        roi_median = flat.median(dim=1).values
        snr = peak_val / roi_median.clamp(min=1.0e-4)
        peak_floor = torch.maximum(
            torch.full_like(roi_max, self.peak_threshold),
            roi_max * float(peak_relative_min),
        )

        conf = torch.clamp(peak_val / roi_max.clamp(min=1.0e-4), 0.0, 1.0)
        detected = (peak_val >= peak_floor) & (snr >= 2.0)
        detected = detected & (conf >= self.conf_threshold)

        azimuth = torch.where(detected, azimuth, torch.zeros_like(azimuth))
        target_range = torch.where(
            detected,
            target_range,
            torch.full_like(target_range, float("inf")),
        )
        conf = torch.where(detected, conf, torch.zeros_like(conf))
        return azimuth, target_range, conf, detected
