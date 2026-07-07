"""Sonar pipeline debug: save raw / processed images and mark detection peak."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def _to_uint8_gray(image: torch.Tensor) -> np.ndarray:
    """[H, W] float tensor -> uint8 grayscale numpy."""
    arr = image.detach().float().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def peak_pixel_from_polar(
    azimuth_rad: float,
    target_range: float,
    *,
    height: int,
    width: int,
    hori_fov_deg: float,
    min_range: float,
    max_range: float,
) -> tuple[int, int]:
    """Map body-frame (azimuth rad, range m) to sonar image (row, col); row 0 ≈ near.

    Body bearing: + = target left (+Y), − = target right (−Y).
    Image columns: col > center = right side of display = negative body bearing.
    """
    az_deg = float(np.rad2deg(azimuth_rad))
    col = int(round(width / 2.0 - az_deg * width / float(hori_fov_deg)))
    col = max(0, min(width - 1, col))
    span = max(float(max_range) - float(min_range), 1.0e-6)
    row = int(round((float(target_range) - float(min_range)) / span * max(height - 1, 0)))
    row = max(0, min(height - 1, row))
    return row, col


def draw_peak_marker(
    rgb: np.ndarray,
    row: int,
    col: int,
    *,
    radius: int = 2,
    color: tuple[int, int, int] = (255, 0, 0),
) -> None:
    """Draw a cross on an H×W×3 RGB image in-place."""
    h, w = rgb.shape[:2]
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = row + dr, col + dc
            if 0 <= r < h and 0 <= c < w and (dr == 0 or dc == 0):
                rgb[r, c] = color


def draw_crosses_on_gray(
    gray: np.ndarray,
    markers: list[tuple[int, int, tuple[int, int, int]]],
    *,
    radius: int = 2,
) -> np.ndarray:
    """Return RGB image with crosses at (row, col, rgb) tuples."""
    rgb = np.stack([gray, gray, gray], axis=-1)
    for row, col, color in markers:
        draw_peak_marker(rgb, row, col, radius=radius, color=color)
    return rgb


def save_sonar_debug_triplet(
    *,
    raw: torch.Tensor,
    processed: torch.Tensor,
    peak_row: int | None,
    peak_col: int | None,
    save_dir: str | Path,
    step: int,
    env_idx: int = 0,
    gt_row: int | None = None,
    gt_col: int | None = None,
) -> None:
    """Save raw.png, processed.png, peak_overlay.png under save_dir.

    Red cross = detection peak; green cross = GT target (sonar frame) when provided.
    """
    try:
        from PIL import Image
    except ImportError:
        return

    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"step{step:06d}_env{env_idx}"

    raw_u8 = _to_uint8_gray(raw)
    proc_u8 = _to_uint8_gray(processed)
    Image.fromarray(raw_u8, mode="L").save(out / f"{prefix}_raw.png")
    Image.fromarray(proc_u8, mode="L").save(out / f"{prefix}_processed.png")

    if peak_row is not None and peak_col is not None:
        markers = [(peak_row, peak_col, (255, 0, 0))]
        if gt_row is not None and gt_col is not None:
            markers.append((gt_row, gt_col, (0, 255, 0)))
        overlay = draw_crosses_on_gray(raw_u8, markers)
        Image.fromarray(overlay, mode="RGB").save(out / f"{prefix}_peak_on_raw.png")
        overlay_p = draw_crosses_on_gray(proc_u8, markers)
        Image.fromarray(overlay_p, mode="RGB").save(out / f"{prefix}_peak_on_processed.png")


def img_stats(image: torch.Tensor) -> tuple[float, float, float]:
    nz = (image > 1.0e-4).float().mean().item()
    return float(image.mean().item()), float(image.max().item()), nz
