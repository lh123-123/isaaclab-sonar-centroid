"""Module E reward terms — simple angle-based reward.

R_detected = +5.0 - 0.15 * |det_az_deg|   (居中=+5, 偏20deg=+2, 偏33deg=0)
R_lost = -5.0
"""

from __future__ import annotations

import torch


def compute_action_reward(
    *,
    action: torch.Tensor,
    detected: torch.Tensor,
    det_azimuth: torch.Tensor | None,
    r_detect: float = 5.0,
    r_deviation_k: float = 0.15,
    r_miss: float = -5.0,
    r_smooth: float = 0.010,
    **_: object,
) -> torch.Tensor:
    return compute_action_reward_breakdown(
        action=action,
        detected=detected,
        det_azimuth=det_azimuth,
        r_detect=r_detect,
        r_deviation_k=r_deviation_k,
        r_miss=r_miss,
        r_smooth=r_smooth,
    )["total"]


def compute_action_reward_breakdown(
    *,
    action: torch.Tensor,
    detected: torch.Tensor,
    det_azimuth: torch.Tensor | None,
    r_detect: float = 5.0,
    r_deviation_k: float = 0.15,
    r_miss: float = -5.0,
    r_smooth: float = 0.010,
    **_: object,
) -> dict[str, torch.Tensor]:
    """Simple angle-based reward.

    When detected: R = r_detect - r_deviation_k * |det_az_deg| - smooth
    When lost:     R = r_miss - smooth
    """
    act = action.squeeze(-1)
    dtype = act.dtype
    device = act.device

    # --- Detect base ---
    r_detect_term = torch.where(
        detected,
        torch.full_like(act, float(r_detect)),
        torch.zeros_like(act),
    )

    # --- Deviation (linear in angle) ---
    r_deviation_term = torch.zeros(detected.shape[0], device=device, dtype=dtype)
    if float(r_deviation_k) != 0.0 and det_azimuth is not None:
        az_deg = torch.rad2deg(torch.abs(det_azimuth))
        r_deviation_term = torch.where(
            detected,
            -float(r_deviation_k) * az_deg,
            torch.zeros_like(az_deg),
        )

    # --- Miss ---
    r_miss_term = torch.where(
        detected,
        torch.zeros(detected.shape[0], device=device, dtype=dtype),
        torch.full_like(act, float(r_miss)),
    )

    # --- Smooth ---
    r_smooth_term = -float(r_smooth) * torch.square(act)

    total = r_detect_term + r_deviation_term + r_miss_term + r_smooth_term
    return {
        "detect": r_detect_term,
        "deviation": r_deviation_term,
        "miss": r_miss_term,
        "smooth": r_smooth_term,
        "total": total,
    }
