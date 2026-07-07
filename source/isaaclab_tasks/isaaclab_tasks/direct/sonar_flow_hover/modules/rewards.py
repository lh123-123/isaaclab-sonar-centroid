# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""IBVS-DRL reward computation."""

from __future__ import annotations

import torch

from ..sparse_feature_tracker import SparseFeatureTracker


def mean_lateral_pixel_error(feature_tracker: SparseFeatureTracker) -> torch.Tensor:
    """Mean |lateral IBVS error|: cart ex or polar Δaz (normalized)."""
    return feature_tracker.mean_lateral_error_abs()


def centroid_lateral_pixel_error(feature_tracker: SparseFeatureTracker) -> torch.Tensor:
    """|Teach-to-track centroid lateral offset| in normalized sonar coords."""
    return feature_tracker.centroid_lateral_error_abs()


def compute_reward_terms(
    *,
    feature_tracker: SparseFeatureTracker,
    actions: torch.Tensor,
    depth: torch.Tensor,
    w1_pixel: float,
    w2_action: float,
    w3_safety: float,
    min_depth: float,
    max_depth: float,
    min_valid_features: int,
) -> dict[str, torch.Tensor]:
    """Per-env reward breakdown (same terms as ``compute_reward``)."""
    pixel_error = feature_tracker.normalized_pixel_error()
    action_penalty = torch.mean(actions**2, dim=1)
    valid_n = feature_tracker.valid_count()
    depth_low = depth < min_depth
    depth_high = depth > max_depth
    too_few_features = valid_n < min_valid_features
    safety = (depth_low | depth_high | too_few_features).float()
    term_pixel = -w1_pixel * pixel_error
    term_action = -w2_action * action_penalty
    term_safety = -w3_safety * safety
    return {
        "pixel_error": pixel_error,
        "action_penalty": action_penalty,
        "valid_n": valid_n.float(),
        "safety": safety,
        "depth_low": depth_low.float(),
        "depth_high": depth_high.float(),
        "too_few_features": too_few_features.float(),
        "term_pixel": term_pixel,
        "term_action": term_action,
        "term_safety": term_safety,
        "total": term_pixel + term_action + term_safety,
    }


def compute_reward(
    *,
    feature_tracker: SparseFeatureTracker,
    actions: torch.Tensor,
    depth: torch.Tensor,
    w1_pixel: float,
    w2_action: float,
    w3_safety: float,
    min_depth: float,
    max_depth: float,
    min_valid_features: int,
) -> torch.Tensor:
    return compute_reward_terms(
        feature_tracker=feature_tracker,
        actions=actions,
        depth=depth,
        w1_pixel=w1_pixel,
        w2_action=w2_action,
        w3_safety=w3_safety,
        min_depth=min_depth,
        max_depth=max_depth,
        min_valid_features=min_valid_features,
    )["total"]


def geometric_centroid_lateral_error_px(
    centroid_err_px: torch.Tensor,
    *,
    use_signed_sq: bool = False,
) -> torch.Tensor:
    """Lateral centroid penalty from bias-corrected signed column error (px)."""
    signed = centroid_err_px[:, 0]
    return signed * signed if use_signed_sq else signed.abs()


def compute_reward_lateral_1d_terms(
    *,
    feature_tracker: SparseFeatureTracker,
    actions: torch.Tensor,
    depth: torch.Tensor,
    w1_pixel: float,
    w2_action: float,
    w3_safety: float,
    min_depth: float,
    max_depth: float,
    min_valid_features: int,
    centroid_err: torch.Tensor | None = None,
    lateral_reward_pixel_sq: bool = False,
) -> dict[str, torch.Tensor]:
    """1D lateral reward: bias-corrected |signed centroid lateral px| + sway + safety."""
    if centroid_err is not None:
        pixel_error = geometric_centroid_lateral_error_px(
            centroid_err, use_signed_sq=lateral_reward_pixel_sq
        )
    else:
        pixel_error = centroid_lateral_pixel_error(feature_tracker)
    action_penalty = actions[:, 0] ** 2 if actions.shape[-1] == 1 else actions[:, 1] ** 2
    valid_n = feature_tracker.valid_count()
    depth_low = depth < min_depth
    depth_high = depth > max_depth
    too_few_features = valid_n < min_valid_features
    safety = (depth_low | depth_high | too_few_features).float()
    term_pixel = -w1_pixel * pixel_error
    term_action = -w2_action * action_penalty
    term_safety = -w3_safety * safety
    return {
        "pixel_error": pixel_error,
        "action_penalty": action_penalty,
        "valid_n": valid_n.float(),
        "safety": safety,
        "depth_low": depth_low.float(),
        "depth_high": depth_high.float(),
        "too_few_features": too_few_features.float(),
        "term_pixel": term_pixel,
        "term_action": term_action,
        "term_safety": term_safety,
        "total": term_pixel + term_action + term_safety,
    }


def compute_reward_lateral_1d(**kwargs) -> torch.Tensor:
    return compute_reward_lateral_1d_terms(**kwargs)["total"]
