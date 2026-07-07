# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sonar capture → preprocess → feature tracking pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from ..sonar_preprocess import preprocess_polar_batch, preprocess_sonar_batch
from ..sparse_feature_tracker import SparseFeatureTracker

if TYPE_CHECKING:
    from ..sonar_ibvs_drl_env_cfg import SonarIbvsDrlEnvCfg

PerceptionFrame = Literal["cart", "polar"]


class SonarPerceptionPipeline:
    """Cached sonar → (polar|cart) → sparse-feature pipeline for one simulation step."""

    def __init__(self, cfg: SonarIbvsDrlEnvCfg, num_envs: int, device: str | torch.device):
        ocfg = cfg.oceansim_sonar_cfg
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.max_range = float(ocfg.max_range)
        self.min_range = float(ocfg.min_range)
        self.hori_fov_deg = float(ocfg.hori_fov)
        self.perception_frame: PerceptionFrame = str(getattr(cfg, "perception_frame", "cart")).lower()  # type: ignore[assignment]
        if self.perception_frame not in ("cart", "polar"):
            raise ValueError(f"perception_frame must be cart|polar, got {self.perception_frame!r}")

        polar_h = int(ocfg.height)
        polar_w = int(ocfg.width)
        cart_size = int(cfg.cart_image_size)

        self.sonar_image = torch.zeros(num_envs, polar_h, polar_w, device=device)
        self.polar_image = torch.zeros(num_envs, polar_h, polar_w, device=device)
        self.cart_image = torch.zeros(num_envs, cart_size, cart_size, device=device)

        if self.perception_frame == "polar":
            track_w, track_h = polar_w, polar_h
        else:
            track_w = track_h = cart_size

        self.feature_tracker = SparseFeatureTracker(
            num_envs,
            device,
            max_extract=int(cfg.max_extract_features),
            max_track=int(cfg.max_track_features),
            image_size=max(track_w, track_h),
            image_width=track_w,
            image_height=track_h,
            fb_error_threshold=float(cfg.fb_error_threshold),
            patch_half_size=int(getattr(cfg, "feature_patch_half_size", 6)),
            relocate_window=int(getattr(cfg, "feature_relocate_window", 18)),
            relocate_min_score=float(getattr(cfg, "feature_relocate_min_score", 0.55)),
            ref_min_self_score=float(getattr(cfg, "feature_ref_min_self_score", 0.72)),
            ref_min_peak_margin=float(getattr(cfg, "feature_ref_min_peak_margin", 0.10)),
            track_min_peak_margin=float(getattr(cfg, "feature_track_min_peak_margin", 0.06)),
            blob_threshold_ratio=float(getattr(cfg, "feature_contour_threshold_ratio", 0.25)),
            blob_min_area=float(getattr(cfg, "feature_contour_min_area", 4.0)),
            tracking_mode=str(getattr(cfg, "feature_tracking_mode", "patch")),
            coord_frame=self.perception_frame,
        )
        self._cache_step: int = -1

    @property
    def track_image(self) -> torch.Tensor:
        """Image used for sparse feature IBVS (polar or cart depending on cfg)."""
        return self.polar_image if self.perception_frame == "polar" else self.cart_image

    def invalidate_cache(self) -> None:
        self._cache_step = -1

    def reset(self, env_ids: torch.Tensor) -> None:
        self.feature_tracker.reset(env_ids)
        self.invalidate_cache()

    def ensure_updated(self, step_counter: int, oceansim_sonar, sim) -> None:
        if self._cache_step == step_counter:
            return
        self._update(oceansim_sonar, sim)
        self._cache_step = step_counter

    def _update(self, oceansim_sonar, sim) -> None:
        if not bool(getattr(oceansim_sonar, "_initialized", False)):
            return
        sim.render()
        dt = float(self.cfg.sim.dt * self.cfg.decimation)
        oceansim_sonar.update(dt=dt)
        self.sonar_image = oceansim_sonar.get_sonar_image()
        self.polar_image = preprocess_polar_batch(
            self.sonar_image,
            skip_median=bool(getattr(self.cfg, "skip_median_filter", False)),
        )
        if self.perception_frame == "cart" or bool(getattr(self.cfg, "keep_cart_for_viz", True)):
            self.cart_image = preprocess_sonar_batch(
                self.sonar_image,
                hori_fov_deg=self.hori_fov_deg,
                max_range=self.max_range,
                cart_size=int(self.cfg.cart_image_size),
            )
        self.feature_tracker.update(self.track_image)
        if bool(getattr(self.cfg.oceansim_sonar_cfg, "sonar_initialize_viewport", False)) and sim.has_gui():
            try:
                import omni.kit.app

                omni.kit.app.get_app().update()
            except Exception:
                pass

    def metrics(self, env_id: int = 0) -> dict[str, float | int | str]:
        ft = self.feature_tracker
        return {
            "valid_pts": int(ft.valid_count()[env_id].item()),
            "pixel_err": float(ft.normalized_pixel_error()[env_id].item()),
            "daz_norm": float(ft.mean_lateral_error_signed()[env_id].item()),
            "dr_norm": float(ft.mean_range_error_signed()[env_id].item()),
            "perception_frame": self.perception_frame,
            "max_track": int(self.cfg.max_track_features),
        }
