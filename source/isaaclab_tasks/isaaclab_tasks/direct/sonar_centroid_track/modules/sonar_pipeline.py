# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Polar-only sonar capture and preprocessing (no cart)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .sonar_preprocess import preprocess_polar_batch

if TYPE_CHECKING:
    from ..sonar_centroid_track_env_cfg import SonarCentroidTrackEnvCfg


class PolarSonarPipeline:
    """OceanSim render → log+median polar; one update per sim step."""

    def __init__(self, cfg: SonarCentroidTrackEnvCfg, num_envs: int, device: str | torch.device):
        ocfg = cfg.oceansim_sonar_cfg
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.polar_height = int(ocfg.height)
        self.polar_width = int(ocfg.width)
        self.max_range = float(ocfg.max_range)
        self.min_range = float(ocfg.min_range)
        self.hori_fov_deg = float(ocfg.hori_fov)

        self.sonar_image = torch.zeros(num_envs, self.polar_height, self.polar_width, device=device)
        self.polar_image = torch.zeros(num_envs, self.polar_height, self.polar_width, device=device)
        self._cache_step: int = -1

    def invalidate_cache(self) -> None:
        self._cache_step = -1

    def ensure_updated(self, step_counter: int, oceansim_sonar, sim) -> torch.Tensor:
        if self._cache_step != step_counter:
            self._update(oceansim_sonar, sim)
            self._cache_step = step_counter
        return self.polar_image

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
        # Do not call omni.kit.app.get_app().update() here: in GUI eval loops it
        # advances Kit an extra tick per sonar read and doubles robot displacement.

    def warmup(self, oceansim_sonar, sim, *, n_frames: int = 4) -> None:
        for _ in range(max(int(n_frames), 1)):
            self.invalidate_cache()
            self._update(oceansim_sonar, sim)
