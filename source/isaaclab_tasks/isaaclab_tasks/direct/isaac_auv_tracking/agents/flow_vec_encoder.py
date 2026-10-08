# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Feature extractor for a pre-structured flow vector: no convolution, no spatial grid.

`MidCNNExtractor` (cam_encoder.py) exists because an image is a *raster*: the information sits
in the pixel layout, so convolutions have to rediscover where things are. Direct-v6-flowvec
instead feeds the five analytic numbers that the optical-flow raster in Direct-v5-flow is merely
a picture of -- (u, v, radius, du, dv) per frame, stacked over `frame_stack` frames. There is no
layout left to exploit, so a convolution would be pure overhead and, worse, would have to undo
the rasterisation first.

This extractor is therefore the trivial-but-correct one: flatten, project to `features_dim`,
normalise, Tanh. It mirrors the tail of `MidCNNExtractor.trunk`
(`Linear -> LayerNorm -> Tanh`) so the two are drop-in comparable at the same `features_dim`,
and it keeps the `observations["image"]` key so `custom_sac_policy.CustomSACPolicy` and the
SAC actor plumbing need no changes at all.

Design note -- why `nn.LayerNorm` and not `nn.BatchNorm2d`: there is no batch of pixels to
statistic over, and LayerNorm is what the CNN counterpart already uses at this point, so the
comparison stays clean.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FlowVectorExtractor(BaseFeaturesExtractor):
    """Flow vector ``(D,)`` -> ``(features_dim,)`` via a single Linear + LayerNorm + Tanh.

    The vector is expected to be already normalised to roughly ``[-1, 1]`` by the env (see
    ``AUVTrackingV6FlowVecEnv._flow_vec_now``), so nothing here rescales it. That is deliberate
    and it is also the one place this setup differs from the raster: the raster's `du`/`dv`
    reach conv1 at their raw ~0.02 magnitude, while this path can be given whatever gain the
    caller chooses. See the `flow_vec_dv_gain` config option.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 512,
        hidden: int | None = None,
    ) -> None:
        super().__init__(observation_space, features_dim=int(features_dim))
        # A Dict space here is the {"image", "state"} wrapper; the vector itself is "image".
        if isinstance(observation_space, spaces.Dict):
            image_space = observation_space["image"]
        else:
            image_space = observation_space
        if len(image_space.shape) != 1:
            raise ValueError(
                f"FlowVectorExtractor expects a 1-D vector, got shape {tuple(image_space.shape)}. "
                "A (C, H, W) input is what MidCNNExtractor is for."
            )
        in_dim = int(image_space.shape[0])

        layers: list[nn.Module] = []
        if hidden is not None and int(hidden) > 0:
            layers += [nn.Linear(in_dim, int(hidden)), nn.LayerNorm(int(hidden)), nn.Tanh()]
            in_dim = int(hidden)
        layers += [
            nn.Linear(in_dim, int(features_dim)),
            nn.LayerNorm(int(features_dim)),
            nn.Tanh(),
        ]
        self.trunk = nn.Sequential(*layers)
        # With no hidden layer this is the whole model, so an empty conv stack is expected; the
        # debug probe asserts a CNN critic has zero convolutions, and this makes the actor
        # symmetric with that expectation.
        self.cnn = nn.Sequential()

    def forward(self, observations: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        vec = observations["image"] if isinstance(observations, dict) else observations
        # (B, D) for the single-env case; keep it robust to a flattened (B, D) that some callers
        # (SAC's obs tensors) already are, and to (B, 1, D) if a wrapper ever adds an axis.
        vec = vec.float().reshape(vec.shape[0], -1)
        return self.trunk(vec)
