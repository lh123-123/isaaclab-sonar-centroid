# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sonar polar image encoder for asymmetric SAC (Holo v1 Actor path)."""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SonarEncoder(BaseFeaturesExtractor):
    """Encode ``obs['image']`` (3×H×W uint8/float) → ``features_dim``.

    Compact CNN suited to 64×64 sonar polar (Holo uses ResNet on RGB; we keep the
    same Actor-only image interface without pulling torchvision ResNet50).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 512,
        resnet_output_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim=1)
        image_space = observation_space["image"]
        n_input = int(image_space.shape[0])
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_input, int(image_space.shape[1]), int(image_space.shape[2]))
            n_flat = int(self.cnn(dummy).shape[1])
        self.trunk = nn.Sequential(
            nn.Linear(n_flat, int(resnet_output_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(resnet_output_dim), int(features_dim)),
            nn.LayerNorm(int(features_dim)),
            nn.Tanh(),
        )
        self._features_dim = int(features_dim)

    def forward(self, observations: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        images = observations["image"] if isinstance(observations, dict) else observations
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.float()
            if float(images.max()) > 1.5:
                images = images / 255.0
        # ImageNet-style norm (Holo Encoder); harmless for replicated sonar channels.
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        if images.shape[1] == 3:
            images = (images - mean) / std
        return self.trunk(self.cnn(images))
