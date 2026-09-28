# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# Portions Copyright (c) Facebook, Inc. and its affiliates (MIT) — DrQ-v2 utils.

"""Minimal DrQ-v2 helpers (schedule, TruncatedNormal, weight init)."""

from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float) -> None:
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def to_torch(xs, device: torch.device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data") and m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data") and m.bias is not None:
            m.bias.data.fill_(0.0)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low: float = -1.0, high: float = 1.0, eps: float = 1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clamped_x.detach()

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps = eps * self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        return self._clamp(self.loc + eps)


def schedule(schdl, step: int) -> float:
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = (float(g) for g in match.groups())
            mix = float(np.clip(step / duration, 0.0, 1.0))
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = (float(g) for g in match.groups())
            if step <= duration1:
                mix = float(np.clip(step / duration1, 0.0, 1.0))
                return (1.0 - mix) * init + mix * final1
            mix = float(np.clip((step - duration1) / duration2, 0.0, 1.0))
            return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)
