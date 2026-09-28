# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""SAC for Direct-v2: Actor and Critic share the flow image ⊕ geometry state."""

from __future__ import annotations

from stable_baselines3.sac.policies import SACPolicy


class FlowSACPolicy(SACPolicy):
    """Actor = CNN(image) ⊕ state; Critic = Q(same features, action).

    ``share_features_extractor=False`` so the twin Qs have their own CNN.
    Features come from ``FlowSonarFeaturesExtractor``.
    """

    def __init__(self, *args, **kwargs):
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)
