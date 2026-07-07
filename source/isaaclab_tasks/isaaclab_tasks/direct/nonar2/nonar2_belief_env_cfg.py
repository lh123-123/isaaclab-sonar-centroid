# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Belief + CNN feature observation config for SAC (design doc §3.4–3.5)."""

from isaaclab.utils import configclass

from .agents.policy_dims import BELIEF_SAC_OBS_DIM
from .nonar2_env_cfg import Nonar2EnvCfg


@configclass
class Nonar2BeliefEnvCfg(Nonar2EnvCfg):
    """262-dim belief state: 6 structured + 256 CNN features."""

    policy_obs_mode: str = "belief_sac"
    sonar_feature_dim: int = 256
    sonar_history_len: int = 3
    # [μx, μy, σx, σy, θ, |v|] + f_t(256)
    observation_space = BELIEF_SAC_OBS_DIM
