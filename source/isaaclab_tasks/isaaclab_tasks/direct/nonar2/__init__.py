# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Nonar2-Direct-v0",
    entry_point=f"{__name__}.nonar2_env:Nonar2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nonar2_env_cfg:Nonar2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Nonar2-Belief-Direct-v0",
    entry_point=f"{__name__}.nonar2_env:Nonar2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nonar2_belief_env_cfg:Nonar2BeliefEnvCfg",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Nonar2-Belief6-Direct-v0",
    entry_point=f"{__name__}.nonar2_env:Nonar2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nonar2_belief6_env_cfg:Nonar2Belief6EnvCfg",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_belief6_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Nonar2-Discrete-PPO-v0",
    entry_point=f"{__name__}.nonar2_env:Nonar2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nonar2_belief6_env_cfg:Nonar2DiscretePpoCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_discrete_cfg.yaml",
    },
)