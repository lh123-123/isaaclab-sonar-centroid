# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Residual RL environment registration."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-ResidualRL-Direct-v0",
    entry_point=f"{__name__}.residual_rl_env:ResidualRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_rl_env_cfg:ResidualRLEnvCfg",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ResidualRL-Stage2-Direct-v0",
    entry_point=f"{__name__}.residual_rl_env:ResidualRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_rl_env_cfg:ResidualRLStage2Cfg",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ResidualRL-Stage3-Direct-v0",
    entry_point=f"{__name__}.residual_rl_env:ResidualRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_rl_env_cfg:ResidualRLStage3Cfg",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ResidualRL-Stage4-Direct-v0",
    entry_point=f"{__name__}.residual_rl_env:ResidualRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_rl_env_cfg:ResidualRLStage4Cfg",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ResidualRL-Test1-Direct-v0",
    entry_point=f"{__name__}.residual_rl_env:ResidualRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_rl_env_cfg:ResidualRLTest1Cfg",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_test1_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ResidualRL-Test1-Train-v0",
    entry_point=f"{__name__}.residual_rl_env:ResidualRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_rl_env_cfg:ResidualRLTest1TrainCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_test1_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_test1_cfg.yaml",
    },
)
