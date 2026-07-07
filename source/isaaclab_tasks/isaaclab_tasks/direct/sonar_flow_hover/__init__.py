# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sonar IBVS + DRL environment registration."""

import gymnasium as gym

from . import agents

_ENV = f"{__name__}.sonar_ibvs_drl_env:SonarIbvsDrlEnv"
_LATERAL1D_SAC = f"{agents.__name__}:skrl_sac_lateral1d_cfg.yaml"
_SAC = f"{agents.__name__}:skrl_sac_cfg.yaml"
_PPO = f"{agents.__name__}:skrl_ppo_cfg.yaml"

gym.register(
    id="Isaac-SonarIbvsDrl-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_ibvs_drl_env_cfg:SonarIbvsDrlEnvCfg",
        "skrl_cfg_entry_point": _SAC,
        "skrl_sac_cfg_entry_point": _SAC,
    },
)

gym.register(
    id="Isaac-SonarIbvsDrl-Stage1-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_ibvs_drl_env_cfg:SonarIbvsDrlStage1Cfg",
        "skrl_cfg_entry_point": _SAC,
        "skrl_sac_cfg_entry_point": _SAC,
    },
)

gym.register(
    id="Isaac-SonarIbvsDrl-Stage2-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_ibvs_drl_env_cfg:SonarIbvsDrlStage2Cfg",
        "skrl_cfg_entry_point": _SAC,
        "skrl_sac_cfg_entry_point": _SAC,
    },
)

gym.register(
    id="Isaac-SonarIbvsDrl-Eval-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_ibvs_drl_env_cfg:SonarIbvsDrlEvalCfg",
        "skrl_cfg_entry_point": _SAC,
        "skrl_sac_cfg_entry_point": _SAC,
    },
)

gym.register(
    id="Isaac-SonarIbvsDrl-Lateral1D-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_ibvs_drl_env_cfg:SonarIbvsDrlLateral1DCfg",
        "skrl_cfg_entry_point": _LATERAL1D_SAC,
        "skrl_sac_cfg_entry_point": _LATERAL1D_SAC,
    },
)

gym.register(
    id="Isaac-SonarIbvsDrl-Lateral1D-Debug-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_ibvs_drl_env_cfg:SonarIbvsDrlLateral1DDebugCfg",
        "skrl_cfg_entry_point": _LATERAL1D_SAC,
        "skrl_sac_cfg_entry_point": _LATERAL1D_SAC,
    },
)

gym.register(
    id="Isaac-SonarIbvsDrl-Debug-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_ibvs_drl_env_cfg:SonarIbvsDrlDebugCfg",
        "skrl_cfg_entry_point": _SAC,
        "skrl_sac_cfg_entry_point": _SAC,
    },
)

# Backward-compatible task IDs (same IBVS-DRL env)
gym.register(
    id="Isaac-SonarFlowHover-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_ibvs_drl_env_cfg:SonarIbvsDrlEnvCfg",
        "skrl_cfg_entry_point": _PPO,
    },
)

gym.register(
    id="Isaac-SonarFlowHover-Template-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_ibvs_drl_env_cfg:SonarIbvsDrlEnvCfg",
        "skrl_cfg_entry_point": _SAC,
        "skrl_sac_cfg_entry_point": _SAC,
    },
)

gym.register(
    id="Isaac-SonarFlowHover-Template-Eval-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_ibvs_drl_env_cfg:SonarIbvsDrlEvalCfg",
        "skrl_cfg_entry_point": _SAC,
        "skrl_sac_cfg_entry_point": _SAC,
    },
)

gym.register(
    id="Isaac-SonarFlowHover-Eval-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_ibvs_drl_env_cfg:SonarIbvsDrlEvalCfg",
        "skrl_cfg_entry_point": _SAC,
        "skrl_sac_cfg_entry_point": _SAC,
    },
)

gym.register(
    id="Isaac-SonarFlowHover-Debug-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_ibvs_drl_env_cfg:SonarIbvsDrlDebugCfg",
        "skrl_cfg_entry_point": _SAC,
        "skrl_sac_cfg_entry_point": _SAC,
    },
)

gym.register(
    id="Isaac-SonarFlowHover-Sweep-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_ibvs_drl_env_cfg:SonarIbvsDrlDebugCfg",
        "skrl_cfg_entry_point": _SAC,
        "skrl_sac_cfg_entry_point": _SAC,
    },
)
