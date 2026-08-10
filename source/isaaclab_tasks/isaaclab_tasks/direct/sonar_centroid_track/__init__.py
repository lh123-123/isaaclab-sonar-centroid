# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for sonar centroid tracking."""

import gymnasium as gym

from . import agents

_ENV = f"{__name__}.sonar_centroid_track_env:SonarCentroidTrackEnv"
_PPO = f"{agents.__name__}:skrl_ppo_discrete_cfg.yaml"

gym.register(
    id="Isaac-SonarCentroidTrack-Debug-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_centroid_track_env_cfg:SonarCentroidTrackDebugCfg",
        "skrl_cfg_entry_point": _PPO,
    },
)

gym.register(
    id="Isaac-SonarCentroidTrack-Discrete-PPO-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_centroid_track_env_cfg:SonarCentroidTrackDiscretePpoCfg",
        "skrl_cfg_entry_point": _PPO,
    },
)

gym.register(
    id="Isaac-SonarCentroidTrack-Continuous-PPO-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_centroid_track_env_cfg:SonarCentroidTrackContinuousPpoCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_continuous_cfg.yaml",
    },
)

gym.register(
    id="Isaac-SonarCentroidTrack-Continuous-Eval-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_centroid_track_env_cfg:SonarCentroidTrackContinuousEvalCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_continuous_cfg.yaml",
    },
)

gym.register(
    id="Isaac-SonarCentroidTrack-Eval-Direct-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_centroid_track_env_cfg:SonarCentroidTrackEvalCfg",
        "skrl_cfg_entry_point": _PPO,
    },
)

_AUV_ENV = f"{__name__}.sonar_centroid_track_auv_env:SonarCentroidTrackAUVEnv"

gym.register(
    id="Isaac-SonarCentroidTrack-AUV-Debug-Direct-v0",
    entry_point=_AUV_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_centroid_track_auv_env_cfg:SonarCentroidTrackAUVDebugCfg",
        "skrl_cfg_entry_point": _PPO,
    },
)

gym.register(
    id="Isaac-SonarCentroidTrack-AUV-PControl-Direct-v0",
    entry_point=_AUV_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_centroid_track_auv_env_cfg:SonarCentroidTrackAUVMotionDebugCfg",
        "skrl_cfg_entry_point": _PPO,
    },
)

gym.register(
    id="Isaac-SonarCentroidTrack-AUV-Residual-PPO-Direct-v0",
    entry_point=_AUV_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_centroid_track_auv_env_cfg:SonarCentroidTrackAUVResidualPpoCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_continuous_auv_cfg.yaml",
    },
)

gym.register(
    id="Isaac-SonarCentroidTrack-AUV-Residual-Eval-Direct-v0",
    entry_point=_AUV_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sonar_centroid_track_auv_env_cfg:SonarCentroidTrackAUVResidualEvalCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_continuous_auv_cfg.yaml",
    },
)
