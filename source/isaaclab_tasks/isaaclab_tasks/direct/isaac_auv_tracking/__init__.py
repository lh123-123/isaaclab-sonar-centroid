# AUV Target Tracking environment for IsaacLab
# Integrates isaac-auv-env physics with RL_AUV_tracking task logic

import gymnasium as gym

from . import agents
from .auv_tracking_env import AUVTrackingEnv, AUVTrackingEnvCfg
from .auv_tracking_vision_env import AUVTrackingVisionEnv, AUVTrackingVisionEnvCfg

gym.register(
    id="Isaac-AUV-Tracking-Direct-v0",
    entry_point="isaaclab_tasks.direct.isaac_auv_tracking:AUVTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AUVTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.AUVTrackingPPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)

# Holo v1-style asymmetric SAC: Actor=sonar image, Critic=belief state
gym.register(
    id="Isaac-AUV-Tracking-Direct-v1",
    entry_point="isaaclab_tasks.direct.isaac_auv_tracking:AUVTrackingVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AUVTrackingVisionEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_vision_cfg.yaml",
    },
)
