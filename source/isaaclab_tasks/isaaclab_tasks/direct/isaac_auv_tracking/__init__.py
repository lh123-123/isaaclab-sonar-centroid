# AUV Target Tracking environment for IsaacLab
# Integrates isaac-auv-env physics with RL_AUV_tracking task logic

import gymnasium as gym

from . import agents
from .auv_tracking_env import AUVTrackingEnv, AUVTrackingEnvCfg

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
