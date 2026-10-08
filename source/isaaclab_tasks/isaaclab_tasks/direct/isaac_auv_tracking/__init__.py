# AUV Target Tracking environment for IsaacLab
# Integrates isaac-auv-env physics with RL_AUV_tracking task logic

import gymnasium as gym

from . import agents
from .auv_tracking_env import AUVTrackingEnv, AUVTrackingEnvCfg
from .auv_tracking_flow_env import (
    AUVTrackingFlowEnv,
    AUVTrackingFlowEnvCfg,
    AUVTrackingFlowImageEnv,
    AUVTrackingFlowImageEnvCfg,
    AUVTrackingFlowStateEnv,
    AUVTrackingFlowStateEnvCfg,
)
from .auv_tracking_vision_env import AUVTrackingVisionEnv, AUVTrackingVisionEnvCfg
from .auv_tracking_cam_env import AUVTrackingCamEnv, AUVTrackingCamEnvCfg
from .auv_tracking_v4_cam_env import AUVTrackingV4CamEnv, AUVTrackingV4CamEnvCfg
from .auv_tracking_v5_flow_env import AUVTrackingV5FlowEnv, AUVTrackingV5FlowEnvCfg
from .auv_tracking_v6_flowvec_env import AUVTrackingV6FlowVecEnv, AUVTrackingV6FlowVecEnvCfg

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
        "sb3_symmetric_cnn_cfg_entry_point": f"{agents.__name__}:sb3_sac_vision_symmetric_cnn_cfg.yaml",
        "sb3_hybrid_cnn_cfg_entry_point": f"{agents.__name__}:sb3_sac_vision_hybrid_cnn_cfg.yaml",
        "sb3_gated_cnn_cfg_entry_point": f"{agents.__name__}:sb3_sac_vision_gated_cnn_cfg.yaml",
        "sb3_asym_cnn_cfg_entry_point": f"{agents.__name__}:sb3_sac_vision_asym_cnn_cfg.yaml",
    },
)

# Stacked polar residual + reference position; reward/done inherit v0
gym.register(
    id="Isaac-AUV-Tracking-Direct-v2",
    entry_point="isaaclab_tasks.direct.isaac_auv_tracking:AUVTrackingFlowEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AUVTrackingFlowEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_flow_cfg.yaml",
        "sb3_resnet_cfg_entry_point": f"{agents.__name__}:sb3_sac_flow_resnet_cfg.yaml",
    },
)

# Ablation: state-only (no image, MLP policy)
gym.register(
    id="Isaac-AUV-Tracking-Direct-v2-state",
    entry_point="isaaclab_tasks.direct.isaac_auv_tracking:AUVTrackingFlowStateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AUVTrackingFlowStateEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_flow_state_cfg.yaml",
    },
)

# Ablation: image-only (no state, CNN-only policy)
gym.register(
    id="Isaac-AUV-Tracking-Direct-v2-image",
    entry_point="isaaclab_tasks.direct.isaac_auv_tracking:AUVTrackingFlowImageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AUVTrackingFlowImageEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_flow_image_cfg.yaml",
    },
)

# Direct-v3-cam: no sonar. Forward camera at the sonar mount pose, K=4 grayscale frame stack,
# second WarpAUV as the target. KF measurement from geometric GT.
gym.register(
    id="Isaac-AUV-Tracking-Direct-v3-cam",
    entry_point="isaaclab_tasks.direct.isaac_auv_tracking:AUVTrackingCamEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AUVTrackingCamEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cam_cnn_cfg.yaml",
    },
)

# Direct-v4-cam: a *copy* of Direct-v0 (kept intact) with only two changes -- the target is a
# second WarpAUV (same BatchHoloTargetMotion, so the motion model is unchanged) and the sonar
# is replaced by the forward camera (K=4 grayscale frames, KF from geometric GT).
gym.register(
    id="Isaac-AUV-Tracking-Direct-v4-cam",
    entry_point="isaaclab_tasks.direct.isaac_auv_tracking:AUVTrackingV4CamEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AUVTrackingV4CamEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_v4_cam_cfg.yaml",
    },
)

# Direct-v5-flow: Direct-v4-cam with ONE change -- the actor's observation is a K-frame GT
# optical-flow stack ((frame_stack * 2, 64, 64) float32) instead of the uint8 RGB stack plus a
# separate 2-channel flow image, and the "flow_image" key is gone. Everything else (second-AUV
# target + motion model, camera-based KF measurement, rewards, reset schedule) is v4-cam
# verbatim, so runs here are comparable to the v4_cam_sb3_sac_800k baseline.
gym.register(
    id="Isaac-AUV-Tracking-Direct-v5-flow",
    entry_point="isaaclab_tasks.direct.isaac_auv_tracking:AUVTrackingV5FlowEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AUVTrackingV5FlowEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_v5_flow_cfg.yaml",
    },
)

# Direct-v6-flowvec: Direct-v5-flow with ONE change -- the actor's observation is the
# (frame_stack * 6,) float32 vector of the analytic flow numbers the v5 raster is a picture
# of (u/W, v/H, rad/12, du, dv, validity per frame), and the encoder is
# flow_vec_encoder.FlowVectorExtractor (Linear + LayerNorm + Tanh) instead of the CNN.
# The KF measurement is geometric GT (kf_measurement="gt"), matching the GT flow.
# Everything else is v5-flow verbatim, so this isolates what the convolutional encoder costs
# or earns on this task.
gym.register(
    id="Isaac-AUV-Tracking-Direct-v6-flowvec",
    entry_point="isaaclab_tasks.direct.isaac_auv_tracking:AUVTrackingV6FlowVecEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AUVTrackingV6FlowVecEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_v6_flowvec_cfg.yaml",
    },
)
