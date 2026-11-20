# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AUV tracking tasks in the manager-based workflow."""

import gymnasium as gym

from .v7_flow_hover.v7_flow_hover_env import V7FlowHoverEnv
from .v7_flow_hover.v7_flow_hover_env_cfg import V7FlowHoverEnvCfg

# Manager-based v7-flow-hover: behaviour-preserving port of the Direct v7 station-keeping task.
# Shares the same SB3 asymmetric SAC policy config as the Direct variant.
gym.register(
    id="Isaac-AUV-Tracking-MB-v7-flow-hover",
    entry_point="isaaclab_tasks.manager_based.isaac_auv_tracking.v7_flow_hover.v7_flow_hover_env:V7FlowHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": V7FlowHoverEnvCfg,
        "sb3_cfg_entry_point": "isaaclab_tasks.direct.isaac_auv_tracking.agents:sb3_sac_v7_flow_hover_cfg.yaml",
    },
)
