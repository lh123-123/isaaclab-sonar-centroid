# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the v7 flow-hover AUV task.

The ``policy`` group is *not* concatenated: it returns a Dict ``{"image", "state"}`` so the
asymmetric SAC policy (image CNN actor / state MLP critic) keeps working unchanged. The
stateful optical-flow / texture-mask buffers live on the env (see ``v7_flow_hover_env.py``);
the terms here only read them via ``env._ensure_flow_updated()`` (a once-per-step lazy update).
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat


def image(env: ManagerBasedRLEnv) -> torch.Tensor:
    """7-channel (RGB + du + dv + |flow| + texture mask) image at policy resolution."""
    env._ensure_flow_updated()
    h = int(env.cfg.cam_height)
    w = int(env.cfg.cam_width)
    if env._tiled_camera is None:
        # First call happens during manager init (before the camera prim exists) only to
        # determine the observation dimension; return a correctly-shaped placeholder.
        return torch.zeros((env.num_envs, 7, h, w), dtype=torch.float32, device=env.device)

    rgb = env._read_camera_frame().to(torch.float32) / 255.0
    flow_mag = torch.sqrt(env._flow_image[:, 0] ** 2 + env._flow_image[:, 1] ** 2).unsqueeze(1)
    return torch.cat([rgb, env._flow_image, flow_mag, env._texture_mask], dim=1)


def ego_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """9-D ego state ``[depth, roll, pitch, yaw, v_surge, v_sway, v_heave, yaw_rate, flow_rms]``."""
    env._ensure_flow_updated()
    robot = env._robot
    _, _, agent_yaw = euler_xyz_from_quat(robot.data.root_quat_w)

    vel_w = robot.data.root_lin_vel_w
    cy, sy = torch.cos(agent_yaw), torch.sin(agent_yaw)
    v_surge = cy * vel_w[:, 0] + sy * vel_w[:, 1]
    v_sway = -sy * vel_w[:, 0] + cy * vel_w[:, 1]
    v_heave = vel_w[:, 2]
    yaw_rate = robot.data.root_ang_vel_w[:, 2]

    roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    depth = robot.data.root_pos_w[:, 2] - float(env.cfg.starting_depth)
    flow_rms_n = (env._flow_rms / float(env.cfg.cam_width)).clamp(0.0, 1.0)
    return torch.stack(
        [
            depth.clamp(-10.0, 10.0),
            roll.clamp(-np.pi, np.pi),
            pitch.clamp(-np.pi, np.pi),
            yaw.clamp(-np.pi, np.pi),
            v_surge.clamp(-3.0, 3.0),
            v_sway.clamp(-3.0, 3.0),
            v_heave.clamp(-3.0, 3.0),
            yaw_rate.clamp(-3.0, 3.0),
            flow_rms_n,
        ],
        dim=-1,
    )
