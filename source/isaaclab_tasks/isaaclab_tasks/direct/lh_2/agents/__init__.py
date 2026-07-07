# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL agent presets for **Isaac-Lh-2-Direct-v0**.

Entry points are wired in :mod:`isaaclab_tasks.direct.lh_2`:

- **RSL-RL (recommended)** — ``rsl_rl_ppo_cfg:PPORunnerCfg`` + beam 1D-CNN policy (:class:`~isaaclab_tasks.direct.lh_2.rsl_rl_fls_beam_actor_critic.Lh2FlsBeamActorCritic`).
- **SKRL** — ``skrl_ppo_cfg.yaml`` (MLP on flattened obs; good for quick experiments, heavier than CNN).
- **Stable-Baselines3** — ``sb3_ppo_cfg.yaml``
- **rl_games** — ``rl_games_ppo_cfg.yaml``

Dimensions (sonar H×W, tail, expected obs size) are centralized in :mod:`policy_dims`
so CNN ``sonar_height`` / ``sonar_width`` track :class:`~isaaclab_tasks.direct.lh_2.lh_2_env_cfg.Lh2EnvCfg`.
"""

from . import policy_dims  # noqa: F401
