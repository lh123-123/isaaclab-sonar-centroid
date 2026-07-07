# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for Nonar2 Phase 0 (belief-state vector policy)."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "nonar2_sonar_tracking"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
