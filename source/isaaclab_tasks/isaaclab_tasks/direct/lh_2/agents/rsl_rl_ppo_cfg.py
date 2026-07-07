# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO defaults for Isaac-Lh-2-Direct-v0（旋转跟随 + 圆环带内轨道目标，beam 1D-CNN + Gaussian）。

Tuning guide (most impact first):
- ``algorithm.learning_rate`` / ``algorithm.entropy_coef`` — exploration vs stability
- ``policy.init_noise_std`` / ``policy.min_std`` — action sampling spread (see also env ``action_scale_linear``)
- ``num_steps_per_env`` × ``num_envs`` — batch size per iteration
- ``policy.cnn_feature_dim`` / ``actor_hidden_dims`` — capacity

Sonar resolution comes from :mod:`policy_dims` (kept in sync with :class:`~isaaclab_tasks.direct.lh_2.lh_2_env_cfg.Lh2EnvCfg`).
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .policy_dims import LH2_BEAM_COUNT


@configclass
class Lh2PpoActorCriticCfg(RslRlPpoActorCriticCfg):
    """Loads :class:`~isaaclab_tasks.direct.lh_2.rsl_rl_fls_beam_actor_critic.Lh2FlsBeamActorCritic`."""

    class_name: str = (
        "__import__('isaaclab_tasks.direct.lh_2.rsl_rl_fls_beam_actor_critic', "
        "fromlist=['Lh2FlsBeamActorCritic']).Lh2FlsBeamActorCritic"
    )
    beam_count: int = LH2_BEAM_COUNT
    beam_channels: int = 3
    beam_conv_channels: list[int] = [32, 64]
    beam_kernel_size: int = 5
    beam_feature_dim: int = 64
    tail_mlp_hidden: int = 32
    # 略提高下限，减轻早期策略塌缩到单边饱和（play 里长期 action≈+1）。
    min_std: float = 0.12


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # --- rollout / logging ---
    num_steps_per_env = 64
    max_iterations = 2000
    save_interval = 100
    experiment_name = "underwater_tracking_sonar"
    empirical_normalization = True
    device = "cpu"
    clip_actions: float = 1.0

    # --- policy / value networks (beam 1D-CNN backbone is inside Lh2FlsBeamActorCritic) ---
    policy = Lh2PpoActorCriticCfg(
        init_noise_std=0.52,
        min_std=0.12,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    # --- PPO hyperparameters ---
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.022,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=0.5,
    )
