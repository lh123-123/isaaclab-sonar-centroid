"""RSL-RL PPO configuration for AUV target tracking."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class AUVTrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Env step ≈ 0.2 s → 24 steps ≈ 4.8 s rollout (aligned with v0 control period)
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "auv_tracking_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # Match Holo SAC net_arch scale [256, 256, 256]
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
