"""Nonar2 tracking pipeline modules (design doc §3)."""

from .belief_estimator import BeliefEstimator
from .reward import (
    compute_action_reward,
    compute_action_reward_breakdown,
)
from .sonar_beam import azimuth_rad_to_beam, azimuth_rad_to_beam_tensor, beam_center
from .sonar_column_profile import beam_intensity_column_profile, column_profile_dim
from .sonar_detector import SonarDetector
from .sonar_encoder import SonarFeatureEncoder
from .sonar_preprocess import preprocess_sonar

__all__ = [
    "BeliefEstimator",
    "SonarDetector",
    "azimuth_rad_to_beam",
    "azimuth_rad_to_beam_tensor",
    "beam_center",
    "beam_intensity_column_profile",
    "column_profile_dim",
    "SonarFeatureEncoder",
    "compute_action_reward",
    "compute_action_reward_breakdown",
    "preprocess_sonar",
]
