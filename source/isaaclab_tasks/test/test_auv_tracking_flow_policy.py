# Isaac-free: FlowSonarFeaturesExtractor + FlowSACPolicy shapes.
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

_AGENTS = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_tasks"
    / "direct"
    / "isaac_auv_tracking"
    / "agents"
)


def _load(name: str, filename: str):
    path = _AGENTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flow_extractor_concat_state():
    enc_mod = _load("flow_sonar_encoder", "sonar_encoder.py")
    obs_space = spaces.Dict(
        {
            "image": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
        }
    )
    enc = enc_mod.FlowSonarFeaturesExtractor(obs_space, features_dim=512, resnet_output_dim=256)
    obs = {
        "image": torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8),
        "state": torch.randn(4, 8),
    }
    feat = enc(obs)
    assert feat.shape == (4, 512)
    assert torch.isfinite(feat).all()


def test_flow_sac_policy_forward():
    enc_mod = _load("flow_sonar_encoder", "sonar_encoder.py")
    pol_mod = _load("flow_sac_policy", "flow_sac_policy.py")
    obs_space = spaces.Dict(
        {
            "image": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    policy = pol_mod.FlowSACPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 1e-4,
        net_arch={"pi": [512, 256], "qf": [512, 256]},
        features_extractor_class=enc_mod.FlowSonarFeaturesExtractor,
        features_extractor_kwargs={"features_dim": 512, "resnet_output_dim": 256},
    )
    obs = {
        "image": torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8),
        "state": torch.randn(2, 8),
    }
    action = policy(obs, deterministic=True)
    assert action.shape == (2, 2)
    q1, q2 = policy.critic(obs, action)
    assert q1.shape == (2, 1)
    assert q2.shape == (2, 1)
