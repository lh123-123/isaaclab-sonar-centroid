# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy / observation shapes for Lh-2, derived from :class:`~isaaclab_tasks.direct.lh_2.lh_2_env_cfg.Lh2EnvCfg`.

Use this module so RSL-RL CNN ``sonar_height`` / ``sonar_width`` stay aligned with the env without hand-copying.
"""

from __future__ import annotations

import warnings

from ..lh_2_env_cfg import Lh2EnvCfg


def _template_cfg() -> Lh2EnvCfg:
    return Lh2EnvCfg()


def lh2_policy_tail_dim(cfg: Lh2EnvCfg | None = None) -> int:
    """Trailing size after sonar image / beams: ωz, pad, last_action [+ lateral hint] [+ orbit w_inst]."""
    c = cfg or _template_cfg()
    n = 3
    if bool(getattr(c, "policy_lateral_hint_enabled", False)):
        n += 1
    if bool(getattr(c, "policy_append_orbit_w_inst", False)) and str(
        getattr(c, "target_motion_mode", "orbit_robot")
    ).lower() == "orbit_robot":
        n += 1
    return n


def lh2_sonar_image_hw(cfg: Lh2EnvCfg | None = None) -> tuple[int, int]:
    c = cfg or _template_cfg()
    o = c.oceansim_sonar_cfg
    return int(o.height), int(o.width)


def lh2_expected_policy_obs_dim(cfg: Lh2EnvCfg | None = None) -> int:
    """Flattened policy observation size (must match gym ``observation_space`` for the same cfg)."""
    c = cfg or _template_cfg()
    mode = str(getattr(c, "policy_obs_mode", "image")).lower()
    if mode == "relative_position":
        return 5
    if mode == "beam":
        return int(c.oceansim_sonar_cfg.beam_count) * 3 + lh2_policy_tail_dim(c)
    h, w = lh2_sonar_image_hw(c)
    return h * w + lh2_policy_tail_dim(c)


def lh2_expected_critic_state_dim(cfg: Lh2EnvCfg | None = None) -> int:
    """Privileged critic vector = policy observation + 1 (body bearing change per env step, rad)."""
    return lh2_expected_policy_obs_dim(cfg) + 1


def warn_if_obs_space_mismatch(cfg: Lh2EnvCfg | None = None) -> None:
    """Call once from agent cfg modules to catch drift between ``observation_space`` and sonar/tail settings."""
    c = cfg or _template_cfg()
    derived = lh2_expected_policy_obs_dim(c)
    if int(c.observation_space) != derived:
        warnings.warn(
            f"Lh2EnvCfg.observation_space={c.observation_space} != derived {derived} "
            f"(policy_obs_mode={c.policy_obs_mode}, sonar {lh2_sonar_image_hw(c)}). "
            "Update observation_space or oceansim_sonar_cfg / tail options.",
            stacklevel=2,
        )
    ss = int(getattr(c, "state_space", 0) or 0)
    if ss > 0:
        c_derived = lh2_expected_critic_state_dim(c)
        if ss != c_derived:
            warnings.warn(
                f"Lh2EnvCfg.state_space={ss} != derived critic dim {c_derived} "
                f"(expected policy_dim+1). Update state_space.",
                stacklevel=2,
            )


# Snapshot at import time for @configclass defaults (matches current Lh2EnvCfg).
_ref = _template_cfg()
LH2_SONAR_HEIGHT: int = lh2_sonar_image_hw(_ref)[0]
LH2_SONAR_WIDTH: int = lh2_sonar_image_hw(_ref)[1]
LH2_BEAM_COUNT: int = int(_ref.oceansim_sonar_cfg.beam_count)
LH2_POLICY_TAIL_DIM: int = lh2_policy_tail_dim(_ref)
LH2_EXPECTED_POLICY_OBS_DIM: int = lh2_expected_policy_obs_dim(_ref)
LH2_EXPECTED_CRITIC_STATE_DIM: int = lh2_expected_critic_state_dim(_ref)

warn_if_obs_space_mismatch(_ref)
