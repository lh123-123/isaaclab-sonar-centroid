#!/usr/bin/env python3
"""One-env check that each ablation flag removes only its own cue.

Steps a sway so flow, the template match, and visodom are all nonzero, then
flips one flag at a time and recomputes the observation and the reward on
that same state.

    ./isaaclab.sh -p scripts/debug_auv_v7_ablation.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 ablation-flag wiring check.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import torch

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)

_FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    if not cond:
        _FAILS.append(name)
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def _act(env, sway: float) -> torch.Tensor:
    scale = float(env.cfg.cmd_vel_scale)
    raw = [0.0, sway / scale, 0.0]
    return torch.tensor([raw], dtype=torch.float32, device=env.device).clamp(-1.0, 1.0)


def _terms(env):
    """The four cue contributions, using the same formulas as the reward."""
    cfg = env.cfg
    present = (env._texture_count >= cfg.min_texture_px).to(torch.float32)
    flow_rms_n = (env._flow_rms / 64.0).clamp(0.0, 1.0)
    flow = present * torch.exp(-(flow_rms_n ** 2) / (cfg.sigma_flow ** 2))
    drift = present * torch.exp(-(env._drift_int.norm(dim=-1) ** 2) / (cfg.sigma_drift ** 2))
    anchor = present * torch.exp(-(env._anchor_disp_m ** 2) / (cfg.sigma_anchor_vis_m ** 2))
    sim = present * ((env._anchor_ncc + 1.0) * 0.5).clamp(0.0, 1.0)
    yaw = euler_xyz_from_quat(env._robot.data.root_quat_w)[2]
    yaw_err = torch.atan2(torch.sin(yaw - env._anchor_yaw), torch.cos(yaw - env._anchor_yaw))
    yaw_term = torch.exp(-(yaw_err ** 2) / (cfg.sigma_yaw_anchor ** 2))
    d_vo = env._visodom_m().norm(dim=-1)
    vo = present * torch.exp(-(d_vo ** 2) / (cfg.sigma_visodom_m ** 2))
    d_gt = torch.norm(env._robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=-1)
    gt = torch.exp(-(d_gt ** 2) / (cfg.sigma_anchor_m ** 2))
    return {
        "flow": float(cfg.rew_flow * flow[0]),
        "drift": float(cfg.rew_drift * drift[0]),
        "template": float(cfg.rew_anchor_vis * anchor[0] + cfg.rew_sim * sim[0] + cfg.rew_yaw_anchor * yaw_term[0]),
        "visodom": float(cfg.rew_anchor * vo[0]),
        "gt": float(cfg.rew_anchor * gt[0]),
        "d_vo": float(d_vo[0]),
        "d_gt": float(d_gt[0]),
    }


def main() -> None:
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
    check("flags default off", not any((cfg.ablate_flow, cfg.ablate_drift, cfg.ablate_template, cfg.ablate_visodom)))
    env.reset()
    for step in range(36):
        sway = 0.5 if step >= 8 else 0.0
        env.step(_act(env, sway))

    base_obs = env._get_observations()["policy"]
    base_img = base_obs["image"][0]
    base_st = base_obs["state"][0]
    base_rew = float(env._compute_step_reward()[0])
    terms = _terms(env)
    print(
        f"[ablate] d_gt={terms['d_gt']:.3f} d_vo={terms['d_vo']:.3f} "
        f"flow={terms['flow']:.3f} drift={terms['drift']:.3f} "
        f"template={terms['template']:.3f} visodom={terms['visodom']:.3f} rew={base_rew:.3f}",
        flush=True,
    )
    check("image is 11ch", tuple(base_img.shape) == (11, 64, 64), str(tuple(base_img.shape)))
    check("state is 17", int(base_st.shape[0]) == 17, str(int(base_st.shape[0])))
    check("flow channels live", float(base_img[3:6].abs().mean()) > 1e-4, f"{float(base_img[3:6].abs().mean()):.5f}")
    check("template channels live", float(base_img[7:9].abs().mean()) > 1e-3, f"{float(base_img[7:9].abs().mean()):.4f}")
    check("visodom channels live", float(base_img[9:11].abs().mean()) > 1e-3, f"{float(base_img[9:11].abs().mean()):.4f}")
    check("gt and visodom differ", abs(terms["d_gt"] - terms["d_vo"]) > 0.02, f"{terms['d_gt']:.3f} vs {terms['d_vo']:.3f}")

    cases = (
        ("ablate_flow", "flow", slice(3, 6), None),
        ("ablate_drift", "drift", None, None),
        ("ablate_template", "template", slice(7, 9), slice(11, 15)),
        ("ablate_visodom", "visodom", slice(9, 11), slice(15, 17)),
    )
    for flag, key, img_sl, st_sl in cases:
        setattr(cfg, flag, True)
        obs = env._get_observations()["policy"]
        img, st = obs["image"][0], obs["state"][0]
        rew = float(env._compute_step_reward()[0])
        setattr(cfg, flag, False)
        check(f"{flag} shape", tuple(img.shape) == (11, 64, 64) and int(st.shape[0]) == 17)
        if img_sl is not None:
            check(f"{flag} image zero", float(img[img_sl].abs().max()) == 0.0, f"max={float(img[img_sl].abs().max()):.5f}")
        if st_sl is not None:
            check(f"{flag} state zero", float(st[st_sl].abs().max()) == 0.0, f"max={float(st[st_sl].abs().max()):.5f}")
        if flag == "ablate_drift":
            check("ablate_drift obs unchanged", torch.equal(img, base_img) and torch.equal(st, base_st))
        drop = base_rew - rew
        check(
            f"{flag} reward drop",
            abs(drop - terms[key]) < 1e-4,
            f"drop={drop:.4f} term={terms[key]:.4f}",
        )
        if flag == "ablate_visodom":
            check(
                "visodom off does not pay GT",
                abs(terms["visodom"] - terms["gt"]) > 1e-4 and abs(drop - terms["gt"]) > abs(drop - terms["visodom"]),
                f"drop={drop:.4f} vo_term={terms['visodom']:.4f} gt_term={terms['gt']:.4f}",
            )
        restored = float(env._compute_step_reward()[0])
        check(f"{flag} restores", abs(restored - base_rew) < 1e-4, f"{restored:.4f} vs {base_rew:.4f}")

    env.close()
    print(f"[ablate] {'FAIL ' + str(len(_FAILS)) if _FAILS else 'ALL PASS'}", flush=True)
    if _FAILS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
