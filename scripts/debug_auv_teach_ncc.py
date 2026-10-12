#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Teach-npz pure-NCC probe: detect_teach vs GT pinhole projection. NO orange anywhere.

Hypotheses (session 756db1):
  H1 detection rate: NCC score often below 0.40 gate -> miss (cx < 0)
  H2 position accuracy: hits but centre pulled off target (background near-tie false positive)
  H3 scale correctness: picked template scale wrong vs GT projected width
  H4 teach freshness: npz photo (09-23) vs today's renders differ in brightness/white balance
  H5 far-field collapse: score dies when projected width < ~20 px
  H6 background false positive: target OUT of view, NCC still fires >= 0.40 gate

  ./scripts/debug_auv_teach_ncc.py --num_envs 1 --steps 120 --headless
"""

from __future__ import annotations

import argparse
import json
import math
import time

_LOG = "/home/lh/IsaacLab/.cursor/debug-756db1.log"
_SID = "756db1"


def _log(hid: str, msg: str, data: dict) -> None:
    # #region agent log
    rec = {
        "sessionId": _SID,
        "runId": "teach-ncc",
        "hypothesisId": hid,
        "location": "debug_auv_teach_ncc.py",
        "message": msg,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass
    # #endregion


def main() -> None:
    parser = argparse.ArgumentParser(description="teach npz pure-NCC vs GT projection probe")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--action_v", type=float, default=0.1)
    parser.add_argument("--action_w", type=float, default=0.0)
    parser.add_argument(
        "--pursuit",
        action="store_true",
        help="each step: yaw-rate = clip(1.5 * gt_bearing, -1, 1), v=0 (keeps target in FOV while it sails away)",
    )
    parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v4-cam")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = True
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        _run(args_cli)
    finally:
        simulation_app.close()


def _run(args_cli) -> None:
    import gymnasium as gym
    import numpy as np
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v4_cam_env import AUVTrackingV4CamEnvCfg
    from isaaclab_tasks.direct.isaac_auv_tracking.teach_match import (
        cut_teach_template,
        detect_teach,
        teach_from_numpy,
    )

    cfg = AUVTrackingV4CamEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=cfg)
    raw = env.unwrapped
    dt = float(raw.cfg.sim.dt) * float(raw.cfg.decimation)

    # --- teach npz: the shipped reference photo (image, points, range_m) ---
    npz_path = raw._teach_npz_path()
    blob = np.load(npz_path)
    teach = teach_from_numpy(blob["image"], blob["points"], float(blob["range_m"]))
    template = cut_teach_template(teach.image)  # colour used ONCE here, on the teach photo only
    th, tw = int(template.shape[-2]), int(template.shape[-1])
    print(f"[ncc] teach npz={npz_path.name} img={tuple(teach.image.shape)} tmpl={th}x{tw}px", flush=True)
    _log("H4", "teach loaded", {"npz": str(npz_path), "tmpl": [th, tw], "range_m": float(blob["range_m"])})

    env.reset()
    actions = torch.zeros((raw.num_envs, raw.action_space.shape[-1]), device=raw.device)
    actions[:, 0] = float(args_cli.action_v)
    actions[:, 1] = float(args_cli.action_w)

    rows: list[dict] = []
    for step in range(int(args_cli.steps)):
        # #region agent log
        if args_cli.pursuit:  # H5 sampler: point the nose at the target so far-field stays in FOV
            ap, ayaw = raw._agent_pose_2d()
            cy_, sy_ = torch.cos(ayaw), torch.sin(ayaw)
            dx_ = raw._target_pos[:, 0] - (ap[:, 0] + 0.35 * cy_)
            dy_ = raw._target_pos[:, 1] - (ap[:, 1] + 0.35 * sy_)
            b_ = float(torch.atan2(-sy_ * dx_ + cy_ * dy_, cy_ * dx_ + sy_ * dy_)[0])
            actions[:, 0] = 0.0
            actions[:, 1] = float(np.clip(1.5 * b_, -1.0, 1.0))
        # #endregion
        env.step(actions)
        rgb = raw._camera_rgb()  # (N, H, W, 3) render-res, no colour math on it below
        h, w = int(rgb.shape[1]), int(rgb.shape[2])  # HWC: W is dim 2, NOT the trailing 3

        # --- GT pinhole projection (truth; no orange) ---
        agent_pos, agent_yaw = raw._agent_pose_2d()
        cos_y, sin_y = torch.cos(agent_yaw), torch.sin(agent_yaw)
        cam_x = agent_pos[:, 0] + 0.35 * cos_y
        cam_y = agent_pos[:, 1] + 0.35 * sin_y
        cdx = raw._target_pos[:, 0] - cam_x
        cdy = raw._target_pos[:, 1] - cam_y
        cam_r = float(torch.sqrt(cdx * cdx + cdy * cdy)[0])
        bearing = float(torch.atan2(-sin_y * cdx + cos_y * cdy, cos_y * cdx + sin_y * cdy)[0])
        hfov = float(raw.cfg.cam_hfov_deg)
        fx = 0.5 * w / math.tan(math.radians(hfov) * 0.5)
        u_gt = 0.5 * (w - 1) - fx * math.tan(bearing)
        v_gt = 0.5 * (h - 1)  # target and tracker share depth
        in_fov = abs(bearing) <= math.radians(hfov) * 0.5 + 0.05
        # expected width from the teach template's own geometry: tw px at range_m
        bw_exp = tw * float(blob["range_m"]) / max(cam_r, 0.3)

        # --- pure NCC: detect_teach applies NO colour test to the live frame ---
        # exp_bw = expected width from GT range (stands in for the KF prior) so the
        # scale-prior gate inside detect_teach is exercised (fix countermeasure c).
        det = detect_teach(
            rgb,
            template,
            exp_bw=torch.full((int(rgb.shape[0]),), float(bw_exp), device=raw.device),
        )
        cx, cy, bw, bh, score = (float(v) for v in det[0].tolist())
        hit = cx >= 0.0
        px_err = math.hypot(cx - u_gt, cy - v_gt) if hit else float("nan")
        bw_ratio = bw / bw_exp if hit and bw_exp > 1 else float("nan")

        row = {
            "step": step + 1,
            "t": round((step + 1) * dt, 2),
            "gt_r": round(cam_r, 3),
            "gt_a_deg": round(math.degrees(bearing), 2),
            "in_fov": bool(in_fov),
            "u_gt": round(float(u_gt), 1),
            "v_gt": round(float(v_gt), 1),
            "bw_exp": round(float(bw_exp), 1),
            "det_cx": round(cx, 1),
            "det_cy": round(cy, 1),
            "det_bw": round(bw, 1),
            "score": round(score, 3),
            "hit": bool(hit),
            "px_err": round(px_err, 1) if px_err == px_err else None,
            "bw_ratio": round(bw_ratio, 3) if bw_ratio == bw_ratio else None,
        }
        rows.append(row)
        _log("H1,H2,H3,H5", "ncc sample", row)

        if step == 0:
            live_mean = float(rgb[0].float().mean())
            teach_mean = float(teach.image.float().mean())
            _log("H4", "brightness teach vs live", {"teach_mean": round(teach_mean, 1), "live_mean": round(live_mean, 1)})
            print(
                f"[ncc] brightness teach={teach_mean:.0f} live={live_mean:.0f} "
                f"fx={fx:.1f} hfov={hfov}",
                flush=True,
            )
        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"[ncc] t={row['t']:.1f} gt_r={row['gt_r']:.2f} a={row['gt_a_deg']:+.1f}° "
                f"fov={int(row['in_fov'])} score={row['score']:.2f} hit={int(row['hit'])} "
                f"px_err={row['px_err']} bw={row['det_bw']}/{row['bw_exp']} "
                f"cx={row['det_cx']:.0f} vs u_gt={row['u_gt']:.0f}",
                flush=True,
            )

    # --- summary ---
    vis = [r for r in rows if r["in_fov"]]
    oov = [r for r in rows if not r["in_fov"]]
    hits = [r for r in vis if r["hit"]]
    far = [r for r in vis if r["bw_exp"] < 20.0]
    near = [r for r in vis if r["bw_exp"] >= 20.0]

    def med(vals):
        vals = [v for v in vals if v is not None]
        return round(float(np.median(vals)), 2) if vals else None

    summary = {
        "steps": len(rows),
        "vis_steps": len(vis),
        "oov_steps": len(oov),
        "hit_rate_vis": round(len(hits) / len(vis), 3) if vis else None,
        "hit_rate_far_lt20px": round(sum(1 for r in far if r["hit"]) / len(far), 3) if far else None,
        "hit_rate_near_ge20px": round(sum(1 for r in near if r["hit"]) / len(near), 3) if near else None,
        "med_score_hits": med([r["score"] for r in hits]),
        "med_score_misses": med([r["score"] for r in vis if not r["hit"]]),
        "med_px_err_hits": med([r["px_err"] for r in hits]),
        "med_bw_ratio_hits": med([r["bw_ratio"] for r in hits]),
        # H6 background false positive: target OUT of view, NCC still fires above 0.40
        "fp_rate_oov": round(sum(1 for r in oov if r["hit"]) / len(oov), 3) if oov else None,
        "med_score_oov_hits": med([r["score"] for r in oov if r["hit"]]),
        "gt_r_range_vis": [round(min(r["gt_r"] for r in vis), 2), round(max(r["gt_r"] for r in vis), 2)] if vis else None,
        "bw_exp_range_vis": [round(min(r["bw_exp"] for r in vis), 1), round(max(r["bw_exp"] for r in vis), 1)] if vis else None,
    }
    _log("H1,H2,H3,H5", "ncc summary", summary)
    print("[ncc] TEACH_NCC_RESULT " + json.dumps(summary), flush=True)
    env.close()


if __name__ == "__main__":
    main()
