# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Debug sonar KF measurement path in Isaac-AUV-Tracking-Direct-v0.

Focuses on the changed boundary: GT geometric (r,α) vs OceanSim centroid meas.
Does not change the policy network.

  ./isaaclab.sh -p scripts/debug_auv_tracking_sonar_meas.py --headless --enable_cameras
  HOLD=1 ./scripts/...  (via args: --hold_agent --still_target)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug sonar vs GT measurement in AUV tracking.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--hold_agent", action="store_true", help="Zero action; keep robot near spawn.")
parser.add_argument("--still_target", action="store_true", help="Freeze target XY after reset.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="SB3 SAC zip; if set, drive AUV with policy under free motion.",
)
parser.add_argument("--range_bias", type=float, default=None, help="Override cfg sonar_range_bias_m.")
parser.add_argument("--gate_deg", type=float, default=None, help="Override cfg sonar_prior_gate_deg.")
parser.add_argument("--out_dir", type=str, default="logs/auv_tracking/sonar_meas_debug")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.direct.isaac_auv_tracking.sonar_meas import body_polar_from_xy, detect_centroid_polar
from isaaclab_tasks.direct.sonar_flow_hover.modules.geometry import beacon_bearing_range, sonar_pos_w
from isaaclab_tasks.utils import parse_env_cfg


def _wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def main() -> None:
    task = "Isaac-AUV-Tracking-Direct-v0"
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.range_bias is not None:
        env_cfg.sonar_range_bias_m = float(args_cli.range_bias)
    if args_cli.gate_deg is not None:
        env_cfg.sonar_prior_gate_deg = float(args_cli.gate_deg)
    if args_cli.still_target:
        env_cfg.target_speed = 0.0
        env_cfg.target_max_speed = 0.0

    env = gym.make(task, cfg=env_cfg)
    raw = env.unwrapped
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "meas_vs_gt.csv"

    policy = None
    if args_cli.checkpoint:
        from stable_baselines3 import SAC

        ckpt = Path(args_cli.checkpoint)
        print(f"[policy] loading SAC {ckpt}", flush=True)
        policy = SAC.load(str(ckpt), device=str(raw.device))

    obs, _ = env.reset()
    # Snapshot freeze pose if holding agent / still target
    hold_pose = raw._robot.data.root_pos_w.clone()
    hold_quat = raw._robot.data.root_quat_w.clone()
    still_xy = raw._target_pos.clone()

    bias = float(raw.cfg.sonar_range_bias_m)
    radius = float(raw.cfg.target_radius)
    fov_half = float(raw._fov_half)
    sensor_r = float(raw.cfg.sensor_range)
    ocfg = raw.cfg.oceansim_sonar_cfg

    rows: list[dict] = []
    hit = 0
    mode = "policy" if policy is not None else ("hold" if args_cli.hold_agent else "zero_action")
    print(
        f"[cfg] mode={mode} still_target={args_cli.still_target} "
        f"bias={bias:.3f} radius={radius:.3f} gate={raw.cfg.sonar_prior_gate_deg}° "
        f"sensor_r={sensor_r} fov_half={math.degrees(fov_half):.1f}° "
        f"oceansim=[{ocfg.min_range},{ocfg.max_range}]m fov={ocfg.hori_fov}° "
        f"target_speed={raw.cfg.target_speed}",
        flush=True,
    )

    for step in range(int(args_cli.steps)):
        if args_cli.still_target:
            raw._target_pos[:] = still_xy
            raw._target_state[:, 0:2] = still_xy
            raw._target_state[:, 2:4] = 0.0
            raw._sync_target_entity()

        if args_cli.hold_agent:
            # Pin BEFORE step so sonar render and GT share the same pose.
            pose = torch.zeros(raw.num_envs, 7, device=raw.device)
            pose[:, :3] = hold_pose
            pose[:, 3:7] = hold_quat
            raw._robot.write_root_pose_to_sim(pose, raw._robot._ALL_INDICES)
            raw._robot.write_root_velocity_to_sim(
                torch.zeros(raw.num_envs, 6, device=raw.device), raw._robot._ALL_INDICES
            )

        if policy is not None:
            pol = obs["policy"] if isinstance(obs, dict) else obs
            act_np, _ = policy.predict(pol.detach().cpu().numpy(), deterministic=True)
            action = torch.as_tensor(act_np, device=raw.device, dtype=torch.float32)
            if action.ndim == 1:
                action = action.unsqueeze(0)
        else:
            action = torch.zeros(raw.num_envs, 2, device=raw.device)
        obs, rew, terminated, truncated, info = env.step(action)

        if args_cli.hold_agent:
            pose = torch.zeros(raw.num_envs, 7, device=raw.device)
            pose[:, :3] = hold_pose
            pose[:, 3:7] = hold_quat
            raw._robot.write_root_pose_to_sim(pose, raw._robot._ALL_INDICES)
            raw._robot.write_root_velocity_to_sim(
                torch.zeros(raw.num_envs, 6, device=raw.device), raw._robot._ALL_INDICES
            )

        # --- Recompute the changed measurement boundary (same logic as _update_belief) ---
        raw._ensure_sonar()
        sonar_xy, yaw = raw._sonar_pose_xy_yaw()
        sonar_pw = sonar_pos_w(
            raw._robot.data.root_pos_w,
            raw._robot.data.root_quat_w,
            raw._sonar_mount_offset_b,
        )
        tgt_w = torch.zeros(raw.num_envs, 3, device=raw.device)
        tgt_w[:, :2] = raw._target_pos
        tgt_w[:, 2] = raw.cfg.starting_depth
        gt_alpha, gt_r = beacon_bearing_range(tgt_w, sonar_pw, raw._robot.data.root_quat_w)
        # Surface range (acoustic) ≈ center − radius on LOS
        gt_r_surf = torch.clamp(gt_r - radius, min=1.0e-3)

        az_pred, r_pred = body_polar_from_xy(raw.kf.state[:, :2], sonar_xy, yaw)
        # Note: after step, kf already predicted in _get_observations; state is prior for NEXT step.
        # For debug of the meas that was just used, reconstruct from pre-predict would need hooks.
        # Here we compare current detect vs GT (image truth), and log KF prior used for NEXT gate.
        r_roi = torch.clamp(r_pred - bias, min=1.0e-3)
        r_s, a_s, valid = detect_centroid_polar(
            raw.pipeline.polar_image,
            raw.gt,
            prior_azimuth=az_pred,
            prior_range=r_roi,
            prior_gate_deg=float(raw.cfg.sonar_prior_gate_deg),
            hori_fov_deg=float(ocfg.hori_fov),
            min_range=float(ocfg.min_range),
            max_range=float(ocfg.max_range),
            range_window_m=float(raw.cfg.sonar_range_window_m),
            detect_max_range=float(sensor_r),
        )
        # Also ungated global detect (isolates gate failures)
        r_g, a_g, valid_g = detect_centroid_polar(
            raw.pipeline.polar_image,
            raw.gt,
            prior_azimuth=None,
            prior_range=None,
            prior_gate_deg=float(raw.cfg.sonar_prior_gate_deg),
            hori_fov_deg=float(ocfg.hori_fov),
            min_range=float(ocfg.min_range),
            max_range=float(ocfg.max_range),
            range_window_m=float(raw.cfg.sonar_range_window_m),
            detect_max_range=float(sensor_r),
        )
        r_corr = r_s + bias
        in_range = r_corr <= sensor_r
        in_fov = torch.abs(a_s) <= fov_half
        observed_re = (valid > 0.5) & in_range & in_fov

        i = 0
        v = float(valid[i].item())
        vg = float(valid_g[i].item())
        if v > 0.5:
            hit += 1
            dr = abs(float(r_corr[i].item()) - float(gt_r[i].item()))
            da = abs(math.degrees(float(_wrap_pi(a_s[i] - gt_alpha[i]).item())))
            dr_surf = abs(float(r_s[i].item()) - float(gt_r_surf[i].item()))
        else:
            dr = da = dr_surf = float("nan")

        # Miss taxonomy
        miss_reason = "ok"
        if v <= 0.5:
            if vg > 0.5:
                miss_reason = "gate_miss"
            elif float(gt_r[i].item()) > float(ocfg.max_range):
                miss_reason = "beyond_sonar_max"
            elif abs(float(gt_alpha[i].item())) > math.radians(float(ocfg.hori_fov) * 0.5):
                miss_reason = "beyond_sonar_fov"
            else:
                miss_reason = "no_blob"
        elif not bool(in_range[i].item()):
            miss_reason = "holo_range_gate"
        elif not bool(in_fov[i].item()):
            miss_reason = "holo_fov_gate"

        row = {
            "step": step,
            "gt_r": float(gt_r[i].item()),
            "gt_r_surf": float(gt_r_surf[i].item()),
            "gt_alpha_deg": math.degrees(float(gt_alpha[i].item())),
            "r_raw": float(r_s[i].item()) if v > 0.5 else float("nan"),
            "r_corr": float(r_corr[i].item()) if v > 0.5 else float("nan"),
            "alpha_deg": math.degrees(float(a_s[i].item())) if v > 0.5 else float("nan"),
            "abs_dr": dr,
            "abs_dr_surf": dr_surf,
            "abs_dalpha_deg": da,
            "valid": v,
            "valid_global": vg,
            "env_observed": float(raw._observed[i].item()),
            "r_pred": float(r_pred[i].item()),
            "az_pred_deg": math.degrees(float(az_pred[i].item())),
            "miss_reason": miss_reason,
            "rew": float(rew[i].item()),
            "polar_max": float(raw.pipeline.polar_image[i].max().item()),
            "polar_mean": float(raw.pipeline.polar_image[i].mean().item()),
            "dist_agent_tgt": float(
                torch.norm(raw._robot.data.root_pos_w[i, :2] - raw._target_pos[i]).item()
            ),
        }
        rows.append(row)

        if step % max(int(args_cli.log_every), 1) == 0 or step + 1 == args_cli.steps:
            print(
                f"[step {step:03d}] gt(r={row['gt_r']:.2f},α={row['gt_alpha_deg']:+.1f}°) "
                f"sonar(r={row['r_corr'] if v>0.5 else float('nan'):.2f},"
                f"α={row['alpha_deg'] if v>0.5 else float('nan'):+.1f}°) "
                f"|Δr|={dr:.3f} |Δα|={da:.2f}° valid={v:.0f}/g={vg:.0f} "
                f"obs={row['env_observed']:.0f} miss={miss_reason} "
                f"dist={row['dist_agent_tgt']:.2f} polar_max={row['polar_max']:.3f}",
                flush=True,
            )

        if bool(terminated[i] or truncated[i]):
            obs, _ = env.reset()
            hold_pose = raw._robot.data.root_pos_w.clone()
            hold_quat = raw._robot.data.root_quat_w.clone()
            still_xy = raw._target_pos.clone()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hits = [r for r in rows if r["valid"] > 0.5]
    env_obs_rate = sum(r["env_observed"] for r in rows) / max(len(rows), 1)
    gate_misses = sum(1 for r in rows if r["miss_reason"] == "gate_miss")
    no_blob = sum(1 for r in rows if r["miss_reason"] == "no_blob")
    beyond = sum(1 for r in rows if r["miss_reason"].startswith("beyond"))
    summary = out_dir / "summary.txt"
    lines = [
        f"mode={mode} still_target={args_cli.still_target}",
        f"steps={len(rows)} hit_rate={hit/len(rows):.3f} ({hit}/{len(rows)})",
        f"env_observed_rate={env_obs_rate:.3f}",
        f"gate_miss={gate_misses} no_blob={no_blob} beyond_fov_or_range={beyond}",
        f"bias={bias} radius={radius}",
    ]
    if hits:
        mean_dr = sum(r["abs_dr"] for r in hits) / len(hits)
        mean_da = sum(r["abs_dalpha_deg"] for r in hits) / len(hits)
        mean_drs = sum(r["abs_dr_surf"] for r in hits) / len(hits)
        lines += [
            f"mean_|Δr|_center={mean_dr:.4f} m",
            f"mean_|Δr|_surf(raw)={mean_drs:.4f} m",
            f"mean_|Δα|={mean_da:.3f} deg",
            f"mean_dist_agent_tgt={sum(r['dist_agent_tgt'] for r in rows)/len(rows):.3f} m",
        ]
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"[saved] {csv_path}", flush=True)
    print(f"[saved] {summary}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
