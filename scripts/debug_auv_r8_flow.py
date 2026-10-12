"""R8 optical-flow probe: RAFT quality vs GT displacement (flow debug).

Replicates ``_update_est_flow_image``'s computation OUTSIDE the env with full
instrumentation, on the exact same (prev, curr) render pair the env used, and
compares against the analytic GT pixel displacement. No env code is modified.

Hypotheses:
  H-F1  RAFT-Small intermittently returns a dead flow field (amax ~ 0) on
        low-contrast frames -> lit < 20 -> flow_ok False although the target is
        clearly visible (gate data: ~25% of good-view frames, all action plans).
  H-F2  The relative threshold (amax * flow_mag_rel_frac, floor flow_mag_min_px)
        kills the target blob when background motion dominates (parallax/spin).
  H-F3  RAFT centroid lands on background motion when the target blob is weak
        (the 23-deg median error frames).

Per frame we log: amax, thresh, lit, ok(ours) vs ok(env), GT target pixel and
displacement, RAFT magnitude inside/outside the GT target disk, centroid error,
and the orange bbox count for context.
"""

import argparse
import json
import math
import os
import sys
import time

_LOG_PATH = "/home/lh/IsaacLab/.cursor/debug-756db1.log"


def _log(hypothesis: str, message: str, data: dict) -> None:
    # #region agent log
    try:
        row = {
            "sessionId": "756db1",
            "runId": "r8-flow",
            "hypothesisId": hypothesis,
            "location": "debug_auv_r8_flow.py",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass
    # #endregion


def main() -> None:
    parser = argparse.ArgumentParser(description="R8 RAFT flow quality probe")
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v5-flow")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = True
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    exit_code = 0
    try:
        _run(args_cli)
    except BaseException:
        exit_code = 1
        import traceback

        print("=== PROBE EXCEPTION (printed before close; close() teardown hangs on this box) ===", flush=True)
        traceback.print_exc()
    # simulation_app.close() 的 teardown 在本机反复挂死且会吞掉尚未打印的 traceback；
    # 探针结果均已 flush，故硬退出。正式训练/评测脚本不要照抄此写法。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def _run(args_cli) -> None:
    import faulthandler
    import signal

    import gymnasium as gym
    import numpy as np
    import torch

    # 挂起诊断：卡住时 kill -USR1 <pid> -> 全线程栈写进日志
    faulthandler.enable()
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except (AttributeError, ValueError):
        pass
    print("[bc] faulthandler on (kill -USR1 to dump)", flush=True)

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.direct.isaac_auv_tracking.agents.raft_flow import estimate_flow
    from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v5_flow_env import (
        AUVTrackingV5FlowEnvCfg,
    )
    from isaaclab_tasks.direct.isaac_auv_tracking.teach_match import orange_extent

    cfg = AUVTrackingV5FlowEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=cfg)
    raw = env.unwrapped
    n = int(raw.num_envs)
    dt = float(raw.cfg.sim.dt) * float(raw.cfg.decimation)
    h, w = int(raw.cfg.cam_height), int(raw.cfg.cam_width)
    rel_frac = float(raw.cfg.flow_mag_rel_frac)
    min_px = float(raw.cfg.flow_mag_min_px)
    min_lit = int(raw.cfg.flow_min_lit_px)
    hfov = float(raw.cfg.cam_hfov_deg)
    fx_c = 0.5 * w / math.tan(math.radians(hfov) * 0.5)  # cam res
    print(
        f"[flow] n={n} steps={args_cli.steps} cam={w}x{h} rel_frac={rel_frac} "
        f"min_px={min_px} min_lit={min_lit} hfov={hfov}",
        flush=True,
    )

    plans = [(0.2, 0.0), (0.0, 0.0)]  # straight (parallax case) + still (starved case)
    actions = torch.zeros((n, raw.action_space.shape[-1]), device=raw.device)
    for i in range(n):
        actions[i, 0], actions[i, 1] = plans[i % len(plans)]

    env.reset()
    print("[bc] reset done", flush=True)
    prev_u: list[float] = [float("nan")] * n
    rows: list[dict] = []
    # GT-disk coordinate grid: constant per run, computed once (was per-frame).
    grid_yy, grid_xx = torch.meshgrid(
        torch.arange(h, device=raw.device, dtype=torch.float32),
        torch.arange(w, device=raw.device, dtype=torch.float32),
        indexing="ij",
    )
    t0 = time.time()
    for step in range(int(args_cli.steps)):
        prev_rgb = raw._prev_rgb.clone()
        prev_ok = raw._prev_rgb_ok.clone()
        agent_pos0, agent_yaw0 = raw._agent_pose_2d()

        torch.cuda.synchronize()
        ta = time.time()
        env.step(actions)
        torch.cuda.synchronize()
        tb = time.time()
        if step < 8:
            print(f"[bc] step {step+1}: post_env {tb-ta:.2f}s", flush=True)

        agent_pos, agent_yaw = raw._agent_pose_2d()
        curr = raw._camera_rgb()
        rh, rw = int(curr.shape[1]), int(curr.shape[2])
        fx_r = 0.5 * rw / math.tan(math.radians(hfov) * 0.5)
        ok_env = raw._flow_est_ok.tolist()
        _bw, _bh, bcount = orange_extent(curr)
        bcount = [float(v) for v in (bcount.tolist() if hasattr(bcount, "tolist") else bcount)]
        if step < 8:
            print(f"[bc] step {step+1}: post_cam_orange", flush=True)

        # GT target pixel at render res (pinhole, target shares depth)
        cam_x = agent_pos[:, 0] + 0.35 * torch.cos(agent_yaw)
        cam_y = agent_pos[:, 1] + 0.35 * torch.sin(agent_yaw)
        cdx = raw._target_pos[:, 0] - cam_x
        cdy = raw._target_pos[:, 1] - cam_y
        gt_r = torch.sqrt(cdx * cdx + cdy * cdy)
        bearing = torch.atan2(-torch.sin(agent_yaw) * cdx + torch.cos(agent_yaw) * cdy,
                              torch.cos(agent_yaw) * cdx + torch.sin(agent_yaw) * cdy)
        u_r = (0.5 * (rw - 1) - fx_r * torch.tan(bearing)).tolist()
        v_r = (0.5 * (rh - 1) * torch.ones_like(bearing)).tolist()
        u_c = [u * (w / rw) for u in u_r]  # cam-res column
        gt_r_l = gt_r.tolist()

        # GT displacement at cam res (consecutive projections)
        du_gt: list[float] = [float("nan")] * n
        for i in range(n):
            if not math.isnan(prev_u[i]):
                du_gt[i] = (u_c[i] - prev_u[i]) * (rw / w) / rw  # normalized like du/w
            prev_u[i] = u_r[i]
        if step < 8:
            print(f"[bc] step {step+1}: post_gtmath", flush=True)

        # env semantics: fresh = ~_prev_rgb_ok (no history); RAFT runs on envs WITH
        # history, i.e. prev_ok == True. (An earlier revision inverted this mask and
        # collected zero frames.)
        live = prev_ok.nonzero().flatten().tolist()
        torch.cuda.synchronize()
        tc = time.time()
        est_all = estimate_flow(prev_rgb[live].to(curr.device), curr[live]) if live else None
        torch.cuda.synchronize()
        td = time.time()
        if step < 8:
            print(f"[bc] step {step+1}: post_raft {td-tc:.2f}s live={len(live)}", flush=True)
        for k, i in enumerate(live):
            small = torch.nn.functional.interpolate(est_all[k : k + 1], size=(h, w), mode="area") * (h / rh)
            if step < 8:
                print(f"[bc]   f{k} interp done", flush=True)
            du, dv = small[0, 0], small[0, 1]
            mag = torch.sqrt(du * du + dv * dv)
            amax = float(mag.max().item())
            if step < 8:
                print(f"[bc]   f{k} amax={amax:.2f}", flush=True)
            thresh = max(amax * rel_frac, min_px)
            mask = mag > thresh
            lit = int(mask.sum().item())
            ok = lit >= min_lit
            if step < 8:
                print(f"[bc]   f{k} lit={lit}", flush=True)
            # centroid
            if lit > 0:
                xs = torch.arange(w, device=mag.device, dtype=torch.float32)
                mx = float((mask * xs.view(1, w)).sum().item() / lit)
            else:
                mx = float("nan")
            # GT disk at cam res (target half-length 0.4 m)
            rad = min(max(0.4 / max(gt_r_l[i], 0.3) * fx_c, 1.0), 12.0)
            disk = (grid_xx - u_c[i]) ** 2 + (grid_yy - (h - 1) / 2) ** 2 <= rad**2
            mag_in = float(mag[disk].mean().item()) if bool(disk.any()) else float("nan")
            mag_out = float(mag[~disk].mean().item())
            if step < 8:
                print(f"[bc]   f{k} disk stats in={mag_in:.2f} out={mag_out:.2f}", flush=True)
            # RAFT vector at the GT pixel
            ui = min(max(int(round(u_c[i])), 0), w - 1)
            raft_du = float(du[ui, (h - 1) // 2].item()) / w
            if step < 8:
                print(f"[bc]   f{k} raft_du={raft_du:.4f}", flush=True)
            in_fov = abs(float(torch.rad2deg(bearing[i]).item())) <= hfov * 0.5 + 1.0
            rows.append(
                {
                    "env": i,
                    "t": round((step + 1) * dt, 2),
                    "gt_r": round(gt_r_l[i], 3),
                    "gt_a_deg": round(float(torch.rad2deg(bearing[i]).item()), 2),
                    "in_fov": in_fov,
                    "amax": round(amax, 3),
                    "thresh": round(thresh, 3),
                    "lit": lit,
                    "ok_mine": bool(ok),
                    "ok_env": bool(ok_env[i]),
                    "bcount": round(bcount[i], 0),
                    "mx": round(mx, 1) if mx == mx else None,
                    "u_gt_cam": round(u_c[i], 1),
                    "cent_err_px": round(mx - u_c[i], 1) if mx == mx else None,
                    "mag_in_disk": round(mag_in, 3),
                    "mag_out_disk": round(mag_out, 3),
                    "raft_du_norm": round(raft_du, 5),
                    "du_gt_norm": round(du_gt[i], 5) if du_gt[i] == du_gt[i] else None,
                    "dead_raft": bool(amax < 0.2),
                }
            )
            _log("H-F1,H-F2,H-F3", "flow sample", rows[-1])
            if step < 8:
                print(f"[bc]   f{k} row logged", flush=True)
        te = time.time()
        if step < 8:
            print(f"[bc] step {step+1}: post_frame {te-td:.2f}s", flush=True)
        if step < 30 or (step + 1) % 25 == 0:
            wall = time.time() - t0
            print(
                f"[flow] step={step + 1} frames={len(rows)} wall={wall:.1f}s "
                f"rate={wall / (step + 1):.2f}s/step "
                f"t_env={tb - ta:.2f} t_raft={td - tc:.2f} t_frm={te - td:.2f} live={len(live)}",
                flush=True,
            )
        if (step + 1) % 25 == 0:
            rec = [r for r in rows if r["t"] > (step + 1) * dt - 10.0]
            print(
                f"[flow] recent ok={sum(1 for r in rec if r['ok_mine'])} "
                f"dead={sum(1 for r in rec if r['dead_raft'])} "
                f"disagree={sum(1 for r in rec if r['ok_mine'] != r['ok_env'])}",
                flush=True,
            )

    _log("H-F1,H-F2,H-F3", "flow rows", {"n": len(rows)})

    vis = [r for r in rows if r["in_fov"]]
    okr = [r for r in vis if r["ok_mine"]]
    dead = [r for r in vis if r["dead_raft"]]
    thr_kill = [
        r for r in vis if (not r["ok_mine"]) and (not r["dead_raft"]) and r["mag_in_disk"] == r["mag_in_disk"]
    ]
    cent = [r for r in okr if r["cent_err_px"] is not None]
    cent_err = sorted(abs(r["cent_err_px"]) for r in cent)
    vec = [r for r in vis if r["du_gt_norm"] is not None and abs(r["du_gt_norm"]) > 0.002]
    vec_err = sorted(abs(r["raft_du_norm"] - r["du_gt_norm"]) for r in vec)
    summary = {
        "frames": len(rows),
        "in_fov": len(vis),
        "ok_rate": round(len(okr) / max(len(vis), 1), 3),
        "dead_raft_rate": round(len(dead) / max(len(vis), 1), 3),
        "thresh_kill_frames": len(thr_kill),
        "mag_in_lt_thresh_when_killed": (
            sum(1 for r in thr_kill if r["mag_in_disk"] < r["thresh"]) if thr_kill else 0
        ),
        "ok_env_agreement": round(
            sum(1 for r in vis if r["ok_mine"] == r["ok_env"]) / max(len(vis), 1), 3
        ),
        "cent_err_med_px": round(cent_err[len(cent_err) // 2], 1) if cent_err else None,
        "cent_err_p90_px": round(cent_err[int(len(cent_err) * 0.9)], 1) if cent_err else None,
        "vec_err_med": round(vec_err[len(vec_err) // 2], 5) if vec_err else None,
        "mag_out_med_when_ok": (
            round(sorted(r["mag_out_disk"] for r in okr)[len(okr) // 2], 3) if okr else None
        ),
        "mag_in_med_when_ok": (
            round(sorted(r["mag_in_disk"] for r in okr if r["mag_in_disk"] == r["mag_in_disk"])[len(okr) // 2], 3)
            if okr
            else None
        ),
    }
    print("[flow] R8_FLOW_RESULT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
