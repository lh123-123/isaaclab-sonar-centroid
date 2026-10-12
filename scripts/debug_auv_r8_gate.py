"""R8 existence-gate probe: false-kill rate (H-B) + stale reward value (H-A).

Runs the real Direct-v5-flow env (flow_source=raft, the R8 configuration) and, each
step, reads the gate internals straight off the env, recomputes the orange bbox from
the SAME render frame the env used, and compares both against GT envelope membership
(the same kf.get_observation call _get_dones uses). No env code is modified.

H-B (gate has no teach branch): a frame is a FALSE KILL when the target is inside
the envelope (r<=sensor_range, |a|<=fov_half) but exist = flow_ok | bbox_hit is
False. The OOV streak then advances; 10 in a row end a healthy episode.

H-A (stale estimate): reset clears _flow_est_ok but NOT _flow_est_r/_a, so the first
post-reset reward uses the PREVIOUS episode's azimuth. Reset puts the target at 0 deg,
so the honest r_bearing would be ~1.0; the stale one can be ~0.1.

NDJSON rows go to the debug-756db1 ingest channel; the RESULT line summarizes.
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
            "runId": "r8-gate",
            "hypothesisId": hypothesis,
            "location": "debug_auv_r8_gate.py",
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
    parser = argparse.ArgumentParser(description="R8 existence-gate / stale-estimate probe")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=400)
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
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def _run(args_cli) -> None:
    import gymnasium as gym
    import numpy as np
    import torch

    import isaaclab_tasks  # noqa: F401
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
    sensor_range = float(raw.cfg.sensor_range)
    fov_half = float(raw._fov_half)
    sig_b = math.radians(float(raw.cfg.rew_bearing_sigma_deg))
    print(
        f"[gate] env={args_cli.task} n={n} dt={dt} sensor_range={sensor_range} "
        f"fov_half_deg={math.degrees(fov_half):.1f} sig_b_deg={math.degrees(sig_b):.1f} "
        f"flow_source={getattr(raw.cfg, 'flow_source', '?')}",
        flush=True,
    )

    # One fixed action per env: straight, spin, diagonal, STILL (the still env is the
    # static-blindness stress case for H-B).
    plans = [
        (0.2, 0.0),
        (0.0, 0.3),
        (0.1, -0.2),
        (0.0, 0.0),
    ]
    actions = torch.zeros((n, raw.action_space.shape[-1]), device=raw.device)
    for i in range(n):
        actions[i, 0], actions[i, 1] = plans[i % len(plans)]
    print(f"[gate] action plans={plans[:n]}", flush=True)

    env.reset()
    rows: list[dict] = []
    for step in range(int(args_cli.steps)):
        # BEFORE the step: fresh mask + the estimate values a fresh reward would use.
        prev_ok = raw._prev_rgb_ok.clone()
        stale_a = raw._flow_est_a.clone()

        env.step(actions)

        agent_pos, agent_yaw = raw._agent_pose_2d()
        _z, gt_r_t, gt_a_t = raw.kf.get_observation(raw._target_pos, agent_pos, agent_yaw)
        gt_r = gt_r_t.tolist()
        gt_a_deg = [math.degrees(float(v)) for v in gt_a_t.tolist()]
        in_env = [
            bool(r <= sensor_range and abs(a) <= math.degrees(fov_half))
            for r, a in zip(gt_r, gt_a_deg)
        ]

        flow_ok = raw._flow_est_ok.tolist()
        curr = raw._camera_rgb()
        _bw, _bh, bcount = orange_extent(curr)
        bcount = [float(v) for v in (bcount.tolist() if hasattr(bcount, "tolist") else bcount)]
        bbox_hit = [c >= 20.0 for c in bcount]
        # env 的存在性分支（去橙化后 = teach NCC 检测）；橙框留作对照列
        det = getattr(raw, "_det", None)
        if det is not None:
            ncc_hit = [bool(v) for v in (det[:, 0] >= 0).tolist()]
            ncc_score = [round(float(v), 3) for v in det[:, 4].tolist()]
            ncc_bw = [round(float(v), 1) for v in det[:, 2].tolist()]
        else:
            ncc_hit = bbox_hit
            ncc_score = [-1.0] * n
            ncc_bw = [float("nan")] * n
        exist = [bool(f | h) for f, h in zip(flow_ok, ncc_hit)]
        oov = raw._oov_streak.tolist()
        est_a_deg = [math.degrees(float(v)) for v in raw._flow_est_a.tolist()]
        est_r = [float(v) for v in raw._flow_est_r.tolist()]
        s_ratio = [float(v) for v in raw._size_ratio.tolist()]
        teach_ok = raw._teach_w_ok.tolist()

        for i in range(n):
            fresh = not bool(prev_ok[i])
            row = {
                "env": i,
                "t": round((step + 1) * dt, 2),
                "gt_r": round(gt_r[i], 3),
                "gt_a_deg": round(gt_a_deg[i], 2),
                "in_env": in_env[i],
                "flow_ok": bool(flow_ok[i]),
                "bbox_hit": bbox_hit[i],
                "bcount": round(bcount[i], 0),
                "ncc_hit": ncc_hit[i],
                "ncc_score": ncc_score[i],
                "ncc_bw": ncc_bw[i],
                "exist": exist[i],
                "false_kill": bool(in_env[i] and not exist[i]),
                "oov_streak": int(oov[i]),
                "est_a_deg": round(est_a_deg[i], 2),
                "est_r": round(est_r[i], 3),
                "size_ratio": round(s_ratio[i], 3),
                "teach_ok": bool(teach_ok[i]),
                "fresh": fresh,
            }
            if fresh:
                a_st = math.degrees(float(stale_a[i]))
                row["stale_a_deg"] = round(a_st, 2)
                row["r_bearing_stale"] = round(math.exp(-((a_st / math.degrees(sig_b)) ** 2)), 4)
                row["r_bearing_honest"] = round(
                    math.exp(-((gt_a_deg[i] / math.degrees(sig_b)) ** 2)), 4
                )
            rows.append(row)
        if (step + 1) % 50 == 0:
            recent = rows[-n * 50 :]
            fk = sum(1 for r in recent if r["false_kill"])
            ie = sum(1 for r in recent if r["in_env"])
            print(
                f"[gate] t={(step + 1) * dt:6.1f} in_env={ie} FK={fk} "
                f"oov_max={max(r['oov_streak'] for r in recent)}",
                flush=True,
            )

    _log("H-B,H-A", "run complete", {"steps": len(rows) // n, "envs": n})

    in_env_rows = [r for r in rows if r["in_env"]]
    fk_rows = [r for r in rows if r["false_kill"]]
    fresh_rows = [r for r in rows if r["fresh"]]
    fk_runs: list[int] = []
    for i in range(n):
        run = 0
        for r in rows:
            if r["env"] != i:
                continue
            if r["false_kill"]:
                run += 1
            else:
                if run:
                    fk_runs.append(run)
                run = 0
        if run:
            fk_runs.append(run)
    ncc_only = [r for r in in_env_rows if (not r["flow_ok"]) and r["ncc_hit"]]
    ncc_a_err = sorted(abs(r["est_a_deg"] - r["gt_a_deg"]) for r in ncc_only)
    ncc_r_err = sorted(abs(r["est_r"] - r["gt_r"]) for r in ncc_only)
    summary = {
        "steps": len(rows) // n,
        "envs": n,
        "in_env_frames": len(in_env_rows),
        "false_kill_frames": len(fk_rows),
        "false_kill_rate": round(len(fk_rows) / max(len(in_env_rows), 1), 4),
        "fk_by_env": {
            str(i): {
                "fk": sum(1 for r in fk_rows if r["env"] == i),
                "in_env": sum(1 for r in in_env_rows if r["env"] == i),
            }
            for i in range(n)
        },
        "fk_max_consecutive": max(fk_runs) if fk_runs else 0,
        "fk_runs_ge10": sum(1 for x in fk_runs if x >= 10),
        "branch_among_in_env": {
            "flow_ok": sum(1 for r in in_env_rows if r["flow_ok"]),
            "ncc_hit": sum(1 for r in in_env_rows if r["ncc_hit"]),
            "orange_hit_ref": sum(1 for r in in_env_rows if r["bbox_hit"]),
            "neither": sum(1 for r in in_env_rows if not r["exist"]),
        },
        "ncc_only_frames": len(ncc_only),
        "ncc_a_err_med_deg": round(ncc_a_err[len(ncc_a_err) // 2], 2) if ncc_only else None,
        "ncc_r_err_med_m": round(ncc_r_err[len(ncc_r_err) // 2], 3) if ncc_only else None,
        "fresh_frames": len(fresh_rows),
        "stale_a_abs_med": (
            round(sorted(abs(r["stale_a_deg"]) for r in fresh_rows)[len(fresh_rows) // 2], 2)
            if fresh_rows
            else None
        ),
        "r_bearing_stale_med": (
            round(sorted(r["r_bearing_stale"] for r in fresh_rows)[len(fresh_rows) // 2], 4)
            if fresh_rows
            else None
        ),
        "r_bearing_honest_med": (
            round(sorted(r["r_bearing_honest"] for r in fresh_rows)[len(fresh_rows) // 2], 4)
            if fresh_rows
            else None
        ),
        "teach_ok_rate": round(sum(1 for r in rows if r["teach_ok"]) / max(len(rows), 1), 4),
        "size_ratio_med_in_env": (
            round(sorted(r["size_ratio"] for r in in_env_rows)[len(in_env_rows) // 2], 3)
            if in_env_rows
            else None
        ),
    }
    print("[gate] R8_GATE_RESULT " + json.dumps(summary), flush=True)
    for r in rows:
        _log("H-B,H-A", "gate sample", r)


if __name__ == "__main__":
    main()
