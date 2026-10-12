# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Motion check for v4 teach-match range/bearing and sparse flow.

Steps a constant surge/yaw long enough to leave the 3 s soft-start and the 5 s leash,
then compares the teach 4-vector and the sparse-flow vector to geometric ground truth.

    ./scripts/debug_auv_v4_flow_teach_motion.sh --num_envs 1 --steps 150
"""

from __future__ import annotations

import argparse
import json
import math
import time

_LOG = "/home/lh/IsaacLab/.cursor/debug-ffd96a.log"


def _log(hypothesis_id: str, message: str, data: dict) -> None:
    # #region agent log
    rec = {
        "sessionId": "ffd96a",
        "runId": "core",
        "hypothesisId": hypothesis_id,
        "location": "debug_auv_v4_flow_teach_motion.py",
        "message": message,
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
    parser = argparse.ArgumentParser(description="v4 teach-match and sparse-flow motion check.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--action_v", type=float, default=0.15)
    parser.add_argument("--action_w", type=float, default=0.08)
    parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v4-cam")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Trained v4 SAC zip. Actions come from this policy; teach-match is still measured.",
    )
    parser.add_argument(
        "--save_flow_dir",
        type=str,
        default="",
        help="If set, write sparse-flow overlays of the 64x64 policy frame every 10 steps.",
    )
    parser.add_argument("--ui", action="store_true", help="Open the Isaac Sim window. Needs a display.")
    parser.add_argument(
        "--save_video",
        type=str,
        default="",
        help="Directory for the camera mp4 and the optical-flow mp4 made from those frames.",
    )
    parser.add_argument(
        "--step_delay",
        type=float,
        default=0.0,
        help="Sleep between steps when --ui is set, so the window is watchable.",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = not bool(args_cli.ui)
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        _run(args_cli)
    finally:
        simulation_app.close()


def _save_flow_overlay(out_dir: str, rgb_chw, xs, ys, du, dv, valid, row: dict) -> None:
    """Draw one averaged flow point on the newest stacked frame. ``rgb_chw`` is ``(3, H, W)`` uint8."""
    import os

    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    rgb = rgb_chw.detach().byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(rgb, mode="RGB")
    scale = 4 if im.size[0] <= 64 else 1
    if scale != 1:
        im = im.resize((im.size[0] * scale, im.size[1] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im)
    xs_np = xs[0].detach().cpu().numpy()
    ys_np = ys[0].detach().cpu().numpy()
    du_np = du[0].detach().cpu().numpy()
    dv_np = dv[0].detach().cpu().numpy()
    ok_np = valid[0].detach().cpu().numpy().astype(bool)
    keep = ok_np & (xs_np >= 0.0) & (ys_np >= 0.0)
    if bool(keep.any()):
        x0 = float(xs_np[keep].mean()) * scale
        y0 = float(ys_np[keep].mean()) * scale
        x1 = x0 + float(du_np[keep].mean()) * scale
        y1 = y0 + float(dv_np[keep].mean()) * scale
        draw.line((x0, y0, x1, y1), fill=(40, 220, 80), width=3)
        draw.ellipse((x1 - 5, y1 - 5, x1 + 5, y1 + 5), fill=(255, 220, 40))
    path = os.path.join(out_dir, f"t{row['t']:.1f}_dx{row['flow_dx_px']:+.1f}_dy{row['flow_dy_px']:+.1f}.png")
    im.save(path)
    print(f"[motion] flow overlay {path} {rgb.shape[1]}x{rgb.shape[0]} valid={row['flow_valid']:.2f}", flush=True)


def _follow_behind(raw) -> None:
    """Put the GUI camera a few metres behind the agent, looking at it."""
    try:
        from omni.kit.viewport.utility import get_viewport_from_window_name
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        from pxr import Gf

        from isaaclab.utils.math import quat_apply

        vp = get_viewport_from_window_name("Viewport")
        if vp is None:
            return
        if vp.get_active_camera() != "/OmniverseKit_Persp":
            vp.set_active_camera("/OmniverseKit_Persp")
        pos = raw._robot.data.root_pos_w[0]
        quat = raw._robot.data.root_quat_w[0]
        # Body +X is forward. Sit behind and above, look at the hull.
        offset = pos.new_tensor([-4.0, 0.0, 1.6])
        eye = quat_apply(quat.unsqueeze(0), offset.unsqueeze(0))[0] + pos
        look = pos + pos.new_tensor([0.0, 0.0, 0.3])
        cam = ViewportCameraState("/OmniverseKit_Persp", vp)
        cam.set_position_world(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])), True)
        cam.set_target_world(Gf.Vec3d(float(look[0]), float(look[1]), float(look[2])), True)
    except Exception as exc:  # noqa: BLE001 - a viewport miss must not stop the run
        print(f"[motion] follow camera: {exc}", flush=True)


def _write_flow_videos(out_dir: str, frames: list, fps: float) -> None:
    """Write the camera video, then optical flow computed from those frames.

    One yellow dot is the mean of the points that are still on the target.
    The green arrow ends on that dot. Short motions are drawn at least 16 px
    long so the direction is visible; the dot stays on the measured position.
    """
    import os

    import cv2
    import numpy as np
    import torch

    from isaaclab_tasks.direct.isaac_auv_tracking.sparse_flow import step_locked_flow

    os.makedirs(out_dir, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    camera_path = os.path.join(out_dir, "camera.mp4")
    flow_path = os.path.join(out_dir, "flow.mp4")
    camera = cv2.VideoWriter(camera_path, fourcc, fps, (w, h))
    flow_writer = cv2.VideoWriter(flow_path, fourcc, fps, (w, h))
    xy = torch.zeros(1, 32, 2)
    alive = torch.zeros(1, 32, dtype=torch.bool)
    prev = None
    on_target = 0
    for frame in frames:
        rgb = np.ascontiguousarray(frame)
        camera.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        curr = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        if prev is None:
            prev = curr
        flow, xs, ys, du, dv, ok, xy, alive = step_locked_flow(prev, curr, xy, alive)
        prev = curr
        keep = (ok[0] & (xs[0] >= 0)).detach().cpu().numpy()
        drawn = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if bool(keep.any()):
            on_target += 1
            xs_k = xs[0].detach().cpu().numpy()[keep]
            ys_k = ys[0].detach().cpu().numpy()[keep]
            du_k = du[0].detach().cpu().numpy()[keep]
            dv_k = dv[0].detach().cpu().numpy()[keep]
            x1 = float((xs_k + du_k).mean())
            y1 = float((ys_k + dv_k).mean())
            u = float(du_k.mean())
            v = float(dv_k.mean())
            length = math.hypot(u, v)
            tip = (int(round(x1)), int(round(y1)))
            if length >= 0.5:
                show = max(length, 16.0)
                tail = (int(round(x1 - u / length * show)), int(round(y1 - v / length * show)))
                cv2.arrowedLine(drawn, tail, tip, (80, 220, 40), 2, tipLength=0.35)
            cv2.circle(drawn, tip, 5, (40, 220, 255), -1)
        flow_writer.write(drawn)
    camera.release()
    flow_writer.release()
    print(
        f"[motion] video {camera_path} and {flow_path} frames={len(frames)} "
        f"fps={fps:.2f} dot_on_target={on_target}/{len(frames)}",
        flush=True,
    )


def _grab_render_frames(raw, frames: list) -> None:
    """Keep one camera frame each time the sim actually renders."""
    rendered = {"flag": False}
    orig_render = raw.sim.render
    orig_update = raw.scene.update

    def _render(*args, **kwargs):
        out = orig_render(*args, **kwargs)
        rendered["flag"] = True
        return out

    def _update(*args, **kwargs):
        out = orig_update(*args, **kwargs)
        if rendered["flag"]:
            rendered["flag"] = False
            try:
                rgb = raw._camera_rgb()
                frames.append(rgb[0].detach().byte().permute(1, 2, 0).cpu().numpy().copy())
            except Exception:
                pass
        return out

    raw.sim.render = _render
    raw.scene.update = _update


def _phase(t_s: float) -> str:
    if t_s < 3.0:
        return "soft"
    if t_s < 5.0:
        return "leash"
    return "free"


def _run(args_cli) -> None:
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v4_cam_env import AUVTrackingV4CamEnvCfg
    from isaaclab_tasks.direct.isaac_auv_tracking.sparse_flow import sparse_flow_points, sparse_flow_vector
    from isaaclab_tasks.direct.isaac_auv_tracking.teach_match import orange_core, orange_extent

    cfg = AUVTrackingV4CamEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    # One control step is 0.2 s. Rendering only then makes the window jump.
    # 4 physics steps is 30 Hz, which is what the video and the GUI show.
    if args_cli.ui or str(args_cli.save_video).strip():
        cfg.sim.render_interval = 4
    env = gym.make(args_cli.task, cfg=cfg)
    raw = env.unwrapped
    dt = float(raw.cfg.sim.dt) * float(raw.cfg.decimation)
    # 90 deg HFOV on the 64 px observation: fx = (W/2) / tan(HFOV/2) = 32.
    fx_obs = 32.0
    half_fov = 0.5 * float(raw.cfg.cam_hfov_deg) * 3.14159265 / 180.0
    video_frames: list = []
    if str(args_cli.save_video).strip():
        _grab_render_frames(raw, video_frames)

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    actions = torch.zeros((raw.num_envs, raw.action_space.shape[-1]), device=raw.device)
    actions[:, 0] = float(args_cli.action_v)
    actions[:, 1] = float(args_cli.action_w)
    agent = None
    if str(args_cli.checkpoint).strip():
        import numpy as np
        from stable_baselines3 import SAC

        from isaaclab_tasks.direct.isaac_auv_tracking.agents.cam_encoder import MidCNNExtractor  # noqa: F401
        from isaaclab_tasks.direct.isaac_auv_tracking.agents.custom_sac_policy import CustomSACPolicy  # noqa: F401

        agent = SAC.load(str(args_cli.checkpoint), device=str(raw.device))
        print(f"[motion] policy {args_cli.checkpoint} obs={agent.observation_space}", flush=True)
        _log("D", "policy loaded", {"ckpt": str(args_cli.checkpoint), "space": str(agent.observation_space)})

    def _policy_action(policy_obs) -> None:
        """Drive with the grayscale stack the checkpoint was trained on."""
        import numpy as np

        buf = raw._frame_buf
        x = buf.float()
        ys = []
        for i in range(4):
            rgb = x[:, 3 * i : 3 * i + 3]
            y = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
            ys.append(y.round().clamp(0, 255))
        image = torch.stack(ys, dim=1).to(torch.uint8)[0].detach().cpu().numpy()
        state = policy_obs["policy"]["state"] if "policy" in policy_obs else policy_obs["state"]
        if torch.is_tensor(state):
            state_np = state.detach().float().cpu().numpy()
        else:
            state_np = np.asarray(state, dtype=np.float32)
        if state_np.ndim == 2:
            state_np = state_np[0]
        act, _ = agent.predict({"image": image, "state": state_np}, deterministic=True)
        act = torch.as_tensor(act, device=raw.device, dtype=actions.dtype).reshape(-1)
        actions[0, : act.numel()] = act

    rows: list[dict] = []
    prev_a = None
    if not args_cli.headless:
        print("[motion] GUI camera follows 4 m behind the agent", flush=True)
        _follow_behind(raw)
    for step in range(int(args_cli.steps)):
        if not args_cli.headless:
            _follow_behind(raw)
        if agent is not None:
            _policy_action(obs)
        step_out = env.step(actions)
        obs = step_out[0]
        agent_pos, agent_yaw = raw._agent_pose_2d()
        diff = raw._target_pos - agent_pos
        cos_y = torch.cos(agent_yaw)
        sin_y = torch.sin(agent_yaw)
        dx = cos_y * diff[:, 0] + sin_y * diff[:, 1]
        dy = -sin_y * diff[:, 0] + cos_y * diff[:, 1]
        gt_r = torch.sqrt(dx * dx + dy * dy + 1.0e-10)
        gt_a = torch.atan2(dy, dx)
        # Camera sits 0.35 m forward of the body origin. Image scale follows this range.
        cam_x = agent_pos[:, 0] + 0.35 * cos_y
        cam_y = agent_pos[:, 1] + 0.35 * sin_y
        cdx = raw._target_pos[:, 0] - cam_x
        cdy = raw._target_pos[:, 1] - cam_y
        cam_r = torch.sqrt(cdx * cdx + cdy * cdy + 1.0e-10)
        tyaw = raw.target_motion.yaw
        los = torch.atan2(cdy, cdx)
        aspect = torch.atan2(torch.sin(tyaw - los), torch.cos(tyaw - los))
        teach = raw._teach_vec[0]
        images = raw._frame_buf
        if images.dtype == torch.uint8:
            images_f = images.float() / 255.0
        else:
            images_f = images.float()
        if hasattr(raw, "_flow_vec"):
            flow = raw._flow_vec[0]
            flow_wh = raw._hi_curr.shape[-1]
        else:
            flow = sparse_flow_vector(images_f)[0]
            flow_wh = images_f.shape[-1]
        t_s = (step + 1) * dt
        gt_r0 = float(gt_r[0])
        gt_a0 = float(gt_a[0])
        d_a = 0.0 if prev_a is None else gt_a0 - prev_a
        prev_a = gt_a0
        width, height, count = orange_extent(raw._camera_rgb())
        ref_w = float(raw._extent_ref_w[0])
        ref_h = float(raw._extent_ref_h[0])
        w0 = float(width[0])
        h0 = float(height[0])
        rw = 2.0 * ref_w / max(w0, 1.0)
        rh = 2.0 * ref_h / max(h0, 1.0)
        sw = abs(math.log(max(w0, 1.0) / max(ref_w, 1.0)))
        sh = abs(math.log(max(h0, 1.0) / max(ref_h, 1.0)))
        r_opt = rw if sw <= sh else rh
        core_rms, core_n, core_w, core_h = orange_core(raw._camera_rgb())
        ref_rms = float(getattr(raw, "_core_ref_rms", torch.zeros(1, device=raw.device))[0])
        ref_n = float(getattr(raw, "_core_ref_n", torch.zeros(1, device=raw.device))[0])
        ref_cw = float(raw._core_ref_w[0])
        ref_ch = float(raw._core_ref_h[0])
        cw0 = float(core_w[0])
        ch0 = float(core_h[0])
        csw = abs(math.log(max(cw0, 1.0) / max(ref_cw, 1.0)))
        csh = abs(math.log(max(ch0, 1.0) / max(ref_ch, 1.0)))
        rcw = 2.0 * ref_cw / max(cw0, 1.0)
        rch = 2.0 * ref_ch / max(ch0, 1.0)
        r_core = rcw if csw <= csh else rch
        r_rms = 2.0 * ref_rms / max(float(core_rms[0]), 0.5)
        r_area = 2.0 * math.sqrt(ref_n / max(float(core_n[0]), 1.0))
        tgt_speed = float(torch.linalg.norm(raw._target_state[0, 2:4]))
        r_hat = float(teach[0])
        a_hat = float(teach[1])
        failed = r_hat == 0.0 and a_hat == 0.0 and float(teach[3]) == 0.0
        row = {
            "step": step + 1,
            "t": round(t_s, 2),
            "phase": _phase(t_s),
            "gt_r": round(gt_r0, 4),
            "cam_r": round(float(cam_r[0]), 4),
            "aspect": round(float(aspect[0]), 4),
            "gt_a": round(gt_a0, 4),
            "r_hat": round(r_hat, 4),
            "a_hat": round(a_hat, 4),
            "r_err": round(r_hat - gt_r0, 4),
            "a_err": round(a_hat - gt_a0, 4),
            "scale": round(float(teach[2]), 4),
            "inlier": round(float(teach[3]), 4),
            "failed": failed,
            "in_fov": abs(gt_a0) < half_fov,
            "flow_dx_px": round(float(flow[0]) * float(flow_wh), 4),
            "flow_dy_px": round(float(flow[1]) * float(flow_wh), 4),
            "flow_valid": round(float(flow[2]), 4),
            "d_a": round(d_a, 4),
            "expect_dx_px": round(-d_a * fx_obs, 4),
            "tgt_speed": round(tgt_speed, 4),
            "act_v": round(float(actions[0, 0]), 4),
            "act_w": round(float(actions[0, 1]), 4),
            "box_w": round(w0, 2),
            "box_h": round(h0, 2),
            "ref_w": round(ref_w, 2),
            "ref_h": round(ref_h, 2),
            "orange_n": round(float(count[0]), 1),
            "r_from_w": round(rw, 4),
            "r_from_h": round(rh, 4),
            "r_opt": round(r_opt, 4),
            "use_w": sw <= sh,
            "r_cos1": round(r_opt / max(math.cos(a_hat), 0.25), 4),
            "r_cos2": round(r_opt / max(math.cos(a_hat), 0.25) ** 2, 4),
            "r_core": round(r_core, 4),
            "r_rms": round(r_rms, 4),
            "r_area": round(r_area, 4),
            "core_w": round(cw0, 2),
            "core_h": round(ch0, 2),
        }
        rows.append(row)
        if row["in_fov"] or (step + 1) % 10 == 0 or step == 0:
            _log("H1,H2,H3", "track sample", row)
        if str(args_cli.save_flow_dir).strip() and ((step + 1) % 10 == 0 or step == 0):
            _save_flow_overlay(
                args_cli.save_flow_dir,
                raw._hi_curr[0],
                raw._flow_xs,
                raw._flow_ys,
                raw._flow_du,
                raw._flow_dv,
                raw._flow_ok,
                row,
            )
        if not args_cli.headless and float(args_cli.step_delay) > 0.0:
            time.sleep(float(args_cli.step_delay))
        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"[motion] t={row['t']:.1f} {row['phase']} gt_r={row['gt_r']:.2f} r_hat={row['r_hat']:.2f} "
                f"gt_a={row['gt_a']:+.3f} a_hat={row['a_hat']:+.3f} inl={row['inlier']:.2f} "
                f"flow=({row['flow_dx_px']:+.2f},{row['flow_dy_px']:+.2f}) v_tgt={row['tgt_speed']:.2f}",
                flush=True,
            )

    def _stats(phase: str) -> dict:
        sel = [r for r in rows if r["phase"] == phase and r["in_fov"]]
        if not sel:
            return {"n": 0}
        n = len(sel)
        fail = sum(1 for r in sel if r["failed"])
        ok = [r for r in sel if not r["failed"]]

        def mean_abs(key: str, src: list[dict]) -> float | None:
            if not src:
                return None
            return round(sum(abs(r[key]) for r in src) / len(src), 4)

        return {
            "n": n,
            "fail_frac": round(fail / n, 3),
            "mean_abs_r_err": mean_abs("r_err", ok),
            "mean_abs_a_err_deg": None
            if not ok
            else round(sum(abs(r["a_err"]) for r in ok) / len(ok) * 57.2958, 3),
            "mean_inlier": round(sum(r["inlier"] for r in sel) / n, 3),
            "mean_abs_flow_dx": mean_abs("flow_dx_px", sel),
            "mean_abs_expect_dx": mean_abs("expect_dx_px", sel),
            "gt_r_min": round(min(r["gt_r"] for r in sel), 3),
            "gt_r_max": round(max(r["gt_r"] for r in sel), 3),
            "gt_a_abs_max_deg": round(max(abs(r["gt_a"]) for r in sel) * 57.2958, 2),
            "mean_tgt_speed": round(sum(r["tgt_speed"] for r in sel) / n, 3),
        }

    summary = {"soft": _stats("soft"), "leash": _stats("leash"), "free": _stats("free"), "steps": len(rows), "dt": dt}
    vis = [r for r in rows if r["in_fov"]]
    if vis:
        summary["track"] = {
            "n": len(vis),
            "mean_abs_gt_a_deg": round(sum(abs(r["gt_a"]) for r in vis) / len(vis) * 57.2958, 2),
            "mean_gt_r": round(sum(r["gt_r"] for r in vis) / len(vis), 3),
            "frac_r_1_to_4": round(sum(1 for r in vis if 1.0 <= r["gt_r"] <= 4.0) / len(vis), 3),
        }
    _log("A,B,C", "track summary", summary)
    print("[motion] FLOW_TEACH_RESULT " + json.dumps(summary), flush=True)
    if video_frames:
        _write_flow_videos(args_cli.save_video, video_frames, fps=1.0 / (float(raw.cfg.sim.dt) * float(raw.cfg.sim.render_interval)))
    env.close()


if __name__ == "__main__":
    main()
