#!/usr/bin/env python3
"""Classical baselines (P / PID / LQR) on the v7 visodom+anchor estimate.

The control group for "why RL at all?". Everything except the policy block is the
training environment itself: the same visodom+template estimator, the same domain
randomization, the same PID -> PWM -> 8-thruster chain. The controllers read only
what the real ROV would have (camera-derived displacement + IMU heading), never GT.

The current is pinned per run instead of drawn, so a sweep across current speed is a
sweep and not a random sample:

    ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_baseline.py --headless \
        --num_envs 8 --current 0.30 --dir_deg 17 --controllers p,pid,lqr

Sign calibration matters more than it looks. A controller with a flipped sign still
runs and still produces a number — it just produces a *fake* baseline that makes the
learned policy look better than it is. So each axis is probed for the sign of
d(estimate)/d(action) before any scored episode, and the gains are built on top of
that measured sign.
"""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v7 classical baselines on the visodom+anchor estimate.")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--current", type=float, default=0.30, help="ocean current speed, m/s")
parser.add_argument(
    "--currents",
    type=str,
    default="",
    help="comma-separated speeds to sweep in ONE Isaac instance (e.g. 0.2,0.3,0.45). "
    "Overrides --current. DR is resampled at every reset, so the cfg can be re-pinned "
    "in place instead of paying a fresh Isaac startup per current.",
)
parser.add_argument("--dir_deg", type=float, default=17.0, help="ocean current heading, deg")
parser.add_argument("--controllers", type=str, default="p,pid,lqr")
parser.add_argument("--steps", type=int, default=225, help="policy steps per episode (225 = 15 s @ 15 Hz)")
parser.add_argument(
    "--episodes",
    type=int,
    default=2,
    help="how many per-reset episode log entries to collect per controller before stopping. "
    "Each entry is already an average over the envs that reset together, so with 8 envs in "
    "lockstep N entries ~= 8N episodes. The default of 2 suits the deterministic pinned-current "
    "sweep; use ~10 with --sample_current to get a usable distribution sample.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--ramp_s",
    type=float,
    default=-1.0,
    help="current ramp-up time, s. -1 keeps the cfg default (10 s, the handover story). "
    "0 disables it, so a step change later in the episode is not masked by the ramp.",
)
parser.add_argument("--step_at", type=float, default=0.0, help="s; apply the current step at this time (0 = off)")
parser.add_argument("--step_to", type=float, default=0.45, help="m/s; current speed after the step")
parser.add_argument("--flip_deg", type=float, default=180.0, help="deg; heading change applied with the step")
parser.add_argument(
    "--blind_at",
    type=float,
    default=0.0,
    help="s; start of the visodom dropout window (0 = off). Models the flow integrator "
    "losing usable frames while the template still matches: the classical controllers lose "
    "their only position cue, the policy still has the anchor channels.",
)
parser.add_argument("--blind_len", type=float, default=5.0, help="s; length of the dropout window")
parser.add_argument("--trace_anchor", action="store_true", help="print the template-match state every N steps")
parser.add_argument("--trace_every", type=int, default=15)
parser.add_argument(
    "--settle",
    type=int,
    default=20,
    help="zero-current warm-up frames after every reset before re-capturing the anchor. "
    "0 disables it, which reproduces the stale-first-frame bug (anchor_valid_frac drops to "
    "0.5 and ncc reads ~half). 3 frames is NOT enough — the render pipeline needs ~20 before "
    "its frames are reliably post-teleport; texture was never the limiting factor.",
)
parser.add_argument("--no_dr", action="store_true", help="disable DR except the pinned current")
parser.add_argument(
    "--sample_current",
    action="store_true",
    help="Leave the ocean current to the DR sampler (speed ~U(0.20,0.45), full-circle heading) "
    "instead of pinning it. This is the distribution-matched mode: use it to compare against an "
    "RL arm that was evaluated under DR. --current/--currents are then ignored and each episode "
    "draws its own current.",
)
parser.add_argument("--no_calib", action="store_true", help="trust the default signs instead of probing")
parser.add_argument(
    "--visodom_f_err",
    type=float,
    default=1.0,
    help="× the estimator's f_px (perception-calibration DR, pinned like the current). "
    "1.0 = nominal. The camera itself stays at cam_hfov_deg; only the metric conversion the "
    "estimator believes in is biased, which is what an uncalibrated intrinsics looks like. "
    "Requires DR to be on (drop --no_dr).",
)
parser.add_argument(
    "--visodom_range_err",
    type=float,
    default=1.0,
    help="× the estimator's initial standoff Z(0). 1.0 = nominal. Same rationale as "
    "--visodom_f_err: an unknown altitude, not a moved wall.",
)
parser.add_argument(
    "--cam_tilt_deg",
    type=float,
    default=0.0,
    help="± camera-mount misalignment in degrees, applied per env as a tilt of the textured "
    "wall (identical image on a planar target, and it cannot corrupt the camera prim). "
    "Baked into the mesh at setup, so it CANNOT be swept inside one Isaac instance.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import numpy as np
import torch

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


class Controller:
    """P / PID / LQR over the same error signal, all outputting normalised actions."""

    def __init__(self, kind: str, env, sign: torch.Tensor, dt: float):
        self.kind = kind
        self.env = env
        self.sign = sign.view(1, 3)   # +1: a positive action increases the estimate
        self.dt = dt
        self.scale = float(env.cfg.cmd_vel_scale)
        self.prev_e = torch.zeros((env.num_envs, 3), device=env.device)
        self.integ = torch.zeros((env.num_envs, 3), device=env.device)
        self.prev = torch.zeros((env.num_envs, 2), device=env.device)
        # Cue dropout state: while blind the flow integrator stops receiving usable frames.
        self.blind = False
        self._held: torch.Tensor | None = None

        # Gain set 1 of 2: position loop bandwidth ~0.5 rad/s, i.e. a ~2 s settling time
        # against a ramp current. Deliberately NOT tuned against the RL curve.
        self.kp = torch.tensor([0.5, 0.5, 0.8], device=env.device)
        self.ki = torch.tensor([0.05, 0.05, 0.02], device=env.device)
        self.kd = torch.tensor([0.15, 0.15, 0.08], device=env.device)

        if kind == "lqr":
            self.K = self._lqr_gain(env, dt)

    @staticmethod
    def _lqr_gain(env, dt: float) -> torch.Tensor:
        """Discrete LQR on [p, v] per axis for a velocity-commanded first-order plant."""
        tau = 0.15  # inner loop (PID -> thrust -> hull) lag, s
        A = np.array([[1.0, dt], [0.0, 1.0 - dt / tau]])
        B = np.array([[0.0], [dt / tau]])
        # Q/R trade: 1 cm position error weighs like ~1/3 of a full command.
        qp, qv, r = (1.0 / 0.05) ** 2, (1.0 / 0.30) ** 2, (1.0 / 0.75) ** 2
        try:
            from scipy.linalg import solve_discrete_are

            P = solve_discrete_are(A, B, np.diag([qp, qv]), np.array([[r]]))
            K = float(np.linalg.solve(np.array([[r]]) + B.T @ P @ B, B.T @ P @ A)[0, 0])
        except Exception:  # scipy missing -> fall back to the P law rather than crash
            print("[baseline] WARN scipy unavailable, LQR falls back to P gains", flush=True)
            K = np.array([[0.5, 0.15]])
            return torch.tensor(K, device=env.device, dtype=torch.float32)
        return torch.tensor([[K, 0.15]], device=env.device, dtype=torch.float32)  # [kp, kd]

    def act(self) -> torch.Tensor:
        d = self.env._visodom_m()
        if self.blind:
            # Cue dropout: flow stops receiving usable frames, so its integrator output
            # freezes at the last good value and its derivative reads zero. Nothing tells the
            # controller this happened — it keeps acting on a stale estimate and coasts off
            # station, which is the whole point of the comparison.
            if self._held is None:
                self._held = d.clone()
            d_used = self._held
            v = torch.zeros_like(d_used)
        else:
            self._held = d.clone()
            d_used = d
            v = (d - self.prev) / self.dt
        self.prev = d.clone()
        yaw = euler_xyz_from_quat(self.env._robot.data.root_quat_w)[2]
        e = torch.stack(
            [d_used[:, 0], d_used[:, 1], _wrap(yaw - self.env._anchor_yaw)], dim=-1
        )
        e_signed = e * self.sign            # positive action now reduces e_signed
        v_signed = v * self.sign[:, :2]

        if self.kind == "lqr":
            kp, kd = self.K[0, 0], self.K[0, 1]
            u = -(kp * e_signed[:, :2] + kd * v_signed)
            u_yaw = -(kp * e_signed[:, 2])
            u = torch.cat([u, u_yaw.unsqueeze(-1)], dim=-1)
        else:
            self.integ = (self.integ + e_signed * self.dt).clamp(-2.0, 2.0)
            de = (e_signed - self.prev_e) / self.dt
            self.prev_e = e_signed.clone()
            u = -(self.kp * e_signed)
            if self.kind == "pid":
                u = u - self.ki * self.integ - self.kd * de
            u = torch.cat([u[:, :2], u[:, 2:3]], dim=-1) if u.shape[-1] == 3 else u

        action = (u / self.scale).clamp(-1.0, 1.0)
        return torch.nan_to_num(action, nan=0.0)


def _pin_current(cfg, speed: float, dir_deg: float) -> None:
    """Make the current deterministic: speed fixed, heading fixed, no vertical.

    Both paths have to be pinned, because they read DIFFERENT fields and only one of
    them is the one you would guess:

      * dr_enable=True  -> _apply_action builds the field from the dr_current_* draws;
      * dr_enable=False -> _apply_action calls _ocean_current_field, which is built from
        ``ocean_current.strength`` and ``params["direction"]``.

    Pinning only the dr_* side leaves ``--no_dr`` runs at the cfg default of 0.3 m/s no
    matter what ``--current`` says — the speed axis of the table would silently collapse
    to one point while still printing three.
    """
    cfg.dr_current_speed_range = (float(speed), float(speed))
    cfg.dr_current_dir_full_circle = False
    cfg.dr_current_dir_range_deg = 0.0
    cfg.dr_current_vert_frac = 0.0
    a = math.radians(float(dir_deg))
    cfg.ocean_current.params["direction"] = [math.cos(a), math.sin(a), 0.0]
    # Non-DR path: uniform field is strength * normalize(direction), so speed == strength.
    cfg.ocean_current.enabled = True
    cfg.ocean_current.type = "uniform"
    cfg.ocean_current.strength = float(speed)
    cfg.ocean_current.level = None
    cfg.ocean_current.params["time_modulation"] = False


def _calibrate(env, amp: float = 0.4, steps: int = 20) -> torch.Tensor:
    """Measure sign(d estimate / d action) per axis.

    Per-env deltas, not a before/after mean: the two means live in the same episode but
    the signal (12–20 steps of motion) is small enough that spawn-to-spawn scatter can
    flip a mean-based decision. The relative spread is reported too, so a weak probe is
    visible instead of silently snapping to a sign.
    """
    dev = env.device
    sign = torch.ones(3, device=dev)
    axes = (("surge", 0), ("sway", 1), ("yaw", 2))
    for axis, idx in axes:
        env.reset()
        if idx == 0:
            before = env._visodom_x.clone()
        elif idx == 1:
            before = env._visodom_m()[:, 1].clone()
        else:
            before = euler_xyz_from_quat(env._robot.data.root_quat_w)[2].clone()
        a = torch.zeros((env.num_envs, 3), device=dev)
        a[:, idx] = amp
        for _ in range(steps):
            env.step(a)
        if idx == 0:
            after = env._visodom_x
        elif idx == 1:
            after = env._visodom_m()[:, 1]
        else:
            after = euler_xyz_from_quat(env._robot.data.root_quat_w)[2]
        delta = (after - before)
        mean = float(delta.mean())
        spread = float(delta.std()) if delta.numel() > 1 else 0.0
        sign[idx] = 1.0 if mean >= 0.0 else -1.0
        print(
            f"[baseline] probe {axis}: d={mean:+.4f} (spread {spread:.4f}, "
            f"amp={amp}, {steps} steps) -> sign={sign[idx]:+.0f}",
            flush=True,
        )
        if abs(mean) <= spread:
            print(
                f"[baseline] WARN probe {axis} is weak (|d| <= spread): the sign may be "
                f"noise. Increase --probe_steps / --probe_amp before trusting a baseline.",
                flush=True,
            )
    return sign


def _apply_step(env, speed: float, dir_deg: float) -> None:
    """Step the current mid-episode.

    With current_ramp_s == 0 this lands instantly — a gust / flywheel reversal that a
    constant-current training run never sees. That is the regime where the classical
    cascade may lose the millimetre-level station keeping it otherwise gets.
    """
    a = math.radians(float(dir_deg))
    env._dr_cur_speed[:] = float(speed)
    env._dr_cur_dir[:, 0] = math.cos(a)
    env._dr_cur_dir[:, 1] = math.sin(a)
    print(f"[baseline] STEP applied: {speed:.2f} m/s @ {dir_deg:.0f} deg", flush=True)


def _settle_camera(env, warm: int = 3) -> None:
    """Let the render pipeline catch up after a teleport, then re-capture the anchor.

    The tiled camera can still be showing the pre-teleport pose on the first frame after
    ``reset()``. The env captures its template on the first textured frame and never
    retakes it, so a stale first frame leaves that env with no usable anchor for the whole
    episode — visible as anchor_valid_frac == 0.5 with four envs, and far more dangerous
    than a crash because it looks like a result: the policy just quietly underperforms.

    The current is zeroed while settling so the warm-up frames cost no displacement; every
    metric here is measured from _spawn_xy, which must stay where the ROV actually spawned.
    """
    n = max(1, int(warm))
    zero = torch.zeros((env.num_envs, 3), device=env.device)
    saved_speed = env._dr_cur_speed.clone()
    saved_vz = env._dr_cur_vz.clone()
    env._dr_cur_speed[:] = 0.0
    env._dr_cur_vz[:] = 0.0
    try:
        for _ in range(n):
            env.step(zero)
        # Discard whatever was captured on the possibly-stale frames and take a fresh one.
        env._anchor_valid[:] = False
        env.step(zero)
    finally:
        env._dr_cur_speed[:] = saved_speed
        env._dr_cur_vz[:] = saved_vz
    if not bool(env._anchor_valid.all()):
        n_bad = int((~env._anchor_valid).sum())
        print(
            f"[baseline] WARN {n_bad}/{env.num_envs} envs still have no anchor after settling "
            f"({n} warm frames) — their ncc will read 0 all episode",
            flush=True,
        )


def _terminal(env) -> dict:
    d = env._visodom_m()
    # Measured from the SPAWN point (the anchor's home), not the env origin: under
    # dr_spawn_xy_range the origin is not where the episode started.
    d_end = torch.norm(env._robot.data.root_pos_w[:, :2] - env._spawn_xy, dim=-1)
    return {
        "disp_end_m": float(d_end.mean()),
        "disp_end_p90": float(d_end.quantile(0.9)),
        "visodom_end_m": float(d.norm(dim=-1).mean()),
        "visodom_err_m": float((d.norm(dim=-1) - d_end).abs().mean()),
        "anchor_disp_m": float(env._anchor_disp_m.mean()),
        "anchor_sim": float(env._anchor_ncc.mean()),
        "ncc_min": float(env._anchor_ncc.min()),
    }


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.dr_enable = not bool(args_cli.no_dr)
    # Calibration DR is pinned rather than drawn, for the same reason as the current: a
    # sensitivity point has to be a point, not a sample.
    cfg.dr_visodom_focal_rel_range = (float(args_cli.visodom_f_err), float(args_cli.visodom_f_err))
    cfg.dr_visodom_range_rel_range = (float(args_cli.visodom_range_err), float(args_cli.visodom_range_err))
    cfg.dr_cam_mount_deg_range = float(args_cli.cam_tilt_deg)
    _calib_active = (
        abs(float(args_cli.visodom_f_err) - 1.0) > 1e-9
        or abs(float(args_cli.visodom_range_err) - 1.0) > 1e-9
        or abs(float(args_cli.cam_tilt_deg)) > 1e-9
    )
    if _calib_active and not cfg.dr_enable:
        print(
            "[baseline] WARN --no_dr was given together with a calibration error, and the DR "
            "sampler is what applies it. The run below is the NOMINAL estimator — drop --no_dr "
            "for the sensitivity point you asked for.",
            flush=True,
        )
    if float(args_cli.ramp_s) >= 0.0:
        cfg.current_ramp_s = float(args_cli.ramp_s)
    if args_cli.sample_current:
        print(
            "[baseline] current left to the DR sampler (distribution-matched mode); "
            "--current/--dir_deg ignored",
            flush=True,
        )
    else:
        _pin_current(cfg, args_cli.current, args_cli.dir_deg)
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)

    dt = float(cfg.decimation) * float(cfg.sim.dt)
    print(
        f"[baseline] {args_cli.num_envs} envs  "
        f"current={'DR-sampled' if args_cli.sample_current else f'{args_cli.current:.2f} m/s @ {args_cli.dir_deg:.0f} deg'}"
        f"  dr={cfg.dr_enable}  dt={dt:.4f}s  steps={args_cli.steps}"
        f"  calib(f_px×{args_cli.visodom_f_err:.3f}, Z0×{args_cli.visodom_range_err:.3f},"
        f" tilt±{args_cli.cam_tilt_deg:.2f}°)",
        flush=True,
    )

    if args_cli.no_calib:
        sign = torch.ones(3, device=env.device)
        print("[baseline] sign calibration SKIPPED (assuming +)", flush=True)
    else:
        sign = _calibrate(env)
        print(f"[baseline] measured signs (surge, sway, yaw) = {sign.tolist()}", flush=True)
        if bool((sign < 0).any()):
            print(
                "[baseline] NOTE a negative sign means the estimate moves opposite to the "
                "command; the gains below are built on the measured sign, so this is fine — "
                "but a controller written without this probe would diverge here.",
                flush=True,
            )

    currents = (
        [float("nan")]  # single pass; the DR sampler owns the current in this mode
        if args_cli.sample_current
        else (
            [float(x) for x in str(args_cli.currents).split(",") if x.strip()]
            if str(args_cli.currents).strip()
            else [float(args_cli.current)]
        )
    )
    kinds = [k.strip().lower() for k in args_cli.controllers.split(",") if k.strip()]

    rows: dict = {}
    for cur in currents:
        # DR is resampled at every reset, so re-pinning the cfg in place sweeps the
        # current without paying a fresh Isaac startup (~80 s) per speed.
        if not args_cli.sample_current:
            _pin_current(env.cfg, cur, args_cli.dir_deg)
        for kind in kinds:
            env.reset()
            if int(args_cli.settle) > 0:
                _settle_camera(env, int(args_cli.settle))
            ctrl = Controller(kind, env, sign, dt)
            # Use the environment's OWN episode diagnostics (extras["log"], written in
            # _reset_idx before the parent zeroes visodom/anchor/spawn). Any read taken
            # after a reset describes a freshly spawned ROV instead of the outcome.
            eps, last_log, peaks = [], None, []
            peak = torch.zeros(env.num_envs, device=env.device)
            hz = 1.0 / dt
            step_idx = int(float(args_cli.step_at) * hz) if float(args_cli.step_at) > 0.0 else -1
            for i in range(max(1, int(args_cli.steps))):
                if i == step_idx:
                    _apply_step(
                        env,
                        float(args_cli.step_to),
                        float(args_cli.dir_deg) + float(args_cli.flip_deg),
                    )
                if float(args_cli.blind_at) > 0.0:
                    # Keyed off episode time, not the loop counter, so the window repeats in
                    # every episode instead of only hitting the first one.
                    t_now = float(env.episode_length_buf.float().mean()) * dt
                    on = (
                        float(args_cli.blind_at)
                        <= t_now
                        < float(args_cli.blind_at) + float(args_cli.blind_len)
                    )
                    if on != ctrl.blind:
                        ctrl.blind = on
                        print(
                            f"[baseline] BLIND {'ON' if on else 'OFF'} at t={t_now:.2f}s",
                            flush=True,
                        )
                env.step(ctrl.act())
                peak = torch.maximum(
                    peak,
                    torch.norm(env._robot.data.root_pos_w[:, :2] - env._spawn_xy, dim=-1),
                )
                if args_cli.trace_anchor and i % max(1, int(args_cli.trace_every)) == 0:
                    ncc = env._anchor_ncc
                    off = env._anchor_off_px
                    disp = torch.norm(env._robot.data.root_pos_w[:, :2] - env._spawn_xy, dim=-1)
                    yaw_err = (
                        euler_xyz_from_quat(env._robot.data.root_quat_w)[2] - env._anchor_yaw
                    )
                    print(
                        f"[trace] {kind} i={i:4d} valid={int(env._anchor_valid.sum())}/{env.num_envs} "
                        f"ncc={float(ncc.mean()):.3f} "
                        f"off=({float(off[:, 0].mean()):+.1f},{float(off[:, 1].mean()):+.1f})px "
                        f"disp={float(disp.mean()):.4f} "
                        f"tex={float(env._texture_count.float().mean()):.0f} "
                        f"yaw_err={float(yaw_err.mean()):+.4f}",
                        flush=True,
                    )
                lg = env.extras.get("log")
                if lg is not None and lg is not last_log:
                    last_log = lg
                    eps.append(dict(lg))
                    # Per-episode peak: snapshot at each reset event. The envs run in
                    # lockstep, so clearing here is exact; if they ever drift apart this
                    # becomes an approximation.
                    peaks.append(float(peak.mean()))
                    peak = torch.zeros(env.num_envs, device=env.device)
                if len(eps) >= max(1, int(args_cli.episodes)):
                    break
            m = (
                {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}
                if eps
                else _terminal(env)
            )
            m["n_episodes"] = len(eps)
            m["peak_disp_m"] = float(peak.mean())
            m["peak_ep_mean_m"] = float(np.mean(peaks)) if peaks else float("nan")
            rows[(cur, kind)] = m
            print(
                f"[baseline] cur={'DR' if args_cli.sample_current else f'{cur:.2f}'} {kind.upper():4s} "
                f"disp_spawn={m.get('disp_spawn_m', float('nan')):.4f} "
                f"peak={m.get('peak_disp_m', float('nan')):.4f} "
                f"disp_origin={m.get('disp_end_m', float('nan')):.4f} "
                f"visodom={m.get('visodom_end_m', float('nan')):.4f} "
                f"err={m.get('visodom_err_m', float('nan')):.4f} "
                f"ncc={m.get('anchor_sim', float('nan')):.3f} "
                f"af={m.get('anchor_valid_frac', float('nan')):.3f} "
                f"eps={len(eps)} "
                f"tex={m.get('term_tex_lost', float('nan')):.2f} "
                f"dep={m.get('term_depth', float('nan')):.2f} "
                f"att={m.get('term_att', float('nan')):.2f} "
                f"tmo={m.get('term_timeout', float('nan')):.2f}",
                flush=True,
            )

    if args_cli.sample_current:
        print(
            "\n=== disp_spawn_m: same env, same DR *distribution* (current sampled per episode), "
            "same estimator ===",
            flush=True,
        )
    else:
        print(
            "\n=== disp_spawn_m: same env, same other-DR knobs, same estimator — but the current "
            "is PINNED, so this is a fixed-current sweep, not a DR-distribution sample ===",
            flush=True,
        )
    print("  current " + " ".join(f"{k.upper():>10s}" for k in kinds), flush=True)
    for cur in currents:
        cells = []
        for kind in kinds:
            v = (rows.get((cur, kind)) or {}).get("disp_spawn_m", float("nan"))
            cells.append(f"{v:10.4f}")
        lbl = "    DR" if args_cli.sample_current else f"{cur:7.2f}"
        print(f"  {lbl} " + " ".join(cells), flush=True)


if __name__ == "__main__":
    main()
    # Isaac hangs on shutdown once tiled cameras are alive: results print, then the process
    # sits there holding the GPU, so a chained run never reaches its next scenario. Closing
    # the app is not enough — exit hard, this is a measurement harness, not a service.
    app_launcher.app.close()
    os._exit(0)
