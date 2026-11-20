#!/usr/bin/env python3
"""How much of visodom's accuracy is actually the ruler underneath it?

visodom turns pixels into metres through numbers that on a real hull get measured once, by
hand: the pixel focal length f_px, the initial standoff Z(0), and the mount angles. In sim
all three are exact, so ``visodom_err`` has been reporting a property of the integrator
while silently assuming the conversion is perfect. This sweeps the conversion and reports
what is left.

The estimator's BELIEF is biased; the camera stays at ``cam_hfov_deg`` and the wall stays at
``plane_distance``. That is the real failure mode — the world is fine, your calibration is
not. Nothing here touches the camera prim.

Every point is pinned, not drawn, and every other hardware knob is pinned too, so a row is a
sensitivity point rather than a sample of a distribution. Spawn DR is off for the same
reason: it changes the TRUE standoff, which would confound the swept belief error.

    ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_visodom_dr.py --headless --num_envs 4 \
        --f_errs 0.90,0.95,1.00,1.05,1.10 --z_errs 1.00 --steps 450 --episodes 2

    # the mount axis cannot be swept inside one process (it is baked into the wall mesh):
    ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_visodom_dr.py --headless --num_envs 4 \
        --f_errs 1.0 --z_errs 1.0 --tilt_deg 2.0

Reading it: lateral displacement scales as Z/f, axial as Z. So a point is expected to show
    lateral error  ~= |z_err / f_err - 1| x lateral travel
    axial   error  ~= |z_err        - 1| x axial travel
``lat_scale`` below is that prediction. If measured error tracks it, the estimator is a pure
scale ladder and a single calibration constant fixes it. If it does not, the error is
structural (the fronto-parallel assumption / the (a, k) fit) and no constant will help.
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="visodom sensitivity to its own calibration.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--f_errs", type=str, default="0.90,0.95,1.00,1.05,1.10", help="× f_px grid")
parser.add_argument("--z_errs", type=str, default="1.00", help="× Z(0) grid")
parser.add_argument("--tilt_deg", type=float, default=0.0, help="± camera mount, baked into the wall")
parser.add_argument("--current", type=float, default=0.30, help="m/s, pinned")
parser.add_argument("--dir_deg", type=float, default=17.0, help="current heading, pinned")
parser.add_argument("--steps", type=int, default=450, help="policy steps per point")
parser.add_argument("--episodes", type=int, default=2, help="episode logs to average per point")
parser.add_argument("--drive", type=float, default=0.35, help="dither amplitude, normalised action")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)

import numpy as np
import torch

from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)


def _pin_current(cfg, speed: float, dir_deg: float) -> None:
    """Fixed current: a sweep point, not a sample.

    Pins BOTH fields the env can read — the dr_current_* draws and the
    ocean_current strength/direction — so the number on the command line is the number
    actually applied whichever path _apply_action takes.
    """
    cfg.dr_current_speed_range = (float(speed), float(speed))
    cfg.dr_current_dir_full_circle = False
    cfg.dr_current_dir_range_deg = 0.0
    cfg.dr_current_vert_frac = 0.0
    a = math.radians(float(dir_deg))
    cfg.ocean_current.params["direction"] = [math.cos(a), math.sin(a), 0.0]
    cfg.ocean_current.enabled = True
    cfg.ocean_current.type = "uniform"
    cfg.ocean_current.strength = float(speed)
    cfg.ocean_current.params["time_modulation"] = False


def _pin_hardware(cfg) -> None:
    """Hold the non-swept knobs still, so the only thing changing is the ruler.

    Camera noise is kept at its mid value rather than zeroed: it is part of visodom's real
    error budget (it corrupts the flow the integrator eats), and removing it would make the
    nominal point look better than any build can be.
    """
    cfg.dr_thrust_gain_range = (1.0, 1.0)
    cfg.dr_drag_gain_range = (0.0, 0.0)
    cfg.dr_cam_brightness_range = (1.0, 1.0)
    cfg.dr_cam_noise_std_range = (0.007, 0.007)
    cfg.dr_spawn_xy_range = 0.0
    cfg.dr_spawn_yaw_range_deg = 0.0
    cfg.dr_spawn_depth_range = 0.0


def _drive(step: int, dt: float, amp: float) -> torch.Tensor:
    """Slow dither on surge and sway: both axes travel, yaw is left alone.

    Yaw is deliberately not commanded: heading motion makes the de-rotation term do the work
    and would fold a second error source into the same number.
    """
    t = float(step) * dt
    return torch.tensor(
        [
            amp * math.sin(2.0 * math.pi * t / 9.0),
            amp * math.sin(2.0 * math.pi * t / 6.5),
            0.0,
        ],
        dtype=torch.float32,
    )


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    # DR stays ON: the sampler is what applies a calibration error. All it will draw here is
    # the pinned values plus the swept ones.
    cfg.dr_enable = True
    cfg.dr_cam_mount_deg_range = float(args_cli.tilt_deg)
    cfg.dr_visodom_focal_rel_range = (1.0, 1.0)
    cfg.dr_visodom_range_rel_range = (1.0, 1.0)
    _pin_current(cfg, args_cli.current, args_cli.dir_deg)
    _pin_hardware(cfg)
    env = AUVTrackingV7FlowHoverGsrovEnv(cfg)

    dt = float(cfg.decimation) * float(cfg.sim.dt)
    ids = torch.arange(env.num_envs, device=env.device)
    f_errs = [float(v) for v in str(args_cli.f_errs).split(",") if v.strip()]
    z_errs = [float(v) for v in str(args_cli.z_errs).split(",") if v.strip()]
    n_eps = max(1, int(args_cli.episodes))
    steps = max(1, int(args_cli.steps))

    print(
        f"[visodom-dr] {env.num_envs} envs  dt={dt:.4f}s  current={args_cli.current:.2f} m/s "
        f"@ {args_cli.dir_deg:.0f} deg  tilt=±{args_cli.tilt_deg:.2f} deg  "
        f"{len(f_errs)}x{len(z_errs)} points",
        flush=True,
    )
    print(
        f"[visodom-dr] {'f_err':>6} {'z_err':>6} {'lat_scale':>9} {'visodom_err':>12} "
        f"{'disp_spawn':>11} {'visodom_end':>12} {'n':>2}",
        flush=True,
    )

    rows: list[tuple[float, float, dict]] = []
    for f_err in f_errs:
        for z_err in z_errs:
            env.cfg.dr_visodom_focal_rel_range = (f_err, f_err)
            env.cfg.dr_visodom_range_rel_range = (z_err, z_err)
            # Redraw the per-env hardware terms from the new ranges, then reset so the
            # estimator starts from the standoff it now believes in.
            env._sample_domain_randomization(ids, per_episode=False)
            env.reset()
            eps: list[dict] = []
            last = env.extras.get("log")
            act = _drive(0, dt, float(args_cli.drive)).unsqueeze(0).repeat(env.num_envs, 1).to(env.device)
            for i in range(steps):
                act[:] = _drive(i, dt, float(args_cli.drive)).to(env.device)
                env.step(act)
                lg = env.extras.get("log")
                if lg is not None and lg is not last:
                    last = lg
                    eps.append(dict(lg))
                if len(eps) >= n_eps:
                    break
            if not eps:
                print(
                    f"[visodom-dr] {f_err:6.3f} {z_err:6.3f}   no episode log after {steps} steps "
                    f"— raise --steps (225 steps = one 15 s episode)",
                    flush=True,
                )
                continue
            m = {k: float(np.mean([e[k] for e in eps])) for k in eps[0] if isinstance(eps[0][k], float)}
            lat_scale = z_err / f_err if f_err else float("nan")
            print(
                f"[visodom-dr] {f_err:6.3f} {z_err:6.3f} {lat_scale:9.3f} "
                f"{m.get('visodom_err_m', float('nan')) * 1000:9.2f} mm "
                f"{m.get('disp_spawn_m', float('nan')) * 1000:8.1f} mm "
                f"{m.get('visodom_end_m', float('nan')) * 1000:9.1f} mm {len(eps):2d}",
                flush=True,
            )
            rows.append((f_err, z_err, m))

    if len(rows) >= 2:
        xs = np.array([z / f for f, z, _ in rows], dtype=np.float64)
        ys = np.array([m.get("visodom_err_m", np.nan) for _, _, m in rows], dtype=np.float64)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() >= 2 and float(np.std(xs[ok])) > 1e-9:
            slope = float(np.polyfit(xs[ok] - 1.0, ys[ok], 1)[0])
            print(
                f"\n[visodom-dr] d(visodom_err)/d(lat_scale-1) = {slope * 1000:.1f} mm per unit\n"
                "[visodom-dr] Compare that slope with the mean travel reported above: if it is "
                "close to disp_spawn, the estimator is a pure scale ladder and one calibration "
                "constant fixes it. If it is much larger, the error is structural and no scalar "
                "correction will absorb it.",
                flush=True,
            )

    print(
        "\n=== visodom_err under a mis-calibrated ruler ===\n"
        "  Report nominal AND the worst point, never nominal alone: the gap between them is the\n"
        "  calibration budget a real build has to meet before the mm-level number means anything.",
        flush=True,
    )


if __name__ == "__main__":
    main()
    # Isaac does not exit once tiled cameras are alive: results print, then it holds the GPU
    # and any chained run starves. Exit hard — this is a measurement harness, not a service.
    app_launcher.app.close()
    import os

    os._exit(0)
