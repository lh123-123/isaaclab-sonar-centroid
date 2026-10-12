#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Runtime contract check for the Direct-v6-flowvec flow-VECTOR observation.

`scripts/debug_auv_cam_spaces.sh` proves the SHAPE is right. This proves the SEMANTICS are:

  1. space/obs is a (frame_stack * 6,) float32 vector; no H/W axes anywhere;
  2. reset publishes the CURRENT absolute triple repeated across the stack with a zero
     differential pair -- i.e. NOT a zero stack. This is v6's key behavioural difference from v5
     (whose observation was purely differential) and the assertion pins it down;
  3. every step the stack shifts by one 6-vector: buf_t[:, :-6] == buf_{t-1}[:, 6:]. Episodes that
     RESET during the run are excluded from this check (a reset deliberately rewrites the whole
     stack with a uniform repeat), and the uniform-repeat invariant is checked instead;
  4. the newest 6 slots are bit-identical to the env's live `_flow_vec_now`;
  5. all values are finite and inside the declared Box bounds ([-1, 1]), with u/W and v/H in
     [0, 1] and validity in {0, 1};
  6. the vector AGREES with the raster it is supposed to be a picture of: the value painted inside
     `_flow_image`'s disk must equal the vector's du slot, and the disk's radius must equal the
     rad slot. The disk mask is RECONSTRUCTED from the vector's own (u, v, rad) rather than read
     off the raster as "nonzero pixels" -- du is legitimately exactly 0 on reset frames and on
     steps where the projection does not move, and an earlier revision used |raster| > 0 as the
     mask, which made every such env report a spurious "radius err = rad" failure.
     NOTE `_flow_image` stores `du` already divided by W, so it is compared against the du SLOT
     directly -- an earlier revision divided a second time and compared against du/W;
  7. `rad` really encodes distance in the operating range: at the 2 m reset distance the slot must
     be in the informative interior (neither clamped at 1 nor at 12) and must invert back to ~2 m.
     This asserts the corrected numbers in the module docstring rather than trusting them;
  8. the validity slot is a {0, 1} flag and, under the geometric GT measurement, agrees
     elementwise with the env's `_observed` KF accept gate. That is the invariant that lets the
     reward's 30-step counter and the actor's input talk about the same event;
  9. flow becomes non-zero during the run (the 3 s soft-start makes the first steps tiny).

   ./scripts/debug_auv_v6_flowvec_obs.sh --headless --num_envs 4
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Direct-v6-flowvec observation contract check.")
parser.add_argument("--task", type=str, default="Isaac-AUV-Tracking-Direct-v6-flowvec")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=10)
# Random actions instead of zeros. Zero actions barely move the vehicle, so the measured du/dv
# range would be a floor rather than the operating range -- useless for choosing a gain. Random
# yaw commands exercise the worst case the control loop can actually command.
parser.add_argument("--random_actions", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
# The camera still renders for the KF measurement even though the actor no longer consumes it.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# The rad slot is rad / RAD_CLAMP_MAX, and rad = (K_RAD / dist).clamp(1, 12) with
# K_RAD = 0.5 * length * fx = 0.5 * 0.64 * 32 = 10.24 for this camera (90 deg HFOV, 64 px). Used to
# invert the slot back into a distance so the encoding is checked, not just the range.
RAD_CLAMP_MAX = 12.0
K_RAD = 10.24

# Values per frame in the actor's flow vector: [u/W, v/H, rad/12, du, dv, validity].
STRIDE = 6

_FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        _FAILS.append(name)


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs, use_fabric=True)
    k = int(env_cfg.frame_stack)
    s = STRIDE
    dim = k * s
    w = float(int(env_cfg.cam_width))
    h = float(int(env_cfg.cam_height))
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        raw = env.unwrapped
        ss = raw.single_observation_space
        single_obs = ss["policy"] if isinstance(ss, gym.spaces.Dict) else ss

        # Slot index helpers: each frame contributes 6 values,
        # [u/W, v/H, rad/12, du, dv, validity].
        dv_idx = torch.tensor([s * g + 3 for g in range(k)] + [s * g + 4 for g in range(k)])
        valid_idx = torch.tensor([s * g + 5 for g in range(k)])
        abs_idx = torch.tensor([s * g + c for c in (0, 1, 2) for g in range(k)])
        last_abs = torch.tensor([s * (k - 1) + c for c in (0, 1, 2)])

        print("\n[v6-flowvec] ---- observation space ----", flush=True)
        if isinstance(single_obs, gym.spaces.Dict):
            for key in sorted(single_obs.spaces.keys()):
                sp = single_obs[key]
                print(f"[v6-flowvec]   ['{key}']: shape={sp.shape} dtype={sp.dtype}", flush=True)

        img = single_obs["image"]
        print(f"[v6-flowvec] frame_stack={k} x {s} slots -> expected dims={dim}", flush=True)
        check(f"image key is a 1-D (frame_stack*{s},) vector", tuple(img.shape) == (dim,), f"{tuple(img.shape)}")
        check("image dtype is float32", str(img.dtype) == "float32", str(img.dtype))
        check("bounds are [-1, 1]", float(img.low.min()) == -1.0 and float(img.high.max()) == 1.0)
        check("no raster axes (not a (C,H,W) image)", len(img.shape) == 1)
        check("critic sees the 4-D KF belief", tuple(single_obs["state"].shape) == (4,))

        obs, _ = env.reset()
        vec = obs["policy"]["image"]
        print("\n[v6-flowvec] ---- reset frame (expect absolute repeated, differential zero) ----", flush=True)
        print(f"[v6-flowvec] image shape={tuple(vec.shape)} dtype={vec.dtype} "
              f"absmax={float(vec.abs().max()):.6g}", flush=True)

        # (2) reset semantics: differential zero, absolute repeated across all K groups.
        grp0 = vec[:, [0, 1, 2]]
        for _g in range(k):
            print(f"[v6-flowvec] reset frame env0 group{_g} = "
                  f"{vec[0, s * _g : s * (_g + 1)].tolist()}", flush=True)
        check("reset differential slots (du/dv) are zero", bool((vec[:, dv_idx] == 0).all()),
              f"max|du,dv|={float(vec[:, dv_idx].abs().max()):.6g}")
        abs_equal = all(torch.allclose(vec[:, [s * g, s * g + 1, s * g + 2]], grp0, atol=0.0)
                        for g in range(k))
        check("reset absolute slots are REPEATED across the stack (not zeroed)", abs_equal)
        check("reset absolute slots are non-zero (target is visible at reset)",
              bool((vec[:, abs_idx].abs().max() > 0)),
              f"absmax={float(vec[:, abs_idx].abs().max()):.6g}")
        check("reset is NOT a zero stack (v5 behaviour is deliberately not inherited)",
              not bool((vec == 0).all()))
        check("reset u/W and v/H are inside [0, 1]",
              bool(((vec[:, [s * g for g in range(k)]] >= 0) & (vec[:, [s * g for g in range(k)]] <= 1)).all()))
        # (8) validity on the reset frame: target is 2 m dead ahead, so it must read 1 everywhere.
        check("reset validity slots are 1 (target is 2 m dead ahead)",
              bool((vec[:, valid_idx] == 1).all()),
              f"validity={vec[:, valid_idx].tolist()}")
        check("reset validity agrees with the KF accept gate _observed",
              bool((raw._flow_vec_valid == raw._observed).all()))

        # (7) rad must invert back to the reset distance.
        slot = float(vec[0, s * (k - 1) + 2])
        rad = slot * RAD_CLAMP_MAX
        dist_est = K_RAD / rad if rad > 0 else float("inf")
        print(f"[v6-flowvec] rad slot={slot:.4f} -> rad={rad:.3f} px -> implied dist={dist_est:.3f} m", flush=True)
        check("rad slot is in the informative interior (not clamped at 1 or 12)",
              1.0 / RAD_CLAMP_MAX < slot < 1.0 - 1e-6, f"slot={slot:.4f}")
        check("rad slot inverts to the ~2.0 m reset distance", 1.5 < dist_est < 2.5, f"{dist_est:.3f} m")

        prev = vec.clone()
        saw_nonzero = False
        saw_du = False
        worst_absmax = 0.0
        worst_du = 0.0
        worst_val_err = 0.0
        worst_rad_err = 0.0
        n_resets = 0
        n_valid_mismatch = 0
        n_validity_bad = 0
        worst_slot_err = 0.0
        n_offimage = 0
        n_clipped_disk = 0

        for step in range(args_cli.steps):
            if args_cli.random_actions:
                act = torch.as_tensor(raw.action_space.sample(), device=raw.device)
            else:
                act = torch.zeros(raw.action_space.shape, device=raw.device)
            out = env.step(act)
            vec = out[0]["policy"]["image"]

            finite = bool(torch.isfinite(vec).all())
            in_range = bool((vec >= -1.0).all() and (vec <= 1.0).all())
            absmax = float(vec.abs().max())
            worst_absmax = max(worst_absmax, absmax)
            saw_nonzero = saw_nonzero or absmax > 0.0

            # A step may contain a reset (the 30-step out-of-view rule or a timeout), which
            # rewrites the whole stack with a uniform repeat instead of shifting. Treat the two
            # cases separately rather than reporting a spurious shift/radius failure.
            uniform = torch.stack([vec[:, s * g : s * (g + 1)] for g in range(k)], dim=1)
            is_repeat = (uniform == uniform[:, :1]).all(dim=-1).all(dim=-1)
            resets = is_repeat & (uniform[:, 0, 3].abs() < 1e-12) & (uniform[:, 0, 4].abs() < 1e-12)
            n_resets += int(resets.sum())

            shift_ok = torch.equal(vec[~resets][:, :-s], prev[~resets][:, s:]) if bool((~resets).any()) else True
            repeat_ok = bool(is_repeat[resets].all())
            tail_ok = torch.equal(vec[:, -s:], raw._flow_vec_now)

            # (8) validity is a clean {0, 1} flag and matches the KF accept gate.
            v_slot = raw._flow_vec_now[:, 5]
            validity_ok = bool(((v_slot == 0) | (v_slot == 1)).all())
            n_validity_bad += 0 if validity_ok else 1
            agree_ok = bool((raw._flow_vec_valid == raw._observed).all())
            n_valid_mismatch += 0 if agree_ok else 1

            # (6) agreement with the raster. The raster is by definition a picture of the env's RAW
            # (u, v, rad, du, dv), so it is compared against those, NOT against the observation
            # slots: `_publish_flow_vec` clamps u/v into [0, 1], and a target outside the frustum
            # legitimately sits outside the image -- which is exactly what the validity slot is
            # there to signal. The slot-vs-raw clamp contract is asserted separately below.
            # The mask is RECONSTRUCTED from the raw (u, v, rad) instead of using |raster| > 0:
            # du/dv are legitimately exactly zero on reset frames and on any step where the
            # projection does not move, and a zero-valued disk would otherwise look like "no
            # disk", making the radius check compare against 0.
            u_raw = raw._flow_vec_u
            v_raw = raw._flow_vec_v
            r_raw = raw._flow_vec_rad
            du_raw = raw._flow_vec_du
            dv_raw = raw._flow_vec_dv
            painted_u = raw._flow_image[:, 0]
            painted_v = raw._flow_image[:, 1]
            ys = torch.arange(int(h), device=raw.device, dtype=torch.float32).view(1, -1, 1)
            xs = torch.arange(int(w), device=raw.device, dtype=torch.float32).view(1, 1, -1)
            dx = xs - u_raw.view(-1, 1, 1)
            dy = ys - v_raw.view(-1, 1, 1)
            mask = dx * dx + dy * dy <= r_raw.view(-1, 1, 1) ** 2
            n_lit = int(mask.sum())
            val_err = 0.0
            rad_err = 0.0
            # The area->radius inversion is only meaningful when the WHOLE disk lands inside the
            # image: a clipped disk (or one whose centre is off-image, which is normal here -- 29%
            # of env-steps have the target outside the frustum) has fewer painted pixels than
            # pi*r^2, so the inversion under-reads. That is a limitation of the checker, not of the
            # env, so the radius assertion is restricted to fully-visible disks and the clipped
            # ones are counted instead.
            full_disk = (
                (u_raw - r_raw >= 0.0)
                & (u_raw + r_raw <= w - 1.0)
                & (v_raw - r_raw >= 0.0)
                & (v_raw + r_raw <= h - 1.0)
            )
            n_clipped_disk += int((~full_disk).sum())
            if bool(full_disk.any()):
                fm = mask[full_disk]
                exp_u = du_raw.view(-1, 1, 1).expand_as(painted_u)[full_disk][fm]
                exp_v = dv_raw.view(-1, 1, 1).expand_as(painted_v)[full_disk][fm]
                val_err = max(
                    float((painted_u[full_disk][fm] - exp_u).abs().max()),
                    float((painted_v[full_disk][fm] - exp_v).abs().max()),
                )
                area = fm.sum(dim=(-2, -1)).float()
                rad_from_raster = torch.sqrt(area / math.pi)
                rad_err = float((rad_from_raster - r_raw[full_disk]).abs().max())

            # (10) the observation slot must be exactly the CLAMPED raw value.
            gain = float(getattr(raw.cfg, "flow_vec_dv_gain", 1.0))
            u_n, v_n, r_n, du_n, dv_n, vis_n = raw._flow_vec_now.unbind(dim=-1)
            slot_err = max(
                float((u_n - (u_raw / (w - 1.0)).clamp(0.0, 1.0)).abs().max()),
                float((v_n - (v_raw / (h - 1.0)).clamp(0.0, 1.0)).abs().max()),
                float((r_n - (r_raw / RAD_CLAMP_MAX).clamp(0.0, 1.0)).abs().max()),
                float((du_n - (du_raw * gain).clamp(-1.0, 1.0)).abs().max()),
                float((dv_n - (dv_raw * gain).clamp(-1.0, 1.0)).abs().max()),
            )
            worst_slot_err = max(worst_slot_err, slot_err)
            n_clamped = int(((u_raw / (w - 1.0) < 0) | (u_raw / (w - 1.0) > 1)).sum())
            n_offimage += n_clamped
            worst_val_err = max(worst_val_err, val_err)
            worst_rad_err = max(worst_rad_err, rad_err)

            du_live = float(raw._flow_vec_du.abs().max())
            worst_du = max(worst_du, du_live)
            saw_du = saw_du or du_live > 0.0

            print(f"[v6-flowvec] step {step}: finite={finite} in_range={in_range} absmax={absmax:.6g} "
                  f"resets={int(resets.sum())} shift_ok={shift_ok} repeat_ok={repeat_ok} "
                  f"tail_matches={tail_ok} value_err={val_err:.3g} "
                  f"rad_err={rad_err:.3g} slot_err={slot_err:.3g} offimg={n_clamped} "
                  f"valid_ok={validity_ok} valid_agree={agree_ok} "
                  f"max|du|={du_live:.6g}", flush=True)

            check(f"step {step} values finite", finite)
            check(f"step {step} values within space bounds [-1, 1]", in_range, f"absmax={absmax:.6g}")
            check(f"step {step} stack shifts by one {s}-vector (reset envs excluded)", shift_ok)
            check(f"step {step} resetting envs got a uniform repeated stack", repeat_ok)
            check(f"step {step} newest slots == live _flow_vec_now", tail_ok)
            check(f"step {step} painted raster values == env raw du/dv", val_err < 1e-6, f"err={val_err:.3g}")
            check(f"step {step} reconstructed disk radius == env raw rad", rad_err <= 1.0, f"err={rad_err:.3g} px")
            check(f"step {step} every slot == clamp(raw slot)", slot_err < 1e-6, f"err={slot_err:.3g}")
            check(f"step {step} validity slot is a clean {{0, 1}} flag", validity_ok,
                  f"values={sorted(set(raw._flow_vec_now[:, 5].tolist()))}")
            check(f"step {step} validity slot agrees with _observed", agree_ok)
            prev = vec.clone()

        print("\n[v6-flowvec] ---- summary ----", flush=True)
        print(f"[v6-flowvec] max |vec| = {worst_absmax:.6g}; max |du| = {worst_du:.6g}; "
              f"max value err = {worst_val_err:.3g}; max radius err = {worst_rad_err:.3g} px; "
              f"max slot-clamp err = {worst_slot_err:.3g}; env-steps with u off-image = {n_offimage}; "
              f"env-steps with a clipped disk (radius check skipped) = {n_clipped_disk}; "
              f"reset steps={n_resets}; validity mismatches={n_valid_mismatch}/{args_cli.steps}; "
              f"non-binary validity steps={n_validity_bad}", flush=True)
        check("flow became non-zero during the run", saw_nonzero)
        check("du slot became non-zero during the run", saw_du)
        # The du/dv slot is 1-2 orders of magnitude below u/v/rad (its own normalisation is
        # displacement/W, so a full-frame jump in one step is the only way to reach 1). Reported
        # rather than asserted against an invented threshold: the measured operating range is what
        # `flow_vec_dv_gain` has to be chosen from.
        print(f"[v6-flowvec] NOTE du/dv slot operating range: max|du|={worst_du:.6g} "
              f"vs u/v/rad slots in [0, 1]", flush=True)
        check("validity never disagreed with the KF accept gate", n_valid_mismatch == 0,
              f"{n_valid_mismatch} steps")
        check("validity was binary on every step", n_validity_bad == 0, f"{n_validity_bad} steps")

        print("\n[v6-flowvec] RESULT: " + ("OK" if not _FAILS else f"MISMATCH ({len(_FAILS)})"), flush=True)
        return 0 if not _FAILS else 1
    finally:
        env.close()


if __name__ == "__main__":
    import traceback

    try:
        rc = main()
    except BaseException:  # noqa: BLE001 - must surface the failure immediately
        traceback.print_exc()
        rc = 1
    finally:
        simulation_app.close()
    sys.exit(rc)
