#!/usr/bin/env python3
"""DR coverage audit for the v7 flow-hover env.

Three checks, each one earned by a bug that already happened:

  A. DEAD KNOB   -- every ``dr_*`` cfg field must be READ somewhere in the env source.
                    ``dr_texture_choice`` sat declared-but-unread for weeks: the docs
                    claimed texture randomization, the code never looked at the knob.
  B. HORIZON     -- a knob drawn "per run" must be identical across episode resets, and
                    one drawn "per episode" must actually be redrawn. A per-run draw that
                    gets re-sampled is a hidden per-episode bias the policy can learn to
                    cancel, which is the failure the whole two-horizon design exists to
                    avoid.
  C. EFFECT      -- the sensor/water knobs must move the quantity they claim to move:
                    heading belief off ground truth, water turbidity lowering contrast,
                    drag anisotropy not identity, camera mount not flat.

Run:
    ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_dr_coverage.py --num_envs 2

Exit code 0 = all pass. ANY FAIL exits 1, so it can gate a training launch.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Audit DR wiring and horizons in the v7 env.")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--resets", type=int, default=3)
parser.add_argument("--turbidity_min_drop", type=float, default=0.10)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.direct.isaac_auv_tracking.agents.cam_encoder import MidCNNExtractor  # noqa: F401,E402
from isaaclab_tasks.direct.isaac_auv_tracking.agents.custom_sac_policy import CustomSACPolicy  # noqa: F401,E402
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (  # noqa: E402
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

ENV_SRC = os.path.join(
    os.path.dirname(os.path.abspath(isaaclab_tasks.__file__)),
    "direct/isaac_auv_tracking/auv_tracking_v7_flow_hover_gsrov_env.py",
)

# knob -> (state attribute, horizon). The table is the contract: it is what B checks
# against, so adding a knob without deciding its horizon fails the first assertion.
HORIZON = {
    "dr_current_speed_range": ("_dr_cur_speed", "episode"),
    "dr_current_dir_full_circle": ("_dr_cur_dir", "episode"),
    "dr_current_dir_range_deg": ("_dr_cur_dir", "episode"),
    "dr_current_vert_frac": ("_dr_cur_vz", "episode"),
    "dr_spawn_xy_range": ("_spawn_xy", "episode"),
    "dr_spawn_yaw_range_deg": ("_anchor_yaw", "episode"),
    "dr_spawn_depth_range": ("_dr_spawn_depth", "episode"),
    "dr_thrust_gain_range": ("_dr_thrust_gain", "run"),
    "dr_drag_gain_range": ("_dr_drag_gain", "run"),
    "dr_drag_aniso_range": ("_drag_aniso", "run"),
    "dr_cam_brightness_range": ("_dr_cam_bright", "run"),
    "dr_cam_noise_std_range": ("_dr_cam_noise", "run"),
    "dr_visodom_focal_rel_range": ("_visodom_f_err", "run"),
    "dr_visodom_range_rel_range": ("_visodom_range_err", "run"),
    "dr_visodom_ep_scale_rel": ("_visodom_ep_scale", "episode"),
    "dr_visodom_bias_m": ("_visodom_bias", "episode"),
    "dr_cam_mount_deg_range": ("_cam_tilt_deg", "run"),
    "dr_imu_yaw_bias_deg": ("_imu_yaw_bias", "run"),
    "dr_imu_yaw_noise_deg": ("_imu_yaw_noise", "run"),
    "dr_turbidity_range": ("_turbidity", "episode"),
    # Payload (tier 1): the pod changes between dives, not between control ticks.
    "dr_payload_mass_frac_range": ("_payload_mass", "episode"),
    "dr_payload_offset_xy_range": ("_payload_off", "episode"),
    "dr_payload_offset_z_range": ("_payload_off", "episode"),
}

# Master switches, not draws: they have no state to sample, so the horizon check does
# not apply. "Switch" here means a bool that turns a block of draws on or off.
SWITCHES = {"dr_enable", "dr_texture_choice"}

# Knobs whose PRODUCTION magnitude is a deliberate no-op (1.0 / 0.0). Wiring them can
# only be audited by forcing a non-zero magnitude for the duration of this script; the
# audit prints that it did so, because a silent override would misrepresent the config.
CANARY = {
    "dr_visodom_focal_rel_range": (0.90, 1.10),
    "dr_visodom_range_rel_range": (0.90, 1.10),
    "dr_visodom_ep_scale_rel": 0.05,
    "dr_visodom_bias_m": 0.02,
    # Payload tier-1 ships disabled (0.0, 0.0): the per-episode draw can only be audited by
    # forcing a non-zero range, and the forced values must stay inside the planned envelope.
    "dr_payload_mass_frac_range": (0.05, 0.30),
    "dr_payload_offset_xy_range": (0.0, 0.08),
    "dr_payload_offset_z_range": (-0.10, 0.15),
}

results: list[tuple[str, str, str]] = []


def record(kind: str, name: str, ok: bool, detail: str = "") -> None:
    results.append((kind, name, "PASS" if ok else "FAIL: " + detail))


def snap(raw, attr: str):
    val = getattr(raw, attr, None)
    if val is None:
        return None
    if torch.is_tensor(val):
        return val.detach().to("cpu").clone()
    return torch.tensor(val)


def check_static(cfg, keys) -> None:
    """A. every declared knob must be read somewhere in the source."""
    with open(ENV_SRC, encoding="utf-8") as fh:
        src = fh.read()
    lines = [ln for ln in src.splitlines() if ln.lstrip().startswith(("#", "//")) is False]
    body = "\n".join(lines)
    for key in keys:
        reads = len(re.findall(rf"cfg\.{key}\b|\"{key}\"", body))
        decl = len(re.findall(rf"\b{key}: ", body))
        record("A DEAD KNOB", key, reads > 0, f"{reads} read site(s), {decl} declaration(s)")


def check_horizon(raw, keys, resets: int) -> None:
    """B. per-run draws survive resets; per-episode draws do not."""
    for key in keys:
        if key in SWITCHES:
            continue
        entry = HORIZON.get(key)
        if entry is None:
            record("B HORIZON", key, False, "no HORIZON entry -- decide per-run or per-episode")
            continue
        attr, horizon = entry
        first = snap(raw, attr)
        if first is None:
            record("B HORIZON", key, False, f"state buffer {attr} missing")
            continue
        changed = False
        identical = True
        for _ in range(resets):
            raw.reset()
            cur = snap(raw, attr)
            if cur is not None:
                changed = changed or not torch.equal(cur, first)
                identical = identical and torch.equal(cur, first)
        if horizon == "run":
            record("B HORIZON", key, identical, f"per-run draw re-sampled on reset ({attr})")
        else:
            record("B HORIZON", key, changed, f"per-episode draw never changed across {resets} resets ({attr})")


def check_effects(raw, turbidity_min_drop: float) -> None:
    """C. the new sensor/water knobs must actually move what they claim."""
    # heading belief must leave ground truth
    gt_yaw = euler_xyz_from_quat(raw._robot.data.root_quat_w)[2]
    raw._sensed_yaw_key = -1
    sensed = raw._sensed_yaw()
    delta = (sensed - gt_yaw).abs().max().item()
    record("C EFFECT", "imu_yaw_bias", delta > 1e-4, f"sensed yaw == GT yaw (delta {delta:.2e} rad)")

    # drag anisotropy must not be identity
    aniso = raw._drag_aniso
    dev = (aniso - 1.0).abs().max().item() if torch.is_tensor(aniso) else 0.0
    record("C EFFECT", "drag_aniso", dev > 1e-4, f"anisotropy is identity (max dev {dev:.2e})")

    # camera mount must not be flat
    tilt = getattr(raw, "_cam_tilt_deg", None)
    tdev = tilt.abs().max().item() if torch.is_tensor(tilt) else 0.0
    record("C EFFECT", "cam_mount", tdev > 1e-4, f"camera mount is flat (max tilt {tdev:.2e} deg)")

    # texture choice must be deterministic per env and resolve to a real file
    picks = [raw._dr_texture_for_env(ei) for ei in range(raw.num_envs)]
    repeated = [raw._dr_texture_for_env(ei) for ei in range(raw.num_envs)]
    if all(p is None for p in picks):
        record(
            "C EFFECT",
            "texture_choice",
            True,
            "no texture candidates configured -- falls back to the single default asset",
        )
    else:
        stable = picks == repeated
        valid = all(isinstance(p, str) and os.path.isfile(p) for p in picks)
        record(
            "C EFFECT",
            "texture_choice",
            stable and valid,
            f"unstable or invalid per-env texture: {picks}",
        )

    # turbidity must attenuate contrast: env0 murky, env1 clear, everything else pinned
    raw._dr_cam_bright[:] = 1.0
    raw._dr_cam_noise[:] = 0.0
    raw._turbidity[0] = 1.0
    if raw.num_envs > 1:
        raw._turbidity[1] = 0.0
    raw._dr_frame_key = -1
    frame = raw._camera_rgb().to(torch.float32)
    std0 = frame[0].std().item()
    std1 = frame[1].std().item() if raw.num_envs > 1 else std0
    drop = 1.0 - (std0 / max(std1, 1e-6))
    record(
        "C EFFECT",
        "turbidity",
        drop >= turbidity_min_drop,
        f"contrast barely changed: std {std0:.1f} vs {std1:.1f} (drop {drop:.0%})",
    )


def check_visodom_guard(raw, steps: int = 20) -> None:
    """D. the integrator must stay inside the physically possible travel budget.

    Arm F lost ~10 h because a corrupted estimate could accumulate past any plausible
    speed: the reward pays that estimate, so the policy degenerated into a bimodal
    strategy. This checks the bound is real, not just declared in the cfg.
    """
    v_max = float(getattr(raw.cfg, "visodom_max_speed_mps", 1.0))
    dt_step = float(raw.cfg.sim.dt) * float(raw.cfg.decimation)
    raw.reset()
    zeros = torch.zeros((raw.num_envs, 3), device=raw.device)
    for _ in range(steps):
        raw.step(zeros)
    est = raw._visodom_m().norm(dim=-1)
    budget = v_max * dt_step * steps + 0.05
    record(
        "D GUARD",
        "visodom_max_speed_mps",
        bool(est.max() <= budget),
        f"|visodom| = {est.max():.3f} m after {steps} steps, budget {budget:.3f} m",
    )


def check_reward_source(raw) -> None:
    """E. the position reward must be insensitive to the estimator when it pays truth.

    This is the whole point of rew_position_source="gt": the harder DR is kept, but the
    learning signal stops being noise-dominated. If the reward still reacts to the
    estimate, the collapse that killed arm F (~70 k) and arm G (~190 k) comes back.
    Cost of running it: _compute_step_reward() twice, which double-counts this env's
    episode diagnostics. Harmless for a one-shot audit.
    """
    src = str(getattr(raw.cfg, "rew_position_source", "gt")).lower()
    raw._visodom_bias.zero_()
    r0 = raw._compute_step_reward().clone()
    raw._visodom_bias[:] = 5.0
    r1 = raw._compute_step_reward().clone()
    raw._visodom_bias.zero_()
    moved = (r1 - r0).abs().max().item()
    if src == "visodom":
        record(
            "E REWARD",
            "rew_position_source=visodom",
            moved > 1e-3,
            f"reward ignored a 5 m estimate shift ({moved:.3e}) -- visodom source is dead",
        )
    else:
        record(
            "E REWARD",
            "rew_position_source=gt",
            moved < 1e-3,
            f"reward moved {moved:.3e} when only the ESTIMATE moved -- gt source leaks the ruler",
        )


def check_action_rate(raw) -> None:
    """F. the rate penalty must bite, AND stay bounded.

    Two-sided on purpose. A check that only asserts "the penalty exists" passed arm J's
    unbounded hinge, which reached ~33/step against a ~2.45/step reward and collapsed the
    critic at 26.9 k steps. The upper bound is the part that would have caught it: even a
    worst-case stick jump must cost less than half a step of reward, which is exactly the
    property MarineGym gets for free by using exp(-x) forms.
    """
    raw._prev_actions.copy_(raw._actions)
    r_same = raw._compute_step_reward().clone()

    raw._prev_actions.copy_(raw._actions + 0.5)
    r_mid = raw._compute_step_reward().clone()

    raw._prev_actions.copy_(raw._actions + 2.0)  # worst case: full-scale reversal
    r_max = raw._compute_step_reward().clone()
    raw._prev_actions.copy_(raw._actions)

    drop_mid = float((r_same - r_mid).mean())
    drop_max = float((r_same - r_max).mean())

    record(
        "F ACTRATE-LB",
        "penalty_bites",
        drop_mid > 1e-3,
        f"reward drop for a 0.5 stick step = {drop_mid:.3f} (must be > 0)",
    )
    record(
        "F ACTRATE-UB",
        "penalty_bounded",
        drop_max < 1.2,
        f"reward drop for a 2.0 stick reversal = {drop_max:.3f} (must stay < 50% of ~2.45/step)",
    )


def check_payload(raw) -> None:
    """G. payload wrench: constant world weight, torque from the CG offset.

    Tier-1 emulation replaces a real attached body, so the two things that must be exact are
    the total weight (m_p * g, attitude-independent) and the moment arm (r x F). If either is
    wrong the DR silently becomes "add random force", which is not a payload.
    """
    raw._payload_mass[:] = 0.0
    raw._payload_off[:] = 0.0
    f0, t0 = raw._payload_wrench()
    zero_ok = float(f0.abs().max()) <= 0.0 and float(t0.abs().max()) <= 0.0
    record("G PAYLOAD", "disabled_is_zero", zero_ok, f"max|F|={float(f0.abs().max()):.3g} N")

    m_p = 2.5
    raw._payload_mass[:] = m_p
    raw._payload_off[:] = torch.tensor([0.05, 0.0, 0.10], device=raw.device)
    f, tau = raw._payload_wrench()
    mag = float(f.norm(dim=-1).mean())
    want = m_p * 9.81
    tau_mag = float(tau.norm(dim=-1).mean())
    want_tau = float(torch.linalg.cross(raw._payload_off, f).norm(dim=-1).mean())
    raw._payload_mass[:] = 0.0
    raw._payload_off[:] = 0.0

    record("G PAYLOAD", "weight_magnitude", abs(mag - want) < 0.05 * want,
           f"|F|={mag:.2f} N vs m_p*g={want:.2f} N")
    record("G PAYLOAD", "cg_torque", tau_mag > 1e-3 and abs(tau_mag - want_tau) < 1e-3,
           f"|tau|={tau_mag:.3f} N*m (r x F = {want_tau:.3f})")


def check_vision_delay(raw) -> None:
    """J. vision latency is specified in ms and converted to whole control steps.

    Same trap as the action delay: a value written in steps silently changes meaning if the
    control period ever moves, so the conversion has to be asserted, not assumed.
    """
    dt = float(raw.cfg.sim.dt) * float(raw.cfg.decimation)
    cases = [(0.0, 0), (66.7, 1), (133.3, 2), (200.0, 3)]
    ok = all(round(ms / (dt * 1e3)) == want for ms, want in cases)
    record(
        "J VISDELAY",
        "ms_to_steps",
        ok,
        "ctrl period %.1f ms -> %s"
        % (dt * 1e3, " ".join("%.0fms=%d" % (ms, round(ms / (dt * 1e3))) for ms, _ in cases)),
    )
    record("J VISDELAY", "disabled_by_default", int(raw._vis_delay_steps) == 0,
           f"_vis_delay_steps={int(raw._vis_delay_steps)}")


def main() -> int:
    cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.use_real_usd = False
    cfg.seed = 66
    cfg.dr_enable = True
    overridden = [
        k for k, v in CANARY.items() if getattr(cfg, k, None) in ((1.0, 1.0), (0.0, 0.0), 0.0)
    ]
    for key in overridden:
        setattr(cfg, key, CANARY[key])
    if overridden:
        print(
            f"[dr-audit] canary overrides (production value is a deliberate no-op): {overridden}",
            flush=True,
        )
    env = gym.make("Isaac-AUV-Tracking-Direct-v7-flow-hover", cfg=cfg)
    raw = env.unwrapped

    keys = sorted(k for k in dir(cfg) if k.startswith("dr_") and isinstance(getattr(cfg, k), (bool, int, float, tuple)))
    print(f"[dr-audit] auditing {len(keys)} dr_* knobs on {raw.num_envs} envs", flush=True)

    check_static(cfg, keys)
    check_horizon(raw, keys, int(args_cli.resets))
    check_effects(raw, float(args_cli.turbidity_min_drop))
    check_visodom_guard(raw)
    check_reward_source(raw)
    check_action_rate(raw)
    check_payload(raw)
    check_vision_delay(raw)

    print("\n[dr-audit] ===== coverage table =====", flush=True)
    print(f"{'check':<14} {'item':<34} {'value':>10}  verdict", flush=True)
    fails = 0
    for kind, name, verdict in results:
        val = getattr(cfg, name, "")
        if isinstance(val, tuple):
            val = f"{val[0]}..{val[1]}"
        shown = f"{val:.3f}" if isinstance(val, float) else str(val)
        if verdict.startswith("FAIL"):
            fails += 1
        print(f"{kind:<14} {name:<34} {shown:>10}  {verdict}", flush=True)

    known = set(HORIZON)
    unknown = [k for k in keys if k not in known and k != "dr_enable"]
    if unknown:
        print(f"\n[dr-audit] knobs with no HORIZON entry: {unknown}", flush=True)

    print(f"\n[dr-audit] {len(results) - fails}/{len(results)} PASS, {fails} FAIL", flush=True)
    print("[dr-audit] ALL PASS" if fails == 0 else "[dr-audit] FAILURES PRESENT", flush=True)
    env.close()
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    code = main()
    app_launcher.app.close()
    sys.exit(code)
