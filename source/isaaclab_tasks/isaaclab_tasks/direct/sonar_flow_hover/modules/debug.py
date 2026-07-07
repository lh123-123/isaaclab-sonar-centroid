# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Debug logging and runtime metrics (usable from env and standalone scripts)."""

from __future__ import annotations

from typing import Any

import torch

from .constants import BODY_STATE_DIM, LATERAL_1D_OBS_NAMES, MAX_TRACK_POINTS
from .geometry import (
    euler_xyz_from_quat,
    robot_lateral_sway_signed,
    station_pos_error_norm,
    target_lateral_offset_dy,
)

BODY_STATE_NAMES = ("roll", "pitch", "yaw", "wx", "wy", "wz", "depth", "depth_rate")


def collect_metrics(
    *,
    pipeline,
    robot,
    station_origin_w: torch.Tensor,
    actions: torch.Tensor,
    reference_mode: str,
    kinematic_offset_w: torch.Tensor | None = None,
    target_kinematic_offset_b: torch.Tensor | None = None,
    centroid_err_px: torch.Tensor | None = None,
    env_id: int = 0,
    lateral_1d: bool = False,
) -> dict[str, Any]:
    """Snapshot key IBVS-DRL signals for logging or CSV export."""
    ft = pipeline.feature_tracker
    offset_m = 0.0
    if kinematic_offset_w is not None:
        offset_m = float(torch.norm(kinematic_offset_w[env_id, :2]).item())
    ref_c = ft.ref_centroid()[env_id].detach().cpu()
    track_c = ft.tracked_centroid()[env_id].detach().cpu()
    if centroid_err_px is not None:
        cent_err_px = centroid_err_px[env_id].detach().cpu()
        centroid_daz_px = float(cent_err_px[0].item())
        centroid_dr_px = float(cent_err_px[1].item())
    else:
        cent_err_px = ft.centroid_errors()[env_id].detach().cpu()
        centroid_daz_px = float(cent_err_px[0].item())
        centroid_dr_px = float(cent_err_px[1].item())
    _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[env_id : env_id + 1])
    robot_sway = float(
        robot_lateral_sway_signed(station_origin_w, robot.data.root_pos_w, yaw)[env_id].item()
    )
    target_dy = (
        float(target_lateral_offset_dy(target_kinematic_offset_b)[env_id].item())
        if target_kinematic_offset_b is not None
        else 0.0
    )
    sway_error = robot_sway - target_dy
    centroid_lat_px = abs(centroid_daz_px)
    pixel_err = centroid_lat_px if lateral_1d else float(ft.normalized_pixel_error()[env_id].item())
    return {
        "valid_pts": int(ft.valid_count()[env_id].item()),
        "pixel_err": pixel_err,
        "centroid_daz_px": centroid_daz_px,
        "centroid_dr_px": centroid_dr_px,
        "centroid_lat_px": centroid_lat_px,
        "robot_sway_m": robot_sway,
        "target_dy_m": target_dy,
        "sway_error_m": sway_error,
        "pos_err_m": float(station_pos_error_norm(station_origin_w, robot.data.root_pos_w)[env_id].item()),
        "offset_m": offset_m,
        "depth_m": float(robot.data.root_pos_w[env_id, 2].item()),
        "vel_m_s": float(torch.norm(robot.data.root_lin_vel_b[env_id]).item()),
        "ref_mode": reference_mode,
        "action": [float(actions[env_id, i].item()) for i in range(actions.shape[1])],
        "ref_centroid_xy": [float(ref_c[0].item()), float(ref_c[1].item())],
        "tracked_centroid_xy": [float(track_c[0].item()), float(track_c[1].item())],
        "centroid_dx": centroid_daz_px,
        "centroid_dy": centroid_dr_px,
    }


def decode_observation(obs_row: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split policy observation into named parts (full or compact 1D layout)."""
    if obs_row.numel() == len(LATERAL_1D_OBS_NAMES):
        return {name: obs_row[i : i + 1] for i, name in enumerate(LATERAL_1D_OBS_NAMES)}
    n = MAX_TRACK_POINTS
    body_start = 3 * n
    body_end = body_start + BODY_STATE_DIM
    return {
        "feat_err": obs_row[: 2 * n],
        "valid_flags": obs_row[2 * n : 3 * n],
        "body": obs_row[body_start:body_end],
        "prev_action": obs_row[body_end:],
    }


def format_obs_debug(obs_row: torch.Tensor, *, env_id: int = 0) -> str:
    """Human-readable breakdown of one policy observation vector."""
    if obs_row.numel() == len(LATERAL_1D_OBS_NAMES):
        vals = obs_row.detach().cpu()
        parts = ", ".join(f"{name}={float(vals[i].item()):+.2f}" for i, name in enumerate(LATERAL_1D_OBS_NAMES))
        return f"  obs dim={obs_row.numel()} (compact 1D)\n  [{parts}]"

    parts = decode_observation(obs_row)
    feat = parts["feat_err"].detach().cpu()
    valid = parts["valid_flags"].detach().cpu()
    body = parts["body"].detach().cpu()
    prev = parts["prev_action"].detach().cpu()

    lines = [
        f"  obs dim={obs_row.numel()}",
        "  [feat_err] valid points (ex, ey normalized):",
    ]
    active = 0
    for i in range(MAX_TRACK_POINTS):
        if valid[i] < 0.5:
            continue
        active += 1
        ex, ey = float(feat[2 * i].item()), float(feat[2 * i + 1].item())
        lines.append(f"    pt{i:02d}: ex={ex:+.4f} ey={ey:+.4f}  |e|={((ex * ex + ey * ey) ** 0.5):.4f}")
    if active == 0:
        lines.append("    (none)")
    lines.append(f"  [valid] sum={int(valid.sum().item())}/{MAX_TRACK_POINTS}")
    body_parts = ", ".join(f"{name}={float(body[j].item()):+.4f}" for j, name in enumerate(BODY_STATE_NAMES))
    lines.append(f"  [body] {body_parts}")
    prev_str = ", ".join(f"a{i}={float(prev[i].item()):+.4f}" for i in range(prev.numel()))
    lines.append(f"  [prev_action] {prev_str}")
    return "\n".join(lines)


def format_reward_debug(terms: dict[str, torch.Tensor], *, env_id: int = 0, cfg=None) -> str:
    """Human-readable reward term breakdown for one env."""
    eid = env_id

    def _scalar(key: str) -> float:
        return float(terms[key][eid].item())

    w1 = float(getattr(cfg, "rew_w1_pixel", 1.0)) if cfg is not None else 1.0
    w2 = float(getattr(cfg, "rew_w2_action", 0.1)) if cfg is not None else 0.1
    w3 = float(getattr(cfg, "rew_w3_safety", 5.0)) if cfg is not None else 5.0
    min_valid = int(getattr(cfg, "min_valid_features", 2)) if cfg is not None else 2

    lines = [
        "  [reward terms]",
        f"    pixel_error={_scalar('pixel_error'):.4f}  -> term={_scalar('term_pixel'):+.4f}  (w1={w1})",
        f"    action_penalty={_scalar('action_penalty'):.4f} -> term={_scalar('term_action'):+.4f}  (w2={w2})",
        f"    valid_n={int(_scalar('valid_n'))} (min={min_valid})  safety={int(_scalar('safety'))}",
        f"      depth_low={int(_scalar('depth_low'))} depth_high={int(_scalar('depth_high'))} "
        f"too_few_features={int(_scalar('too_few_features'))}",
        f"    safety term={_scalar('term_safety'):+.4f}  (w3={w3})",
        f"    total reward={_scalar('total'):+.4f}",
    ]
    return "\n".join(lines)


def obs_to_dict(obs_row: torch.Tensor) -> dict[str, float]:
    """Named scalar dict for compact or full observation vectors."""
    if obs_row.numel() == len(LATERAL_1D_OBS_NAMES):
        return {name: float(obs_row[i].item()) for i, name in enumerate(LATERAL_1D_OBS_NAMES)}
    parts = decode_observation(obs_row)
    out: dict[str, float] = {
        "valid_sum": float(parts["valid_flags"].sum().item()),
        "feat_err_nz": float((parts["feat_err"].abs() > 1e-8).float().mean().item()),
    }
    for j, name in enumerate(BODY_STATE_NAMES):
        out[name] = float(parts["body"][j].item())
    for i in range(parts["prev_action"].numel()):
        out[f"prev_a{i}"] = float(parts["prev_action"][i].item())
    return out


class ObsValidityTracker:
    """Collect obs snapshots and run PASS/FAIL checks after a short rollout."""

    def __init__(self) -> None:
        self._rows: list[dict[str, float | int]] = []

    def record(
        self,
        *,
        step: int,
        obs_row: torch.Tensor,
        unwrapped,
        env_id: int = 0,
    ) -> None:
        d = obs_to_dict(obs_row)
        d["step"] = step
        if obs_row.numel() == len(LATERAL_1D_OBS_NAMES):
            amp = max(float(unwrapped.cfg.lateral_target_amp_m), 1.0e-3)
            d["valid_n"] = int(unwrapped.feature_tracker.valid_count()[env_id].item())
            d["pixel_err"] = float(abs(d.get("centroid_daz_px", d.get("centroid_lat_px", 0.0))))
        else:
            d["valid_n"] = int(d.get("valid_sum", 0))
            d["pixel_err"] = float(unwrapped.feature_tracker.normalized_pixel_error()[env_id].item())
        self._rows.append(d)

    def summarize(self, *, cfg, compact: bool) -> list[tuple[str, bool, str]]:
        if not self._rows:
            return [("has samples", False, "no observations recorded")]

        checks: list[tuple[str, bool, str]] = []
        warmup = [r for r in self._rows if int(r["step"]) >= 10]
        use = warmup if warmup else self._rows

        def _fin(key: str) -> bool:
            vals = [float(r[key]) for r in use if key in r]
            return bool(vals) and all(abs(v) < 1.0e6 for v in vals)

        if compact:
            checks.append(
                (
                    "finite values",
                    _fin("centroid_daz_px") and _fin("centroid_dr_px") and _fin("prev_sway"),
                    "all obs scalars finite",
                )
            )
            min_valid = int(getattr(cfg, "min_valid_features", 2))
            max_track = 20
            min_ratio = min_valid / max_track
            vr = [float(r["valid_ratio"]) for r in use]
            checks.append(
                (
                    "valid_ratio above minimum",
                    max(vr) >= min_ratio - 1e-4,
                    f"max valid_ratio={max(vr):.3f} need>={min_ratio:.3f}",
                )
            )
            vn = [int(r["valid_n"]) for r in use]
            checks.append(
                (
                    "tracker valid_n matches ratio",
                    all(abs(int(r["valid_n"]) / max_track - float(r["valid_ratio"])) < 0.05 for r in use),
                    f"valid_n range=[{min(vn)},{max(vn)}]",
                )
            )
            daz = [abs(float(r["centroid_daz_px"])) for r in use]
            dr = [abs(float(r["centroid_dr_px"])) for r in use]
            checks.append(
                (
                    "centroid offset visible",
                    max(daz + dr, default=0.0) > 0.01,
                    f"|daz| max={max(daz) if daz else 0.0:.4f}  |dr| max={max(dr) if dr else 0.0:.4f}",
                )
            )
            if len(use) >= 2:
                d0 = torch.tensor([float(use[0][k]) for k in LATERAL_1D_OBS_NAMES])
                d1 = torch.tensor([float(use[-1][k]) for k in LATERAL_1D_OBS_NAMES])
                delta = float(torch.linalg.norm(d1 - d0).item())
                checks.append(
                    (
                        "obs reacts over time",
                        delta > 0.05,
                        f"L2(first,last)={delta:.4f} in rollout",
                    )
                )
            pe = [float(r["pixel_err"]) for r in use]
            checks.append(
                (
                    "centroid error consistent with reward",
                    max(pe) > 0.01 and max(daz, default=0.0) > 0.01,
                    f"pixel_err max={max(pe):.4f}",
                )
            )
        else:
            checks.append(
                (
                    "valid features present",
                    max(int(r["valid_n"]) for r in use) >= int(getattr(cfg, "min_valid_features", 2)),
                    f"valid_n max={max(int(r['valid_n']) for r in use)}",
                )
            )
            checks.append(
                (
                    "feat_err non-zero when valid",
                    max(float(r.get("feat_err_nz", 0.0)) for r in use) > 0.0,
                    "sparse feat_err has non-zero entries",
                )
            )

        return checks


def print_obs_validity_report(checks: list[tuple[str, bool, str]]) -> bool:
    print("\n======== OBS VALIDITY ========", flush=True)
    all_ok = True
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}: {detail}", flush=True)
        all_ok = all_ok and ok
    if all_ok:
        print("\n✓ Observation data is valid and live for training.", flush=True)
    else:
        print("\n✗ Observation issues detected — see FAIL lines.", flush=True)
    print("==============================\n", flush=True)
    return all_ok


def maybe_print_ibvs_debug(
    *,
    step_counter: int,
    interval: int,
    pipeline,
    robot,
    station_origin_w: torch.Tensor,
    actions: torch.Tensor,
    reference_mode: str,
    max_track_features: int,
    kinematic_offset_w: torch.Tensor | None = None,
    target_kinematic_offset_b: torch.Tensor | None = None,
    centroid_err_px: torch.Tensor | None = None,
    lateral_1d: bool = False,
    env_id: int = 0,
) -> None:
    if step_counter % max(interval, 1) != 0:
        return
    m = collect_metrics(
        pipeline=pipeline,
        robot=robot,
        station_origin_w=station_origin_w,
        actions=actions,
        reference_mode=reference_mode,
        kinematic_offset_w=kinematic_offset_w,
        target_kinematic_offset_b=target_kinematic_offset_b,
        centroid_err_px=centroid_err_px,
        lateral_1d=lateral_1d,
        env_id=env_id,
    )
    act = m["action"]
    act_str = ",".join(f"{a:+.2f}" for a in act)
    print(
        f"[SonarIbvsDrl] step={step_counter} "
        f"frame={m.get('perception_frame', 'cart')} "
        f"centroid_daz_px={m['centroid_daz_px']:+.2f} "
        f"sway={m['robot_sway_m']:+.3f} target_dy={m['target_dy_m']:+.3f} "
        f"sway_err={m['sway_error_m']:+.3f} "
        f"valid_pts={m['valid_pts']}/{max_track_features} "
        f"depth={m['depth_m']:.2f} offset={m['offset_m']:.3f}m "
        f"action=({act_str}) "
        f"ref_mode={reference_mode}",
        flush=True,
    )
