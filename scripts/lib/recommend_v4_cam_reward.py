#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Recommend Direct-v4-cam reward overrides from an eval JSON.

Prints Hydra-style ``env.rew_penalty_*=`` tokens and writes a markdown note.

Fallback defaults (used only when the eval JSON carries no ``reward_cfg``) mirror the live task
defaults: range=0.05, bearing=0.05, action_delta=0.02, lost=0.50.

The task's cost clipping was turned off on 2026-09-28 (``env.rew_penalty_clip=0``), so the costs
are now quadratic and unbounded: weight 0.05 already means 0.2/step at 1.2 m range error and
8.9/step at 10 m separation. The bump sizes and ceilings below were calibrated against the old
clipped 0.20/0.30 weights, so they are now *large* relative to the base. They are only a nudge;
prefer changing ``rew_penalty_clip`` / the sigmas over raising these weights further, and re-check
the numbers in the ``AUVTrackingV4CamEnvCfg`` comment before accepting an override.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def recommend(metrics: dict) -> tuple[dict[str, float], list[str]]:
    cfg = dict(metrics.get("reward_cfg") or {})
    base = {
        "rew_penalty_range": float(cfg.get("rew_penalty_range", 0.05)),
        "rew_penalty_bearing": float(cfg.get("rew_penalty_bearing", 0.05)),
        "rew_penalty_action_delta": float(cfg.get("rew_penalty_action_delta", 0.02)),
        "rew_penalty_lost": float(cfg.get("rew_penalty_lost", 0.50)),
    }
    out = dict(base)
    reasons: list[str] = []

    med_a = float(metrics.get("med_abs_bearing_deg", float("nan")))
    frac_far = float(metrics.get("frac_r_gt3", float("nan")))
    lost = float(metrics.get("lost_rate", float("nan")))
    da_p90 = float(metrics.get("da_p90", float("nan")))
    mean_r = float(metrics.get("mean_gt_r", float("nan")))

    # Bearing: previous failure mode was off-centre parallel tracking.
    if med_a == med_a:  # not NaN
        if med_a > 25.0:
            out["rew_penalty_bearing"] = min(base["rew_penalty_bearing"] + 0.20, 0.60)
            reasons.append(f"med_|α|={med_a:.1f}° > 25 → bump bearing cost to {out['rew_penalty_bearing']:.2f}")
        elif med_a > 15.0:
            out["rew_penalty_bearing"] = min(base["rew_penalty_bearing"] + 0.10, 0.50)
            reasons.append(f"med_|α|={med_a:.1f}° > 15 → bump bearing cost to {out['rew_penalty_bearing']:.2f}")
        else:
            reasons.append(f"med_|α|={med_a:.1f}° OK — keep bearing={out['rew_penalty_bearing']:.2f}")

    if frac_far == frac_far:
        if frac_far > 0.20:
            out["rew_penalty_range"] = min(base["rew_penalty_range"] + 0.15, 0.45)
            reasons.append(f"frac_r>3={frac_far:.1%} > 20% → bump range cost to {out['rew_penalty_range']:.2f}")
        elif frac_far > 0.12:
            out["rew_penalty_range"] = min(base["rew_penalty_range"] + 0.10, 0.40)
            reasons.append(f"frac_r>3={frac_far:.1%} > 12% → bump range cost to {out['rew_penalty_range']:.2f}")
        else:
            reasons.append(f"frac_r>3={frac_far:.1%} OK — keep range={out['rew_penalty_range']:.2f}")

    if mean_r == mean_r and mean_r < 1.2:
        # Too close: strengthen range so standoff ~2 m competes with bearing.
        out["rew_penalty_range"] = max(out["rew_penalty_range"], min(base["rew_penalty_range"] + 0.10, 0.40))
        reasons.append(f"mean_r={mean_r:.2f}m < 1.2 → raise range cost to {out['rew_penalty_range']:.2f}")

    if lost == lost:
        if lost > 0.25:
            out["rew_penalty_lost"] = min(base["rew_penalty_lost"] + 0.25, 0.90)
            reasons.append(f"lost={lost:.1%} > 25% → bump lost cost to {out['rew_penalty_lost']:.2f}")
        elif lost > 0.12:
            out["rew_penalty_lost"] = min(base["rew_penalty_lost"] + 0.15, 0.80)
            reasons.append(f"lost={lost:.1%} > 12% → bump lost cost to {out['rew_penalty_lost']:.2f}")
        else:
            reasons.append(f"lost={lost:.1%} OK — keep lost={out['rew_penalty_lost']:.2f}")

    if da_p90 == da_p90:
        if da_p90 > 0.35:
            out["rew_penalty_action_delta"] = min(base["rew_penalty_action_delta"] + 0.04, 0.10)
            reasons.append(f"da_p90={da_p90:.3f} > 0.35 → bump Δaction cost to {out['rew_penalty_action_delta']:.2f}")
        elif da_p90 > 0.22:
            out["rew_penalty_action_delta"] = min(base["rew_penalty_action_delta"] + 0.02, 0.08)
            reasons.append(f"da_p90={da_p90:.3f} > 0.22 → bump Δaction cost to {out['rew_penalty_action_delta']:.2f}")
        else:
            reasons.append(f"da_p90={da_p90:.3f} OK — keep Δaction={out['rew_penalty_action_delta']:.2f}")

    changed = {k: v for k, v in out.items() if abs(v - base[k]) > 1e-9}
    if not changed:
        reasons.append("No weight changes recommended; second train keeps current reward.")
    return out, reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", type=str, required=True)
    ap.add_argument("--out_overrides", type=str, required=True, help="Write one env.*= per line.")
    ap.add_argument("--out_report", type=str, required=True, help="Markdown note.")
    args = ap.parse_args()

    metrics = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    weights, reasons = recommend(metrics)
    base = metrics.get("reward_cfg") or {}
    lines = []
    for k, v in weights.items():
        old = float(base.get(k, v))
        if abs(v - old) > 1e-9:
            lines.append(f"env.{k}={v}")
    Path(args.out_overrides).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    md = [
        "# v4-cam reward recommendation",
        "",
        f"- checkpoint: `{metrics.get('checkpoint')}`",
        f"- ep_ret: {metrics.get('ep_ret_mean'):.1f} ± {metrics.get('ep_ret_std'):.1f}",
        f"- med_|α|: {metrics.get('med_abs_bearing_deg'):.1f}°",
        f"- mean_r: {metrics.get('mean_gt_r'):.2f} m",
        f"- frac_r>3: {metrics.get('frac_r_gt3'):.1%}",
        f"- lost: {metrics.get('lost_rate'):.1%}",
        f"- da_p90: {metrics.get('da_p90'):.3f}",
        "",
        "## Reasons",
        "",
    ]
    md.extend(f"- {r}" for r in reasons)
    md.append("")
    md.append("## Overrides for second train")
    md.append("")
    if lines:
        md.extend(f"- `{ln}`" for ln in lines)
    else:
        md.append("- *(none — reuse defaults)*")
    md.append("")
    Path(args.out_report).write_text("\n".join(md), encoding="utf-8")
    print(f"[reward] overrides → {args.out_overrides}: {lines or '(none)'}", flush=True)
    print(f"[reward] report → {args.out_report}", flush=True)


if __name__ == "__main__":
    main()
