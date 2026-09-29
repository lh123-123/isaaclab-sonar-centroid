#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Normalize v0 / v2 play JSON into one table row schema (D3).

  python3 scripts/align_auv_eval_metrics.py /tmp/v2_ablation_eval/*.json /tmp/v0_s66.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CORE = ("r_err", "gt_a", "tot", "ep_ret")


def _mean(d: dict, *keys: str) -> float | None:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                v = float(d[k])
            except (TypeError, ValueError):
                continue
            if v == v:  # not NaN
                return v
    return None


def normalize_eval_json(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    track = raw.get("tracking") or {}
    pooled = raw.get("pooled") or {}
    ep_ret = None
    if "ep_ret" in pooled:
        ep_ret = float(pooled["ep_ret"].get("mean", 0.0))
    eps = raw.get("episodes") or []
    if ep_ret is None and eps:
        ep_ret = sum(float(e["ep_ret"]) for e in eps) / len(eps)

    n_ep = int(track.get("n_episodes") or raw.get("n_episodes") or 0)
    row = {
        "source": str(path),
        "label": track.get("tag") or raw.get("label") or path.stem,
        "checkpoint": track.get("checkpoint") or "",
        "n_episodes": n_ep,
        "n_steps": int(track.get("n_steps") or raw.get("n_steps") or 0),
        "r_err": _mean(track, "r_err_mean"),
        "gt_a": _mean(track, "gt_a_mean"),
        "tot": _mean(track, "tot_mean"),
        "ep_ret": ep_ret,
        "ep_len": _mean(pooled.get("ep_len") or {}, "mean"),
        "gt_r": _mean(pooled.get("gt_r") or {}, "mean"),
        "r_star_m": track.get("r_star_m", 2.0),
        "r_err_note": track.get("r_err_note", "v2 r_err = |centroid_r - r*|"),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsons", nargs="+", type=Path)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()
    rows = [normalize_eval_json(p) for p in args.jsons]
    text = json.dumps(rows, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"[align] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
