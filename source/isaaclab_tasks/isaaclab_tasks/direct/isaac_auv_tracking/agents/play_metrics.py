# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Episode-level play metrics for AUV tracking (numpy-only, no Isaac)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_EP_LEN = 199
SAT_THRESH = 0.9


def _as_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def split_episodes(
    n: int,
    *,
    done: np.ndarray | None = None,
    ep_len: int = DEFAULT_EP_LEN,
) -> list[tuple[int, int]]:
    """Return half-open [start, end) slices for completed episodes."""
    if done is not None:
        d = np.asarray(done, dtype=bool).reshape(-1)
        ends = np.flatnonzero(d) + 1
        slices: list[tuple[int, int]] = []
        start = 0
        for end in ends.tolist():
            if end > start:
                slices.append((start, int(end)))
            start = int(end)
        return slices
    return [(i, min(i + ep_len, n)) for i in range(0, n, ep_len) if min(i + ep_len, n) - i >= ep_len]


def _sat(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x) > SAT_THRESH)) if x.size else 0.0


def _signflip(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.mean(np.diff(np.sign(x)) != 0))


def episode_record(
    a_v: np.ndarray,
    a_w: np.ndarray,
    da: np.ndarray,
    auv_spd: np.ndarray,
    tgt_spd: np.ndarray,
    gt_r: np.ndarray,
    rew: np.ndarray,
    *,
    gate: np.ndarray | None = None,
) -> dict[str, float]:
    speed_err = np.abs(auv_spd - tgt_spd)
    rec = {
        "ep_len": int(a_v.size),
        "ep_ret": float(np.sum(rew)),
        "step_rew": float(np.mean(rew)),
        "auv_spd": float(np.mean(auv_spd)),
        "tgt_spd": float(np.mean(tgt_spd)),
        "speed_err": float(np.mean(speed_err)),
        "gt_r": float(np.mean(gt_r)),
        "abs_a_v": float(np.mean(np.abs(a_v))),
        "abs_a_w": float(np.mean(np.abs(a_w))),
        "a_v_sat": _sat(a_v),
        "a_w_sat": _sat(a_w),
        "da": float(np.mean(da)),
        "da_p90": float(np.percentile(da, 90)) if da.size else 0.0,
        "a_v_signflip": _signflip(a_v),
        "full_episode": float(a_v.size >= DEFAULT_EP_LEN),
    }
    if gate is not None and gate.size:
        rec["gate"] = float(np.mean(gate))
    return rec


def mean_std(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1) if x.size > 1 else 0.0), "n": int(x.size)}


def summarize_play(
    arrays: dict[str, np.ndarray],
    *,
    ep_len: int = DEFAULT_EP_LEN,
    label: str = "",
) -> dict[str, Any]:
    a_v = _as_1d(arrays["a_v"])
    a_w = _as_1d(arrays["a_w"])
    da = _as_1d(arrays["da"])
    auv = _as_1d(arrays["auv_spd"])
    tgt = _as_1d(arrays["tgt_spd"])
    gtr = _as_1d(arrays["gt_r"])
    rew = _as_1d(arrays["rew"])
    n = int(a_v.size)
    done = arrays.get("done")
    gate = _as_1d(arrays["gate"]) if "gate" in arrays else None
    slices = split_episodes(n, done=done, ep_len=ep_len)
    episodes = []
    for lo, hi in slices:
        g_slice = gate[lo:hi] if gate is not None else None
        episodes.append(
            episode_record(
                a_v[lo:hi],
                a_w[lo:hi],
                da[lo:hi],
                auv[lo:hi],
                tgt[lo:hi],
                gtr[lo:hi],
                rew[lo:hi],
                gate=g_slice,
            )
        )
    keys = [
        "ep_len",
        "ep_ret",
        "step_rew",
        "auv_spd",
        "tgt_spd",
        "speed_err",
        "gt_r",
        "abs_a_v",
        "abs_a_w",
        "a_v_sat",
        "a_w_sat",
        "da",
        "da_p90",
        "a_v_signflip",
        "full_episode",
    ]
    if episodes and "gate" in episodes[0]:
        keys.append("gate")
    pooled = {k: mean_std([float(ep[k]) for ep in episodes]) for k in keys}
    return {
        "label": label,
        "n_steps": n,
        "n_episodes": len(episodes),
        "pooled": pooled,
        "episodes": episodes,
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"[eval] {summary.get('label') or 'play'}  "
        f"steps={summary['n_steps']}  episodes={summary['n_episodes']}"
    ]
    for key, stats in summary["pooled"].items():
        lines.append(f"[eval] {key:14s}  {stats['mean']:.4g} ± {stats['std']:.4g}  (n={stats['n']})")
    return "\n".join(lines)


def write_summary_json(summary: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
