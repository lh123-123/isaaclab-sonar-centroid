#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Aggregate aligned eval rows and compare groups (D4).

Input: JSON from align_auv_eval_metrics.py (list of rows) or raw play JSONs.
When each group has < ~8 samples, use Wilcoxon rank-sum (Mann–Whitney).
With 3 seeds, report mean±std and p-values; do not overclaim.

  python3 scripts/auv_eval_significance.py --group v2-full a.json b.json \\
      --group v2-image c.json --compare v2-full v2-image
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_auv_eval_metrics import CORE, normalize_eval_json


def _load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            rows.extend(raw)
        else:
            rows.append(normalize_eval_json(p))
    return rows


def _stars(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _compare(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    out = {
        "n_a": int(a.size),
        "n_b": int(b.size),
        "mean_a": float(a.mean()) if a.size else None,
        "mean_b": float(b.mean()) if b.size else None,
        "test": None,
        "p": None,
        "sig": "n/a",
    }
    if a.size < 2 or b.size < 2:
        return out
    if a.size == b.size and a.size >= 3:
        try:
            _w, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="pratt")
            out["test"] = "wilcoxon_signed"
            out["p"] = float(p)
            out["sig"] = _stars(float(p))
            return out
        except ValueError:
            pass
    try:
        _u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        out["test"] = "mannwhitneyu"
        out["p"] = float(p)
        out["sig"] = _stars(float(p))
    except ValueError:
        pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        action="append",
        nargs="+",
        metavar=("NAME", "JSON"),
        help="Repeatable: --group NAME file.json [file2.json ...]",
    )
    parser.add_argument(
        "--compare",
        action="append",
        nargs=2,
        metavar=("A", "B"),
        help="Repeatable pair of group names, e.g. --compare v2-full v2-image",
    )
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()
    if not args.group:
        parser.error("need at least one --group")

    groups: dict[str, list[dict]] = {}
    for chunk in args.group:
        name, *files = chunk
        groups[name] = _load_rows([Path(f) for f in files])

    agg = {}
    for name, rows in groups.items():
        agg[name] = {}
        for key in CORE:
            xs = np.array([r[key] for r in rows if r.get(key) is not None], dtype=np.float64)
            agg[name][key] = {
                "mean": float(xs.mean()) if xs.size else None,
                "std": float(xs.std(ddof=1)) if xs.size > 1 else 0.0,
                "n": int(xs.size),
                "values": [float(x) for x in xs],
            }

    comparisons = []
    pairs = args.compare or []
    for a_name, b_name in pairs:
        if a_name not in groups or b_name not in groups:
            raise SystemExit(f"unknown group in --compare {a_name} vs {b_name}")
        block = {"a": a_name, "b": b_name, "metrics": {}}
        for key in CORE:
            xa = [r[key] for r in groups[a_name] if r.get(key) is not None]
            xb = [r[key] for r in groups[b_name] if r.get(key) is not None]
            block["metrics"][key] = _compare(np.asarray(xa), np.asarray(xb))
        comparisons.append(block)

    report = {"groups": agg, "comparisons": comparisons}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"[sig] wrote {args.out}", flush=True)

    print("\n[table-1 preview]")
    hdr = f"{'group':20s}  r_err     gt_a      tot       ep_ret    n"
    print(hdr)
    for name, mets in agg.items():
        print(
            f"{name:20s}  "
            + "  ".join(
                f"{(mets[k]['mean'] if mets[k]['mean'] is not None else float('nan')):8.3f}"
                for k in CORE
            )
            + f"  {mets['r_err']['n']}"
        )
    for block in comparisons:
        print(f"\n{block['a']} vs {block['b']}")
        for key, st in block["metrics"].items():
            p = st["p"]
            p_s = f"{p:.4g}" if p is not None else "n/a"
            print(f"  {key:8s}  {st['test'] or '-':16s} p={p_s}  {st['sig']}")


if __name__ == "__main__":
    main()
