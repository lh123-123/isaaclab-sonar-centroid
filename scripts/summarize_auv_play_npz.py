#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Summarize a play dump (.npz) into episode mean±std JSON. No Isaac required.

  python3 scripts/summarize_auv_play_npz.py /tmp/play_v1_gated_cnn_2250k_actions.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from auv_play_metrics import format_summary, summarize_play, write_summary_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=str)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--ep_len", type=int, default=199)
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()
    path = Path(args.npz)
    data = np.load(path)
    arrays = {k: data[k] for k in data.files}
    summary = summarize_play(arrays, ep_len=args.ep_len, label=args.label or str(path))
    out = args.out or str(path.with_suffix(".json"))
    write_summary_json(summary, out)
    print(format_summary(summary))
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
