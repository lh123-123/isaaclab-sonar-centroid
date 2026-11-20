#!/usr/bin/env python3
"""Manually crop the teach template from the stored teach photo.

No colour involved: you draw (or pass) the box, the script cuts the template,
writes it back into the npz as ``template`` + ``box_xywh``, and the env prefers
the stored template when present.

Usage:
  # interactive: click top-left then bottom-right on the popped window
  python3 scripts/crop_teach_template.py --npz models/auv_tracking/teach_v5_2m_0deg.npz
  # non-interactive:
  python3 scripts/crop_teach_template.py --npz models/auv_tracking/teach_v5_2m_0deg.npz \
      --box 100,120,60,36
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--box", type=str, default="",
                    help="'x0,y0,x1,y1' pixels in the stored image; omit for GUI click")
    ap.add_argument("--margin", type=int, default=2)
    args = ap.parse_args()

    blob = np.load(args.npz)
    image = np.ascontiguousarray(blob["image"])
    h, w = image.shape[:2]

    if args.box.strip():
        x0, y0, x1, y1 = (int(v) for v in args.box.split(","))
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(image.astype(np.uint8) if image.max() > 1.5 else image)
        ax.set_title("click TOP-LEFT then BOTTOM-RIGHT of the target")
        pts = plt.ginput(2, timeout=0)
        plt.close(fig)
        if len(pts) < 2:
            print("need 2 clicks", file=sys.stderr)
            sys.exit(1)
        (x0, y0), (x1, y1) = pts
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    m = int(args.margin)
    x0 = max(min(x0, x1) - m, 0)
    y0 = max(min(y0, y1) - m, 0)
    x1 = min(max(x0, x1) + m, w)
    y1 = min(max(y0, y1) + m, h)
    crop = image[y0:y1, x0:x1].astype(np.float32)
    if crop.max() > 1.5:
        crop = crop / 255.0
    template = np.ascontiguousarray(crop.transpose(2, 0, 1))[None]  # (1,C,th,tw)
    box = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0], dtype=np.float32)

    out = {k: np.ascontiguousarray(blob[k]) for k in blob.files}
    out["template"] = template
    out["box_xywh"] = box
    np.savez(args.npz, **out)
    print(f"template {template.shape} box_cxcywh={box.tolist()} -> {args.npz}")


if __name__ == "__main__":
    main()
