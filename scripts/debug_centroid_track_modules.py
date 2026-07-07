#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline layer-by-layer debug for sonar_centroid_track modules (no Isaac Sim).

Usage:
  python scripts/debug_centroid_track_modules.py --layer coordinates
  python scripts/debug_centroid_track_modules.py --layer all
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO / "source/isaaclab_tasks/isaaclab_tasks/direct/sonar_centroid_track"
FLOW_ROOT = REPO / "source/isaaclab_tasks/isaaclab_tasks/direct/sonar_flow_hover"
PKG = "isaaclab_tasks.direct.sonar_centroid_track"


def _ensure_packages() -> None:
    """Minimal package tree so relative imports work without Isaac Lab."""
    tasks = REPO / "source/isaaclab_tasks/isaaclab_tasks"
    for name, path in [
        ("isaaclab_tasks", tasks),
        ("isaaclab_tasks.direct", tasks / "direct"),
        (PKG, PKG_ROOT),
        (f"{PKG}.modules", PKG_ROOT / "modules"),
        (f"{PKG}.metrics", PKG_ROOT / "metrics"),
        ("isaaclab_tasks.direct.sonar_flow_hover", FLOW_ROOT),
        ("isaaclab_tasks.direct.sonar_flow_hover.modules", FLOW_ROOT / "modules"),
    ]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            sys.modules[name] = mod


def _load(full_name: str, rel_path: Path):
    spec = importlib.util.spec_from_file_location(full_name, rel_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_all():
    _ensure_packages()
    mods = {}
    for stem in (
        "constants",
        "coordinates",
        "sonar_preprocess",
        "sonar_gt",
        "fixed_point",
        "sonar_centroid_tracker",
        "centroid_bus",
        "observation",
        "actions",
        "rewards",
        "debug",
    ):
        mods[stem] = _load(f"{PKG}.modules.{stem}", PKG_ROOT / "modules" / f"{stem}.py")
    mods["validation"] = _load(f"{PKG}.metrics.validation", PKG_ROOT / "metrics" / "validation.py")
    return mods


def _synthetic_polar_blob(
    *,
    az_col: float = 72.0,
    range_row: float = 28.0,
    shift_col: float = 0.0,
    shift_row: float = 0.0,
    width: int = 128,
    height: int = 64,
) -> np.ndarray:
    polar = np.zeros((height, width), dtype=np.float32)
    cc = az_col + shift_col
    rr = range_row + shift_row
    for r in range(height):
        for c in range(width):
            polar[r, c] = np.exp(-((c - cc) ** 2 / 200.0 + (r - rr) ** 2 / 80.0))
    polar += np.random.rand(height, width).astype(np.float32) * 0.04
    return polar


def build_tests(m):
    polar_point = m["coordinates"].polar_point
    to_fixed_frame = m["coordinates"].to_fixed_frame
    mirror_azimuth_col = m["coordinates"].mirror_azimuth_col

    def test_coordinates() -> None:
        p_fixed = polar_point(70.0, 30.0)
        p_abs = polar_point(74.0, 31.0)
        p_rel = to_fixed_frame(p_abs, p_fixed)
        assert abs(p_rel.col - 4.0) < 1e-5 and abs(p_rel.row - 1.0) < 1e-5
        mirrored = mirror_azimuth_col(10.0, 128)
        assert abs(mirrored - 117.0) < 1e-5
        print(f"[coordinates] p_rel=({p_rel.col:.1f},{p_rel.row:.1f}) mirror(10)={mirrored:.1f} OK")

    def test_preprocess() -> None:
        raw = torch.rand(2, 64, 128)
        polar = m["sonar_preprocess"].preprocess_polar_batch(raw)
        assert polar.shape == (2, 64, 128)
        print(f"[preprocess] shape={tuple(polar.shape)} range=[{polar.min():.3f},{polar.max():.3f}] OK")

    def test_sonar_gt() -> None:
        polar = _synthetic_polar_blob(az_col=80.0, range_row=32.0)
        det = m["sonar_gt"].SonarGtDetector(apply_azimuth_mirror=False)
        p = det.detect_abs(polar)
        assert p is not None
        print(f"[sonar_gt] detected=({p.col:.1f},{p.row:.1f}) expected≈(80,32) OK")

    def test_fixed_point() -> None:
        polar = _synthetic_polar_blob()
        store = m["fixed_point"].FixedPointStore()
        p0 = store.calibrate_from_polar(polar)
        p_rel = store.to_relative(polar_point(p0.col + 3.0, p0.row))
        assert p_rel is not None and abs(p_rel.col - 3.0) < 0.5
        out = REPO / "logs/centroid_track/_debug_fixed.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        store.save(out)
        store2 = m["fixed_point"].FixedPointStore()
        p_load = store2.load(out)
        assert abs(p_load.col - p0.col) < 1e-4
        print(f"[fixed_point] p_fixed_abs=({p0.col:.1f},{p0.row:.1f}) rel_test du≈3 saved={out.name} OK")

    def _make_tracker(store, gt, polar_np):
        import torch

        tr = m["sonar_centroid_tracker"].SonarCentroidTracker(
            gt, store, polar_width=61, polar_height=110
        )
        tr.bind_polar(torch.from_numpy(polar_np).unsqueeze(0))
        return tr

    def test_sonar_centroid_tracker() -> None:
        polar = _synthetic_polar_blob(az_col=65.0, range_row=30.0)
        gt = m["sonar_gt"].SonarGtDetector(apply_azimuth_mirror=False, polar_width=110)
        store = m["fixed_point"].FixedPointStore(sonar_gt=gt)
        store.calibrate_from_polar(polar, mode="manual", manual_col_row=(65.0, 30.0))
        tr = _make_tracker(store, gt, polar)
        p0 = tr.detected_rel(0)
        shifted = _synthetic_polar_blob(az_col=70.0, range_row=30.0)
        tr.bind_polar(__import__("torch").from_numpy(shifted).unsqueeze(0))
        p1 = tr.detected_rel(0)
        assert p0 is not None and p1 is not None
        assert abs(p1.col) > abs(p0.col) * 0.5
        print(
            f"[sonar_centroid_tracker] rel0=({p0.col:+.1f},{p0.row:+.1f}) "
            f"rel1=({p1.col:+.1f},{p1.row:+.1f}) valid={tr.valid_ratio():.2f} OK"
        )

    def test_centroid_bus() -> None:
        polar = _synthetic_polar_blob()
        gt = m["sonar_gt"].SonarGtDetector(apply_azimuth_mirror=False)
        p_sonar = gt.detect_abs(polar)
        store = m["fixed_point"].FixedPointStore()
        store.calibrate_from_polar(polar)
        frame = m["centroid_bus"].make_centroid_frame(
            p_sonar_abs=p_sonar,
            p_perc_abs=polar_point(p_sonar.col + 1.5, p_sonar.row) if p_sonar else None,
            p_fixed_abs=store.p_fixed_abs,
            valid_ratio=1.0 if p_sonar else 0.0,
        )
        print(f"[centroid_bus] {frame.summary()}")

    def test_observation() -> None:
        polar = _synthetic_polar_blob()
        gt = m["sonar_gt"].SonarGtDetector(apply_azimuth_mirror=False, polar_width=110)
        store = m["fixed_point"].FixedPointStore(sonar_gt=gt)
        store.calibrate_from_polar(polar)
        tr = _make_tracker(store, gt, polar)
        builder = m["observation"].DiscreteObservationBuilder(1, "cpu", polar_width=61, polar_height=110, image_only=True)
        obs = builder.build(
            tr,
            robot_sway_m=torch.tensor([0.2]),
            target_dy_m=torch.tensor([0.0]),
            target_dy_vel_m=torch.tensor([0.1]),
        )
        parts = m["debug"].decode_observation(obs[0])
        print(f"[observation] dim={obs.shape[-1]} decoded={parts} OK")

    def test_reward() -> None:
        c = m["constants"]
        polar = _synthetic_polar_blob()
        gt = m["sonar_gt"].SonarGtDetector(apply_azimuth_mirror=False, polar_width=110)
        store = m["fixed_point"].FixedPointStore(sonar_gt=gt)
        store.calibrate_from_polar(polar)
        tr = _make_tracker(store, gt, polar)
        actions = torch.tensor([c.ACTION_STAY])
        prev = torch.tensor([c.ACTION_LEFT])
        r = m["rewards"].compute_reward(
            tr, actions, prev, min_depth=1.0, max_depth=20.0, depth=torch.tensor([10.0])
        )
        print(f"[reward] r={r.item():+.4f} (aligned frame, expect near 0) OK")

    def test_validation() -> None:
        report = m["validation"].ValidationReport()
        for du in np.linspace(-2, 2, 11):
            report.add_perc_sample(
                polar_point(70.0 + du, 30.0),
                polar_point(70.0, 30.0),
                label=f"du={du:+.1f}",
            )
        ok, msg = report.perc_verdict()
        print(f"[validation] perc PASS={ok} {msg}")

    def test_integration() -> None:
        c = m["constants"]
        polar_teach = _synthetic_polar_blob(az_col=68.0, range_row=29.0)
        gt = m["sonar_gt"].SonarGtDetector(apply_azimuth_mirror=False, polar_width=110)
        store = m["fixed_point"].FixedPointStore(sonar_gt=gt)
        store.calibrate_from_polar(polar_teach)
        polar_shift = _synthetic_polar_blob(az_col=74.0, range_row=29.0)
        tr = _make_tracker(store, gt, polar_shift)
        p_sonar = gt.detect_abs(polar_shift)
        p_rel = tr.detected_rel(0)
        frame = m["centroid_bus"].make_centroid_frame(
            p_sonar_abs=p_sonar,
            p_perc_abs=p_sonar,
            p_fixed_abs=store.p_fixed_abs,
            valid_ratio=tr.valid_ratio(),
        )
        builder = m["observation"].DiscreteObservationBuilder(1, "cpu", polar_width=61, polar_height=110, image_only=True)
        obs = builder.build(
            tr,
            robot_sway_m=torch.tensor([0.0]),
            target_dy_m=torch.tensor([0.0]),
            target_dy_vel_m=torch.tensor([0.0]),
        )
        actions = torch.tensor([c.ACTION_LEFT if p_rel and p_rel.col > 0 else c.ACTION_RIGHT])
        r = m["rewards"].compute_reward(
            tr, actions, torch.tensor([c.ACTION_STAY]), min_depth=1.0, max_depth=20.0, depth=torch.tensor([10.0])
        )
        print(f"[integration] {frame.summary()}")
        print(f"[integration] obs_u={obs[0, 0].item():+.3f} reward={r.item():+.3f} action={int(actions.item())} OK")

    def test_obs_correspondence() -> None:
        c = m["constants"]
        oc = _load(f"{PKG}.modules.obs_correspondence", PKG_ROOT / "modules" / "obs_correspondence.py")
        polar = _synthetic_polar_blob(az_col=35.0, range_row=40.0, width=61, height=110)
        gt = m["sonar_gt"].SonarGtDetector(apply_azimuth_mirror=False, polar_width=61)
        store = m["fixed_point"].FixedPointStore(sonar_gt=gt)
        store.calibrate_from_polar(polar, mode="manual", manual_col_row=(30.0, 40.0))
        tr = _make_tracker(store, gt, polar)
        for act, expected in (
            (c.ACTION_LEFT, -1.0),
            (c.ACTION_STAY, 0.0),
            (c.ACTION_RIGHT, 1.0),
        ):
            rep = oc.correspondence_report(tr, env_id=0, prev_action=act, polar_width=61, polar_height=110)
            assert rep["prev_act"] == expected, (act, rep["prev_act"])
        assert rep["u_match"] and rep["v_match"]
        assert abs(rep["d_col_px"] - 5.0) < 1.5
        assert abs(rep["u_norm"] - rep["d_col_px"] / 61.0) < 1.0e-5
        builder = m["observation"].DiscreteObservationBuilder(1, "cpu", polar_width=61, polar_height=110, image_only=True)
        builder.set_prev_action(torch.tensor([c.ACTION_RIGHT]))
        obs = builder.build(
            tr,
            robot_sway_m=torch.tensor([0.0]),
            target_dy_m=torch.tensor([0.0]),
            target_dy_vel_m=torch.tensor([0.0]),
        )
        ref = rep["policy_obs_4d"]
        ref[3] = 1.0  # prev_act from builder (RIGHT)
        assert all(abs(obs[0, i].item() - ref[i]) < 1.0e-5 for i in range(4))
        print(
            f"[obs_correspondence] d_col={rep['d_col_px']:+.1f}px u={rep['u_norm']:+.6f} "
            f"prev={rep['prev_action']} builder_match OK"
        )

    return {
        "coordinates": test_coordinates,
        "preprocess": test_preprocess,
        "sonar_gt": test_sonar_gt,
        "fixed_point": test_fixed_point,
        "sonar_centroid_tracker": test_sonar_centroid_tracker,
        "centroid_bus": test_centroid_bus,
        "observation": test_observation,
        "obs_correspondence": test_obs_correspondence,
        "reward": test_reward,
        "validation": test_validation,
        "integration": test_integration,
    }


def main() -> None:
    mods = _load_all()
    layer_fn = build_tests(mods)
    parser = argparse.ArgumentParser(description="Debug sonar_centroid_track modules offline")
    parser.add_argument("--layer", type=str, default="all", choices=[*layer_fn.keys(), "all"])
    args = parser.parse_args()

    layers = list(layer_fn.keys()) if args.layer == "all" else [args.layer]
    print("=== sonar_centroid_track module debug ===\n")
    for name in layers:
        print(f"--- {name} ---")
        layer_fn[name]()
        print()
    print("All selected layers passed.")


if __name__ == "__main__":
    main()
