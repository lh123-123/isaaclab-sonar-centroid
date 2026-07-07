#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline debug for Sonar IBVS-DRL modules (no Isaac Sim required).

Run individual layers:
  python scripts/debug_ibvs_modules.py --layer preprocess
  python scripts/debug_ibvs_modules.py --layer tracker
  python scripts/debug_ibvs_modules.py --layer obs
  python scripts/debug_ibvs_modules.py --layer reward
  python scripts/debug_ibvs_modules.py --layer integration  # synthetic teach→track chain
  python scripts/debug_ibvs_modules.py --layer all
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "source/isaaclab_tasks/isaaclab_tasks/direct/sonar_flow_hover"
PKG_NAME = "sonar_flow_hover"


def _setup_package() -> None:
    """Register a minimal package tree so relative imports work without Isaac Lab."""
    import types

    if PKG_NAME not in sys.modules:
        root = types.ModuleType(PKG_NAME)
        root.__path__ = [str(PKG)]
        sys.modules[PKG_NAME] = root
    mods_name = f"{PKG_NAME}.modules"
    if mods_name not in sys.modules:
        sub = types.ModuleType(mods_name)
        sub.__path__ = [str(PKG / "modules")]
        sys.modules[mods_name] = sub


def _load(module_name: str, rel_path: str):
    path = PKG / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_modules():
    _setup_package()
    constants = _load(f"{PKG_NAME}.modules.constants", "modules/constants.py")
    geometry = _load(f"{PKG_NAME}.modules.geometry", "modules/geometry.py")
    trk_mod = _load(f"{PKG_NAME}.sparse_feature_tracker", "sparse_feature_tracker.py")
    pre = _load(f"{PKG_NAME}.sonar_preprocess", "sonar_preprocess.py")
    obs_mod = _load(f"{PKG_NAME}.modules.observation", "modules/observation.py")
    rew_mod = _load(f"{PKG_NAME}.modules.rewards", "modules/rewards.py")
    dbg_mod = _load(f"{PKG_NAME}.modules.debug", "modules/debug.py")
    return constants, pre, trk_mod, obs_mod, rew_mod, dbg_mod


def test_preprocess(pre) -> None:
    raw = torch.rand(2, 64, 128)
    polar = pre.preprocess_polar_batch(raw)
    cart = pre.preprocess_sonar_batch(raw, hori_fov_deg=90.0, max_range=8.0, cart_size=128)
    assert polar.shape == (2, 64, 128), polar.shape
    assert cart.shape == (2, 128, 128), cart.shape
    print(
        f"[preprocess] polar={tuple(polar.shape)} cart={tuple(cart.shape)} "
        f"polar_range=[{polar.min():.3f},{polar.max():.3f}]"
    )


def test_polar_tracker(trk_mod, pre) -> None:
    """Polar IBVS: lateral sway shifts azimuth column (index 0)."""
    SparseFeatureTracker = trk_mod.SparseFeatureTracker
    raw = torch.zeros(1, 64, 128)
    raw[0, 18:46, 55:72] = 1.2
    raw[0, 22:42, 60:67] = 2.0
    polar = pre.preprocess_polar_batch(raw)
    tracker = SparseFeatureTracker(
        1,
        "cpu",
        max_track=10,
        image_width=128,
        image_height=64,
        coord_frame="polar",
        tracking_mode="lk",
        ref_min_self_score=0.55,
        ref_min_peak_margin=0.05,
    )
    u8 = tracker.to_u8_image(polar[0].numpy())
    pts = tracker.extract_features(u8, roi_center=(64.0, 32.0), roi_radius=40.0)
    if len(pts) < 2:
        pts = np.array([[64.0, 25.0], [64.0, 35.0], [67.0, 30.0]], dtype=np.float32)
    tracker.set_reference(0, pts, ref_cart=polar[0].numpy())
    tracker.update(polar)
    shifted = torch.roll(polar.clone(), shifts=4, dims=2)
    tracker.update(shifted)
    daz = float(tracker.mean_lateral_error_signed()[0].item())
    ref_c = tracker.ref_centroid()[0]
    track_c = tracker.tracked_centroid()[0]
    cent_norm = tracker.tracked_centroid_norm()[0]
    print(
        f"[polar] teach_pts={len(pts)} valid={int(tracker.valid[0].sum())} mean_daz={daz:+.4f} "
        f"centroid=({track_c[0]:.1f},{track_c[1]:.1f}) norm=({cent_norm[0]:.3f},{cent_norm[1]:.3f})"
    )
    assert int(tracker.valid[0].sum()) >= 1, "polar tracking should keep at least one feature"
    assert abs(daz) > 0.003, "polar lateral error should reflect azimuth shift"


def test_tracker(trk_mod) -> None:
    SparseFeatureTracker = trk_mod.SparseFeatureTracker
    tracker = SparseFeatureTracker(1, "cpu", max_track=20)
    u8 = np.random.randint(32, 220, (128, 128), dtype=np.uint8)
    pts = tracker.extract_features(u8, roi_center=(64.0, 64.0), roi_radius=35.0)
    print(f"[tracker] extracted {len(pts)} FAST points")
    cart_t = torch.from_numpy(u8.astype(np.float32) / 255.0)
    tracker.set_reference(0, pts[: min(10, len(pts))], ref_cart=cart_t.numpy())
    for step in range(5):
        noise = torch.randn_like(cart_t) * 0.02
        tracker.update((cart_t + noise).clamp(0, 1).unsqueeze(0))
        err, flags = tracker.observation_features()
        print(
            f"[tracker] step={step} valid={int(flags.sum().item())} "
            f"pixel_err={tracker.normalized_pixel_error()[0].item():.4f}"
        )
    assert err.shape == (1, 40) and flags.shape == (1, 20)


def test_contour_blob(trk_mod) -> None:
    """Smooth blob: keep only distinctive patches, not dense contour spam."""
    SparseFeatureTracker = trk_mod.SparseFeatureTracker
    tracker = SparseFeatureTracker(1, "cpu", max_track=20, fast_min_distance=4.0)
    u8 = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(u8, (72, 64), 14, 180, -1)
    cv2.circle(u8, (72, 64), 4, 255, -1)
    cv2.GaussianBlur(u8, (5, 5), 0, dst=u8)
    pts = tracker.extract_features(u8, roi_center=(72.0, 64.0), roi_radius=35.0)
    print(f"[accuracy] smooth blob -> {len(pts)} validated features")
    assert len(pts) <= 12, f"accuracy gate should not over-fill smooth blob, got {len(pts)}"


def test_obs(constants, trk_mod, obs_mod, dbg_mod) -> None:
    SparseFeatureTracker = trk_mod.SparseFeatureTracker
    ObservationBuilder = obs_mod.ObservationBuilder

    class _FakeRobot:
        device = torch.device("cpu")

        class data:
            root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            root_ang_vel_b = torch.zeros(1, 3)
            root_pos_w = torch.tensor([[0.0, 0.0, 10.0]])

    builder = ObservationBuilder(1, "cpu", ang_vel_scale=1.0, depth_scale=10.0, depth_rate_scale=1.0, step_dt=0.1)
    builder.reset_depth(torch.tensor([0]), torch.tensor([10.0]))
    tracker = SparseFeatureTracker(1, "cpu", max_track=constants.MAX_TRACK_POINTS)
    tracker.set_reference(0, np.array([[64.0, 64.0], [70.0, 60.0]], dtype=np.float32))
    actions = torch.zeros(1, constants.POLICY_ACTION_DIM)
    obs = builder.build(tracker, _FakeRobot(), actions)
    parts = dbg_mod.decode_observation(obs[0])
    print(
        f"[obs] dim={obs.shape[-1]} (expected {constants.POLICY_OBS_DIM}) "
        f"valid={parts['valid_flags'].sum().item():.0f} body={parts['body'].tolist()}"
    )
    assert obs.shape[-1] == constants.POLICY_OBS_DIM


def test_lateral_obs(constants, trk_mod, obs_mod) -> None:
    SparseFeatureTracker = trk_mod.SparseFeatureTracker
    Lateral1DObservationBuilder = obs_mod.Lateral1DObservationBuilder

    class _FakeRobot:
        device = torch.device("cpu")

        class data:
            root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            root_pos_w = torch.tensor([[0.0, 0.5, 10.0]])

    builder = Lateral1DObservationBuilder(1, "cpu", max_track_points=10)
    tracker = SparseFeatureTracker(
        1, "cpu", max_track=10, image_width=110, image_height=61, coord_frame="polar"
    )
    ref_cart = np.zeros((61, 110), dtype=np.float32)
    for x in (26, 37):
        ref_cart[30:40, x - 3 : x + 4] = 1.0
    tracker.set_reference(
        0,
        np.array([[26.0, 35.0], [37.0, 35.0]], dtype=np.float32),
        ref_cart=ref_cart,
    )
    if float(tracker.ref_points[0, 0, 0].item()) < 1.0e-3:
        tracker.ref_points[0, 0] = torch.tensor([26.0, 35.0])
        tracker.ref_points[0, 1] = torch.tensor([37.0, 35.0])
    tracker.curr_points[0, 0] = torch.tensor([28.0, 35.0])
    tracker.curr_points[0, 1] = torch.tensor([39.0, 35.0])
    tracker.valid[0, :2] = True
    obs = builder.build(tracker, _FakeRobot(), torch.zeros(1, 1))
    print(
        f"[lateral_obs] dim={obs.shape[-1]} (expected {constants.LATERAL_1D_COMPACT_OBS_DIM}) "
        f"daz={obs[0, 0].item():+.4f} dr={obs[0, 1].item():+.4f}"
    )
    assert obs.shape[-1] == constants.LATERAL_1D_COMPACT_OBS_DIM
    assert abs(obs[0, 0].item()) > 1.0e-4


def test_reward(constants, trk_mod, rew_mod) -> None:
    SparseFeatureTracker = trk_mod.SparseFeatureTracker
    tracker = SparseFeatureTracker(1, "cpu", max_track=20)
    tracker.set_reference(0, np.array([[64.0, 64.0]], dtype=np.float32))
    actions = torch.tensor([[0.5, -0.2, 0.1, 0.0]])
    depth = torch.tensor([10.0])
    r = rew_mod.compute_reward(
        feature_tracker=tracker,
        actions=actions,
        depth=depth,
        w1_pixel=1.0,
        w2_action=0.1,
        w3_safety=5.0,
        min_depth=1.0,
        max_depth=20.0,
        min_valid_features=constants.MIN_VALID_FEATURES,
    )
    print(f"[reward] r={r.item():+.4f} (expect safety penalty: valid_pts=1 < {constants.MIN_VALID_FEATURES})")


def _synthetic_polar_blob(
    height: int = 64,
    width: int = 128,
    *,
    az_center: float = 0.55,
    range_center: float = 0.45,
    sigma_az: float = 0.04,
    sigma_r: float = 0.06,
    shift_az: float = 0.0,
    shift_r: float = 0.0,
) -> torch.Tensor:
    """Polar sonar with Gaussian target blob (for integration debug without OceanSim)."""
    az = torch.linspace(-0.5, 0.5, width)
    rg = torch.linspace(0.0, 1.0, height)
    azg, rgg = torch.meshgrid(az, rg, indexing="xy")
    azc = az_center + shift_az
    rc = range_center + shift_r
    blob = torch.exp(-((azg - azc) ** 2 / (2 * sigma_az ** 2) + (rgg - rc) ** 2 / (2 * sigma_r ** 2)))
    noise = torch.rand_like(blob) * 0.05
    return (blob + noise).clamp(0, 1).T  # (H_range, W_az)


def test_integration(constants, pre, trk_mod, obs_mod, rew_mod, dbg_mod) -> None:
    """Synthetic teach → track → obs → reward chain (simulates GUI teach without Isaac)."""
    from pathlib import Path

    SparseFeatureTracker = trk_mod.SparseFeatureTracker
    ObservationBuilder = obs_mod.ObservationBuilder

    out_dir = REPO / "logs/station_ref"
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_path = out_dir / "station_ref_synthetic.npz"
    png_path = out_dir / "station_ref_synthetic_cart.png"

    # --- Teach: extract ref features from synthetic sonar ---
    polar_teach = _synthetic_polar_blob().unsqueeze(0)
    cart_teach = pre.preprocess_sonar_batch(polar_teach, hori_fov_deg=90.0, max_range=8.0, cart_size=128)
    tracker = SparseFeatureTracker(1, "cpu", max_track=constants.MAX_TRACK_POINTS)
    u8_teach = tracker.to_u8_image(cart_teach[0].numpy())
    ref_pts = tracker.extract_features(u8_teach, roi_center=(64.0, 72.0), roi_radius=35.0)
    assert len(ref_pts) >= constants.MIN_VALID_FEATURES, (
        f"teach extracted only {len(ref_pts)} features (need >= {constants.MIN_VALID_FEATURES})"
    )
    tracker.set_reference(0, ref_pts[: constants.MAX_TRACK_POINTS], ref_cart=cart_teach[0].numpy())
    print(f"[integration] teach: {len(ref_pts)} FAST features, using {int(tracker.valid[0].sum())} for tracking")

    # Save synthetic snapshot (same format as real teach)
    snap_mod = _load(f"{PKG_NAME}.reference_snapshot", "reference_snapshot.py")
    snap_mod.ReferenceSnapshot(
        ref_points=tracker.ref_points[0, tracker.valid[0]].numpy(),
        sonar_image=polar_teach[0].numpy(),
        cart_image=cart_teach[0].numpy(),
        track_image=polar_teach[0].numpy(),
        perception_frame="polar",
        ref_yaw=0.0,
    ).save(snap_path)

    try:
        from PIL import Image

        Image.fromarray(u8_teach).save(png_path)
    except ImportError:
        png_path = None

    # --- Track: shift target blob over 10 steps ---
    class _FakeRobot:
        device = torch.device("cpu")

        class data:
            root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            root_ang_vel_b = torch.zeros(1, 3)
            root_pos_w = torch.tensor([[0.0, 0.0, 10.0]])

    builder = ObservationBuilder(1, "cpu", ang_vel_scale=1.0, depth_scale=10.0, depth_rate_scale=1.0, step_dt=0.1)
    builder.reset_depth(torch.tensor([0]), torch.tensor([10.0]))
    actions = torch.zeros(1, constants.POLICY_ACTION_DIM)

    print("[integration] track loop (synthetic AUV offset → blob shift):")
    for step in range(10):
        shift_az = step * 0.008
        shift_r = step * 0.003
        polar = _synthetic_polar_blob(shift_az=shift_az, shift_r=shift_r).unsqueeze(0)
        cart = pre.preprocess_sonar_batch(polar, hori_fov_deg=90.0, max_range=8.0, cart_size=128)
        tracker.update(cart)
        obs = builder.build(tracker, _FakeRobot(), actions)
        r = rew_mod.compute_reward(
            feature_tracker=tracker,
            actions=actions,
            depth=torch.tensor([10.0]),
            w1_pixel=1.0,
            w2_action=0.1,
            w3_safety=5.0,
            min_depth=1.0,
            max_depth=20.0,
            min_valid_features=constants.MIN_VALID_FEATURES,
        )
        valid_n = int(tracker.valid_count()[0].item())
        pixel_err = float(tracker.normalized_pixel_error()[0].item())
        print(f"  step={step:2d} valid={valid_n:2d} pixel_err={pixel_err:.4f} reward={r.item():+.4f} obs_dim={obs.shape[-1]}")
        assert obs.shape[-1] == constants.POLICY_OBS_DIM
        if valid_n >= constants.MIN_VALID_FEATURES:
            assert r.item() > -1.0, "reward should not hit safety when enough valid features"

    loaded = snap_mod.ReferenceSnapshot.load(snap_path)
    assert len(loaded.ref_points) >= constants.MIN_VALID_FEATURES
    print(f"[integration] snapshot OK: {snap_path} ({len(loaded.ref_points)} ref_points)")
    if png_path:
        print(f"[integration] cart preview: {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Sonar IBVS-DRL modules offline.")
    parser.add_argument(
        "--layer",
        choices=("preprocess", "tracker", "obs", "reward", "integration", "all"),
        default="all",
    )
    args = parser.parse_args()

    constants, pre, trk_mod, obs_mod, rew_mod, dbg_mod = _load_modules()

    runners = {
        "preprocess": lambda: test_preprocess(pre),
        "tracker": lambda: (test_tracker(trk_mod), test_contour_blob(trk_mod), test_polar_tracker(trk_mod, pre)),
        "obs": lambda: (test_obs(constants, trk_mod, obs_mod, dbg_mod), test_lateral_obs(constants, trk_mod, obs_mod)),
        "reward": lambda: test_reward(constants, trk_mod, rew_mod),
        "integration": lambda: test_integration(constants, pre, trk_mod, obs_mod, rew_mod, dbg_mod),
    }
    if args.layer == "all":
        for name, fn in runners.items():
            print(f"\n--- {name} ---")
            fn()
    else:
        runners[args.layer]()
    print("\nAll requested checks passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
