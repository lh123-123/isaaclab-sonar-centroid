# test_auv_tracking_flow_obs.py — Isaac-free unit tests for Direct-v2 flow obs.
import importlib.util
from pathlib import Path

import torch

_FLOW_OBS_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_tasks"
    / "direct"
    / "isaac_auv_tracking"
    / "flow_obs.py"
)
_spec = importlib.util.spec_from_file_location("auv_tracking_flow_obs", _FLOW_OBS_PATH)
assert _spec and _spec.loader
_flow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_flow)

MIN_RANGE = 0.5
MAX_RANGE = 4.0
HORI_FOV_DEG = 120.0
H, W = 70, 61


def test_reset_flow_invalid():
    r = torch.tensor([2.0])
    a = torch.tensor([0.1])
    valid = torch.tensor([True])
    ready = torch.tensor([False])
    dr, dalpha, flow_valid = _flow.centroid_displacement(r, a, valid, r, a, valid, ready)
    assert not bool(flow_valid[0])
    assert float(dr[0]) == 0.0
    assert float(dalpha[0]) == 0.0


def test_synthetic_blob_delta_range_and_bearing():
    # Mid-lobe blob, then shift +4 range bins and -3 azimuth columns.
    row0, col0 = 30.0, 30.0
    d_row, d_col = 4.0, -3.0
    prev = _flow.make_gaussian_blob(height=H, width=W, row=row0, col=col0).unsqueeze(0)
    curr = _flow.make_gaussian_blob(height=H, width=W, row=row0 + d_row, col=col0 + d_col).unsqueeze(0)

    r0, a0, v0 = _flow.intensity_centroid_polar(
        prev, min_range=MIN_RANGE, max_range=MAX_RANGE, hori_fov_deg=HORI_FOV_DEG
    )
    r1, a1, v1 = _flow.intensity_centroid_polar(
        curr, min_range=MIN_RANGE, max_range=MAX_RANGE, hori_fov_deg=HORI_FOV_DEG
    )
    assert bool(v0[0]) and bool(v1[0])

    ready = torch.ones(1, dtype=torch.bool)
    dr, dalpha, flow_valid = _flow.centroid_displacement(r1, a1, v1, r0, a0, v0, ready)
    assert bool(flow_valid[0])

    h_max = float(H - 1)
    span = MAX_RANGE - MIN_RANGE
    expected_dr = d_row / h_max * span
    # col_to_az: az_deg = (W/2 - col) * fov / W  (sonar_meas convention)
    expected_da = -d_col * (HORI_FOV_DEG / float(W)) * (3.141592653589793 / 180.0)

    assert abs(float(dr[0]) - expected_dr) < 0.05
    assert float(dr[0]) > 0.0
    assert abs(float(dalpha[0]) - expected_da) < 0.05
    assert float(dalpha[0]) > 0.0


def test_residual_zero_then_nonzero():
    blob = _flow.make_gaussian_blob(height=H, width=W, row=28.0, col=30.0)
    curr = _flow.normalize_polar_fixed(blob.unsqueeze(0))
    prev = curr.clone()
    ready0 = torch.zeros(1, dtype=torch.bool)
    res0 = _flow.photometric_residual(curr, prev, ready0)
    assert float(res0.abs().max()) == 0.0

    shifted = _flow.make_gaussian_blob(height=H, width=W, row=34.0, col=30.0)
    curr2 = _flow.normalize_polar_fixed(shifted.unsqueeze(0))
    ready1 = torch.ones(1, dtype=torch.bool)
    res1 = _flow.photometric_residual(curr2, prev, ready1)
    assert float(res1.abs().max()) > 0.05

    img = _flow.pack_flow_image(curr2, res1, size=(64, 64))
    assert tuple(img.shape) == (1, 3, 64, 64)
    assert img.dtype == torch.uint8
    # No-motion residual maps to 128; motion should leave ch1 away from 128 somewhere.
    assert int((img[0, 1].int() != 128).sum()) > 0


def test_reference_pixel_dead_ahead_mid_range():
    row, col = _flow.reference_polar_pixel(
        r_star=2.0,
        min_range=MIN_RANGE,
        max_range=MAX_RANGE,
        polar_height=H,
        polar_width=W,
    )
    assert abs(col - 0.5 * (W - 1)) < 1e-6
    assert 25.0 < row < 35.0
