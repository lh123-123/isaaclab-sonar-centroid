# test_ocean_current.py — lives outside the isaaclab_tasks package so bare pytest
# does not import Isaac Sim via isaaclab_tasks.__init__.
import importlib.util
from pathlib import Path

import torch

_OCEAN_CURRENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_tasks"
    / "direct"
    / "isaac_auv_tracking"
    / "ocean_current.py"
)
_spec = importlib.util.spec_from_file_location("ocean_current", _OCEAN_CURRENT_PATH)
assert _spec and _spec.loader
_ocean_current = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ocean_current)

clip_current_velocity = _ocean_current.clip_current_velocity
create_ocean_current_field = _ocean_current.create_ocean_current_field
resolve_strength = _ocean_current.resolve_strength
LEVEL_STRENGTH = _ocean_current.LEVEL_STRENGTH


def _make_cfg(**overrides):
    class Cfg:
        enabled = True
        type = "uniform"
        strength = 0.5
        level = None
        clip_speed = 5.0
        params = {}

    for key, value in overrides.items():
        setattr(Cfg, key, value)
    return Cfg()


def test_resolve_strength_level():
    assert resolve_strength(None, "mild") == LEVEL_STRENGTH["mild"]
    assert resolve_strength(0.5, None) == 0.5
    assert resolve_strength(0.5, "strong") == 0.5  # explicit strength wins


def test_clip():
    v = torch.tensor([[3.0, 4.0, 0.0]])  # norm 5
    out = clip_current_velocity(v, clip_speed=2.5)
    assert torch.allclose(out.norm(dim=-1), torch.tensor([2.5]), atol=1e-5)


def test_uniform_batch_ignores_holo_depth_sign():
    # Isaac z≈8 must still get current (unlike Holo z>0 → 0)
    field = create_ocean_current_field(
        _make_cfg(type="uniform", params={"direction": [1.0, 0.0, 0.0]})
    )
    pos = torch.tensor([[0.0, 0.0, 8.0], [1.0, 0.0, 8.0]])
    v = field(pos, 0.0)
    assert v.shape == (2, 3)
    assert torch.allclose(v[:, 0], torch.tensor([0.5, 0.5]), atol=1e-5)


def test_disabled_returns_none():
    assert create_ocean_current_field(_make_cfg(enabled=False)) is None


def test_vortex_shape_finite():
    field = create_ocean_current_field(_make_cfg(type="vortex", strength=0.4))
    pos = torch.tensor([[1.0, 0.0, 8.0], [0.0, 2.0, 8.0], [-3.0, 1.0, 8.0]])
    v = field(pos, 0.0)
    assert v.shape == (3, 3)
    assert torch.isfinite(v).all()


def test_sine_wave_shape_finite():
    field = create_ocean_current_field(
        _make_cfg(type="sine_wave", strength=0.3, params={"direction": [1.0, 0.0, 0.0]})
    )
    pos = torch.tensor([[0.0, 0.0, 8.0], [2.5, 0.0, 8.0]])
    v = field(pos, 1.0)
    assert v.shape == (2, 3)
    assert torch.isfinite(v).all()


def test_random_shape_finite():
    field = create_ocean_current_field(
        _make_cfg(type="random", strength=0.25, params={"seed": 42.0})
    )
    pos = torch.tensor([[0.0, 0.0, 8.0], [1.0, -1.0, 8.0], [3.0, 2.0, 8.0]])
    v = field(pos, 0.5)
    assert v.shape == (3, 3)
    assert torch.isfinite(v).all()


def test_factory_default_clip_speed():
    field = create_ocean_current_field(
        _make_cfg(
            type="uniform",
            strength=100.0,
            params={"direction": [1.0, 0.0, 0.0]},
        )
    )
    pos = torch.tensor([[0.0, 0.0, 8.0]])
    v = field(pos, 0.0)
    assert torch.allclose(v.norm(dim=-1), torch.tensor([5.0]), atol=1e-5)
