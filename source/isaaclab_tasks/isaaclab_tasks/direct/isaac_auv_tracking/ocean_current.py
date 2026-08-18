"""Batched torch ocean current fields for Isaac-AUV-Tracking.

World frame (Z-up, depth ~8). Unlike HoloOcean, positive Z does not zero out currents.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

LEVEL_STRENGTH: dict[str, float] = {
    "off": 0.0,
    "mild": 0.15,
    "medium": 0.4,
    "strong": 0.8,
}


@runtime_checkable
class OceanCurrentField(Protocol):
    """Callable field: (pos_w[N,3], t) -> v[N,3] in world frame m/s."""

    def __call__(self, pos_w: Tensor, t: Tensor | float) -> Tensor: ...


def resolve_strength(strength: float | None, level: str | None) -> float:
    """Resolve effective current strength; explicit ``strength`` wins over ``level``."""
    if strength is not None:
        return float(strength)
    if level is not None:
        if level not in LEVEL_STRENGTH:
            raise ValueError(f"Unknown ocean current level: {level!r}. Available: {list(LEVEL_STRENGTH)}")
        return LEVEL_STRENGTH[level]
    return 0.0


def clip_current_velocity(v: Tensor, clip_speed: float) -> Tensor:
    """Clip velocity vectors to ``clip_speed`` (m/s) preserving direction."""
    norms = v.norm(dim=-1, keepdim=True)
    scale = torch.where(
        norms > clip_speed,
        clip_speed / norms.clamp_min(1e-6),
        torch.ones_like(norms),
    )
    return v * scale


def _cfg_attr(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _as_tensor(values: list[float] | tuple[float, ...], device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor(list(values), device=device, dtype=dtype)


def _normalize(v: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return v / v.norm(dim=dim, keepdim=True).clamp_min(eps)


def _time_tensor(t: Tensor | float, n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(t, Tensor):
        if t.ndim == 0:
            return t.to(device=device, dtype=dtype).expand(n)
        return t.to(device=device, dtype=dtype)
    return torch.full((n,), float(t), device=device, dtype=dtype)


def _uniform_field(pos_w: Tensor, t: Tensor | float, strength: float, params: dict[str, Any]) -> Tensor:
    n = pos_w.shape[0]
    device, dtype = pos_w.device, pos_w.dtype

    if "velocity" in params:
        base = _as_tensor(params["velocity"], device, dtype)
    else:
        direction = _as_tensor(params.get("direction", [1.0, 0.0, 0.0]), device, dtype)
        base = strength * _normalize(direction)

    v = base.unsqueeze(0).expand(n, -1)

    if params.get("time_modulation", False):
        t_vec = _time_tensor(t, n, device, dtype)
        period = float(params.get("modulation_period", 10.0))
        amplitude = float(params.get("modulation_amplitude", 0.5))
        factor = 1.0 + amplitude * torch.sin(2.0 * math.pi * t_vec / period)
        v = v * factor.unsqueeze(-1)

    return v


def _vortex_field(pos_w: Tensor, t: Tensor | float, strength: float, params: dict[str, Any]) -> Tensor:
    device, dtype = pos_w.device, pos_w.dtype
    n = pos_w.shape[0]
    center = _as_tensor(params.get("center", [0.0, 0.0, 8.0]), device, dtype)
    radius = float(params.get("radius", 5.0))
    angular_velocity = float(params.get("angular_velocity", 0.0))
    decay_rate = float(params.get("decay_rate", 1.0))

    rel_pos = pos_w - center.unsqueeze(0)
    x = rel_pos[..., 0]
    y = rel_pos[..., 1]

    r_horizontal_squared = x * x + y * y + 1e-6
    r_horizontal = torch.sqrt(r_horizontal_squared)
    velocity_magnitude = strength * (radius / (r_horizontal + radius)) ** decay_rate

    t_vec = _time_tensor(t, n, device, dtype)
    time_phase = angular_velocity * t_vec
    cos_phase = torch.cos(time_phase)
    sin_phase = torch.sin(time_phase)

    dx_base = -y / r_horizontal * velocity_magnitude
    dy_base = x / r_horizontal * velocity_magnitude
    dx = dx_base * cos_phase - dy_base * sin_phase
    dy = dx_base * sin_phase + dy_base * cos_phase
    dz = 0.2 * strength * torch.cos(0.1 * r_horizontal_squared + time_phase)

    return torch.stack([dx, dy, dz], dim=-1)


def _sine_wave_field(pos_w: Tensor, t: Tensor | float, strength: float, params: dict[str, Any]) -> Tensor:
    device, dtype = pos_w.device, pos_w.dtype
    n = pos_w.shape[0]
    direction = _normalize(_as_tensor(params.get("direction", [1.0, 0.0, 0.0]), device, dtype))
    wavelength = float(params.get("wavelength", 10.0))
    period = float(params.get("period", 5.0))
    phase = float(params.get("phase", 0.0))

    wave_number = 2.0 * math.pi / wavelength
    angular_frequency = 2.0 * math.pi / period
    position_along_wave = (pos_w * direction).sum(dim=-1)
    t_vec = _time_tensor(t, n, device, dtype)
    wave_phase = wave_number * position_along_wave - angular_frequency * t_vec + phase

    velocity_magnitude = strength * torch.sin(wave_phase)
    velocity = direction.unsqueeze(0) * velocity_magnitude.unsqueeze(-1)

    if abs(direction[0].item()) > 0.1 or abs(direction[1].item()) > 0.1:
        perp = torch.stack([-direction[1], direction[0], torch.zeros((), device=device, dtype=dtype)])
        perp = _normalize(perp)
        perp_magnitude = strength * 0.3 * torch.cos(wave_phase)
        velocity = velocity + perp.unsqueeze(0) * perp_magnitude.unsqueeze(-1)

    return velocity


def _random_field(pos_w: Tensor, t: Tensor | float, strength: float, params: dict[str, Any]) -> Tensor:
    device, dtype = pos_w.device, pos_w.dtype
    n = pos_w.shape[0]
    max_velocity = float(params.get("max_velocity", strength))
    spatial_scale = float(params.get("spatial_scale", 10.0))
    temporal_scale = float(params.get("temporal_scale", 5.0))
    correlation_length = float(params.get("correlation_length", 3.0))
    seed = float(params.get("seed", 0.0))

    t_vec = _time_tensor(t, n, device, dtype)
    x_coord = pos_w[..., 0] / spatial_scale + t_vec / temporal_scale + seed
    y_coord = pos_w[..., 1] / spatial_scale + t_vec / temporal_scale + seed * 1.37
    z_coord = pos_w[..., 2] / spatial_scale + t_vec / temporal_scale + seed * 2.71

    dx = max_velocity * torch.sin(x_coord * 2.3 + torch.cos(y_coord * 1.7)) * torch.cos(t_vec / temporal_scale)
    dy = max_velocity * torch.sin(y_coord * 1.9 + torch.cos(z_coord * 2.1)) * torch.cos(t_vec / temporal_scale + 1.0)
    dz = (
        max_velocity
        * 0.3
        * torch.sin(z_coord * 2.7 + torch.cos(x_coord * 1.5))
        * torch.cos(t_vec / temporal_scale + 2.0)
    )

    dist_factor = torch.exp(-pos_w[..., :2].norm(dim=-1) / correlation_length)
    return torch.stack([dx, dy, dz], dim=-1) * dist_factor.unsqueeze(-1)


_FIELD_BUILDERS = {
    "uniform": _uniform_field,
    "vortex": _vortex_field,
    "sine_wave": _sine_wave_field,
    "random": _random_field,
}


def create_ocean_current_field(cfg: Any) -> OceanCurrentField | None:
    """Build a clipped current field from config, or ``None`` when disabled."""
    if not _cfg_attr(cfg, "enabled", False):
        return None

    field_type = _cfg_attr(cfg, "type", "uniform")
    if field_type not in _FIELD_BUILDERS:
        raise ValueError(f"Unknown ocean current type: {field_type!r}. Available: {list(_FIELD_BUILDERS)}")

    strength = resolve_strength(_cfg_attr(cfg, "strength", None), _cfg_attr(cfg, "level", None))
    params = _cfg_attr(cfg, "params", {}) or {}
    clip_speed = float(_cfg_attr(cfg, "clip_speed", 5.0))
    builder = _FIELD_BUILDERS[field_type]

    def field(pos_w: Tensor, t: Tensor | float) -> Tensor:
        pos = pos_w.to(dtype=torch.float32) if pos_w.dtype != torch.float32 else pos_w
        v = builder(pos, t, strength, params)
        return clip_current_velocity(v, clip_speed)

    return field
