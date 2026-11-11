# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thruster allocation: body wrench (Fx,Fy,Fz,Mx,My,Mz) -> 6 PWM commands.

Uses WarpAUV thruster geometry from :mod:`isaac-auv-env.thruster_dynamics` and a
Moore-Penrose pseudoinverse (same wrench model as ``WarpAUVEnv._compute_dynamics``).

Under thruster saturation, depth + attitude (Fz, Mx, My, Mz) are allocated first;
surge/sway (Fx, Fy) are scaled into remaining PWM headroom so lateral/range demands
cannot erase station-keeping in Z / attitude.
"""

from __future__ import annotations

import importlib

import torch


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vector by quaternion (w, x, y, z)."""
    q_w = quat[..., 0:1]
    q_vec = quat[..., 1:4]
    t = 2.0 * torch.cross(q_vec, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_vec, t, dim=-1)


# 缓存推进器几何，避免每次 env 创建时重复 import/print
_ALLOC_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

# DoF priority: heave first; attitude optional. Fx/Fy fill leftover PWM.
_PRIORITY_HEAVE = (2,)  # Fz
_PRIORITY_ATT = (3, 4, 5)  # Mx, My, Mz
_SECONDARY_XY = (0, 1)  # Fx, Fy


def build_thruster_allocation_matrix(
    device: torch.device | str,
    damping: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``B`` and ``B_pinv`` with ``wrench_b = B @ u`` (6x6)."""
    key = f"{device}"
    if key in _ALLOC_CACHE:
        return _ALLOC_CACHE[key]

    td = importlib.import_module("isaaclab_tasks.direct.isaac-auv-env.thruster_dynamics")
    offsets, quats = td.get_thruster_com_and_orientations(device)

    force_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    cols: list[torch.Tensor] = []
    for i in range(6):
        f_b = _quat_apply(quats[i : i + 1], force_axis.unsqueeze(0)).squeeze(0)
        tau_b = torch.cross(offsets[i], f_b, dim=-1)
        cols.append(torch.cat([f_b, tau_b]))

    b_mat = torch.stack(cols, dim=1)
    gram = b_mat @ b_mat.T + damping * torch.eye(6, device=device, dtype=b_mat.dtype)
    b_pinv = b_mat.T @ torch.linalg.inv(gram)
    _ALLOC_CACHE[key] = (b_mat, b_pinv)
    return b_mat, b_pinv


def build_allocation_from_geometry(
    offsets: torch.Tensor,
    quats: torch.Tensor,
    damping: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``B`` / ``B_pinv`` for an arbitrary thruster set.

    ``offsets`` is ``(n, 3)`` COM->thruster vectors in metres and ``quats`` is ``(n, 4)``
    mount rotations carrying the local +X axis onto the thrust direction. Returns ``B``
    of shape ``(6, n)`` with ``wrench_b = B @ u`` and its Moore-Penrose pseudoinverse
    ``(n, 6)``, so the same code serves the 6-thruster WarpAUV and the 8-thruster GS-ROV.
    """
    n = int(offsets.shape[0])
    force_axis = torch.tensor([1.0, 0.0, 0.0], device=offsets.device, dtype=offsets.dtype)
    f_b = _quat_apply(quats, force_axis.unsqueeze(0).expand(n, 3))
    tau_b = torch.cross(offsets, f_b, dim=-1)
    b_mat = torch.cat([f_b, tau_b], dim=1).transpose(0, 1)  # (6, n)
    gram = b_mat @ b_mat.T + damping * torch.eye(6, device=b_mat.device, dtype=b_mat.dtype)
    b_pinv = b_mat.T @ torch.linalg.inv(gram)  # (n, 6)
    return b_mat, b_pinv


def _max_secondary_scale(u_pri: torch.Tensor, u_sec: torch.Tensor, clamp: float) -> torch.Tensor:
    """Per-env scale in [0, 1] so ``|u_pri + s * u_sec| <= clamp`` on all channels."""
    eps = 1.0e-6
    # Positive u_sec: s <= (clamp - u_pri) / u_sec
    # Negative u_sec: s <= (-clamp - u_pri) / u_sec
    room_hi = (clamp - u_pri) / u_sec.clamp(min=eps)
    room_lo = (-clamp - u_pri) / u_sec.clamp(max=-eps)
    lim = torch.where(u_sec > eps, room_hi, torch.where(u_sec < -eps, room_lo, torch.ones_like(u_sec)))
    # Already over clamp from priority alone → no secondary budget on that channel.
    lim = torch.where(lim < 0.0, torch.zeros_like(lim), lim)
    return lim.amin(dim=-1, keepdim=True).clamp(0.0, 1.0)


def allocate_thrusters(
    wrench_b: torch.Tensor,
    b_pinv: torch.Tensor,
    clamp: float = 1.0,
    *,
    priority: bool = True,
    priority_attitude: bool = True,
) -> torch.Tensor:
    """Map desired body wrench ``(N, 6)`` to PWM ``(N, 6)``.

    When ``priority=True`` (default), allocate heave (+ optional attitude) first,
    then scale surge/sway into leftover thruster headroom. Without priority, plain
    pinv + clamp (legacy) can starve depth when horizontal residual saturates.
    """
    if not priority:
        return torch.clamp(wrench_b @ b_pinv.T, -clamp, clamp)

    pri_dofs = _PRIORITY_HEAVE + (_PRIORITY_ATT if priority_attitude else ())
    w_pri = torch.zeros_like(wrench_b)
    w_sec = torch.zeros_like(wrench_b)
    for i in pri_dofs:
        w_pri[:, i] = wrench_b[:, i]
    for i in _SECONDARY_XY:
        w_sec[:, i] = wrench_b[:, i]
    if not priority_attitude:
        for i in _PRIORITY_ATT:
            w_sec[:, i] = wrench_b[:, i]

    u_pri = torch.clamp(w_pri @ b_pinv.T, -clamp, clamp)
    u_sec = w_sec @ b_pinv.T
    scale = _max_secondary_scale(u_pri, u_sec, clamp)
    return torch.clamp(u_pri + scale * u_sec, -clamp, clamp)
