# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thruster allocation: body wrench (Fx,Fy,Fz,Mx,My,Mz) -> 6 PWM commands.

Uses WarpAUV thruster geometry from :mod:`isaac-auv-env.thruster_dynamics` and a
Moore-Penrose pseudoinverse (same wrench model as ``WarpAUVEnv._compute_dynamics``).
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


def allocate_thrusters(
    wrench_b: torch.Tensor,
    b_pinv: torch.Tensor,
    clamp: float = 1.0,
) -> torch.Tensor:
    """Map desired body wrench ``(N, 6)`` to PWM ``(N, 6)`` via ``u = wrench @ B_pinv.T``."""
    pwm = wrench_b @ b_pinv.T
    return torch.clamp(pwm, -clamp, clamp)
