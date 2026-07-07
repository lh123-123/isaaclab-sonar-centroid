# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Target lateral motion profiles (shared by env + eval scripts)."""

from __future__ import annotations

import math


def motion_leg_steps(sway_amp: float, vy: float, step_dt: float) -> int:
    return max(int(round(abs(sway_amp) / max(abs(vy) * step_dt, 1.0e-6))), 1)


def round_trip_leg_steps(sway_amp: float, vy: float, step_dt: float) -> int:
    return motion_leg_steps(sway_amp, vy, step_dt)


def round_trip_vy(step: int, *, sway_amp: float, vy: float, step_dt: float) -> float:
    leg = motion_leg_steps(sway_amp, vy, step_dt)
    if step < leg:
        return float(vy)
    if step < 2 * leg:
        return float(-vy)
    return 0.0


def round_trip_dy(step: int, *, sway_amp: float, vy: float, step_dt: float) -> float:
    dy = 0.0
    for s in range(step):
        dy += round_trip_vy(s, sway_amp=sway_amp, vy=vy, step_dt=step_dt) * step_dt
    total = 2 * motion_leg_steps(sway_amp, vy, step_dt)
    return float(max(0.0, min(sway_amp, dy)) if step <= total else 0.0)


def round_trip_cycle_steps(sway_amp: float, vy: float, step_dt: float) -> int:
    return 2 * motion_leg_steps(sway_amp, vy, step_dt)


def triangle_vy(step: int, *, sway_amp: float, vy: float, step_dt: float) -> float:
    """Constant-speed legs: 0 → +A → 0 → −A → 0."""
    leg = motion_leg_steps(sway_amp, vy, step_dt)
    if step < leg:
        return float(vy)
    if step < 2 * leg:
        return float(-vy)
    if step < 3 * leg:
        return float(-vy)
    if step < 4 * leg:
        return float(vy)
    return 0.0


def triangle_dy(step: int, *, sway_amp: float, vy: float, step_dt: float) -> float:
    dy = 0.0
    for s in range(step):
        dy += triangle_vy(s, sway_amp=sway_amp, vy=vy, step_dt=step_dt) * step_dt
    total = triangle_cycle_steps(sway_amp, vy, step_dt)
    return float(max(-sway_amp, min(sway_amp, dy)) if step <= total else 0.0)


def triangle_cycle_steps(sway_amp: float, vy: float, step_dt: float) -> int:
    return 4 * motion_leg_steps(sway_amp, vy, step_dt)


def triangle_cycle_s(sway_amp: float, vy: float, step_dt: float) -> float:
    return triangle_cycle_steps(sway_amp, vy, step_dt) * step_dt


def sine_peak_vy(amp: float, period_s: float) -> float:
    """Peak |vy| for dy = amp·sin(2πt/T)."""
    return abs(amp) * 2.0 * math.pi / max(period_s, 1.0e-3)


def sine_period_for_peak_vy(amp: float, vy_peak: float) -> float:
    """Period (s) for one round trip 0 → +amp → 0 → −amp → 0 at the given peak speed."""
    return 2.0 * math.pi * abs(amp) / max(abs(vy_peak), 1.0e-6)


def sine_cycle_s(amp: float, vy_peak: float) -> float:
    """Episode duration for one sine round trip."""
    return sine_period_for_peak_vy(amp, vy_peak)


def sine_dy_and_vy(
    step: int, *, amp: float, period_s: float, step_dt: float, phase: float = 0.0
) -> tuple[float, float]:
    omega = 2.0 * math.pi / max(period_s, 1.0e-3)
    t = step * step_dt
    dy = amp * math.sin(omega * t + phase)
    vy = amp * omega * math.cos(omega * t + phase)
    return dy, vy
