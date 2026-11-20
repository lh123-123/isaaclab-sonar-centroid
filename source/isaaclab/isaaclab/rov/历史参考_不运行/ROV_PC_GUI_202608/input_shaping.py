#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared input shaping helpers for physical joystick axes."""

from __future__ import annotations

import math


def remap_deadzone(value: float, deadzone: float = 0.15) -> float:
    """Remove a centered deadzone and continuously remap the remainder to 0..1.

    Values at the edge of the deadzone map to zero, while a full-scale input
    still maps to full scale.  This avoids the old 0 -> deadzone-sized jump.
    """
    value = float(value)
    deadzone = float(deadzone)
    if not math.isfinite(value):
        return 0.0
    if not 0.0 <= deadzone < 1.0:
        raise ValueError("deadzone must be in the range 0 <= deadzone < 1")

    value = max(-1.0, min(1.0, value))
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0

    remapped = (magnitude - deadzone) / (1.0 - deadzone)
    return math.copysign(remapped, value)


def apply_response_curve(value: float, exponent: float = 2.2) -> float:
    """Make small manual inputs gentler while retaining full-scale output.

    ``value`` is expected to have already passed through ``remap_deadzone``.
    An exponent above 1.0 reduces small and mid-travel commands; 1.0 is the
    original linear response.  Full input remains exactly full output.
    """
    value = float(value)
    exponent = float(exponent)
    if not math.isfinite(value):
        return 0.0
    if not math.isfinite(exponent) or not 1.0 <= exponent <= 4.0:
        raise ValueError("response exponent must be in the range 1.0..4.0")

    value = max(-1.0, min(1.0, value))
    return math.copysign(abs(value) ** exponent, value)


def select_cardinal_axis(x: float, y: float) -> tuple[float, float]:
    """Keep only one stick axis, yielding four single-action sectors.

    The stick disc is split by the two 45-degree diagonals.  Up/down sectors
    keep ``y`` and suppress ``x``; left/right sectors keep ``x`` and suppress
    ``y``.  An exact diagonal deterministically selects up/down, so a stick
    can never produce two simultaneous motion commands.
    """
    x = float(x)
    y = float(y)
    if not math.isfinite(x) or not math.isfinite(y):
        return 0.0, 0.0
    if abs(y) >= abs(x):
        return 0.0, y
    return x, 0.0
