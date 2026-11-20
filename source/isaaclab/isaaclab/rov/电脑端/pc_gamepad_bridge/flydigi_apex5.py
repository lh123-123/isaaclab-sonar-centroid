"""Flydigi APEX 5 raw-HID decoding and ArduSub MANUAL_CONTROL mapping.

The vendor interface keeps M1-M4/LM/RM independent.  QGroundControl 5.1.4
only exposes the standard XInput buttons for this controller, so the PC bridge
uses this interface directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import FrozenSet


VENDOR_ID = 0x37D7
PRODUCT_ID = 0x2501
VENDOR_USAGE_PAGE = 0xFFA0

EXTRA_BUTTON_INDEX = {
    "M1": 15,
    "M2": 16,
    "M3": 17,
    "M4": 18,
    "LM": 19,
    "RM": 20,
}

_EXTRA_MASKS = {
    "M1": 0x04,
    "M2": 0x08,
    "M3": 0x10,
    "M4": 0x20,
    "LM": 0x40,
    "RM": 0x80,
}


def _signed16(low: int, high: int) -> int:
    value = low | (high << 8)
    return value - 0x10000 if value & 0x8000 else value


def _axis(value: int) -> float:
    if value < 0:
        return max(-1.0, value / 32768.0)
    return min(1.0, value / 32767.0)


def _radial_deadzone(x: float, y: float, size: float) -> tuple[float, float]:
    """Apply one circular deadzone while preserving the requested direction."""
    magnitude = math.hypot(x, y)
    if magnitude <= size:
        return 0.0, 0.0
    output_magnitude = min(1.0, (magnitude - size) / (1.0 - size))
    scale = output_magnitude / magnitude
    return x * scale, y * scale


def _soft_axis_snap(
    x: float,
    y: float,
    inner_deg: float,
    outer_deg: float,
) -> tuple[float, float]:
    """Suppress small cross-axis errors near cardinals without losing diagonals."""
    if not 0.0 <= inner_deg < outer_deg < 45.0:
        raise ValueError("axis snap angles must satisfy 0 <= inner < outer < 45")

    abs_x = abs(x)
    abs_y = abs(y)
    if abs_x == 0.0 or abs_y == 0.0:
        return x, y

    x_is_major = abs_x >= abs_y
    major = abs_x if x_is_major else abs_y
    minor = abs_y if x_is_major else abs_x
    angle_deg = math.degrees(math.atan2(minor, major))
    if angle_deg <= inner_deg:
        minor_scale = 0.0
    elif angle_deg >= outer_deg:
        return x, y
    else:
        progress = (angle_deg - inner_deg) / (outer_deg - inner_deg)
        minor_scale = progress * progress * (3.0 - 2.0 * progress)

    if x_is_major:
        snapped = (x, y * minor_scale)
    else:
        snapped = (x * minor_scale, y)
    # Assistance changes direction, not requested magnitude. A full stick
    # near a cardinal must still reach full authority after the radial deadzone.
    scale = math.hypot(x, y) / math.hypot(*snapped)
    return snapped[0] * scale, snapped[1] * scale


def _trigger_deadzone(value: float, size: float) -> float:
    if value <= size:
        return 0.0
    return min(1.0, (value - size) / (1.0 - size))


@dataclass(frozen=True)
class Apex5State:
    left_x: float
    left_y: float
    right_x: float
    right_y: float
    left_trigger: float
    right_trigger: float
    buttons: int
    extra_buttons: FrozenSet[str]

    @property
    def vertical(self) -> float:
        """Continuous vertical demand: positive up, negative down."""
        return self.right_trigger - self.left_trigger

    @classmethod
    def neutral(cls) -> "Apex5State":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, frozenset())


@dataclass(frozen=True)
class ManualControlCommand:
    x: int
    y: int
    z: int
    r: int
    buttons: int
    buttons2: int
    enabled_extensions: int
    s: int
    t: int
    aux1: int = 0
    aux2: int = 0
    aux3: int = 0
    aux4: int = 0
    aux5: int = 0
    aux6: int = 0


def parse_input_report(report: bytes) -> Apex5State:
    """Decode a 32-byte APEX 5 v2 vendor input report."""
    if len(report) < 32 or report[1:4] != bytes((0x5A, 0xA5, 0xEF)):
        raise ValueError("not a Flydigi APEX 5 input report")

    # SDL's APEX 5 driver uses the same axis orientation.  Positive Y means
    # the physical stick is pushed forward/up.
    left_x = _axis(_signed16(report[4], report[5]))
    left_y = _axis(-_signed16(report[6], report[7]))
    right_x = _axis(_signed16(report[8], report[9]))
    right_y = _axis(-_signed16(report[10], report[11]))

    buttons = 0
    byte11 = report[12]
    byte12 = report[13]
    byte13 = report[14]
    byte14 = report[15]

    standard = (
        (0, byte11, 0x10),  # A
        (1, byte11, 0x20),  # B
        (2, byte11, 0x80),  # X
        (3, byte12, 0x01),  # Y
        (4, byte11, 0x40),  # Back/View
        (5, byte14, 0x08),  # Guide
        (6, byte12, 0x02),  # Start/Menu
        (7, byte12, 0x40),  # L3
        (8, byte12, 0x80),  # R3
        (9, byte12, 0x04),  # LB
        (10, byte12, 0x08), # RB
    )
    for index, value, mask in standard:
        if value & mask:
            buttons |= 1 << index

    dpad = byte11 & 0x0F
    for index, mask in ((11, 0x01), (12, 0x04), (13, 0x08), (14, 0x02)):
        if dpad & mask:
            buttons |= 1 << index

    pressed_extra = set()
    for name, mask in _EXTRA_MASKS.items():
        if byte13 & mask:
            pressed_extra.add(name)
            buttons |= 1 << EXTRA_BUTTON_INDEX[name]

    return Apex5State(
        left_x=left_x,
        left_y=left_y,
        right_x=right_x,
        right_y=right_y,
        left_trigger=report[16] / 255.0,
        right_trigger=report[17] / 255.0,
        buttons=buttons,
        extra_buttons=frozenset(pressed_extra),
    )


def build_manual_control(
    state: Apex5State,
    *,
    stick_deadzone: float = 0.04,
    trigger_deadzone: float = 0.0,
    vertical_scale: float = 0.4,
    motion_scale: float = 1.0,
    vertical_motion_scale: float | None = None,
    axis_snap_inner_deg: float = 8.0,
    axis_snap_outer_deg: float = 18.0,
    yaw_response_exponent: float = 1.0,
    allow_arm: bool = False,
    button_mask: int | None = None,
) -> ManualControlCommand:
    """Map controller state to ArduSub's MANUAL_CONTROL fields.

    RT is up and LT is down.  Both sticks use smooth cardinal-axis assistance:
    small accidental cross-axis input is removed, but deliberate diagonals are
    preserved.  Manual roll is intentionally unassigned.
    """
    if not all(math.isfinite(value) for value in (
            state.left_x, state.left_y, state.right_x, state.right_y,
            state.left_trigger, state.right_trigger)):
        raise ValueError("controller axes must be finite")
    for deadzone in (stick_deadzone, trigger_deadzone):
        if not math.isfinite(deadzone) or not 0 <= deadzone < 1:
            raise ValueError("deadzones must be finite in [0,1)")
    if not math.isfinite(vertical_scale) or not 0 < vertical_scale <= 1:
        raise ValueError("vertical_scale must be finite in (0,1]")
    if not math.isfinite(yaw_response_exponent) or not 0.5 <= yaw_response_exponent <= 2:
        raise ValueError("yaw_response_exponent must be finite in [0.5,2]")
    lx, ly = _radial_deadzone(state.left_x, state.left_y, stick_deadzone)
    rx, ry = _radial_deadzone(state.right_x, state.right_y, stick_deadzone)
    lx, ly = _soft_axis_snap(lx, ly, axis_snap_inner_deg, axis_snap_outer_deg)
    rx, ry = _soft_axis_snap(rx, ry, axis_snap_inner_deg, axis_snap_outer_deg)
    # Curve yaw AFTER neutral deadzone/snap, BEFORE gear scaling. This raises
    # mid-stick response without increasing full-scale authority or noise creep.
    rx = math.copysign(abs(rx) ** yaw_response_exponent, rx)
    lt = _trigger_deadzone(state.left_trigger, trigger_deadzone)
    rt = _trigger_deadzone(state.right_trigger, trigger_deadzone)

    full_buttons = state.buttons
    if button_mask is not None:
        full_buttons &= button_mask
    if not allow_arm:
        full_buttons &= ~(1 << 0)  # Never forward A/Arm until explicitly enabled.

    if not 0.0 < motion_scale <= 1.0:
        raise ValueError("motion_scale must be in (0, 1]")
    if vertical_motion_scale is None:
        vertical_motion_scale = motion_scale
    if not 0.0 < vertical_motion_scale <= 1.0:
        raise ValueError("vertical_motion_scale must be in (0, 1]")

    return ManualControlCommand(
        x=round(ly * 1000 * motion_scale),
        y=round(lx * 1000 * motion_scale),
        z=round(500 + ((rt - lt) * 500 * vertical_scale * vertical_motion_scale)),
        r=round(rx * 1000 * motion_scale),
        buttons=full_buttons & 0xFFFF,
        buttons2=(full_buttons >> 16) & 0xFFFF,
        enabled_extensions=0x03,
        # ArduSub 4.7.1 applies s/t directly around PWM 1500, so constrain the
        # extension axes to +/-400 instead of sending +/-1000.
        s=round(-ry * 400 * motion_scale),
        t=0,
    )
