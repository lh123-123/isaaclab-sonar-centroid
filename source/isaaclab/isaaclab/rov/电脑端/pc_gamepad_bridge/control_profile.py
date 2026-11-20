"""Competition controller profile and edge-triggered local actions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_PROFILE = Path(__file__).with_name("controller_profile.json")


def load_profile(path: str | Path = DEFAULT_PROFILE) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        profile = json.load(stream)
    startup = profile.setdefault("thruster_startup", {"reference_offset_us": 50})
    if not isinstance(startup, dict):
        raise ValueError("thruster_startup须为配置对象")
    offset = startup.get("reference_offset_us")
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 100:
        raise ValueError("推进器启转参考偏移须为0～100整数μs")
    presets = profile["gimbal"]["presets_deg"]
    if not presets or presets != sorted(set(presets)):
        raise ValueError("gimbal presets must be unique and ascending")
    if any(not -90 <= value <= 20 for value in presets):
        raise ValueError("gimbal presets must stay in the configured -90..20 deg range")
    quick_angle = profile["gimbal"].get("quick_angle_deg", -20)
    if (type(quick_angle) not in (int, float) or not math.isfinite(quick_angle)
            or not -90 <= quick_angle <= 20):
        raise ValueError("云台快捷角度必须在 -90～20° 之间")
    for group in ("motion_gain", "vertical_gain"):
        levels = profile[group]["levels"]
        if not levels or levels != sorted(set(levels)):
            raise ValueError("motion gain levels must be unique and ascending")
        if any(not 0.0 < float(value) <= 1.0 for value in levels):
            raise ValueError("motion gain levels must be in (0, 1]")
        if profile[group]["initial"] not in levels:
            raise ValueError("default motion gain must be one of the configured levels")
    gain_buttons = (profile["motion_gain"]["cycle_button"],
                    profile["vertical_gain"]["cycle_button"],
                    profile["motion_gain"]["reset_button"])
    if len(set(gain_buttons)) != 3:
        raise ValueError("motion cycle/reset buttons must be independent")
    inner = float(profile["axis"]["axis_snap_inner_deg"])
    outer = float(profile["axis"]["axis_snap_outer_deg"])
    yaw_exponent = profile["axis"].get("yaw_response_exponent", 1.0)
    if (type(yaw_exponent) not in (int, float) or not math.isfinite(yaw_exponent)
            or not 0.5 <= yaw_exponent <= 2.0):
        raise ValueError("yaw response exponent must be finite in [0.5,2]")
    if not 0.0 <= inner < outer < 45.0:
        raise ValueError("axis snap angles must satisfy 0 <= inner < outer < 45")
    servos = profile["servos"]
    channels = []
    direction_buttons = []
    for name in ("gripper", "angle", "rotation"):
        item = servos[name]
        minimum = int(item["min_pwm"])
        maximum = int(item["max_pwm"])
        initial = int(item["initial_pwm"])
        if not 500 <= minimum < initial < maximum <= 2500:
            raise ValueError(f"invalid guarded PWM range for {name}")
        channels.append(int(item["channel"]))
        direction_buttons.extend(
            (int(item["decrease_button"]), int(item["increase_button"]))
        )
    if len(set(channels)) != len(channels):
        raise ValueError("servo output channels must be unique")
    if len(set(direction_buttons)) != len(direction_buttons):
        raise ValueError("servo direction buttons must be unique")
    reset_pwm = servos["angle"].get("reset_pwm", 920)
    if (type(reset_pwm) is not int or
            not int(servos["angle"]["min_pwm"]) <= reset_pwm <= int(servos["angle"]["max_pwm"])):
        raise ValueError("上下舵机复位PWM必须在已设定的舵机范围内")
    dpad_buttons = (
        profile["gimbal"]["up_button"], profile["gimbal"]["down_button"],
        profile["gimbal"].get("quick_button", 13),
        servos["angle"].get("reset_button", 14),
    )
    if len(set(dpad_buttons)) != 4 or set(dpad_buttons) & set(direction_buttons):
        raise ValueError("十字键与舵机方向键不能重复")
    gripper = servos["gripper"]
    if not int(gripper["min_pwm"]) <= int(gripper["closed_pwm"]) < int(gripper["open_pwm"]) <= int(gripper["max_pwm"]):
        raise ValueError("gripper closed/open PWM endpoints are invalid")
    return profile


def button_mask(indices: Iterable[int]) -> int:
    mask = 0
    for index in indices:
        if not 0 <= int(index) <= 31:
            raise ValueError(f"button index out of range: {index}")
        mask |= 1 << int(index)
    return mask


def save_default_motion_gain(profile_path: str | Path, gain: float) -> None:
    """Persist a common startup/LB default without changing servo calibration."""
    path = Path(profile_path)
    profile = load_profile(path)
    for group in ("motion_gain", "vertical_gain"):
        if gain not in profile[group]["levels"]:
            raise ValueError("默认档位必须是两组都支持的档位")
        profile[group]["initial"] = float(gain)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_quick_targets(profile_path: str | Path, gimbal_angle_deg: float,
                       angle_servo_pwm: int) -> None:
    """Save the two D-pad targets atomically without changing other settings."""
    if type(angle_servo_pwm) is not int:
        raise ValueError("上下舵机复位PWM必须是整数")
    path = Path(profile_path)
    profile = load_profile(path)
    profile["gimbal"]["quick_angle_deg"] = float(gimbal_angle_deg)
    profile["servos"]["angle"]["reset_pwm"] = angle_servo_pwm
    profile["buttons"]["13"]["action"] = f"云台快捷角度 {float(gimbal_angle_deg):+g}°"
    profile["buttons"]["14"]["action"] = f"夹爪上下复位 {angle_servo_pwm}µs"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        load_profile(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class RisingEdgeButtons:
    def __init__(self) -> None:
        self._previous = 0

    def update(self, current: int) -> int:
        current &= 0xFFFFFFFF
        rising = current & ~self._previous
        self._previous = current
        return rising


@dataclass
class GimbalPresetController:
    presets_deg: tuple[float, ...]
    target_deg: float

    @classmethod
    def from_profile(cls, profile: dict) -> "GimbalPresetController":
        values = tuple(float(value) for value in profile["gimbal"]["presets_deg"])
        initial = float(profile["gimbal"].get("initial_deg", 0))
        return cls(values, initial)

    @property
    def angle_deg(self) -> float:
        return self.target_deg

    def move_up(self) -> float:
        self.target_deg = next((v for v in self.presets_deg if v > self.target_deg),
                               self.presets_deg[-1])
        return self.angle_deg

    def move_down(self) -> float:
        self.target_deg = next((v for v in reversed(self.presets_deg) if v < self.target_deg),
                               self.presets_deg[0])
        return self.angle_deg

    def center(self) -> float:
        self.target_deg = min(self.presets_deg, key=abs)
        return self.angle_deg

    def quick(self, angle_deg: float) -> float:
        self.target_deg = float(angle_deg)
        return self.angle_deg


@dataclass
class MotionGainController:
    levels: tuple[float, ...]
    index: int
    default_index: int

    @classmethod
    def from_profile(cls, profile: dict, group: str = "motion_gain") -> "MotionGainController":
        values = tuple(float(value) for value in profile[group]["levels"])
        initial = float(profile[group]["initial"])
        nearest = min(range(len(values)), key=lambda i: abs(values[i] - initial))
        return cls(values, nearest, nearest)

    @property
    def gain(self) -> float:
        return self.levels[self.index]

    def cycle(self) -> float:
        self.index = (self.index + 1) % len(self.levels)
        return self.gain

    def reset(self) -> float:
        self.index = self.default_index
        return self.gain


@dataclass(frozen=True)
class ActionUpdate:
    gimbal_angle_deg: float | None = None
    motion_gain: float | None = None
    mode_action: str | None = None
    vertical_gain: float | None = None
    angle_servo_reset_pwm: int | None = None


class RepeatedButtonPulser:
    """Turn held buttons into short repeated rising edges for ArduSub increments."""

    def __init__(self, indices: Iterable[int], rate_hz: float, pulse_seconds: float):
        self.indices = tuple(int(index) for index in indices)
        self.period = 1.0 / float(rate_hz)
        self.pulse_seconds = float(pulse_seconds)
        if not 0.0 < self.pulse_seconds < self.period:
            raise ValueError("pulse_seconds must be between 0 and the repeat period")
        self._states: dict[int, tuple[bool, float]] = {}

    @classmethod
    def from_profile(cls, profile: dict) -> "RepeatedButtonPulser":
        settings = profile["repeat_buttons"]
        return cls(
            settings["indices"],
            float(settings["rate_hz"]),
            float(settings["pulse_seconds"]),
        )

    def update(self, buttons: int, now: float) -> int:
        output = buttons
        for index in self.indices:
            bit = 1 << index
            if not buttons & bit:
                self._states.pop(index, None)
                output &= ~bit
                continue

            state = self._states.get(index)
            if state is None:
                pulse_on = True
                next_transition = now + self.pulse_seconds
            else:
                pulse_on, next_transition = state
                while now >= next_transition:
                    if pulse_on:
                        pulse_on = False
                        next_transition += self.period - self.pulse_seconds
                    else:
                        pulse_on = True
                        next_transition += self.pulse_seconds
            self._states[index] = (pulse_on, next_transition)
            if pulse_on:
                output |= bit
            else:
                output &= ~bit
        return output


class LocalActionController:
    """Consumes local-only buttons and reports discrete control changes."""

    def __init__(self, profile: dict) -> None:
        self.profile = profile
        self.edges = RisingEdgeButtons()
        self.gimbal = GimbalPresetController.from_profile(profile)
        self.motion = MotionGainController.from_profile(profile)
        self.vertical = MotionGainController.from_profile(profile, "vertical_gain")

    def set_default_gain(self, gain: float) -> None:
        if gain not in self.motion.levels or gain not in self.vertical.levels:
            raise ValueError("默认档位必须是两组都支持的档位")
        self.motion.default_index = self.motion.levels.index(gain)
        self.vertical.default_index = self.vertical.levels.index(gain)

    def gain_text(self) -> str:
        return f"摇杆 {self.motion.gain:.0%}  ·  升潜 {self.vertical.gain:.0%}"

    def update(self, buttons: int) -> ActionUpdate:
        rising = self.edges.update(buttons)
        if buttons & (1 << 1):
            return ActionUpdate()
        gimbal_settings = self.profile["gimbal"]
        gimbal_angle = None
        # Ignore diagonals until released; do not execute two camera actions.
        quick_button = int(gimbal_settings.get("quick_button", 13))
        reset_button = int(self.profile["servos"]["angle"].get("reset_button", 14))
        dpad_mask = button_mask((gimbal_settings["up_button"], gimbal_settings["down_button"],
                                 quick_button, reset_button))
        if (buttons & dpad_mask).bit_count() != 1:
            rising &= ~dpad_mask
        angle_servo_reset_pwm = None
        if rising & (1 << quick_button):
            gimbal_angle = self.gimbal.quick(gimbal_settings.get("quick_angle_deg", -20))
        elif rising & (1 << reset_button):
            angle_servo_reset_pwm = int(self.profile["servos"]["angle"].get("reset_pwm", 920))
        elif rising & (1 << int(gimbal_settings["up_button"])):
            gimbal_angle = self.gimbal.move_up()
        elif rising & (1 << int(gimbal_settings["down_button"])):
            gimbal_angle = self.gimbal.move_down()

        gain_settings = self.profile["motion_gain"]
        motion_gain = None
        vertical_gain = None
        if rising & (1 << int(gain_settings["reset_button"])):
            motion_gain = self.motion.reset()
            vertical_gain = self.vertical.reset()
        else:
            if rising & (1 << int(gain_settings["cycle_button"])):
                motion_gain = self.motion.cycle()
            if rising & (1 << int(self.profile["vertical_gain"]["cycle_button"])):
                vertical_gain = self.vertical.cycle()

        mode = self.profile["flight_mode"]
        mode_action = None
        if rising & (1 << int(mode["stabilize_button"])):
            mode_action = "attitude_toggle"
        elif rising & (1 << int(mode["depth_toggle_button"])):
            mode_action = "depth_toggle"
        return ActionUpdate(gimbal_angle, motion_gain, mode_action, vertical_gain,
                            angle_servo_reset_pwm)


def resolve_button_conflicts(buttons: int) -> int:
    """Disarm wins; opposing servo directions cancel; slow grip beats quick."""
    if buttons & (1 << 1):
        return 1 << 1
    if buttons & button_mask((17, 18)):
        buttons &= ~(1 << 10)
    for pair in ((15, 16), (17, 18), (19, 20)):
        mask = button_mask(pair)
        if buttons & mask == mask:
            buttons &= ~mask
    return buttons
