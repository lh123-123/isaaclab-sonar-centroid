"""Direct, rate-limited control for the three non-thruster AUX servos.

ArduSub's joystick actuator increment functions run once per MANUAL_CONTROL
packet while a button is held.  This module instead owns an explicit PWM
target and sends MAV_CMD_DO_SET_SERVO at a bounded rate, so motion speed does
not depend on the joystick packet rate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


SERVO_ABSOLUTE_MIN_PWM = 500
SERVO_ABSOLUTE_MAX_PWM = 2500
SERVO_MIN_SLOW_RATE = 20.0
SERVO_MAX_SLOW_RATE = 500.0


def validate_servo_speed(value: float) -> float:
    speed = float(value)
    if not math.isfinite(speed) or not SERVO_MIN_SLOW_RATE <= speed <= SERVO_MAX_SLOW_RATE:
        raise ValueError("舵机慢速速度必须在20～500µs/s之间")
    return speed


def validate_servo_range(minimum_pwm: int, maximum_pwm: int) -> tuple[int, int]:
    minimum = int(minimum_pwm)
    maximum = int(maximum_pwm)
    if not SERVO_ABSOLUTE_MIN_PWM <= minimum < maximum <= SERVO_ABSOLUTE_MAX_PWM:
        raise ValueError("舵机范围必须满足 500 <= 最小值 < 最大值 <= 2500")
    return minimum, maximum


def save_servo_ranges(profile_path: str | Path, ranges: dict[str, tuple[int, int]],
                      speeds: dict[str, float] | None = None) -> None:
    """Persist operator-calibrated ranges without touching other controls."""
    path = Path(profile_path)
    profile = json.loads(path.read_text(encoding="utf-8"))
    if speeds is not None:
        if set(speeds) != {"gripper", "angle", "rotation"}:
            raise ValueError("请设置三路舵机的慢速速度")
        speeds = {name: validate_servo_speed(value) for name, value in speeds.items()}
    for name in ("gripper", "angle", "rotation"):
        if name not in ranges:
            raise ValueError(f"缺少{name}舵机范围")
        minimum, maximum = validate_servo_range(*ranges[name])
        item = profile["servos"][name]
        item["min_pwm"] = minimum
        item["max_pwm"] = maximum
        if speeds is not None:
            item["slow_rate_us_per_s"] = speeds[name]
        initial = int(item.get("initial_pwm", round((minimum + maximum) / 2)))
        item["initial_pwm"] = initial if minimum < initial < maximum else round((minimum + maximum) / 2)
        if name == "gripper":
            item["closed_pwm"] = minimum
            item["open_pwm"] = maximum
        elif name == "angle" and "reset_pwm" in item:
            item["reset_pwm"] = min(max(int(item["reset_pwm"]), minimum), maximum)
            profile["buttons"]["14"]["action"] = f"夹爪上下复位 {item['reset_pwm']}µs"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class ServoCommand:
    channel: int
    pwm: int
    reason: str


@dataclass
class ServoAxis:
    name: str
    channel: int
    decrease_button: int
    increase_button: int
    minimum_pwm: int
    maximum_pwm: int
    target_pwm: float
    next_send: float = 0.0
    last_sent_pwm: int | None = None
    has_commanded: bool = False
    motion_credit: float = 0.0
    last_direction: int = 0
    slow_rate_us_per_s: float | None = None

    def clamp(self, value: float) -> float:
        return min(float(self.maximum_pwm), max(float(self.minimum_pwm), value))


class ServoController:
    """Convert held local buttons into slow, bounded servo PWM commands."""

    def __init__(
        self,
        axes: tuple[ServoAxis, ...],
        *,
        slow_rate_us_per_s: float = 50.0,
        command_rate_hz: float = 50.0,
        toggle_button: int = 10,
        gripper_name: str = "gripper",
        gripper_open_pwm: int = 2500,
        gripper_closed_pwm: int | None = 500,
    ) -> None:
        if slow_rate_us_per_s <= 0 or command_rate_hz <= 0:
            raise ValueError("servo rates must be positive")
        self.axes = {axis.name: axis for axis in axes}
        self.channels = {axis.channel: axis for axis in axes}
        if gripper_name not in self.axes:
            raise ValueError("gripper axis is required")
        self.gripper_name = gripper_name
        self.slow_rate_us_per_s = validate_servo_speed(slow_rate_us_per_s)
        for axis in axes:
            if axis.slow_rate_us_per_s is not None:
                validate_servo_speed(axis.slow_rate_us_per_s)
        self.command_period = 1.0 / float(command_rate_hz)
        self.toggle_button = int(toggle_button)
        self.gripper_open_pwm = int(gripper_open_pwm)
        self.gripper_closed_pwm = (
            None if gripper_closed_pwm is None else int(gripper_closed_pwm)
        )
        self._previous_buttons = 0
        self._last_update: float | None = None
        self.next_gripper_action = "close"
        self._refresh_next_action_from_position()

    @classmethod
    def for_competition_defaults(cls) -> "ServoController":
        return cls(
            (
                ServoAxis("gripper", 9, 18, 17, 500, 2500, 1500),
                ServoAxis("angle", 10, 15, 16, 500, 2500, 1500),
                ServoAxis("rotation", 11, 19, 20, 500, 2500, 1500),
            )
        )

    @classmethod
    def from_profile(cls, profile: dict) -> "ServoController":
        settings = profile["servos"]
        axes = []
        for name in ("gripper", "angle", "rotation"):
            item = settings[name]
            axes.append(
                ServoAxis(
                    name=name,
                    channel=int(item["channel"]),
                    decrease_button=int(item["decrease_button"]),
                    increase_button=int(item["increase_button"]),
                    minimum_pwm=int(item["min_pwm"]),
                    maximum_pwm=int(item["max_pwm"]),
                    target_pwm=float(item.get("initial_pwm", 1500)),
                    slow_rate_us_per_s=(float(item["slow_rate_us_per_s"])
                                        if "slow_rate_us_per_s" in item else None),
                )
            )
        return cls(
            tuple(axes),
            slow_rate_us_per_s=float(settings["slow_rate_us_per_s"]),
            command_rate_hz=float(settings["command_rate_hz"]),
            toggle_button=int(settings["gripper"]["toggle_button"]),
            gripper_open_pwm=int(settings["gripper"]["open_pwm"]),
            gripper_closed_pwm=settings["gripper"].get("closed_pwm"),
        )

    def channel_pwm(self, channel: int) -> int:
        return int(round(self.channels[int(channel)].target_pwm))

    @property
    def next_gripper_action_cn(self) -> str:
        if self.next_gripper_action == "open":
            return "快速张开"
        if self.gripper_closed_pwm is None:
            return "闭合端未标定"
        return "快速闭合"

    def status_text(self) -> str:
        gripper = self.axes["gripper"]
        angle = self.axes["angle"]
        rotation = self.axes["rotation"]
        return (
            f"夹爪 {self.channel_pwm(9)}us [{gripper.minimum_pwm}-{gripper.maximum_pwm}]"
            f"｜RB下一次：{self.next_gripper_action_cn}\n"
            f"夹爪上下 {self.channel_pwm(10)}us [{angle.minimum_pwm}-{angle.maximum_pwm}]\n"
            f"末端旋转 {self.channel_pwm(11)}us [{rotation.minimum_pwm}-{rotation.maximum_pwm}]"
            f"\n慢速 开合/上下/旋转：{self.speed_for('gripper'):g}/"
            f"{self.speed_for('angle'):g}/{self.speed_for('rotation'):g}µs/s"
        )

    def speed_for(self, name: str) -> float:
        value = self.axes[name].slow_rate_us_per_s
        return self.slow_rate_us_per_s if value is None else value

    def command_preset(self, name: str, pwm: int, now: float) -> ServoCommand:
        """Apply one guarded absolute target; subsequent slow commands start here."""
        if name not in self.axes:
            raise ValueError(f"未知舵机：{name}")
        axis = self.axes[name]
        target = int(pwm)
        if not axis.minimum_pwm <= target <= axis.maximum_pwm:
            raise ValueError("快捷PWM超出已设定的舵机范围")
        axis.target_pwm = float(target)
        axis.last_sent_pwm = target
        axis.has_commanded = True
        axis.next_send = float(now) + self.command_period
        axis.motion_credit = 0.0
        axis.last_direction = 0
        return ServoCommand(axis.channel, target, f"{name}-preset")

    def configure_speeds(self, speeds: dict[str, float]) -> None:
        # Validate the whole transaction before changing any live setting.
        if set(speeds) != set(self.axes):
            raise ValueError("请设置三路舵机的慢速速度")
        validated = {name: validate_servo_speed(value) for name, value in speeds.items()}
        for name, speed in validated.items():
            self.axes[name].slow_rate_us_per_s = speed
            self.axes[name].motion_credit = 0.0
        self._last_update = None

    def _refresh_next_action_from_position(self) -> None:
        gripper = self.axes[self.gripper_name]
        midpoint = (gripper.minimum_pwm + gripper.maximum_pwm) / 2.0
        self.next_gripper_action = "open" if gripper.target_pwm < midpoint else "close"

    def set_gripper_closed_pwm(self, pwm: int) -> None:
        gripper = self.axes[self.gripper_name]
        value = int(pwm)
        if not gripper.minimum_pwm <= value <= gripper.maximum_pwm:
            raise ValueError("closed PWM is outside the guarded discovery range")
        self.gripper_closed_pwm = value

    def configure_ranges(self, ranges: dict[str, tuple[int, int]]) -> None:
        for name, values in ranges.items():
            if name not in self.axes:
                raise ValueError(f"未知舵机：{name}")
            minimum, maximum = validate_servo_range(*values)
            axis = self.axes[name]
            axis.minimum_pwm = minimum
            axis.maximum_pwm = maximum
            axis.target_pwm = axis.clamp(axis.target_pwm)
            axis.last_sent_pwm = None
            if name == self.gripper_name:
                self.gripper_closed_pwm = minimum
                self.gripper_open_pwm = maximum
        self._refresh_next_action_from_position()

    def sync_channel(self, channel: int, pwm: int) -> bool:
        axis = self.channels.get(int(channel))
        if axis is None or not axis.minimum_pwm <= int(pwm) <= axis.maximum_pwm:
            return False
        if axis.has_commanded:
            return False
        axis.target_pwm = float(pwm)
        axis.last_sent_pwm = int(pwm)
        if axis.name == self.gripper_name:
            self._refresh_next_action_from_position()
        return True

    def observe(self, message, _now: float) -> None:
        if message.get_type() != "SERVO_OUTPUT_RAW":
            return
        for channel in self.channels:
            pwm = getattr(message, f"servo{channel}_raw", 0)
            self.sync_channel(channel, int(pwm))

    def update(self, buttons: int, now: float, *, enabled: bool) -> tuple[ServoCommand, ...]:
        buttons = int(buttons)
        rising = buttons & ~self._previous_buttons
        self._previous_buttons = buttons
        if self._last_update is None:
            elapsed = 0.0
        else:
            # A paused UI or delayed packet must never become one large jump.
            elapsed = min(self.command_period, max(0.0, float(now) - self._last_update))
        self._last_update = float(now)
        if not enabled:
            for axis in self.axes.values():
                axis.motion_credit = 0.0
                axis.last_direction = 0
            return ()

        commands: list[ServoCommand] = []
        gripper = self.axes[self.gripper_name]
        if rising & (1 << self.toggle_button):
            if self.next_gripper_action == "open":
                gripper.target_pwm = float(self.gripper_open_pwm)
                self.next_gripper_action = "close"
                reason = "gripper-fast-open"
            else:
                if self.gripper_closed_pwm is None:
                    return ()
                gripper.target_pwm = float(self.gripper_closed_pwm)
                self.next_gripper_action = "open"
                reason = "gripper-fast-close"
            command = ServoCommand(gripper.channel, self.channel_pwm(gripper.channel), reason)
            gripper.last_sent_pwm = command.pwm
            gripper.has_commanded = True
            gripper.next_send = float(now) + self.command_period
            gripper.motion_credit = 0.0
            commands.append(command)

        for axis in self.axes.values():
            decrease = bool(buttons & (1 << axis.decrease_button))
            increase = bool(buttons & (1 << axis.increase_button))
            direction = int(increase) - int(decrease)
            if direction != axis.last_direction:
                axis.motion_credit = 0.0
                if direction:
                    axis.next_send = max(axis.next_send, float(now) + self.command_period)
            axis.last_direction = direction
            if direction == 0:
                axis.motion_credit = 0.0
                continue
            if axis.name == self.gripper_name:
                self.next_gripper_action = "close" if direction > 0 else "open"
            # Time-based small steps. Keep sub-microsecond rounding credit,
            # but at most one normal packet's travel after a delayed callback.
            rate = self.speed_for(axis.name)
            max_step = max(1, math.ceil(rate * self.command_period))
            axis.motion_credit = min(rate * self.command_period + 1.0 - 1e-9,
                                     axis.motion_credit + rate * elapsed)
            if axis.motion_credit + 1e-9 < 1.0 or float(now) + 1e-9 < axis.next_send:
                continue
            step = min(max_step, int(axis.motion_credit + 1e-9))
            pwm = int(axis.clamp(self.channel_pwm(axis.channel) + direction * step))
            if pwm != self.channel_pwm(axis.channel):
                axis.target_pwm = float(pwm)
                commands.append(ServoCommand(axis.channel, pwm, f"{axis.name}-slow"))
                axis.last_sent_pwm = pwm
                axis.has_commanded = True
                axis.next_send = float(now) + self.command_period
            axis.motion_credit = max(0.0, axis.motion_credit - step)
        return tuple(commands)
