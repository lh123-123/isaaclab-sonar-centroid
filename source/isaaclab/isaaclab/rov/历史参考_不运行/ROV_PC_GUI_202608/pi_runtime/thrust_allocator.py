#!/usr/bin/env python3
"""
8-thruster ROV thrust allocator with group-ratio-preserving PWM safety limits.

Coordinate convention (right-handed):
    +x: forward
    +y: starboard / right
    +z: down
    +Mx: roll right-hand moment about +x
    +My: pitch right-hand moment about +y
    +Mz: yaw right-hand moment about +z

Thruster order:
    0 lateral_fl
    1 lateral_fr
    2 lateral_rl
    3 lateral_rr
    4 vertical_fl
    5 vertical_fr
    6 vertical_rl
    7 vertical_rr

The four lateral thrusters use an X layout at 45 degrees. The four vertical
thrusters point in the +z direction in the nominal geometry. If a real
thruster's motor command is reversed, change MOTOR_SIGNS instead of changing
the joystick mapping.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


THRUSTER_NAMES: Tuple[str, ...] = (
    "lateral_fl",
    "lateral_fr",
    "lateral_rl",
    "lateral_rr",
    "vertical_fl",
    "vertical_fr",
    "vertical_rl",
    "vertical_rr",
)
VERTICAL_THRUSTER_NAMES: Tuple[str, ...] = THRUSTER_NAMES[4:8]

# Replace these values with the actual PCA9685 channel mapping.
DEFAULT_CHANNELS: Dict[str, int] = {
    "lateral_fl": 0,
    "lateral_fr": 1,
    "lateral_rl": 2,
    "lateral_rr": 3,
    "vertical_fl": 4,
    "vertical_fr": 5,
    "vertical_rl": 6,
    "vertical_rr": 7,
}

# Set an entry to -1 only when that motor/ESC's command direction is reversed
# relative to the nominal force direction used by the TAM.
DEFAULT_MOTOR_SIGNS: Dict[str, int] = {
    name: 1 for name in THRUSTER_NAMES
}


def world_down_force_component(body_force, *, roll_deg: float, pitch_deg: float) -> float:
    """Project a body-frame force onto world down; yaw cancels identically."""
    force = np.asarray(body_force, dtype=float)
    if force.shape != (3,) or not np.all(np.isfinite(force)):
        raise ValueError("body_force must contain three finite values")
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    if not all(math.isfinite(value) for value in (roll, pitch)):
        raise ValueError("roll/pitch must be finite")
    world_down_in_body = np.asarray(
        (
            -math.sin(pitch),
            math.sin(roll) * math.cos(pitch),
            math.cos(roll) * math.cos(pitch),
        ),
        dtype=float,
    )
    return float(world_down_in_body @ force)


@dataclass(frozen=True)
class AllocationResult:
    """One allocation cycle's calculated and commanded values."""

    requested_tau: np.ndarray
    achieved_tau: np.ndarray
    nominal_thrusts: np.ndarray
    motor_thrusts: np.ndarray
    pwm_us: np.ndarray
    scale: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "requested_tau": self.requested_tau.tolist(),
            "achieved_tau": self.achieved_tau.tolist(),
            "nominal_thrusts": dict(zip(THRUSTER_NAMES, self.nominal_thrusts.tolist())),
            "motor_thrusts": dict(zip(THRUSTER_NAMES, self.motor_thrusts.tolist())),
            "pwm_us": dict(zip(THRUSTER_NAMES, self.pwm_us.tolist())),
            "scale": self.scale,
        }


class PCA9685Output:
    """
    Adapter for the project's existing ``Thrusters.PCA9685`` smbus driver.

    The import happens only when hardware output is enabled, so desktop
    allocation tests do not require Raspberry Pi I2C modules.
    """

    def __init__(
        self,
        address: int = 0x40,
        frequency_hz: int = 50,
        debug: bool = False,
    ) -> None:
        try:
            from Thrusters import PCA9685
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Cannot import PCA9685 from Thrusters.py. Keep Thrusters.py "
                "beside thrust_allocator.py and make sure python3-smbus is installed."
            ) from exc

        self.frequency_hz = int(frequency_hz)
        if self.frequency_hz <= 0:
            raise ValueError("PWM frequency must be > 0.")
        self._pca = PCA9685(address=int(address), debug=bool(debug))
        self._pca.setPWMFreq(self.frequency_hz)
        # A PCA9685 keeps its register state after a process exits. Explicitly
        # disable every channel before any servo or ESC pulse is issued.
        for channel in range(16):
            self._pca.setPWM(channel, 0, 4096)

    def set_pulse_us(self, channel: int, pulse_us: int) -> None:
        self._pca.setServoPulse(int(channel), int(pulse_us))

    def set_motor_speed(
        self,
        channels: Union[Sequence[int], int],
        pulse_us: int,
    ) -> None:
        """Compatibility API used by the existing BlueROV main program."""
        if isinstance(channels, (int, np.integer)):
            channels = (int(channels),)
        self._pca.set_motor_speed(
            [int(channel) for channel in channels],
            int(pulse_us),
        )

    def initialize_esc(
        self,
        channels: Union[Sequence[int], int],
        *,
        neutral_pulse: int = 1540,
        duration: float = 5.0,
    ) -> None:
        """Hold every ESC at neutral for its arming/initialization interval."""
        duration = float(duration)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("ESC initialization duration must be finite and >= 0.")
        if isinstance(channels, (int, np.integer)):
            channels = (int(channels),)
        self._pca.initialize_esc(
            [int(channel) for channel in channels],
            neutral_pulse=int(neutral_pulse),
            duration=duration,
        )

    def close(self) -> None:
        bus = getattr(self._pca, "bus", None)
        close = getattr(bus, "close", None)
        if callable(close):
            close()


class PWMCompatibilityOutput:
    """
    Compatibility wrapper for an injected `pwm_writer(channel, pulse_us)`.

    In calculation-only mode writer is None, so these calls intentionally do
    nothing and allow desktop allocation tests without Raspberry Pi hardware.
    """

    def __init__(
        self,
        writer: Optional[Callable[[int, int], None]],
    ) -> None:
        self._writer = writer

    def set_pulse_us(self, channel: int, pulse_us: int) -> None:
        if self._writer is not None:
            self._writer(int(channel), int(pulse_us))

    def set_motor_speed(
        self,
        channels: Union[Sequence[int], int],
        pulse_us: int,
    ) -> None:
        if isinstance(channels, (int, np.integer)):
            channels = (int(channels),)
        for channel in channels:
            self.set_pulse_us(int(channel), int(pulse_us))

    def initialize_esc(
        self,
        channels: Union[Sequence[int], int],
        *,
        neutral_pulse: int = 1540,
        duration: float = 5.0,
    ) -> None:
        duration = float(duration)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("ESC initialization duration must be finite and >= 0.")
        self.set_motor_speed(channels, int(neutral_pulse))
        if self._writer is not None:
            time.sleep(duration)


class ThrustAllocator:
    """
    Full 8-thruster allocator with asymmetric thrust limits.

    Public methods used by a typical main loop:
        allocate(...)          calculate and optionally write all 8 channels
        allocate_thrust(...)   compatibility alias
        stop_all()             immediately write neutral PWM to all 8 channels

    `pwm_writer` must be callable as `pwm_writer(channel, pulse_us)`.
    Hardware output is enabled by default for compatibility with main.py.
    Pass enable_hardware=False for calculation-only desktop tests.
    """

    def __init__(
        self,
        pwm_address: int = 0x40,
        channels_dict: Optional[Mapping[str, int]] = None,
        *,
        simple_mode: bool = False,
        debug: bool = False,
        enable_hardware: bool = True,
        pwm_writer: Optional[Callable[[int, int], None]] = None,
        motor_signs: Optional[Mapping[str, int]] = None,
        lateral_x_m: float = 0.18,
        lateral_y_m: float = 0.18,
        vertical_x_m: float = 0.18,
        vertical_y_m: float = 0.18,
        vertical_positions_m: Optional[Mapping[str, Sequence[float]]] = None,
        vertical_directions_body: Optional[Mapping[str, Sequence[float]]] = None,
        lateral_angle_deg: float = 45.0,
        pwm_min_us: int = 1100,
        pwm_neutral_us: int = 1540,
        pwm_max_us: int = 1900,
        pwm_output_min_us: int = 1200,
        pwm_output_max_us: int = 1850,
        thruster_pwm_limits: Optional[Mapping[str, Tuple[int, int]]] = None,
        reverse_start_us: Optional[int] = None,
        forward_start_us: Optional[int] = None,
        lateral_reverse_start_us: Optional[int] = None,
        lateral_forward_start_us: Optional[int] = None,
        reverse_max_thrust_n: float = 55.0,
        forward_max_thrust_n: float = 65.0,
        horizontal_limit_fraction: Optional[float] = None,
        vertical_limit_fraction: Optional[float] = None,
        vertical_correction_limit_fraction: float = 0.5,
        forward_front_rear_trim_percent: float = 0.0,
        backward_front_rear_trim_percent: float = 0.0,
        left_lateral_surge_trim_percent: float = 0.0,
        right_lateral_surge_trim_percent: float = 0.0,
        rcond: float = 1e-8,
    ) -> None:
        if simple_mode:
            raise ValueError(
                "This allocator is intentionally full 8-thruster mode; "
                "set simple_mode=False."
            )

        self.debug = bool(debug)
        self.thruster_names = list(THRUSTER_NAMES)
        self.channels = self._validate_channels(
            DEFAULT_CHANNELS if channels_dict is None else channels_dict
        )
        self.motor_signs = self._validate_motor_signs(
            DEFAULT_MOTOR_SIGNS if motor_signs is None else motor_signs
        )

        self.pwm_min_us = int(pwm_min_us)
        self.pwm_neutral_us = int(pwm_neutral_us)
        self.pwm_max_us = int(pwm_max_us)
        self.pwm_output_min_us = int(pwm_output_min_us)
        self.pwm_output_max_us = int(pwm_output_max_us)
        self.reverse_start_us = (
            None if reverse_start_us is None else int(reverse_start_us)
        )
        self.forward_start_us = (
            None if forward_start_us is None else int(forward_start_us)
        )
        self.lateral_reverse_start_us = (
            None if lateral_reverse_start_us is None else int(lateral_reverse_start_us)
        )
        self.lateral_forward_start_us = (
            None if lateral_forward_start_us is None else int(lateral_forward_start_us)
        )
        self.reverse_max_thrust_n = float(reverse_max_thrust_n)
        self.forward_max_thrust_n = float(forward_max_thrust_n)
        self._validate_pwm_and_thrust_config()
        self._lock = threading.RLock()
        self.forward_front_rear_trim_percent = self._validated_trim_percent(
            forward_front_rear_trim_percent,
            "forward_front_rear_trim_percent",
        )
        self.backward_front_rear_trim_percent = self._validated_trim_percent(
            backward_front_rear_trim_percent,
            "backward_front_rear_trim_percent",
        )
        self.left_lateral_surge_trim_percent = self._validated_trim_percent(
            left_lateral_surge_trim_percent,
            "left_lateral_surge_trim_percent",
        )
        self.right_lateral_surge_trim_percent = self._validated_trim_percent(
            right_lateral_surge_trim_percent,
            "right_lateral_surge_trim_percent",
        )
        self.vertical_x_m = float(vertical_x_m)
        self.vertical_y_m = float(vertical_y_m)
        self.vertical_correction_limit_fraction = self._validated_limit_fraction(
            vertical_correction_limit_fraction,
            "vertical_correction_limit_fraction",
        )
        (
            self.vertical_positions_m,
            self.vertical_directions_body,
        ) = self._validate_vertical_geometry(
            vertical_positions_m,
            vertical_directions_body,
            self.vertical_x_m,
            self.vertical_y_m,
        )
        if (horizontal_limit_fraction is None) != (vertical_limit_fraction is None):
            raise ValueError(
                "horizontal and vertical limit fractions must be provided together"
            )
        if horizontal_limit_fraction is None:
            self.horizontal_limit_fraction = None
            self.vertical_limit_fraction = None
            self.thruster_pwm_limits = self._validate_thruster_pwm_limits(
                self._make_default_thruster_pwm_limits()
                if thruster_pwm_limits is None
                else thruster_pwm_limits
            )
            self._refresh_force_limits_from_pwm()
        else:
            self._set_command_fractions_unlocked(
                horizontal_limit_fraction, vertical_limit_fraction
            )

        # Backward-compatible names used by the existing BlueROV main program
        # and the previous allocator implementation.
        self.PWM_min = self.pwm_min_us
        self.PWM_mid = self.pwm_neutral_us
        self.PWM_max = self.pwm_max_us
        self.PWM_output_min = self.pwm_output_min_us
        self.PWM_output_max = self.pwm_output_max_us
        self.ccw_max_thrust = -self.reverse_max_thrust_n
        self.cw_max_thrust = self.forward_max_thrust_n

        self.tam = self._build_tam(
            lateral_x_m=float(lateral_x_m),
            lateral_y_m=float(lateral_y_m),
            vertical_x_m=float(vertical_x_m),
            vertical_y_m=float(vertical_y_m),
            vertical_positions_m=self.vertical_positions_m,
            vertical_directions_body=self.vertical_directions_body,
            angle_deg=float(lateral_angle_deg),
        )
        if np.linalg.matrix_rank(self.tam) < 6:
            raise ValueError("TAM is rank-deficient; check geometry parameters.")
        self.tam_pinv = np.linalg.pinv(self.tam, rcond=float(rcond))
        self.vertical_control_matrix = self.tam[np.ix_((2, 3, 4), range(4, 8))]
        if np.linalg.matrix_rank(self.vertical_control_matrix) < 3:
            raise ValueError(
                "vertical thruster geometry cannot independently control heave/roll/pitch"
            )
        self.vertical_control_pinv = np.linalg.pinv(
            self.vertical_control_matrix, rcond=float(rcond)
        )
        self.TAM = self.tam
        self.TAM_pinv = self.tam_pinv

        self._hardware: Optional[PCA9685Output] = None
        if pwm_writer is not None:
            self._pwm_writer = pwm_writer
            self.pwm = PWMCompatibilityOutput(self._pwm_writer)
        elif enable_hardware:
            self._hardware = PCA9685Output(
                address=pwm_address,
                frequency_hz=50,
                debug=self.debug,
            )
            self._pwm_writer = self._hardware.set_pulse_us
            self.pwm = self._hardware
        else:
            self._pwm_writer = None
            self.pwm = PWMCompatibilityOutput(None)

        self.last_result: Optional[AllocationResult] = None

    @staticmethod
    def _validate_channels(channels: Mapping[str, int]) -> Dict[str, int]:
        missing = [name for name in THRUSTER_NAMES if name not in channels]
        if missing:
            raise ValueError(f"Missing PCA9685 channels: {missing}")

        result = {name: int(channels[name]) for name in THRUSTER_NAMES}
        values = list(result.values())
        if any(channel < 0 or channel > 15 for channel in values):
            raise ValueError("Every PCA9685 channel must be in the range 0..15.")
        if len(set(values)) != len(values):
            raise ValueError("PCA9685 channel numbers must be unique.")
        return result

    @staticmethod
    def _validate_motor_signs(signs: Mapping[str, int]) -> np.ndarray:
        missing = [name for name in THRUSTER_NAMES if name not in signs]
        if missing:
            raise ValueError(f"Missing motor direction signs: {missing}")
        values = np.asarray([int(signs[name]) for name in THRUSTER_NAMES], dtype=float)
        if not np.all(np.isin(values, (-1.0, 1.0))):
            raise ValueError("Every motor direction sign must be +1 or -1.")
        return values

    def _validate_thruster_pwm_limits(
        self,
        limits: Mapping[str, Tuple[int, int]],
    ) -> Dict[str, Tuple[int, int]]:
        missing = [name for name in THRUSTER_NAMES if name not in limits]
        if missing:
            raise ValueError(f"Missing per-thruster PWM limits: {missing}")

        result: Dict[str, Tuple[int, int]] = {}
        for name in THRUSTER_NAMES:
            pair = limits[name]
            if len(pair) != 2:
                raise ValueError(f"{name} PWM limit must be (min_us, max_us).")
            low, high = int(pair[0]), int(pair[1])
            if not (
                self.pwm_output_min_us
                <= low
                < self.pwm_neutral_us
                < high
                <= self.pwm_output_max_us
            ):
                raise ValueError(
                    f"{name} PWM limit must stay inside the global output "
                    "window and contain neutral."
                )
            result[name] = (low, high)
        return result

    def _make_default_thruster_pwm_limits(self) -> Dict[str, Tuple[int, int]]:
        """
        Build directional limits inside the configured global PWM window.

        Full left strafe commands lateral_fl/lateral_rr in reverse and
        lateral_fr/lateral_rl forward. Those four directions receive the extra
        protection reported necessary during physical testing.
        """
        limits = {
            name: (self.pwm_output_min_us, self.pwm_output_max_us)
            for name in THRUSTER_NAMES
        }
        protected_reverse_min = max(self.pwm_output_min_us, 1250)
        protected_forward_max = min(self.pwm_output_max_us, 1750)
        limits["lateral_fl"] = (
            protected_reverse_min,
            self.pwm_output_max_us,
        )
        limits["lateral_rr"] = (
            protected_reverse_min,
            self.pwm_output_max_us,
        )
        limits["lateral_fr"] = (
            self.pwm_output_min_us,
            protected_forward_max,
        )
        limits["lateral_rl"] = (
            self.pwm_output_min_us,
            protected_forward_max,
        )
        return limits

    @staticmethod
    def _validated_limit_fraction(value: float, label: str) -> float:
        fraction = float(value)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 0.5:
            raise ValueError(f"{label} must be finite and in 0..0.5")
        return fraction

    @staticmethod
    def _validated_trim_percent(value: float, label: str) -> float:
        percent = float(value)
        if not math.isfinite(percent) or not -100.0 <= percent <= 100.0:
            raise ValueError(f"{label} must be finite and in -100..100")
        return percent

    def _refresh_force_limits_from_pwm(self) -> None:
        self.reverse_output_limits_n = np.asarray(
            [
                self._reverse_limit_from_pwm(self.thruster_pwm_limits[name][0], name)
                for name in THRUSTER_NAMES
            ],
            dtype=float,
        )
        self.forward_output_limits_n = np.asarray(
            [
                self._forward_limit_from_pwm(self.thruster_pwm_limits[name][1], name)
                for name in THRUSTER_NAMES
            ],
            dtype=float,
        )
        self.reverse_output_limit_n = float(np.min(self.reverse_output_limits_n))
        self.forward_output_limit_n = float(np.min(self.forward_output_limits_n))
        self.base_reverse_output_limits_n = self.reverse_output_limits_n.copy()
        self.base_forward_output_limits_n = self.forward_output_limits_n.copy()

    def _set_command_fractions_unlocked(
        self, horizontal_fraction: float, vertical_fraction: float
    ) -> None:
        horizontal = self._validated_limit_fraction(
            horizontal_fraction, "horizontal_limit_fraction"
        )
        vertical = self._validated_limit_fraction(
            vertical_fraction, "vertical_limit_fraction"
        )
        if vertical > self.vertical_correction_limit_fraction + 1e-12:
            raise ValueError(
                "vertical_limit_fraction cannot exceed the reserved correction ceiling"
            )
        base_fractions = [horizontal] * 4 + [vertical] * 4
        hard_fractions = [horizontal] * 4 + [
            self.vertical_correction_limit_fraction
        ] * 4
        base_force_limits = np.asarray(
            [self.reverse_max_thrust_n * value for value in base_fractions],
            dtype=float,
        )
        hard_force_limits = np.asarray(
            [self.reverse_max_thrust_n * value for value in hard_fractions],
            dtype=float,
        )
        pwm_limits = {}
        for name, force_limit in zip(THRUSTER_NAMES, hard_force_limits):
            reverse_start = self._start_pulse_us("reverse", name)
            forward_start = self._start_pulse_us("forward", name)
            reverse_ratio = force_limit / self.reverse_max_thrust_n
            forward_ratio = force_limit / self.forward_max_thrust_n
            low = round(
                reverse_start
                - reverse_ratio * (reverse_start - self.pwm_min_us)
            )
            high = round(
                forward_start
                + forward_ratio * (self.pwm_max_us - forward_start)
            )
            pwm_limits[name] = (int(low), int(high))

        self.thruster_pwm_limits = self._validate_thruster_pwm_limits(pwm_limits)
        # Base motion remains inside the operator's group fractions.  Only the
        # zero-sum vertical correction layer may use the larger hard window.
        # Exact force values avoid PWM round-trip asymmetry.
        self.base_reverse_output_limits_n = base_force_limits.copy()
        self.base_forward_output_limits_n = base_force_limits.copy()
        self.reverse_output_limits_n = hard_force_limits.copy()
        self.forward_output_limits_n = hard_force_limits.copy()
        self.reverse_output_limit_n = float(np.min(hard_force_limits))
        self.forward_output_limit_n = float(np.min(hard_force_limits))
        self.horizontal_limit_fraction = horizontal
        self.vertical_limit_fraction = vertical

    def set_command_fractions(
        self, horizontal_fraction: float, vertical_fraction: float
    ) -> None:
        """Apply group force limits immediately without touching arm state."""
        with self._lock:
            self._set_command_fractions_unlocked(
                horizontal_fraction, vertical_fraction
            )

    def set_forward_front_rear_trim_percent(self, value: float) -> None:
        """Apply forward-only vertical front/rear trim immediately."""
        percent = self._validated_trim_percent(
            value, "forward_front_rear_trim_percent"
        )
        with self._lock:
            self.forward_front_rear_trim_percent = percent

    def set_backward_front_rear_trim_percent(self, value: float) -> None:
        """Apply backward-only vertical front/rear trim immediately."""
        percent = self._validated_trim_percent(
            value, "backward_front_rear_trim_percent"
        )
        with self._lock:
            self.backward_front_rear_trim_percent = percent

    def set_left_lateral_surge_trim_percent(self, value: float) -> None:
        """Apply the surge feed-forward used only while translating left."""
        percent = self._validated_trim_percent(
            value, "left_lateral_surge_trim_percent"
        )
        with self._lock:
            self.left_lateral_surge_trim_percent = percent

    def set_right_lateral_surge_trim_percent(self, value: float) -> None:
        """Apply the surge feed-forward used only while translating right."""
        percent = self._validated_trim_percent(
            value, "right_lateral_surge_trim_percent"
        )
        with self._lock:
            self.right_lateral_surge_trim_percent = percent

    def _validate_pwm_and_thrust_config(self) -> None:
        if not self.pwm_min_us < self.pwm_neutral_us < self.pwm_max_us:
            raise ValueError("PWM must satisfy min < neutral < max.")
        if not (
            self.pwm_min_us
            <= self.pwm_output_min_us
            < self.pwm_neutral_us
            < self.pwm_output_max_us
            <= self.pwm_max_us
        ):
            raise ValueError(
                "Safe PWM must satisfy physical_min <= output_min < neutral "
                "< output_max <= physical_max."
            )
        if self.reverse_max_thrust_n <= 0.0 or self.forward_max_thrust_n <= 0.0:
            raise ValueError("Forward and reverse maximum thrust must be positive.")
        for name, start in (("reverse_start_us", self.reverse_start_us),
                            ("lateral_reverse_start_us", self.lateral_reverse_start_us)):
            if start is not None and not self.pwm_min_us < start < self.pwm_neutral_us:
                raise ValueError(
                    f"{name} must be strictly between min and neutral."
                )
            if start is not None and self.pwm_output_min_us >= start:
                raise ValueError(f"pwm_output_min_us must be below {name}.")
        for name, start in (("forward_start_us", self.forward_start_us),
                            ("lateral_forward_start_us", self.lateral_forward_start_us)):
            if start is not None and not self.pwm_neutral_us < start < self.pwm_max_us:
                raise ValueError(
                    f"{name} must be strictly between neutral and max."
                )
            if start is not None and self.pwm_output_max_us <= start:
                raise ValueError(f"pwm_output_max_us must be above {name}.")

    def _start_pulse_us(self, direction: str, thruster_name: Optional[str]) -> int:
        if direction not in ("forward", "reverse"):
            raise ValueError("direction must be forward or reverse")
        lateral = thruster_name is not None and thruster_name.startswith("lateral_")
        specific = (
            self.lateral_forward_start_us
            if direction == "forward" else self.lateral_reverse_start_us
        )
        generic = self.forward_start_us if direction == "forward" else self.reverse_start_us
        neutral = self.pwm_neutral_us
        if lateral and specific is not None:
            return specific
        return neutral if generic is None else generic

    def _reverse_limit_from_pwm(self, pwm_limit_us: int, thruster_name: Optional[str] = None) -> float:
        """Maximum safe reverse thrust magnitude implied by output PWM minimum."""
        start = self._start_pulse_us("reverse", thruster_name)
        ratio = (
            (start - int(pwm_limit_us))
            / float(start - self.pwm_min_us)
        )
        return self.reverse_max_thrust_n * max(0.0, min(1.0, ratio))

    def _forward_limit_from_pwm(self, pwm_limit_us: int, thruster_name: Optional[str] = None) -> float:
        """Maximum safe forward thrust implied by output PWM maximum."""
        start = self._start_pulse_us("forward", thruster_name)
        ratio = (
            (int(pwm_limit_us) - start)
            / float(self.pwm_max_us - start)
        )
        return self.forward_max_thrust_n * max(0.0, min(1.0, ratio))

    @staticmethod
    def _validate_vertical_geometry(
        positions: Optional[Mapping[str, Sequence[float]]],
        directions: Optional[Mapping[str, Sequence[float]]],
        vertical_x_m: float,
        vertical_y_m: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if (positions is None) != (directions is None):
            raise ValueError(
                "vertical positions and directions must be configured together"
            )
        if positions is None:
            x_arm = float(vertical_x_m)
            y_arm = float(vertical_y_m)
            positions = {
                "vertical_fl": (x_arm, -y_arm, 0.0),
                "vertical_fr": (x_arm, y_arm, 0.0),
                "vertical_rl": (-x_arm, -y_arm, 0.0),
                "vertical_rr": (-x_arm, y_arm, 0.0),
            }
            directions = {
                name: (0.0, 0.0, 1.0) for name in VERTICAL_THRUSTER_NAMES
            }

        missing_positions = [
            name for name in VERTICAL_THRUSTER_NAMES if name not in positions
        ]
        missing_directions = [
            name for name in VERTICAL_THRUSTER_NAMES if name not in directions
        ]
        extras = sorted(
            (set(positions) | set(directions)) - set(VERTICAL_THRUSTER_NAMES)
        )
        if missing_positions or missing_directions or extras:
            raise ValueError(
                "vertical geometry keys must be exactly vertical_fl/fr/rl/rr"
            )

        position_rows = []
        direction_rows = []
        for name in VERTICAL_THRUSTER_NAMES:
            position = np.asarray(positions[name], dtype=float)
            direction = np.asarray(directions[name], dtype=float)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError(f"{name} position must contain three finite metres")
            if direction.shape != (3,) or not np.all(np.isfinite(direction)):
                raise ValueError(f"{name} direction must contain three finite values")
            if np.any(np.abs(position) > 2.0):
                raise ValueError(f"{name} position must be relative to the COM in metres")
            magnitude = float(np.linalg.norm(direction))
            if abs(magnitude - 1.0) > 1e-3:
                raise ValueError(f"{name} direction must be a unit vector")
            direction = direction / magnitude
            if direction[2] < 0.5:
                raise ValueError(
                    f"{name} nominal direction must retain a positive body-z component"
                )
            position_rows.append(position)
            direction_rows.append(direction)
        return np.asarray(position_rows), np.asarray(direction_rows)

    @staticmethod
    def _build_tam(
        *,
        lateral_x_m: float,
        lateral_y_m: float,
        vertical_x_m: float,
        vertical_y_m: float,
        vertical_positions_m: np.ndarray,
        vertical_directions_body: np.ndarray,
        angle_deg: float,
    ) -> np.ndarray:
        """
        Build a 6x8 TAM where tau = TAM @ nominal_thrusts.

        Positions are symmetric about the vehicle center:
            FL=(+x,-y), FR=(+x,+y), RL=(-x,-y), RR=(-x,+y)
        """
        dimensions = (
            lateral_x_m,
            lateral_y_m,
            vertical_x_m,
            vertical_y_m,
        )
        if not all(math.isfinite(v) and v > 0.0 for v in dimensions):
            raise ValueError("All x/y geometry dimensions must be finite and > 0.")
        if not math.isfinite(angle_deg) or not 0.0 < angle_deg < 90.0:
            raise ValueError("lateral_angle_deg must be between 0 and 90.")

        angle = math.radians(angle_deg)
        c = math.cos(angle)
        s = math.sin(angle)
        yaw_arm = lateral_x_m * s + lateral_y_m * c

        tam = np.zeros((6, 8), dtype=float)

        # Four 45-degree lateral thrusters.
        tam[0, 0:4] = (c, c, c, c)                         # Fx
        tam[1, 0:4] = (s, -s, -s, s)                       # Fy
        tam[5, 0:4] = (yaw_arm, -yaw_arm, yaw_arm, -yaw_arm)  # Mz

        # Every vertical column is built from its measured body-frame thrust
        # line.  Both the lever arm and force rotate with the vehicle, so the
        # body wrench remains [d_i; r_i x d_i] even while the ROV is tilted.
        for local, (position, direction) in enumerate(
            zip(vertical_positions_m, vertical_directions_body)
        ):
            column = 4 + local
            tam[0:3, column] = direction
            tam[3:6, column] = np.cross(position, direction)
        return tam

    @staticmethod
    def _finite_tau(values: Sequence[float]) -> np.ndarray:
        tau = np.asarray(values, dtype=float)
        if tau.shape != (6,):
            raise ValueError("tau must contain exactly [Fx, Fy, Fz, Mx, My, Mz].")
        if not np.all(np.isfinite(tau)):
            raise ValueError("tau contains NaN or Infinity.")
        return tau

    def _scale_for_indices(
        self, motor_thrusts: np.ndarray, indices, *, base_limits: bool = True
    ) -> float:
        """Return the safe common scale for one independent thruster group."""
        forward_limits = (
            self.base_forward_output_limits_n
            if base_limits else self.forward_output_limits_n
        )
        reverse_limits = (
            self.base_reverse_output_limits_n
            if base_limits else self.reverse_output_limits_n
        )
        scale = 1.0
        for index in indices:
            value = motor_thrusts[index]
            if value > 0.0:
                scale = min(
                    scale,
                    forward_limits[index] / float(value),
                )
            elif value < 0.0:
                scale = min(
                    scale,
                    reverse_limits[index] / abs(float(value)),
                )
        return max(0.0, min(1.0, scale))

    def _vertical_nominal_thrusts(
        self, tau_z: float, M_x: float, M_y: float
    ) -> np.ndarray:
        """Allocate one consistent heave/roll/pitch request to four actuators.

        The equal command is retained as the known-good heave reference.  A
        minimum-norm correction then solves the exact configured
        ``[Fz, Mx, My]`` matrix made from each ``[d_i; r_i x d_i]`` column.
        Roll-only, pitch-only and dual hold are projections of this same solve.
        """
        requested = np.asarray((tau_z, M_x, M_y), dtype=float)
        if not np.all(np.isfinite(requested)):
            raise ValueError("vertical wrench request must be finite")
        baseline = self._vertical_heave_baseline(tau_z)
        residual = requested - self.vertical_control_matrix @ baseline
        correction = self.vertical_control_pinv @ residual
        return baseline + correction

    def _vertical_heave_baseline(self, tau_z: float) -> np.ndarray:
        z_effectiveness = float(np.sum(self.vertical_control_matrix[0]))
        if z_effectiveness <= 1e-9:
            raise ValueError("vertical thrusters have no positive heave authority")
        return np.full(4, float(tau_z) / z_effectiveness, dtype=float)

    def _desaturate_vertical_correction(
        self, nominal_vertical: np.ndarray, nominal_baseline: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Keep safe equal heave first, then fit the combined attitude vector.

        A common factor is applied to the zero-sum roll/pitch correction.  This
        preserves both requested moment directions and their ratio.  It also
        prevents a corner limit from scaling away the known-good equal hover
        baseline.  Only if the baseline itself is outside the motor window is
        that baseline reduced first.
        """
        nominal = np.asarray(nominal_vertical, dtype=float)
        baseline_nominal = np.asarray(nominal_baseline, dtype=float)
        if (
            nominal.shape != (4,)
            or baseline_nominal.shape != (4,)
            or not np.all(np.isfinite(nominal))
            or not np.all(np.isfinite(baseline_nominal))
        ):
            raise ValueError("vertical thrust vector must contain four finite values")

        nominal_correction = nominal - baseline_nominal
        signs = self.motor_signs[4:8]
        motor_baseline = signs * baseline_nominal

        baseline_scale = 1.0
        for local, index in enumerate(range(4, 8)):
            value = motor_baseline[local]
            if value > 0.0:
                baseline_scale = min(
                    baseline_scale,
                    self.base_forward_output_limits_n[index] / value,
                )
            elif value < 0.0:
                baseline_scale = min(
                    baseline_scale,
                    self.base_reverse_output_limits_n[index] / abs(value),
                )
        baseline_scale = max(0.0, min(1.0, baseline_scale))
        motor_baseline *= baseline_scale

        motor_correction = signs * nominal_correction
        correction_scale = 1.0
        for local, index in enumerate(range(4, 8)):
            base = motor_baseline[local]
            correction = motor_correction[local]
            low = -self.base_reverse_output_limits_n[index]
            high = self.base_forward_output_limits_n[index]
            if correction > 0.0:
                correction_scale = min(
                    correction_scale, max(0.0, (high - base) / correction)
                )
            elif correction < 0.0:
                correction_scale = min(
                    correction_scale, max(0.0, (low - base) / correction)
                )
        correction_scale = max(0.0, min(1.0, correction_scale))
        return (
            motor_baseline + correction_scale * motor_correction,
            min(baseline_scale, correction_scale),
        )

    def _add_vertical_priority_correction(
        self, motor_base: np.ndarray, motor_correction: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Add attitude/feed-forward correction inside the independent hard window.

        The base vector has already been limited by ``vertical_limit_fraction``.
        A common scale now gives the four zero-sum correction commands access to
        the reserved window up to ``vertical_correction_limit_fraction`` while
        preserving roll/pitch direction and ratio.
        """
        base = np.asarray(motor_base, dtype=float)
        correction = np.asarray(motor_correction, dtype=float)
        if (
            base.shape != (4,)
            or correction.shape != (4,)
            or not np.all(np.isfinite(base))
            or not np.all(np.isfinite(correction))
        ):
            raise ValueError("vertical priority vectors must contain four finite values")
        scale = 1.0
        for local, index in enumerate(range(4, 8)):
            low = -self.reverse_output_limits_n[index]
            high = self.forward_output_limits_n[index]
            delta = correction[local]
            if delta > 0.0:
                scale = min(scale, max(0.0, (high - base[local]) / delta))
            elif delta < 0.0:
                scale = min(scale, max(0.0, (low - base[local]) / delta))
        scale = max(0.0, min(1.0, scale))
        return base + scale * correction, scale

    def thrust_to_pwm(
        self,
        thrust_n: float,
        thruster_name: Optional[str] = None,
    ) -> int:
        """Map -55..+65 N to 1100..1540..1900 us by default."""
        thrust = float(thrust_n)
        if not math.isfinite(thrust):
            raise ValueError("Thruster command is NaN or Infinity.")
        thrust = max(
            -self.reverse_max_thrust_n,
            min(self.forward_max_thrust_n, thrust),
        )

        if abs(thrust) < 1e-9:
            return self.pwm_neutral_us

        if thrust > 0.0:
            start = self._start_pulse_us("forward", thruster_name)
            ratio = thrust / self.forward_max_thrust_n
            pwm = start + ratio * (self.pwm_max_us - start)
        else:
            start = self._start_pulse_us("reverse", thruster_name)
            ratio = abs(thrust) / self.reverse_max_thrust_n
            pwm = start - ratio * (start - self.pwm_min_us)

        # Final hardware-side safety clamp. The vector has already been scaled
        # to these bounds above, so this is only a last line of protection and
        # normally does not change the requested thrust ratio.
        output_min = self.pwm_output_min_us
        output_max = self.pwm_output_max_us
        if thruster_name is not None:
            if thruster_name not in self.thruster_pwm_limits:
                raise ValueError(f"Unknown thruster name: {thruster_name}")
            output_min, output_max = self.thruster_pwm_limits[thruster_name]

        return int(
            round(
                max(
                    output_min,
                    min(output_max, pwm),
                )
            )
        )

    def _write_all(self, pwm_us: np.ndarray) -> None:
        if self._pwm_writer is None:
            return
        for name, pulse in zip(THRUSTER_NAMES, pwm_us):
            self._pwm_writer(self.channels[name], int(pulse))

    def allocate(
        self,
        tau_x: float = 0.0,
        tau_y: float = 0.0,
        tau_z: float = 0.0,
        M_z: float = 0.0,
        M_x: float = 0.0,
        M_y: float = 0.0,
        *,
        priority_M_x: float = 0.0,
        priority_M_y: float = 0.0,
        write_output: bool = True,
    ) -> AllocationResult:
        """
        Allocate [Fx, Fy, Fz, Mx, My, Mz] to all eight thrusters.

        For your Mode-2 joystick:
            left Y  -> tau_x
            left X  -> M_z
            right Y -> tau_z
            right X -> tau_y
        """
        tau = self._finite_tau((tau_x, tau_y, tau_z, M_x, M_y, M_z))
        priority = np.asarray((float(priority_M_x), float(priority_M_y)))
        if not np.all(np.isfinite(priority)):
            raise ValueError("priority roll/pitch moments must be finite")

        with self._lock:
            lateral_surge_trim = 0.0
            if tau[1] > 1e-12:
                lateral_surge_trim = self.right_lateral_surge_trim_percent
            elif tau[1] < -1e-12:
                lateral_surge_trim = self.left_lateral_surge_trim_percent
            if abs(lateral_surge_trim) > 1e-12:
                tau = tau.copy()
                tau[0] += abs(float(tau[1])) * lateral_surge_trim / 100.0
            nominal_unscaled = self.tam_pinv @ tau
            # Use the explicit four-corner mechanics instead of relying on a
            # numerically opaque pseudoinverse for the vertical group.
            vertical_baseline = self._vertical_heave_baseline(tau[2])
            vertical_base = self._vertical_nominal_thrusts(
                tau[2], tau[3] - priority[0], tau[4] - priority[1]
            )
            vertical_priority = self.vertical_control_pinv @ np.asarray(
                (0.0, priority[0], priority[1]), dtype=float
            )
            surge_trim = 0.0
            if tau[0] > 1e-12:
                surge_trim = self.forward_front_rear_trim_percent
            elif tau[0] < -1e-12:
                surge_trim = self.backward_front_rear_trim_percent
            if abs(surge_trim) > 1e-12:
                # Keep the four diagonal thrusters on their normal equal Fx
                # allocation. Direction-specific operator trim adds a
                # zero-net-Fz pitch correction on the four vertical thrusters:
                # negative strengthens the front pair, positive the rear.
                pure_surge_tau = np.zeros(6, dtype=float)
                pure_surge_tau[0] = tau[0]
                pure_surge = self.tam_pinv @ pure_surge_tau
                pure_surge_motor = self.motor_signs * pure_surge
                surge_scale = self._scale_for_indices(
                    pure_surge_motor, range(0, 4)
                )
                surge_per_thruster = float(
                    np.mean(np.abs(pure_surge[0:4]))
                ) * surge_scale
                vertical_delta = (
                    surge_per_thruster
                    * surge_trim
                    / 200.0
                )
                vertical_priority[0:2] -= vertical_delta
                vertical_priority[2:4] += vertical_delta

            nominal_unscaled[4:8] = vertical_base + vertical_priority

            # Convert nominal geometry force to actual motor command direction.
            # Horizontal saturation is independent.  For the vertical group,
            # retain the equal heave/hover baseline and scale only the combined
            # zero-sum roll/pitch correction when corner authority is exhausted.
            motor_unscaled = self.motor_signs * nominal_unscaled
            motor_thrusts = motor_unscaled.copy()
            horizontal_scale = self._scale_for_indices(
                motor_thrusts, range(0, 4)
            )
            motor_thrusts[0:4] *= horizontal_scale
            motor_vertical_base, vertical_base_scale = (
                self._desaturate_vertical_correction(
                    vertical_base, vertical_baseline
                )
            )
            motor_vertical_priority = self.motor_signs[4:8] * vertical_priority
            motor_thrusts[4:8], vertical_priority_scale = (
                self._add_vertical_priority_correction(
                    motor_vertical_base, motor_vertical_priority
                )
            )
            scale = min(
                horizontal_scale, vertical_base_scale, vertical_priority_scale
            )
            nominal_thrusts = self.motor_signs * motor_thrusts
            pwm_us = np.asarray(
                [
                    self.thrust_to_pwm(value, name)
                    for name, value in zip(THRUSTER_NAMES, motor_thrusts)
                ],
                dtype=int,
            )
            achieved_tau = self.tam @ nominal_thrusts

            result = AllocationResult(
                requested_tau=tau.copy(),
                achieved_tau=achieved_tau,
                nominal_thrusts=nominal_thrusts,
                motor_thrusts=motor_thrusts,
                pwm_us=pwm_us,
                scale=scale,
            )

            try:
                if write_output:
                    self._write_all(pwm_us)
            except Exception:
                # Best-effort neutral command before propagating the hardware error.
                try:
                    self._write_all(
                        np.full(len(THRUSTER_NAMES), self.pwm_neutral_us, dtype=int)
                    )
                finally:
                    raise

            self.last_result = result
            if self.debug:
                print(result.as_dict())
            return result

    def allocate_thrust(
        self,
        tau_x: float = 0.0,
        tau_y: float = 0.0,
        tau_z: float = 0.0,
        M_z: float = 0.0,
        M_x: float = 0.0,
        M_y: float = 0.0,
        *,
        priority_M_x: float = 0.0,
        priority_M_y: float = 0.0,
        write_output: bool = True,
    ) -> AllocationResult:
        """Compatibility alias for allocate()."""
        return self.allocate(
            tau_x=tau_x,
            tau_y=tau_y,
            tau_z=tau_z,
            M_z=M_z,
            M_x=M_x,
            M_y=M_y,
            priority_M_x=priority_M_x,
            priority_M_y=priority_M_y,
            write_output=write_output,
        )

    def set_wrench(
        self,
        tau_x: float = 0.0,
        tau_y: float = 0.0,
        tau_z: float = 0.0,
        M_x: float = 0.0,
        M_y: float = 0.0,
        M_z: float = 0.0,
        *,
        priority_M_x: float = 0.0,
        priority_M_y: float = 0.0,
        verbose: bool = False,
        write_output: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        """
        Allocate and output one wrench using the interface expected by main.py.

        Returns:
            force_dict: signed motor thrust command for every named thruster
            pwm_dict: PWM pulse in microseconds for every named thruster
        """
        result = self.allocate(
            tau_x=tau_x,
            tau_y=tau_y,
            tau_z=tau_z,
            M_x=M_x,
            M_y=M_y,
            M_z=M_z,
            priority_M_x=priority_M_x,
            priority_M_y=priority_M_y,
            write_output=write_output,
        )
        force_dict = {
            name: float(value)
            for name, value in zip(self.thruster_names, result.motor_thrusts)
        }
        pwm_dict = {
            name: int(value)
            for name, value in zip(self.thruster_names, result.pwm_us)
        }
        if verbose:
            for name in self.thruster_names:
                print(
                    f"{name:>14s}: {force_dict[name]:+7.2f} N"
                    f" -> {pwm_dict[name]:4d} us"
                )
        return force_dict, pwm_dict

    def stop_all(self) -> None:
        """Immediately command neutral PWM on every configured channel."""
        neutral = np.full(len(THRUSTER_NAMES), self.pwm_neutral_us, dtype=int)
        with self._lock:
            self._write_all(neutral)

    def set_aux_pwm(self, channel: int, pulse_us: int) -> None:
        """Write one non-thruster PCA9685 channel without touching motors 0..7."""
        channel = int(channel)
        pulse_us = int(pulse_us)
        if not 0 <= channel <= 15:
            raise ValueError("auxiliary PWM channel must be in 0..15")
        if channel in self.channels.values():
            raise ValueError("auxiliary PWM channel must not overlap a thruster channel")
        if not 500 <= pulse_us <= 2500:
            raise ValueError("auxiliary PWM pulse must be in 500..2500 us")
        with self._lock:
            self.pwm.set_pulse_us(channel, pulse_us)

    def close(self, *, stop_thrusters: bool = True) -> None:
        """Optionally neutralize thrusters before closing the hardware driver."""
        with self._lock:
            if stop_thrusters:
                self.stop_all()
            if self._hardware is not None:
                self._hardware.close()
                self._hardware = None


if __name__ == "__main__":
    # Desktop numerical self-test; no hardware is accessed.
    allocator = ThrustAllocator(debug=True, enable_hardware=False)
    allocator.allocate(tau_x=100.0, write_output=False)
    allocator.allocate(tau_y=100.0, write_output=False)
    allocator.allocate(tau_z=100.0, write_output=False)
    allocator.allocate(M_z=20.0, write_output=False)
