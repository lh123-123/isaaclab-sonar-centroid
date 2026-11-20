#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROV closed-loop control: depth and opt-in attitude/speed loops."""

from __future__ import annotations

import math
import time

from Heading_Control import HeadingControl
from config_loader import DEFAULT_CONFIG_PATH, load_config, update_config_values


def _clamp(value, lower, upper):
    return max(lower, min(upper, float(value)))


class PIDController:
    """Small PID with anti-windup and an explicit output direction."""

    def __init__(self, kp, ki, kd, *, sample_time=0.1, output_limit=200.0,
                 windup_guard=200.0, direction=1.0):
        self.Kp = float(kp)
        self.Ki = float(ki)
        self.Kd = float(kd)
        self.sample_time = float(sample_time)
        self.output_limit = abs(float(output_limit))
        self.windup_guard = abs(float(windup_guard))
        self.direction = float(direction)
        self.clear()

    def clear(self):
        self.last_time = time.monotonic()
        self.last_error = 0.0
        self.integral = 0.0
        self.output = 0.0

    def prime(self, target, feedback):
        """Reset without a derivative kick when a loop is enabled."""
        self.last_time = time.monotonic()
        self.last_error = float(target) - float(feedback)
        self.integral = 0.0
        self.output = 0.0

    @staticmethod
    def _wrapped_error(target, feedback):
        return (float(target) - float(feedback) + 180.0) % 360.0 - 180.0

    def prime_angle(self, target, feedback):
        self.last_time = time.monotonic()
        self.last_error = self._wrapped_error(target, feedback)
        self.integral = 0.0
        self.output = 0.0

    def set_output_limit(self, value):
        limit = abs(float(value))
        if not math.isfinite(limit) or not 0.1 <= limit <= 20.0:
            raise ValueError("attitude moment limit must be in 0.1..20 Nm")
        self.output_limit = limit
        self.windup_guard = limit
        self.integral = _clamp(self.integral, -limit, limit)
        self.output = _clamp(self.output, -limit, limit)

    def set_pid(self, values):
        self.Kp, self.Ki, self.Kd = map(float, values)
        self.clear()

    def update(self, target, feedback):
        error = float(target) - float(feedback)
        now = time.monotonic()
        dt = now - self.last_time
        if dt < self.sample_time:
            return self.output

        self.integral = _clamp(
            self.integral + self.Ki * error * dt,
            -self.windup_guard,
            self.windup_guard,
        )
        derivative = (error - self.last_error) / dt if dt > 0.0 else 0.0
        raw = self.Kp * error + self.integral + self.Kd * derivative
        self.output = _clamp(
            self.direction * raw, -self.output_limit, self.output_limit
        )
        self.last_error = error
        self.last_time = now
        return self.output

    def update_angle(self, target, feedback):
        """PID for bounded Euler angles without a +/-180 derivative jump."""
        error = self._wrapped_error(target, feedback)
        now = time.monotonic()
        dt = now - self.last_time
        if dt < self.sample_time:
            return self.output
        self.integral = _clamp(
            self.integral + self.Ki * error * dt,
            -self.windup_guard,
            self.windup_guard,
        )
        error_step = (error - self.last_error + 180.0) % 360.0 - 180.0
        derivative = error_step / dt if dt > 0.0 else 0.0
        raw = self.Kp * error + self.integral + self.Kd * derivative
        self.output = _clamp(
            self.direction * raw, -self.output_limit, self.output_limit
        )
        self.last_error = error
        self.last_time = now
        return self.output


class DepthPID(PIDController):
    """Depth PI-D to vertical force; +z/down matches the vehicle convention.

    The derivative term damps the measured vertical speed instead of
    differentiating the quantized pressure depth.  Conditional integration and
    an explicit force limit prevent hidden wind-up behind the allocator's
    lower vertical-thrust ceiling.
    """

    def __init__(
        self,
        Kp=120.0,
        Ki=10.0,
        Kd=0.0,
        sample_time=0.1,
        output_limit=60.0,
        rate_filter_alpha=1.0,
        rate_deadband_mps=0.01,
        integral_limit_fraction=0.5,
        release_brake_exit_speed_mps=0.03,
    ):
        self.rate_filter_alpha = float(rate_filter_alpha)
        self.rate_deadband_mps = float(rate_deadband_mps)
        self.integral_limit_fraction = float(integral_limit_fraction)
        self.release_brake_exit_speed_mps = float(
            release_brake_exit_speed_mps
        )
        if (
            not math.isfinite(self.release_brake_exit_speed_mps)
            or self.release_brake_exit_speed_mps < self.rate_deadband_mps
        ):
            raise ValueError(
                "release brake exit speed must be finite and at least the rate deadband"
            )
        self.filtered_rate_mps = 0.0
        self._release_brake_pending = False
        self._release_braking = False
        # A deeper target is positive depth error and therefore positive/down Fz.
        super().__init__(Kp, Ki, Kd, sample_time=sample_time,
                         output_limit=output_limit, direction=1.0)

    def clear(self):
        super().clear()
        self.filtered_rate_mps = 0.0
        self._release_brake_pending = False
        self._release_braking = False

    def set_release_brake_exit_speed(self, value):
        speed = abs(float(value))
        if (
            not math.isfinite(speed)
            or speed < self.rate_deadband_mps
            or speed > 0.30
        ):
            raise ValueError(
                "release brake exit speed must be in rate_deadband..0.30 m/s"
            )
        self.release_brake_exit_speed_mps = speed

    def set_force_limit(self, value):
        limit = abs(float(value))
        if not math.isfinite(limit) or not 1.0 <= limit <= 200.0:
            raise ValueError("depth force limit must be in 1..200 N")
        self.output_limit = limit
        self.windup_guard = limit
        self.integral = _clamp(self.integral, -limit, limit)
        self.output = _clamp(self.output, -limit, limit)

    def freeze_for_manual_override(self, vertical_speed_mps=0.0):
        """Freeze P/I state while retaining velocity braking for stick release."""
        measured_rate = float(vertical_speed_mps)
        if not math.isfinite(measured_rate):
            raise ValueError("vertical speed must be finite")
        self.filtered_rate_mps = measured_rate
        self._release_brake_pending = True
        self._release_braking = False
        damped_rate = (
            0.0
            if abs(measured_rate) <= self.rate_deadband_mps
            else measured_rate
        )
        applied_integral = self.integral
        if applied_integral * damped_rate > 0.0:
            applied_integral = 0.0
        self.output = _clamp(
            applied_integral - self.Kd * damped_rate,
            -self.output_limit,
            self.output_limit,
        )
        self.last_error = 0.0
        self.last_time = time.monotonic()
        return self.output

    def update_depth(
        self,
        target_m,
        feedback_m,
        vertical_speed_mps=0.0,
        position_deadband_m=0.0,
    ):
        raw_error = float(target_m) - float(feedback_m)
        measured_rate = float(vertical_speed_mps)
        deadband = abs(float(position_deadband_m))
        if not all(
            math.isfinite(value) for value in (raw_error, measured_rate, deadband)
        ):
            raise ValueError(
                "depth error, vertical speed and deadband must be finite"
            )

        # The position deadband suppresses pressure-sensor chatter, but must not
        # disable velocity braking or erase the learned neutral-buoyancy bias.
        error = 0.0 if abs(raw_error) <= deadband else raw_error

        now = time.monotonic()
        dt = now - self.last_time
        if dt < self.sample_time:
            return self.output

        alpha = _clamp(self.rate_filter_alpha, 0.0, 1.0)
        self.filtered_rate_mps += alpha * (
            measured_rate - self.filtered_rate_mps
        )
        damped_rate = (
            0.0
            if abs(self.filtered_rate_mps) <= self.rate_deadband_mps
            else self.filtered_rate_mps
        )

        if self._release_brake_pending:
            self._release_brake_pending = False
            self._release_braking = True
        if (
            self._release_braking
            and abs(damped_rate) <= self.release_brake_exit_speed_mps
        ):
            self._release_braking = False

        proportional_and_damping = self.Kp * error - self.Kd * damped_rate

        integral_limit = self.output_limit * _clamp(
            self.integral_limit_fraction, 0.0, 1.0
        )
        if not self._release_braking:
            candidate_integral = _clamp(
                self.integral + self.Ki * error * dt,
                -integral_limit,
                integral_limit,
            )
            candidate_raw = proportional_and_damping + candidate_integral
            saturated_high = candidate_raw > self.output_limit
            saturated_low = candidate_raw < -self.output_limit
            if not (saturated_high or saturated_low) or (
                saturated_high and error < 0.0
            ) or (saturated_low and error > 0.0):
                self.integral = candidate_integral

        applied_integral = self.integral
        if (
            self._release_braking
            and applied_integral * damped_rate > 0.0
        ):
            # Keep the learned static buoyancy/weight bias in memory, but do not
            # let it push in the same direction as post-stick inertia.  Once Vz
            # is nearly settled, the full bias returns automatically.
            applied_integral = 0.0

        self.output = _clamp(
            proportional_and_damping + applied_integral,
            -self.output_limit,
            self.output_limit,
        )
        self.last_error = error
        self.last_time = now
        return self.output


class AttitudeAxisController(PIDController):
    """One axis of the yaw-free reduced-attitude PI-D controller.

    ``error_deg`` is supplied by the geometric tilt calculation below.  The
    derivative term uses the calibrated body angular rate directly, avoiding
    the noise amplification and timing sensitivity of differentiating Euler
    samples.  The integral is a bounded moment contribution and is accepted
    only when it does not drive an already saturated output farther outward.
    """

    def __init__(
        self,
        kp,
        ki,
        kd,
        *,
        sample_time=0.1,
        output_limit=2.0,
        error_deadband_deg=0.15,
        rate_deadband_dps=0.25,
        rate_filter_alpha=0.35,
        integral_limit_fraction=0.35,
    ):
        self.error_deadband_deg = float(error_deadband_deg)
        self.rate_deadband_dps = float(rate_deadband_dps)
        self.rate_filter_alpha = float(rate_filter_alpha)
        self.integral_limit_fraction = float(integral_limit_fraction)
        self.filtered_rate_dps = 0.0
        super().__init__(
            kp,
            ki,
            kd,
            sample_time=sample_time,
            output_limit=output_limit,
            windup_guard=abs(float(output_limit)),
        )

    @staticmethod
    def _deadband(value, width):
        value = float(value)
        width = max(0.0, float(width))
        if abs(value) <= width:
            return 0.0
        return math.copysign(abs(value) - width, value)

    def clear(self):
        super().clear()
        self.filtered_rate_dps = 0.0

    def prime_geometric(self, error_deg, body_rate_dps):
        self.last_time = time.monotonic()
        self.last_error = float(error_deg)
        self.filtered_rate_dps = float(body_rate_dps)
        self.integral = 0.0
        self.output = 0.0

    def update_geometric(self, error_deg, body_rate_dps):
        error = self._deadband(error_deg, self.error_deadband_deg)
        measured_rate = float(body_rate_dps)
        if not all(math.isfinite(value) for value in (error, measured_rate)):
            raise ValueError("attitude error and body rate must be finite")

        now = time.monotonic()
        dt = now - self.last_time
        if dt < self.sample_time:
            return self.output

        alpha = _clamp(self.rate_filter_alpha, 0.0, 1.0)
        self.filtered_rate_dps += alpha * (
            measured_rate - self.filtered_rate_dps
        )
        damped_rate = self._deadband(
            self.filtered_rate_dps, self.rate_deadband_dps
        )
        proportional_and_damping = self.Kp * error - self.Kd * damped_rate

        integral_limit = self.output_limit * _clamp(
            self.integral_limit_fraction, 0.0, 1.0
        )
        candidate_integral = _clamp(
            self.integral + self.Ki * error * dt,
            -integral_limit,
            integral_limit,
        )
        candidate_raw = proportional_and_damping + candidate_integral
        saturated_high = candidate_raw > self.output_limit
        saturated_low = candidate_raw < -self.output_limit
        # Integrate while unsaturated, or while the current error unwinds an
        # already saturated command.  This prevents both permanent bias error
        # and classical integral windup.
        if not (saturated_high or saturated_low) or (
            saturated_high and error < 0.0
        ) or (saturated_low and error > 0.0):
            self.integral = candidate_integral

        raw = proportional_and_damping + self.integral
        self.output = _clamp(raw, -self.output_limit, self.output_limit)
        self.last_error = error
        self.last_time = now
        return self.output


class Controller:
    """Compute only the loops whose feature, mode and sensor validity allow it."""

    PID_TYPE_TO_PREFIX = {
        1: "depth",
        2: "heading",
        3: "speed",
        4: "roll",
        5: "pitch",
        6: "yaw",
    }

    def __init__(self, config_path=DEFAULT_CONFIG_PATH, feature_overrides=None):
        self.config_path = str(config_path)
        self.config = load_config(config_path)
        self.feature_overrides = dict(feature_overrides or {})
        self.depth_hold_deadband_m = 0.0
        self.depth_hold_filter_alpha = 1.0
        self.depth_max_force_n = 60.0
        self.depth_release_brake_exit_speed_mps = 0.03
        self._depth_hold_filtered_m = None

        self.heading_pid_params = (2.0, 0.2, 0.3)
        self.depth_pid_params = (120.0, 10.0, 0.0)
        self.speed_pid_params = (40.0, 2.0, 0.0)
        self.roll_pid_params = (0.16, 0.02, 0.08)
        self.pitch_pid_params = (0.12, 0.015, 0.09)
        self.yaw_pid_params = (0.2, 0.01, 0.08)

        self.heading_controller = HeadingControl(sample_time=0.1)
        self.depth_controller = DepthPID(sample_time=0.1)
        self.speed_controller = PIDController(40.0, 2.0, 0.0,
                                              sample_time=0.1, output_limit=200.0)
        self.roll_controller = AttitudeAxisController(
            0.16, 0.02, 0.08, sample_time=0.1, output_limit=4.0
        )
        self.pitch_controller = AttitudeAxisController(
            0.12, 0.015, 0.09, sample_time=0.1, output_limit=3.0
        )
        self.yaw_controller = AttitudeAxisController(
            0.2, 0.01, 0.08, sample_time=0.1, output_limit=4.0
        )
        self.roll_max_moment_nm = 4.0
        self.pitch_max_moment_nm = 3.0
        self.yaw_max_moment_nm = 4.0
        self._roll_hold_active = False
        self._pitch_hold_active = False
        self._yaw_hold_active = False
        self.load_all_pids()
        self.update_controller_pids(verbose=False)

    def _feature(self, name):
        if name in self.feature_overrides:
            return bool(self.feature_overrides[name])
        return self.config.getboolean("features", name)

    def load_all_pids(self):
        self.config = load_config(self.config_path)

        def triple(prefix, defaults):
            section = self.config["pid"]
            return tuple(
                section.getfloat(f"{prefix}_{suffix}", fallback=default)
                for suffix, default in zip(("kp", "ki", "kd"), defaults)
            )

        self.depth_pid_params = triple("depth", (120.0, 10.0, 0.0))
        self.depth_hold_deadband_m = self.config.getfloat(
            "control", "depth_hold_deadband_m", fallback=0.0
        )
        self.depth_hold_filter_alpha = self.config.getfloat(
            "control", "depth_hold_filter_alpha", fallback=1.0
        )
        self.depth_max_force_n = self.config.getfloat(
            "control", "depth_max_force_n", fallback=60.0
        )
        self.depth_release_brake_exit_speed_mps = self.config.getfloat(
            "control", "depth_release_brake_exit_speed_mps", fallback=0.03
        )
        self.heading_pid_params = triple("heading", (2.0, 0.2, 0.3))
        self.speed_pid_params = triple("speed", (40.0, 2.0, 0.0))
        self.roll_pid_params = triple("roll", (0.16, 0.02, 0.08))
        self.pitch_pid_params = triple("pitch", (0.12, 0.015, 0.09))
        self.yaw_pid_params = triple("yaw", (0.2, 0.01, 0.08))
        self.roll_max_moment_nm = self.config.getfloat(
            "control", "roll_max_moment_nm", fallback=4.0
        )
        self.pitch_max_moment_nm = self.config.getfloat(
            "control", "pitch_max_moment_nm", fallback=3.0
        )
        self.yaw_max_moment_nm = self.config.getfloat(
            "control", "yaw_max_moment_nm", fallback=4.0
        )

    def update_controller_pids(self, verbose=True):
        self.heading_controller.set_pid(self.heading_pid_params)
        self.depth_controller.set_pid(self.depth_pid_params)
        self.depth_controller.set_force_limit(self.depth_max_force_n)
        self.depth_controller.set_release_brake_exit_speed(
            self.depth_release_brake_exit_speed_mps
        )
        self.speed_controller.set_pid(self.speed_pid_params)
        self.roll_controller.set_pid(self.roll_pid_params)
        self.pitch_controller.set_pid(self.pitch_pid_params)
        self.yaw_controller.set_pid(self.yaw_pid_params)
        self.roll_controller.set_output_limit(self.roll_max_moment_nm)
        self.pitch_controller.set_output_limit(self.pitch_max_moment_nm)
        self.yaw_controller.set_output_limit(self.yaw_max_moment_nm)
        self._roll_hold_active = False
        self._pitch_hold_active = False
        self._yaw_hold_active = False
        if verbose:
            print(f"[PID] heading={self.heading_pid_params} depth={self.depth_pid_params}")
            print(f"[PID] speed(EXPERIMENTAL)={self.speed_pid_params} "
                  f"roll={self.roll_pid_params} pitch={self.pitch_pid_params} "
                  f"yaw={self.yaw_pid_params}")

    def set_pid_runtime(self, pid_flag, values):
        """Apply one PID immediately without writing it to disk."""
        prefix = self.PID_TYPE_TO_PREFIX.get(int(pid_flag))
        if prefix is None:
            raise ValueError(f"unsupported pid type: {pid_flag}")
        kp, ki, kd = map(float, values)
        if not all(math.isfinite(value) for value in (kp, ki, kd)):
            raise ValueError("PID values must be finite")
        limits = (1000.0, 200.0, 500.0)
        if any(value < 0.0 or value > limit
               for value, limit in zip((kp, ki, kd), limits)):
            raise ValueError("PID gains must be non-negative and within safe limits")
        setattr(self, f"{prefix}_pid_params", (kp, ki, kd))
        self.update_controller_pids(verbose=False)

    def write_pid_to_config(self, pid_flag, values):
        """Persist one PID after it has been explicitly applied by the operator."""
        prefix = self.PID_TYPE_TO_PREFIX.get(int(pid_flag))
        if prefix is None:
            raise ValueError(f"unsupported pid type: {pid_flag}")
        kp, ki, kd = map(float, values)
        if not all(math.isfinite(value) for value in (kp, ki, kd)):
            raise ValueError("PID values must be finite")
        self.set_pid_runtime(pid_flag, (kp, ki, kd))
        update_config_values(
            self.config_path,
            "pid",
            {f"{prefix}_kp": kp, f"{prefix}_ki": ki, f"{prefix}_kd": kd},
        )
        # Keep the in-memory config in sync without reloading and accidentally
        # discarding another PID that is temporarily applied but not yet saved.
        self.config.set("pid", f"{prefix}_kp", str(kp))
        self.config.set("pid", f"{prefix}_ki", str(ki))
        self.config.set("pid", f"{prefix}_kd", str(kd))

    def clear_attitude_holds(self):
        self.roll_controller.clear()
        self.pitch_controller.clear()
        self.yaw_controller.clear()
        self._roll_hold_active = False
        self._pitch_hold_active = False
        self._yaw_hold_active = False

    def set_attitude_moment_limits(self, roll_nm, pitch_nm, yaw_nm=None):
        roll = float(roll_nm)
        pitch = float(pitch_nm)
        yaw = self.yaw_max_moment_nm if yaw_nm is None else float(yaw_nm)
        if not all(math.isfinite(value) and 0.1 <= value <= 20.0
                   for value in (roll, pitch, yaw)):
            raise ValueError("attitude moment limits must be in 0.1..20 Nm")
        self.roll_max_moment_nm = roll
        self.pitch_max_moment_nm = pitch
        self.yaw_max_moment_nm = yaw
        self.roll_controller.set_output_limit(roll)
        self.pitch_controller.set_output_limit(pitch)
        self.yaw_controller.set_output_limit(yaw)

    def get_pid_snapshot(self):
        snapshot = {
            "depth": dict(zip(("kp", "ki", "kd"), self.depth_pid_params)),
            "heading": dict(zip(("kp", "ki", "kd"), self.heading_pid_params)),
            "speed": dict(zip(("kp", "ki", "kd"), self.speed_pid_params)),
            "roll": dict(zip(("kp", "ki", "kd"), self.roll_pid_params)),
            "pitch": dict(zip(("kp", "ki", "kd"), self.pitch_pid_params)),
            "yaw": dict(zip(("kp", "ki", "kd"), self.yaw_pid_params)),
        }
        snapshot["depth"]["deadband_m"] = self.depth_hold_deadband_m
        snapshot["depth"]["filter_alpha"] = self.depth_hold_filter_alpha
        snapshot["depth"]["max_force_n"] = self.depth_max_force_n
        snapshot["depth"]["release_brake_exit_speed_mps"] = (
            self.depth_release_brake_exit_speed_mps
        )
        snapshot["roll"]["max_moment_nm"] = self.roll_max_moment_nm
        snapshot["pitch"]["max_moment_nm"] = self.pitch_max_moment_nm
        snapshot["yaw"]["max_moment_nm"] = self.yaw_max_moment_nm
        return snapshot

    def _filter_depth_for_hold(self, raw_depth_m):
        raw = float(raw_depth_m)
        if not math.isfinite(raw):
            raise ValueError("depth must be finite")
        if self._depth_hold_filtered_m is None:
            self._depth_hold_filtered_m = raw
        else:
            alpha = self.depth_hold_filter_alpha
            self._depth_hold_filtered_m += alpha * (raw - self._depth_hold_filtered_m)
        return self._depth_hold_filtered_m

    @staticmethod
    def reduced_attitude_error_deg(roll_deg, pitch_deg):
        """Return yaw-free level error and total tilt on the sphere S2.

        Let ``g_b = R.T @ e3`` be world-down expressed in vehicle-body axes.
        The shortest restoring direction is ``e3 x g_b``.  Scaling that cross
        product by ``angle/sin(angle)`` gives a rotation-vector error in
        degrees.  Near level it reduces smoothly to ``(-roll, -pitch)`` while
        avoiding Euler-axis coupling and the pitch singularity.
        """
        roll = math.radians(float(roll_deg))
        pitch = math.radians(float(pitch_deg))
        if not all(math.isfinite(value) for value in (roll, pitch)):
            raise ValueError("attitude angles must be finite")

        down_x = -math.sin(pitch)
        down_y = math.sin(roll) * math.cos(pitch)
        down_z = math.cos(roll) * math.cos(pitch)
        cross_x = -down_y
        cross_y = down_x
        sine = math.hypot(cross_x, cross_y)
        tilt_rad = math.atan2(sine, down_z)
        if sine < 1e-12:
            return 0.0, 0.0, math.degrees(tilt_rad)
        scale = math.degrees(tilt_rad) / sine
        return cross_x * scale, cross_y * scale, math.degrees(tilt_rad)

    @staticmethod
    def normalize_mode(mode):
        if isinstance(mode, dict):
            if bool(mode.get("heading_hold", False)):
                raise ValueError(
                    "heading_hold is temporarily disabled and must be false"
                )
            return {
                "heading_hold": False,
                "depth_hold": bool(mode.get("depth_hold", False)),
                "speed_hold": bool(mode.get("speed_hold", False)),
            }
        values = list(mode or [])
        # Legacy six-element arrays are accepted, but obsolete mission bits are ignored.
        if len(values) > 0 and bool(values[0]):
            raise ValueError(
                "heading_hold is temporarily disabled and must be false"
            )
        return {
            "heading_hold": False,
            "depth_hold": bool(values[1]) if len(values) > 1 else False,
            "speed_hold": False,
        }

    @staticmethod
    def normalize_target(target):
        if isinstance(target, dict):
            return {
                "heading_deg": float(target.get("heading_deg", 0.0)),
                "depth_m": float(target.get("depth_m", 0.0)),
                "speed_mps": float(target.get("speed_mps", 0.0)),
            }
        values = list(target or [])
        return {
            "heading_deg": float(values[0]) if len(values) > 0 else 0.0,
            "depth_m": float(values[1]) if len(values) > 1 else 0.0,
            # Compatibility with the old [h,d,x,y,time,speed,plan_h] target.
            "speed_mps": float(values[5]) if len(values) > 5 else 0.0,
        }

    def compute_control(self, current_state, mode, target_state, sensor_valid=None,
                        attitude_hold=None, depth_manual_override=False,
                        attitude_targets=None):
        mode = self.normalize_mode(mode)
        target = self.normalize_target(target_state)
        valid = {"imu": True, "depth": True}
        valid.update(sensor_valid or {})
        attitude_hold = dict(attitude_hold or {})
        attitude_targets = dict(attitude_targets or {})
        roll_hold = bool(attitude_hold.get("roll", False))
        pitch_hold = bool(attitude_hold.get("pitch", False))
        yaw_hold = bool(attitude_hold.get("yaw", False))
        inhibited = []

        tau_x = tau_y = tau_z = 0.0
        M_x = M_y = M_z = 0.0

        # Magnetic-heading hold remains intentionally dormant.  The separate
        # IMU yaw-to-calibrated-zero loop is evaluated below.
        self.heading_controller.clear()

        if mode["depth_hold"]:
            if valid["depth"]:
                if depth_manual_override:
                    # The Pi tracks the target separately while the stick is
                    # displaced.  Keep the learned buoyancy bias, but do not
                    # let the old target advance P or I behind the manual force.
                    self._depth_hold_filtered_m = float(
                        current_state.get("depth", 0.0)
                    )
                    tau_z = self.depth_controller.freeze_for_manual_override(
                        current_state.get("vertical_speed", 0.0)
                    )
                    inhibited.append("depth_manual_override")
                else:
                    filtered_depth = self._filter_depth_for_hold(
                        current_state.get("depth", 0.0)
                    )
                    tau_z = self.depth_controller.update_depth(
                        target["depth_m"],
                        filtered_depth,
                        current_state.get("vertical_speed", 0.0),
                        position_deadband_m=self.depth_hold_deadband_m,
                    )
            else:
                self.depth_controller.clear()
                self._depth_hold_filtered_m = None
                inhibited.append("depth_stale")
        else:
            self.depth_controller.clear()
            self._depth_hold_filtered_m = None

        if mode["speed_hold"]:
            if not self._feature("speed_closed_loop"):
                self.speed_controller.clear()
                inhibited.append("speed_feature_disabled")
            elif not valid["imu"]:
                self.speed_controller.clear()
                inhibited.append("speed_stale")
            else:
                tau_x = self.speed_controller.update(
                    target["speed_mps"], float(current_state.get("est_vx", 0.0))
                )
        else:
            self.speed_controller.clear()

        roll_error, pitch_error, tilt_deg = self.reduced_attitude_error_deg(
            current_state.get("roll", 0.0), current_state.get("pitch", 0.0)
        )
        roll_rate = float(current_state.get("gyro_x", 0.0))
        pitch_rate = float(current_state.get("gyro_y", 0.0))
        yaw = float(current_state.get("yaw", 0.0))
        yaw_rate = float(current_state.get("gyro_z", 0.0))
        yaw_target = float(attitude_targets.get("yaw", 0.0))
        if not all(math.isfinite(value) for value in (
            roll_rate, pitch_rate, yaw, yaw_rate, yaw_target
        )):
            valid["imu"] = False

        if roll_hold:
            if valid["imu"]:
                if not self._roll_hold_active:
                    self.roll_controller.prime_geometric(roll_error, roll_rate)
                    self._roll_hold_active = True
                M_x = self.roll_controller.update_geometric(
                    roll_error, roll_rate
                )
            else:
                self.roll_controller.clear()
                self._roll_hold_active = False
                inhibited.append("roll_stale")
        else:
            self.roll_controller.clear()
            self._roll_hold_active = False

        if pitch_hold:
            if valid["imu"]:
                if not self._pitch_hold_active:
                    self.pitch_controller.prime_geometric(
                        pitch_error, pitch_rate
                    )
                    self._pitch_hold_active = True
                M_y = self.pitch_controller.update_geometric(
                    pitch_error, pitch_rate
                )
            else:
                self.pitch_controller.clear()
                self._pitch_hold_active = False
                inhibited.append("pitch_stale")
        else:
            self.pitch_controller.clear()
            self._pitch_hold_active = False

        if yaw_hold:
            if valid["imu"]:
                yaw_error = PIDController._wrapped_error(yaw_target, yaw)
                if not self._yaw_hold_active:
                    self.yaw_controller.prime_geometric(yaw_error, yaw_rate)
                    self._yaw_hold_active = True
                M_z = self.yaw_controller.update_geometric(
                    yaw_error, yaw_rate
                )
            else:
                self.yaw_controller.clear()
                self._yaw_hold_active = False
                inhibited.append("yaw_stale")
        else:
            self.yaw_controller.clear()
            self._yaw_hold_active = False

        output = (
            _clamp(tau_x, -200.0, 200.0),
            _clamp(tau_y, -200.0, 200.0),
            _clamp(tau_z, -200.0, 200.0),
            _clamp(M_x, -40.0, 40.0),
            _clamp(M_y, -40.0, 40.0),
            _clamp(M_z, -40.0, 40.0),
        )
        return output, inhibited
