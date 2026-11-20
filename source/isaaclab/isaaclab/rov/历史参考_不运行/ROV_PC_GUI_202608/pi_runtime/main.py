#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eight-thruster ROV lower controller with fail-closed safety handling.

Canonical command protocol (legacy heading/depth arrays remain accepted):
    {"arm": bool, "enable": bool,
     "mode": {"heading_hold": bool, "depth_hold": bool, "speed_hold": bool},
     "target": {"heading_deg": float, "depth_m": float, "speed_mps": float},
     "save_data": bool, "emergency_stop": bool,
     "reset_emergency_stop": bool, "pid_set": {...}, "pid_save": {...}}

The fixed direct-Ethernet addresses and every runtime setting are in config.ini.
"""

from __future__ import annotations

from collections import deque
import logging
from logging.handlers import RotatingFileHandler
import math
import signal
import socket
import sys
import threading
import time

from command_handler import CommandHandler
from attitude_math import (
    IDENTITY_QUATERNION,
    average_quaternions,
    attitude_from_raw_euler,
    inverse as quaternion_inverse,
    mount_from_axis_captures,
    mount_from_pitch_yaw_captures,
    multiply as quaternion_multiply,
    normalize as normalize_quaternion,
    quaternion_distance_deg,
    raw_euler_to_vehicle_axes,
    rotation_angle_deg,
    rotation_axis,
    tilt_compensated_magnetic_heading_deg,
    vehicle_body_rate_from_raw,
    vehicle_vector_from_raw,
)
from config_loader import (
    config_fingerprint,
    get_int,
    load_config,
    resolve_path,
    update_config_sections,
)
from controller import Controller
from imu_depth_combined import DepthSensor, IMUReader
from joystick_handler import JoystickHandler, NEUTRAL_JOYSTICK
from secure_protocol import encode_telemetry_packet, new_session_id
from thrust_allocator import THRUSTER_NAMES, ThrustAllocator
import save
from yaw_feedback import YawFeedbackEstimator


CONFIG = load_config()
MOUNT_CAPTURE_MIN_ANGLE_DEG = 20.0
MOUNT_CAPTURE_MAX_ANGLE_DEG = 45.0
MOUNT_CAPTURE_MIN_AXIS_SEPARATION_DEG = 60.0
MOUNT_CAPTURE_SAMPLE_WINDOW_SECONDS = 0.35
MOUNT_CAPTURE_MIN_SAMPLES = 5
MOUNT_CAPTURE_MAX_SPREAD_DEG = 1.5
MOUNT_FINALIZE_MAX_RETURN_ERROR_DEG = 5.0
YAW_RELEASE_SETTLE_DWELL_SECONDS = 0.20
YAW_RELEASE_MAX_COAST_SECONDS = 2.0
ATTITUDE_MIN_PLAUSIBLE_STEP_DEG = 45.0
ATTITUDE_MAX_PLAUSIBLE_STEP_DEG = 90.0
ATTITUDE_STEP_RATE_SCALE = 2.5
_log_path = resolve_path(CONFIG, "logging", "log_file")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(
            _log_path,
            maxBytes=CONFIG.getint("logging", "log_max_bytes"),
            backupCount=CONFIG.getint("logging", "log_backup_count"),
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.critical("uncaught main-thread exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


def vertical_command_with_hover(correction_n, hover_per_thruster_n):
    """Add the fixed upward compensation in a body frame whose +Z is down."""
    correction = float(correction_n)
    hover = float(hover_per_thruster_n)
    if not math.isfinite(correction) or not math.isfinite(hover):
        raise ValueError("vertical command values must be finite")
    return correction - 4.0 * hover


def tilt_compensated_hover_per_thruster(
    hover_per_thruster_n, roll_deg, pitch_deg, max_scale,
    vertical_directions_body=None,
):
    """Bound the loss of world-vertical hover force while the body is tilted.

    Four body-z thrusters provide only ``cos(roll)*cos(pitch)`` of their force
    along world down.  Only the known-good equal hover feed-forward is scaled;
    feedback moments remain zero-sum.  The hard scale cap protects attitude
    authority and prevents a large-angle command surge.
    """
    hover = float(hover_per_thruster_n)
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    limit = float(max_scale)
    if not all(math.isfinite(value) for value in (hover, roll, pitch, limit)):
        raise ValueError("tilt hover compensation values must be finite")
    if not 1.0 <= limit <= 1.5:
        raise ValueError("tilt hover compensation max scale must be in 1.0..1.5")
    world_down_in_body = (
        -math.sin(pitch),
        math.sin(roll) * math.cos(pitch),
        math.cos(roll) * math.cos(pitch),
    )
    if vertical_directions_body is None:
        effectiveness = world_down_in_body[2]
    else:
        directions = tuple(vertical_directions_body)
        if len(directions) != 4:
            raise ValueError("four vertical thruster directions are required")
        projections = []
        for direction in directions:
            values = tuple(float(value) for value in direction)
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError("vertical thruster directions must be finite 3-vectors")
            projections.append(sum(a * b for a, b in zip(world_down_in_body, values)))
        effectiveness = sum(projections) / 4.0
    if effectiveness <= 0.0:
        return hover * limit
    return hover * min(limit, 1.0 / effectiveness)


def vertical_front_rear_trim_moment(
    vertical_force_n, trim_percent, vertical_x_m
):
    """Convert an operator-visible front/rear balance into a pitch moment.

    The allocator then produces front/rear pair factors ``1-p/200`` and
    ``1+p/200`` while preserving the requested total vertical force. Negative
    trim strengthens the front pair; positive trim strengthens the rear pair.
    """
    vertical_force = float(vertical_force_n)
    trim = float(trim_percent)
    arm = float(vertical_x_m)
    if not all(math.isfinite(value) for value in (vertical_force, trim, arm)):
        raise ValueError("vertical trim values must be finite")
    if not -100.0 <= trim <= 100.0:
        raise ValueError("vertical trim percent must be in -100..100")
    if arm <= 0.0:
        raise ValueError("vertical thruster x arm must be positive")
    return arm * vertical_force * trim / 200.0


def select_vertical_front_rear_trim(
    vertical_correction_n, ascent_trim_percent, descent_trim_percent, deadband_n,
    pitch_hold=False,
):
    """Select a trim from motion direction before fixed hover is added.

    The equal fixed hover command is intentionally not classified as ascent.
    A requested pitch hold suppresses this open-loop pitch bias so the IMU
    loop is the only source of automatic pitch correction during heave.
    Negative correction is ascent and positive correction is descent because
    the vehicle body convention is +Z down.
    """
    correction = float(vertical_correction_n)
    deadband = abs(float(deadband_n))
    values = (correction, ascent_trim_percent, descent_trim_percent, deadband)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("vertical direction trim values must be finite")
    if bool(pitch_hold):
        return 0.0
    if correction < -deadband:
        return float(ascent_trim_percent)
    if correction > deadband:
        return float(descent_trim_percent)
    return 0.0


def fuse_pitch_heave_feedforward(
    pid_moment_nm,
    slewed_vertical_force_n,
    hover_per_thruster_n,
    ascent_trim_percent,
    descent_trim_percent,
    deadband_n,
    vertical_x_m,
    feedforward_scale,
    pitch_moment_limit_nm,
):
    """Fuse direction-specific heave balance with the closed-loop pitch PID.

    The force input must be the slew-limited vertical request.  This keeps the
    feed-forward ramp synchronized with the thrust ramp instead of applying the
    full front/rear bias on the first control cycle.  The returned contribution
    is the amount that remains after the combined safety limit is enforced.
    """
    pid_moment = float(pid_moment_nm)
    vertical_force = float(slewed_vertical_force_n)
    hover = float(hover_per_thruster_n)
    scale = float(feedforward_scale)
    moment_limit = float(pitch_moment_limit_nm)
    values = (pid_moment, vertical_force, hover, scale, moment_limit)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pitch feed-forward values must be finite")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("pitch feed-forward scale must be in 0..1")
    if moment_limit <= 0.0:
        raise ValueError("pitch moment limit must be positive")

    # Recover the dynamic heave component so fixed equal hover alone never
    # selects the ascent/descent trim.  Once moving, the trim is applied to the
    # total slew-limited vertical force, matching the actual four-thruster load.
    dynamic_vertical_force = vertical_force + 4.0 * hover
    trim = select_vertical_front_rear_trim(
        dynamic_vertical_force,
        ascent_trim_percent,
        descent_trim_percent,
        deadband_n,
    )
    requested_feedforward = scale * vertical_front_rear_trim_moment(
        vertical_force, trim, vertical_x_m
    )
    limited_pid = max(-moment_limit, min(moment_limit, pid_moment))
    total = max(
        -moment_limit,
        min(moment_limit, limited_pid + requested_feedforward),
    )
    return total, total - limited_pid


class ROVController:
    def __init__(self, config_path=None):
        self.config = load_config(config_path)
        self.config_path = self.config.config_path

        network = self.config["network"]
        hardware = self.config["hardware"]
        safety = self.config["safety"]
        thrusters = self.config["thrusters"]
        timeout = self.config["timeouts"]
        control = self.config["control"]

        self.pi_ip = network["pi_ip"]
        self.pc_ip = network["pc_ip"]
        self.bind_host = network["bind_host"]
        self.cmd_port = network.getint("command_port")
        self.pc_port = network.getint("telemetry_port")
        self.joy_port = network.getint("joystick_port")

        self.imu_port = hardware["imu_port"]
        self.imu_baud = hardware.getint("imu_baud")
        self.imu_report_rate = hardware.getint("imu_report_rate_hz")
        self.depth_port = hardware["depth_port"]
        self.depth_baud = hardware.getint("depth_baud")
        self.depth_addr = hardware.getint("depth_address")
        self.depth_raw_zero_mpa = hardware.getfloat("depth_raw_zero_mpa", fallback=0.0)
        self.depth_m_per_mpa = hardware.getfloat(
            "depth_m_per_mpa", fallback=DepthSensor.MPA_TO_WATER_DEPTH_M
        )
        self.imu_roll_inverted = hardware.getboolean("imu_roll_inverted")
        self.imu_pitch_inverted = hardware.getboolean("imu_pitch_inverted")
        self.imu_yaw_inverted = hardware.getboolean("imu_yaw_inverted")
        self.imu_heading_inverted = hardware.getboolean("imu_heading_inverted")
        self.imu_roll_offset_deg = hardware.getfloat("imu_roll_offset_deg")
        self.imu_pitch_offset_deg = hardware.getfloat("imu_pitch_offset_deg")
        self.imu_yaw_offset_deg = hardware.getfloat("imu_yaw_offset_deg")
        self.imu_heading_offset_deg = hardware.getfloat("imu_heading_offset_deg")
        self.imu_3d_mount_calibrated = hardware.getboolean(
            "imu_3d_mount_calibrated", fallback=False
        )
        self.imu_mount_quaternion = self._quaternion_from_config("imu_mount_quat")
        self.imu_reference_quaternion = self._quaternion_from_config("imu_reference_quat")
        self.pwm_address = get_int(self.config, "hardware", "pwm_i2c_address")
        self.pwm_neutral_us = hardware.getint("pwm_neutral_us")
        servo = self.config["servo"]
        self.servo_output_enabled = servo.getboolean("output_enabled")
        self.servo_channel = servo.getint("channel")
        self.servo_pulse_min_us = servo.getint("pulse_min_us")
        self.servo_pulse_center_us = servo.getint("pulse_center_us")
        self.servo_pulse_max_us = servo.getint("pulse_max_us")
        self.servo_angle_min_deg = servo.getfloat("angle_min_deg")
        self.servo_angle_max_deg = servo.getfloat("angle_max_deg")
        self.servo_slew_deg_per_sec = servo.getfloat("slew_deg_per_sec")
        self.servo_min_position = servo.getfloat("closed_position")
        self.servo_failsafe_position = servo.getfloat("failsafe_position")

        self.cmd_timeout = safety.getint("command_timeout_ms") / 1000.0
        self.joy_timeout = self.cmd_timeout
        self.arm_neutral_threshold = safety.getfloat("arm_neutral_threshold")
        self.arm_neutral_dwell = safety.getint("arm_neutral_dwell_ms") / 1000.0
        self.motion_output_enabled = safety.getboolean("motion_output_enabled")
        self.allowed_ground_ip = safety["allowed_ground_ip"]
        self.protocol_version = safety.getint("protocol_version")
        self.shared_key = safety["shared_key"]
        self.max_command_packet_bytes = safety.getint("max_command_packet_bytes")
        self.allow_legacy_unsigned = safety.getboolean("allow_legacy_unsigned")
        self.legacy_joystick_enabled = safety.getboolean("legacy_joystick_enabled")
        self.imu_freshness = timeout.getfloat("imu_freshness_seconds")
        self.depth_freshness = timeout.getfloat("depth_freshness_seconds")
        self.control_interval = 1.0 / control.getfloat("loop_hz")
        self.telemetry_interval = 1.0 / control.getfloat("telemetry_hz")
        self.depth_query_interval = 1.0 / control.getfloat("depth_query_hz")
        self.manual_deadband = control.getfloat("manual_deadband")
        self.yaw_release_rate_threshold_dps = control.getfloat(
            "yaw_release_rate_threshold_dps", fallback=3.0
        )
        self.max_force_step = control.getfloat("max_force_step_per_cycle")
        self.max_moment_step = control.getfloat("max_moment_step_per_cycle")
        self.max_command_fraction = thrusters.getfloat(
            "max_command_fraction", fallback=0.5
        )
        self.horizontal_limit_fraction = thrusters.getfloat(
            "horizontal_limit_fraction", fallback=self.max_command_fraction
        )
        self.vertical_limit_fraction = thrusters.getfloat(
            "vertical_limit_fraction", fallback=self.max_command_fraction
        )
        self.vertical_hover_force_per_thruster_n = control.getfloat(
            "vertical_hover_force_per_thruster_n", fallback=0.0
        )
        self.attitude_control_mapping_validated = control.getboolean(
            "attitude_control_mapping_validated", fallback=False
        )
        self.attitude_hover_tilt_max_scale = control.getfloat(
            "attitude_hover_tilt_max_scale", fallback=1.15
        )
        legacy_vertical_trim = control.getfloat(
            "vertical_front_rear_trim_percent", fallback=0.0
        )
        legacy_horizontal_trim = control.getfloat(
            "lateral_front_rear_trim_percent", fallback=0.0
        )
        self.vertical_ascent_front_rear_trim_percent = control.getfloat(
            "vertical_ascent_front_rear_trim_percent", fallback=0.0
        )
        self.vertical_descent_front_rear_trim_percent = control.getfloat(
            "vertical_descent_front_rear_trim_percent",
            fallback=legacy_vertical_trim,
        )
        self.pitch_heave_feedforward_scale = control.getfloat(
            "pitch_heave_feedforward_scale", fallback=0.0
        )
        self.forward_front_rear_trim_percent = control.getfloat(
            "forward_front_rear_trim_percent", fallback=legacy_horizontal_trim
        )
        self.backward_front_rear_trim_percent = control.getfloat(
            "backward_front_rear_trim_percent", fallback=legacy_horizontal_trim
        )
        self.left_lateral_surge_trim_percent = control.getfloat(
            "left_lateral_surge_trim_percent", fallback=0.0
        )
        self.right_lateral_surge_trim_percent = control.getfloat(
            "right_lateral_surge_trim_percent", fallback=0.0
        )
        self.velocity_leak = control.getfloat("velocity_leak_factor")
        self.is_running = False
        self.thrusters_armed = False
        self.control_enabled = False
        self.emergency_stop_latched = False
        self.emergency_stop_source = ""
        self.rearm_required = True
        self._arm_release_seen = True
        self.link_timed_out = True
        self.active_control_session = None
        self.last_control_packet_time = 0.0
        self.last_cmd_time = 0.0
        self.last_joy_time = 0.0
        self._joystick_centered_since = None
        self.last_command_ack = {"ok": True, "message": "startup_locked", "timestamp": time.time()}
        self.last_servo_ack = {
            "ok": True,
            "message": "servo_startup_locked",
            "timestamp": time.time(),
        }

        self.imu = None
        self.depth_sensor = None
        self.allocator = None
        self.controller = Controller(config_path=self.config_path)
        self.performance_mode = self.config.get(
            "control", "performance_mode", fallback="balance"
        ).strip().lower()
        if self.performance_mode not in ("balance", "rage"):
            self.performance_mode = "balance"
        self.performance_profiles = self._load_performance_profiles()
        # Runtime-only operator switches. Restart, lock, E-stop, link loss, and
        # invalid IMU return all three loops to OFF. Roll/pitch target the
        # calibrated vehicle-frame zero angles; yaw captures the current angle
        # on enable and after every manual-yaw release.
        self.attitude_hold = {"roll": False, "pitch": False, "yaw": False}
        self.attitude_hold_target_deg = {
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        }
        self._yaw_manual_override_active = False
        self._yaw_input_active = False
        self._yaw_release_pending = False
        self._yaw_release_started_at = None
        self._yaw_release_below_threshold_since = None
        self.attitude_hold_reason = "startup_disabled"
        self.attitude_hold_output = {
            "roll_moment_nm": 0.0,
            "pitch_moment_nm": 0.0,
            "yaw_moment_nm": 0.0,
            "roll_manual_override": False,
            "pitch_manual_override": False,
            "yaw_manual_override": False,
        }
        self.udp_sock = None
        self.telemetry_session = new_session_id()
        self.telemetry_seq = 0
        self.cmd_handler = None
        self.joy_handler = None
        self.state_lock = threading.RLock()

        self._prev_depth = None
        self._prev_depth_time = 0.0
        self._vert_speed = 0.0
        self._est_vx = 0.0
        self._est_vy = 0.0
        self._last_imu_update = 0.0
        self._last_heading_update = 0.0
        self._latest_raw_euler = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self._yaw_feedback_estimator = YawFeedbackEstimator()
        self._last_accepted_attitude = None
        self._last_accepted_attitude_time = 0.0
        self._attitude_rejected_samples = 0
        self._attitude_fault_reason = ""
        self._recent_raw_quaternions = deque(
            maxlen=max(
                MOUNT_CAPTURE_MIN_SAMPLES,
                int(math.ceil(
                    self.imu_report_rate * MOUNT_CAPTURE_SAMPLE_WINDOW_SECONDS
                )),
            )
        )
        self._attitude_mount_capture = None
        self._last_attitude_mount_workflow = "none"
        self._last_depth_update = 0.0
        self._last_depth_pressure_update = 0.0
        self._depth_rejected_samples = 0
        self._depth_fault_reason = "awaiting_depth_sample"
        self._auto_inhibited = []
        self.depth_manual_override = False

        self.sensor_state = {
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "heading": 0.0,
            "yaw_control": 0.0, "heading_vehicle": 0.0,
            "attitude_sample_rejected": False,
            "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
            "acc_x": 0.0, "acc_y": 0.0, "acc_z": 0.0,
            "depth": 0.0, "temperature": 0.0,
            "pressure_mpa": 0.0,
        }
        self.mode = {"heading_hold": False, "depth_hold": False, "speed_hold": False}
        self.target = {"heading_deg": 0.0, "depth_m": 0.0, "speed_mps": 0.0}
        self.joy_force = dict(NEUTRAL_JOYSTICK)
        self.joy_active = False
        self.servo_target_position = self.servo_failsafe_position
        self.servo_position = self.servo_failsafe_position
        self.servo_angle_deg = self._servo_position_to_angle(
            self.servo_failsafe_position
        )
        self.servo_pwm_us = self.servo_pulse_center_us
        self._servo_last_update = time.monotonic()
        self.force_state = {
            "tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0,
            "M_x": 0.0, "M_y": 0.0, "M_z": 0.0,
        }
        self.pwm_state = {}
        self.thrust_state = {}
        self._last_tau = [0.0] * 6
        self.save_data_flag = False

        def _csv_values(option, cast):
            values = [cast(value.strip()) for value in thrusters[option].split(",")]
            if len(values) != len(THRUSTER_NAMES):
                raise ValueError(
                    f"thrusters.{option} must contain {len(THRUSTER_NAMES)} comma-separated values"
                )
            return values

        self.thruster_channels = dict(zip(THRUSTER_NAMES, _csv_values("channels", int)))
        self.thruster_signs = dict(zip(THRUSTER_NAMES, _csv_values("motor_signs", int)))
        pwm_lows = _csv_values("pwm_min_us", int)
        pwm_highs = _csv_values("pwm_max_us", int)
        self.thruster_pwm_limits = dict(
            zip(THRUSTER_NAMES, zip(pwm_lows, pwm_highs))
        )
        self.thruster_geometry = {
            "lateral_x_m": thrusters.getfloat("lateral_x_m"),
            "lateral_y_m": thrusters.getfloat("lateral_y_m"),
            "vertical_x_m": thrusters.getfloat("vertical_x_m"),
            "vertical_y_m": thrusters.getfloat("vertical_y_m"),
            "lateral_angle_deg": thrusters.getfloat("lateral_angle_deg"),
        }
        vertical_names = THRUSTER_NAMES[4:8]
        geometry_options = tuple(
            f"{name}_{suffix}"
            for name in vertical_names
            for suffix in ("position_m", "direction_body")
        )
        if all(thrusters.get(option, fallback=None) is not None
               for option in geometry_options):
            def _vector3(option):
                values = tuple(
                    float(value.strip()) for value in thrusters[option].split(",")
                )
                if len(values) != 3 or not all(math.isfinite(value) for value in values):
                    raise ValueError(f"thrusters.{option} must contain three finite values")
                return values

            self.thruster_geometry["vertical_positions_m"] = {
                name: _vector3(f"{name}_position_m") for name in vertical_names
            }
            self.thruster_geometry["vertical_directions_body"] = {
                name: _vector3(f"{name}_direction_body") for name in vertical_names
            }

    # ----------------------------- safety -----------------------------
    def _set_ack(self, ok, message, **details):
        with self.state_lock:
            self.last_command_ack = {
                "ok": bool(ok), "message": str(message), "timestamp": time.time(), **details
            }

    def _set_servo_ack(self, ok, message, **details):
        """Keep auxiliary-servo status separate from transactional command ACKs."""
        with self.state_lock:
            self.last_servo_ack = {
                "ok": bool(ok), "message": str(message),
                "timestamp": time.time(), **details,
            }

    def _motion_tuning_snapshot(self):
        return {
            "horizontal_limit_fraction": float(self.horizontal_limit_fraction),
            "vertical_limit_fraction": float(self.vertical_limit_fraction),
            "hover_force_per_thruster_n": float(
                self.vertical_hover_force_per_thruster_n
            ),
            "vertical_ascent_front_rear_trim_percent": float(
                self.vertical_ascent_front_rear_trim_percent
            ),
            "vertical_descent_front_rear_trim_percent": float(
                self.vertical_descent_front_rear_trim_percent
            ),
            "forward_front_rear_trim_percent": float(
                self.forward_front_rear_trim_percent
            ),
            "backward_front_rear_trim_percent": float(
                self.backward_front_rear_trim_percent
            ),
        }

    def _load_performance_profiles(self):
        """Load complete balance/rage profiles with legacy config fallbacks."""
        controller_pid = self.controller.get_pid_snapshot()
        balance_pid = {
            axis: {
                field: controller_pid[axis][field]
                for field in ("kp", "ki", "kd")
            }
            for axis in ("roll", "pitch", "yaw")
        }
        legacy_motion = self._motion_tuning_snapshot()
        legacy_moments = {
            "roll_max_moment_nm": float(self.controller.roll_max_moment_nm),
            "pitch_max_moment_nm": float(self.controller.pitch_max_moment_nm),
            "yaw_max_moment_nm": float(self.controller.yaw_max_moment_nm),
        }
        defaults = {
            "balance": {
                "horizontal_limit_fraction": self.horizontal_limit_fraction,
                "vertical_limit_fraction": self.vertical_limit_fraction,
                **{
                    key: value for key, value in legacy_motion.items()
                    if key not in ("horizontal_limit_fraction", "vertical_limit_fraction")
                },
                "left_lateral_surge_trim_percent": (
                    self.left_lateral_surge_trim_percent
                ),
                "right_lateral_surge_trim_percent": (
                    self.right_lateral_surge_trim_percent
                ),
                **legacy_moments,
                "roll_pid": balance_pid["roll"],
                "pitch_pid": balance_pid["pitch"],
                "yaw_pid": balance_pid["yaw"],
            },
            "rage": {
                "horizontal_limit_fraction": min(
                    self.max_command_fraction, self.horizontal_limit_fraction * 1.5
                ),
                # Rage never raises the vertical ceiling implicitly.  Older
                # configs without a mode section retain their commissioned
                # global vertical limit (currently 0.30).
                "vertical_limit_fraction": min(
                    self.max_command_fraction, self.vertical_limit_fraction
                ),
                **{
                    key: value for key, value in legacy_motion.items()
                    if key not in ("horizontal_limit_fraction", "vertical_limit_fraction")
                },
                **legacy_moments,
                # Conservative commissioning presets. They are optional
                # feed-forward only; zero remains a fully supported value.
                "forward_front_rear_trim_percent": 8.0,
                "backward_front_rear_trim_percent": -8.0,
                "left_lateral_surge_trim_percent": 0.0,
                "right_lateral_surge_trim_percent": 0.0,
                "pitch_max_moment_nm": max(
                    3.5, legacy_moments["pitch_max_moment_nm"]
                ),
                "roll_pid": {"kp": 0.32, "ki": 0.02, "kd": 0.11},
                "pitch_pid": {"kp": 0.38, "ki": 0.015, "kd": 0.16},
                "yaw_pid": {"kp": 0.27, "ki": 0.01, "kd": 0.11},
            },
        }
        profiles = {}
        for mode, default in defaults.items():
            section = f"mode_{mode}"
            values = {
                "mode": mode,
                "horizontal_limit_fraction": float(default["horizontal_limit_fraction"]),
                "vertical_limit_fraction": float(default["vertical_limit_fraction"]),
                **{
                    field: float(default[field])
                    for field in (
                        "hover_force_per_thruster_n",
                        "vertical_ascent_front_rear_trim_percent",
                        "vertical_descent_front_rear_trim_percent",
                        "forward_front_rear_trim_percent",
                        "backward_front_rear_trim_percent",
                        "left_lateral_surge_trim_percent",
                        "right_lateral_surge_trim_percent",
                        "roll_max_moment_nm",
                        "pitch_max_moment_nm",
                        "yaw_max_moment_nm",
                    )
                },
                "roll_pid": dict(default["roll_pid"]),
                "pitch_pid": dict(default["pitch_pid"]),
                "yaw_pid": dict(default["yaw_pid"]),
            }
            if self.config.has_section(section):
                values["horizontal_limit_fraction"] = self.config.getfloat(
                    section, "horizontal_limit_fraction", fallback=values["horizontal_limit_fraction"]
                )
                values["vertical_limit_fraction"] = self.config.getfloat(
                    section, "vertical_limit_fraction", fallback=values["vertical_limit_fraction"]
                )
                for field in (
                    "hover_force_per_thruster_n",
                    "vertical_ascent_front_rear_trim_percent",
                    "vertical_descent_front_rear_trim_percent",
                    "forward_front_rear_trim_percent",
                    "backward_front_rear_trim_percent",
                    "left_lateral_surge_trim_percent",
                    "right_lateral_surge_trim_percent",
                    "roll_max_moment_nm",
                    "pitch_max_moment_nm",
                    "yaw_max_moment_nm",
                ):
                    values[field] = self.config.getfloat(
                        section, field, fallback=values[field]
                    )
                for axis in ("roll", "pitch", "yaw"):
                    values[f"{axis}_pid"] = {
                        field: self.config.getfloat(
                            section, f"{axis}_{field}",
                            fallback=values[f"{axis}_pid"][field],
                        )
                        for field in ("kp", "ki", "kd")
                    }
            profiles[mode] = self._validated_performance_profile(values)
        return profiles

    def _validated_performance_profile(self, values):
        if not isinstance(values, dict):
            raise ValueError("performance_profile must be an object")
        legacy_required = {
            "mode", "horizontal_limit_fraction", "vertical_limit_fraction",
            "roll_pid", "pitch_pid", "yaw_pid",
        }
        previous_required = legacy_required | {
            "hover_force_per_thruster_n",
            "vertical_ascent_front_rear_trim_percent",
            "vertical_descent_front_rear_trim_percent",
            "forward_front_rear_trim_percent",
            "backward_front_rear_trim_percent",
            "roll_max_moment_nm", "pitch_max_moment_nm", "yaw_max_moment_nm",
        }
        required = previous_required | {
            "left_lateral_surge_trim_percent",
            "right_lateral_surge_trim_percent",
        }
        if set(values) not in (
            legacy_required, previous_required, required
        ) or values["mode"] not in (
            "balance", "rage"
        ):
            raise ValueError("performance_profile fields are incomplete or unknown")
        existing = getattr(self, "performance_profiles", {}).get(
            values["mode"], {}
        )
        lateral_defaults = {
            field: existing.get(field, getattr(self, field, 0.0))
            for field in (
                "left_lateral_surge_trim_percent",
                "right_lateral_surge_trim_percent",
            )
        }
        if set(values) == legacy_required:
            values = {
                **values,
                **{
                    key: value for key, value in self._motion_tuning_snapshot().items()
                    if key not in ("horizontal_limit_fraction", "vertical_limit_fraction")
                },
                "roll_max_moment_nm": self.controller.roll_max_moment_nm,
                "pitch_max_moment_nm": self.controller.pitch_max_moment_nm,
                "yaw_max_moment_nm": self.controller.yaw_max_moment_nm,
                **lateral_defaults,
            }
        elif set(values) == previous_required:
            values = {**values, **lateral_defaults}
        horizontal = float(values["horizontal_limit_fraction"])
        vertical = float(values["vertical_limit_fraction"])
        if not all(math.isfinite(value) for value in (horizontal, vertical)):
            raise ValueError("performance thrust limits must be finite")
        if not 0.0 <= horizontal <= self.max_command_fraction:
            raise ValueError("horizontal performance limit exceeds the global ceiling")
        if not 0.0 <= vertical <= self.max_command_fraction:
            raise ValueError("vertical performance limit exceeds the global ceiling")
        profile = {
            "mode": values["mode"],
            "horizontal_limit_fraction": horizontal,
            "vertical_limit_fraction": vertical,
        }
        hover = float(values["hover_force_per_thruster_n"])
        if not math.isfinite(hover) or not 0.0 <= hover <= 10.0:
            raise ValueError("performance hover force must be in 0..10 N")
        if hover > 55.0 * vertical + 1e-9:
            raise ValueError("performance hover force exceeds the vertical limit")
        profile["hover_force_per_thruster_n"] = hover
        for field in (
            "vertical_ascent_front_rear_trim_percent",
            "vertical_descent_front_rear_trim_percent",
            "forward_front_rear_trim_percent",
            "backward_front_rear_trim_percent",
            "left_lateral_surge_trim_percent",
            "right_lateral_surge_trim_percent",
        ):
            value = float(values[field])
            if not math.isfinite(value) or not -100.0 <= value <= 100.0:
                raise ValueError(f"{field} must be in -100..100 percent")
            profile[field] = value
        for field in (
            "roll_max_moment_nm", "pitch_max_moment_nm", "yaw_max_moment_nm"
        ):
            value = float(values[field])
            if not math.isfinite(value) or not 0.1 <= value <= 20.0:
                raise ValueError(f"{field} must be in 0.1..20 Nm")
            profile[field] = value
        for axis in ("roll", "pitch", "yaw"):
            pid = values[axis + "_pid"]
            if not isinstance(pid, dict) or set(pid) != {"kp", "ki", "kd"}:
                raise ValueError(f"{axis} performance PID fields are incomplete")
            gains = {field: float(pid[field]) for field in ("kp", "ki", "kd")}
            if not all(math.isfinite(value) and value >= 0.0 for value in gains.values()):
                raise ValueError(f"{axis} performance PID values are invalid")
            if gains["kp"] > 1000.0 or gains["ki"] > 200.0 or gains["kd"] > 500.0:
                raise ValueError(f"{axis} performance PID exceeds the safe GUI range")
            profile[axis + "_pid"] = gains
        return profile

    def _performance_profiles_snapshot(self):
        return {
            mode: dict(profile, **{
                axis + "_pid": dict(profile[axis + "_pid"])
                for axis in ("roll", "pitch", "yaw")
            })
            for mode, profile in self.performance_profiles.items()
        }

    def _activate_performance_profile(self, profile):
        """Atomically select all mode-owned limits, feed-forward and PIDs."""
        if self.allocator is not None:
            self.allocator.set_command_fractions(
                profile["horizontal_limit_fraction"],
                profile["vertical_limit_fraction"],
            )
            self.allocator.set_forward_front_rear_trim_percent(
                profile["forward_front_rear_trim_percent"]
            )
            self.allocator.set_backward_front_rear_trim_percent(
                profile["backward_front_rear_trim_percent"]
            )
            self.allocator.set_left_lateral_surge_trim_percent(
                profile["left_lateral_surge_trim_percent"]
            )
            self.allocator.set_right_lateral_surge_trim_percent(
                profile["right_lateral_surge_trim_percent"]
            )
        with self.state_lock:
            self.horizontal_limit_fraction = profile["horizontal_limit_fraction"]
            self.vertical_limit_fraction = profile["vertical_limit_fraction"]
            self.vertical_hover_force_per_thruster_n = profile[
                "hover_force_per_thruster_n"
            ]
            self.vertical_ascent_front_rear_trim_percent = profile[
                "vertical_ascent_front_rear_trim_percent"
            ]
            self.vertical_descent_front_rear_trim_percent = profile[
                "vertical_descent_front_rear_trim_percent"
            ]
            self.forward_front_rear_trim_percent = profile[
                "forward_front_rear_trim_percent"
            ]
            self.backward_front_rear_trim_percent = profile[
                "backward_front_rear_trim_percent"
            ]
            self.left_lateral_surge_trim_percent = profile[
                "left_lateral_surge_trim_percent"
            ]
            self.right_lateral_surge_trim_percent = profile[
                "right_lateral_surge_trim_percent"
            ]
        self.controller.set_attitude_moment_limits(
            profile["roll_max_moment_nm"],
            profile["pitch_max_moment_nm"],
            profile["yaw_max_moment_nm"],
        )
        for pid_type, axis in ((4, "roll"), (5, "pitch"), (6, "yaw")):
            pid = profile[axis + "_pid"]
            self.controller.set_pid_runtime(
                pid_type, (pid["kp"], pid["ki"], pid["kd"])
            )

    def _apply_performance_profile(self, values):
        profile = self._validated_performance_profile(values)
        mode = profile["mode"]
        self.performance_profiles[mode] = profile
        update_config_sections(
            self.config_path,
            {
                f"mode_{mode}": {
                    "horizontal_limit_fraction": profile["horizontal_limit_fraction"],
                    "vertical_limit_fraction": profile["vertical_limit_fraction"],
                    **{
                        field: profile[field]
                        for field in (
                            "hover_force_per_thruster_n",
                            "vertical_ascent_front_rear_trim_percent",
                            "vertical_descent_front_rear_trim_percent",
                            "forward_front_rear_trim_percent",
                            "backward_front_rear_trim_percent",
                            "left_lateral_surge_trim_percent",
                            "right_lateral_surge_trim_percent",
                            "roll_max_moment_nm",
                            "pitch_max_moment_nm",
                            "yaw_max_moment_nm",
                        )
                    },
                    **{
                        f"{axis}_{field}": profile[axis + "_pid"][field]
                        for axis in ("roll", "pitch", "yaw")
                        for field in ("kp", "ki", "kd")
                    },
                }
            },
        )
        if mode == self.performance_mode:
            self._activate_performance_profile(profile)
        return profile

    def _apply_performance_mode(self, mode):
        mode = str(mode).strip().lower()
        if mode not in self.performance_profiles:
            raise ValueError("performance_mode must be balance or rage")
        self.performance_mode = mode
        self._activate_performance_profile(self.performance_profiles[mode])
        self.config.set("control", "performance_mode", mode)
        update_config_sections(self.config_path, {"control": {"performance_mode": mode}})
        return mode

    def _validated_motion_tuning(self, values):
        if not isinstance(values, dict):
            raise ValueError("motion_tuning must be an object")
        required = {
            "horizontal_limit_fraction",
            "vertical_limit_fraction",
            "hover_force_per_thruster_n",
            "vertical_ascent_front_rear_trim_percent",
            "vertical_descent_front_rear_trim_percent",
            "forward_front_rear_trim_percent",
            "backward_front_rear_trim_percent",
        }
        if set(values) != required:
            raise ValueError("motion_tuning fields are incomplete or unknown")
        horizontal = float(values["horizontal_limit_fraction"])
        vertical = float(values["vertical_limit_fraction"])
        hover = float(values["hover_force_per_thruster_n"])
        ascent_trim = float(values["vertical_ascent_front_rear_trim_percent"])
        descent_trim = float(values["vertical_descent_front_rear_trim_percent"])
        forward_trim = float(values["forward_front_rear_trim_percent"])
        backward_trim = float(values["backward_front_rear_trim_percent"])
        if not all(
            math.isfinite(value)
            for value in (
                horizontal, vertical, hover, ascent_trim, descent_trim,
                forward_trim, backward_trim,
            )
        ):
            raise ValueError("motion_tuning values must be finite")
        if not 0.0 <= horizontal <= self.max_command_fraction:
            raise ValueError("horizontal thrust fraction is outside the safe range")
        if not 0.0 <= vertical <= self.max_command_fraction:
            raise ValueError("vertical thrust fraction is outside the safe range")
        if not 0.0 <= hover <= 10.0:
            raise ValueError("hover force per thruster must be in 0..10 N")
        if hover > 55.0 * vertical + 1e-9:
            raise ValueError("hover force exceeds the vertical thrust limit")
        if not all(
            -100.0 <= value <= 100.0
            for value in (ascent_trim, descent_trim, forward_trim, backward_trim)
        ):
            raise ValueError("front/rear trims must be in -100..100 percent")
        return {
            "horizontal_limit_fraction": horizontal,
            "vertical_limit_fraction": vertical,
            "hover_force_per_thruster_n": hover,
            "vertical_ascent_front_rear_trim_percent": ascent_trim,
            "vertical_descent_front_rear_trim_percent": descent_trim,
            "forward_front_rear_trim_percent": forward_trim,
            "backward_front_rear_trim_percent": backward_trim,
        }

    def _apply_motion_tuning(self, values):
        """Apply and persist all live motion tuning as one accepted command."""
        tuning = self._validated_motion_tuning(values)
        old = self._motion_tuning_snapshot()
        try:
            if self.allocator is not None:
                self.allocator.set_command_fractions(
                    tuning["horizontal_limit_fraction"],
                    tuning["vertical_limit_fraction"],
                )
                self.allocator.set_forward_front_rear_trim_percent(
                    tuning["forward_front_rear_trim_percent"]
                )
                self.allocator.set_backward_front_rear_trim_percent(
                    tuning["backward_front_rear_trim_percent"]
                )
            with self.state_lock:
                self.horizontal_limit_fraction = tuning["horizontal_limit_fraction"]
                self.vertical_limit_fraction = tuning["vertical_limit_fraction"]
                self.vertical_hover_force_per_thruster_n = tuning[
                    "hover_force_per_thruster_n"
                ]
                self.vertical_ascent_front_rear_trim_percent = tuning[
                    "vertical_ascent_front_rear_trim_percent"
                ]
                self.vertical_descent_front_rear_trim_percent = tuning[
                    "vertical_descent_front_rear_trim_percent"
                ]
                self.forward_front_rear_trim_percent = tuning[
                    "forward_front_rear_trim_percent"
                ]
                self.backward_front_rear_trim_percent = tuning[
                    "backward_front_rear_trim_percent"
                ]
            update_config_sections(
                self.config_path,
                {
                    "thrusters": {
                        "horizontal_limit_fraction": tuning[
                            "horizontal_limit_fraction"
                        ],
                        "vertical_limit_fraction": tuning[
                            "vertical_limit_fraction"
                        ],
                    },
                    "control": {
                        "vertical_hover_force_per_thruster_n": tuning[
                            "hover_force_per_thruster_n"
                        ],
                        "vertical_ascent_front_rear_trim_percent": tuning[
                            "vertical_ascent_front_rear_trim_percent"
                        ],
                        "vertical_descent_front_rear_trim_percent": tuning[
                            "vertical_descent_front_rear_trim_percent"
                        ],
                        "forward_front_rear_trim_percent": tuning[
                            "forward_front_rear_trim_percent"
                        ],
                        "backward_front_rear_trim_percent": tuning[
                            "backward_front_rear_trim_percent"
                        ],
                    },
                },
            )
        except Exception:
            if self.allocator is not None:
                self.allocator.set_command_fractions(
                    old["horizontal_limit_fraction"],
                    old["vertical_limit_fraction"],
                )
                self.allocator.set_forward_front_rear_trim_percent(
                    old["forward_front_rear_trim_percent"]
                )
                self.allocator.set_backward_front_rear_trim_percent(
                    old["backward_front_rear_trim_percent"]
                )
            with self.state_lock:
                self.horizontal_limit_fraction = old["horizontal_limit_fraction"]
                self.vertical_limit_fraction = old["vertical_limit_fraction"]
                self.vertical_hover_force_per_thruster_n = old[
                    "hover_force_per_thruster_n"
                ]
                self.vertical_ascent_front_rear_trim_percent = old[
                    "vertical_ascent_front_rear_trim_percent"
                ]
                self.vertical_descent_front_rear_trim_percent = old[
                    "vertical_descent_front_rear_trim_percent"
                ]
                self.forward_front_rear_trim_percent = old[
                    "forward_front_rear_trim_percent"
                ]
                self.backward_front_rear_trim_percent = old[
                    "backward_front_rear_trim_percent"
                ]
            raise

        self.config.set(
            "thrusters", "horizontal_limit_fraction",
            str(tuning["horizontal_limit_fraction"]),
        )
        self.config.set(
            "thrusters", "vertical_limit_fraction",
            str(tuning["vertical_limit_fraction"]),
        )
        self.config.set(
            "control", "vertical_hover_force_per_thruster_n",
            str(tuning["hover_force_per_thruster_n"]),
        )
        self.config.set(
            "control", "vertical_ascent_front_rear_trim_percent",
            str(tuning["vertical_ascent_front_rear_trim_percent"]),
        )
        self.config.set(
            "control", "vertical_descent_front_rear_trim_percent",
            str(tuning["vertical_descent_front_rear_trim_percent"]),
        )
        self.config.set(
            "control", "forward_front_rear_trim_percent",
            str(tuning["forward_front_rear_trim_percent"]),
        )
        self.config.set(
            "control", "backward_front_rear_trim_percent",
            str(tuning["backward_front_rear_trim_percent"]),
        )
        return self._motion_tuning_snapshot()

    def _attitude_tuning_snapshot(self):
        return {
            "roll_max_moment_nm": float(self.controller.roll_max_moment_nm),
            "pitch_max_moment_nm": float(self.controller.pitch_max_moment_nm),
            "yaw_max_moment_nm": float(self.controller.yaw_max_moment_nm),
            "yaw_release_rate_threshold_dps": float(
                self.yaw_release_rate_threshold_dps
            ),
        }

    @staticmethod
    def _validated_attitude_tuning(values):
        required = {
            "roll_max_moment_nm", "pitch_max_moment_nm", "yaw_max_moment_nm",
            "yaw_release_rate_threshold_dps",
        }
        if not isinstance(values, dict) or set(values) != required:
            raise ValueError("attitude_tuning fields are incomplete or unknown")
        tuning = {key: float(values[key]) for key in required}
        if not all(
            math.isfinite(tuning[key]) and 0.1 <= tuning[key] <= 20.0
            for key in (
                "roll_max_moment_nm", "pitch_max_moment_nm",
                "yaw_max_moment_nm",
            )
        ):
            raise ValueError("attitude moment limits must be in 0.1..20 Nm")
        if not 0.5 <= tuning["yaw_release_rate_threshold_dps"] <= 30.0:
            raise ValueError("yaw release rate threshold must be in 0.5..30 deg/s")
        return tuning

    def _apply_attitude_tuning(self, values):
        tuning = self._validated_attitude_tuning(values)
        old = self._attitude_tuning_snapshot()
        mode = self.performance_mode
        updated_profile = dict(self.performance_profiles[mode])
        for field in (
            "roll_max_moment_nm", "pitch_max_moment_nm", "yaw_max_moment_nm"
        ):
            updated_profile[field] = tuning[field]
        self.controller.set_attitude_moment_limits(
            tuning["roll_max_moment_nm"], tuning["pitch_max_moment_nm"],
            tuning["yaw_max_moment_nm"],
        )
        with self.state_lock:
            self.yaw_release_rate_threshold_dps = tuning[
                "yaw_release_rate_threshold_dps"
            ]
        try:
            update_config_sections(self.config_path, {
                "control": tuning,
                f"mode_{mode}": {
                    field: tuning[field]
                    for field in (
                        "roll_max_moment_nm",
                        "pitch_max_moment_nm",
                        "yaw_max_moment_nm",
                    )
                },
            })
        except Exception:
            self.controller.set_attitude_moment_limits(
                old["roll_max_moment_nm"], old["pitch_max_moment_nm"],
                old["yaw_max_moment_nm"],
            )
            with self.state_lock:
                self.yaw_release_rate_threshold_dps = old[
                    "yaw_release_rate_threshold_dps"
                ]
            raise
        self.performance_profiles[mode] = updated_profile
        for key, value in tuning.items():
            self.config.set("control", key, str(value))
        for field in (
            "roll_max_moment_nm", "pitch_max_moment_nm", "yaw_max_moment_nm"
        ):
            self.config.set(f"mode_{mode}", field, str(tuning[field]))
        return self._attitude_tuning_snapshot()

    def _disable_attitude_hold(self, reason):
        with self.state_lock:
            self.attitude_hold = {
                "roll": False, "pitch": False, "yaw": False
            }
            self.attitude_hold_reason = str(reason)
            self.attitude_hold_output = {
                "roll_moment_nm": 0.0,
                "pitch_moment_nm": 0.0,
                "yaw_moment_nm": 0.0,
                "roll_manual_override": False,
                "pitch_manual_override": False,
                "yaw_manual_override": False,
            }
            self._yaw_manual_override_active = False
            self._yaw_input_active = False
            self._yaw_release_pending = False
            self._yaw_release_started_at = None
            self._yaw_release_below_threshold_since = None
        self.controller.clear_attitude_holds()

    def _attitude_calibration_config_snapshot(self):
        """Expose only non-secret posture calibration values to the GUI."""
        with self.state_lock:
            values = {
                "imu_3d_mount_calibrated": bool(
                    self.imu_3d_mount_calibrated
                ),
                "mapping_validated": bool(
                    self.attitude_control_mapping_validated
                ),
            }
            for prefix, quaternion in (
                ("imu_mount_quat", self.imu_mount_quaternion),
                ("imu_reference_quat", self.imu_reference_quaternion),
            ):
                for component, value in zip(("w", "x", "y", "z"), quaternion):
                    values[f"{prefix}_{component}"] = float(value)
            return values

    def _set_attitude_mapping_validated(self, enabled):
        """Persist the operator acceptance gate for closed-loop attitude output."""
        if not isinstance(enabled, bool):
            raise ValueError("attitude_mapping_validated_must_be_boolean")

        if not enabled:
            # Runtime safety takes priority over persistence: an operator can
            # always revoke acceptance, including while a hold is active.
            with self.state_lock:
                self.attitude_control_mapping_validated = False
                self.config.set(
                    "control", "attitude_control_mapping_validated", "false"
                )
            self._disable_attitude_hold("mapping_unvalidated")
            update_config_sections(self.config_path, {
                "control": {"attitude_control_mapping_validated": "false"},
            })
            return False

        now = time.monotonic()
        with self.state_lock:
            if self.thrusters_armed:
                raise ValueError(
                    "attitude_mapping_rejected_thrusters_must_be_locked"
                )
            if self.control_enabled:
                raise ValueError(
                    "attitude_mapping_rejected_autonomous_control_must_be_off"
                )
            if any(self.attitude_hold.values()):
                raise ValueError("attitude_mapping_rejected_holds_must_be_off")
            if not self.imu_3d_mount_calibrated:
                raise ValueError("attitude_mapping_rejected_mount_uncalibrated")
            if now - self._last_imu_update > self.imu_freshness:
                raise ValueError("attitude_mapping_rejected_imu_not_fresh")

        update_config_sections(self.config_path, {
            "control": {"attitude_control_mapping_validated": "true"},
        })
        with self.state_lock:
            self.attitude_control_mapping_validated = True
            self.config.set(
                "control", "attitude_control_mapping_validated", "true"
            )
            self.attitude_hold_reason = "operator_disabled"
        return True

    def _set_attitude_hold(self, values):
        # Direct maintenance callers from the previous build may still provide
        # roll/pitch only; authenticated network packets require all 3 axes.
        if not isinstance(values, dict) or set(values) not in (
            {"roll", "pitch"}, {"roll", "pitch", "yaw"}
        ):
            raise ValueError("attitude_hold must contain roll/pitch[/yaw]")
        requested = {
            key: bool(values.get(key, False))
            for key in ("roll", "pitch", "yaw")
        }
        health = self._sensor_health()
        with self.state_lock:
            if any(requested.values()):
                if self.emergency_stop_latched:
                    self.attitude_hold_reason = "emergency_stop"
                    raise ValueError("attitude_hold_rejected_emergency_stop")
                if not self.thrusters_armed:
                    self.attitude_hold_reason = "latched_or_locked"
                    raise ValueError("attitude_hold_rejected_thrusters_locked")
                if self.link_timed_out:
                    self.attitude_hold_reason = "control_link_timeout"
                    raise ValueError("attitude_hold_rejected_control_link")
                if not health["imu_valid"]:
                    self.attitude_hold_reason = "imu_invalid"
                    raise ValueError("attitude_hold_rejected_imu_invalid")
                if not self.imu_3d_mount_calibrated:
                    self.attitude_hold_reason = "mount_uncalibrated"
                    raise ValueError("attitude_hold_rejected_mount_uncalibrated")
                if not self.attitude_control_mapping_validated:
                    self.attitude_hold_reason = "mapping_unvalidated"
                    raise ValueError("attitude_hold_rejected_mapping_unvalidated")
            changed = requested != self.attitude_hold
            if requested["yaw"] and not self.attitude_hold.get("yaw", False):
                yaw = float(self.sensor_state.get("yaw", 0.0))
                if not math.isfinite(yaw):
                    self.attitude_hold_reason = "imu_invalid"
                    raise ValueError("attitude_hold_rejected_imu_invalid")
                self.attitude_hold_target_deg["yaw"] = self._wrap_angle_180(yaw)
            self.attitude_hold = requested
            self.attitude_hold_reason = "operator_enabled" if any(
                requested.values()
            ) else "operator_disabled"
            self.attitude_hold_output = {
                "roll_moment_nm": 0.0,
                "pitch_moment_nm": 0.0,
                "yaw_moment_nm": 0.0,
                "roll_manual_override": False,
                "pitch_manual_override": False,
                "yaw_manual_override": False,
            }
            if changed:
                self._yaw_manual_override_active = False
                self._yaw_input_active = False
                self._yaw_release_pending = False
                self._yaw_release_started_at = None
                self._yaw_release_below_threshold_since = None
        if changed:
            # Prime happens on the next control sample; clearing here guarantees
            # a zero first output even when Kd is non-zero.
            self.controller.clear_attitude_holds()
        return dict(requested)

    def _servo_position_to_angle(self, position):
        position = max(0.0, min(1.0, float(position)))
        return self.servo_angle_min_deg + position * (
            self.servo_angle_max_deg - self.servo_angle_min_deg
        )

    def _servo_angle_to_position(self, angle_deg):
        angle_deg = max(self.servo_angle_min_deg, min(self.servo_angle_max_deg, float(angle_deg)))
        return (angle_deg - self.servo_angle_min_deg) / (
            self.servo_angle_max_deg - self.servo_angle_min_deg
        )

    def _servo_position_to_pulse(self, position):
        position = max(0.0, min(1.0, float(position)))
        return int(round(
            self.servo_pulse_min_us
            + position * (self.servo_pulse_max_us - self.servo_pulse_min_us)
        ))

    def _write_servo_position(self, position):
        """Write the configured auxiliary PWM channel and retain telemetry state."""
        position = max(self.servo_min_position, min(1.0, float(position)))
        angle = self._servo_position_to_angle(position)
        pulse = self._servo_position_to_pulse(position)
        with self.state_lock:
            self.servo_position = position
            self.servo_angle_deg = angle
            self.servo_pwm_us = pulse
        if self.servo_output_enabled and self.allocator:
            self.allocator.set_aux_pwm(self.servo_channel, pulse)

    def _set_servo_target(self, position):
        with self.state_lock:
            self.servo_target_position = max(
                self.servo_min_position, min(1.0, float(position))
            )

    def _servo_worker(self):
        """Slew the auxiliary servo and center it whenever motion is unsafe."""
        interval = 0.05
        try:
            while self.is_running:
                now = time.monotonic()
                with self.state_lock:
                    target = (
                        self.servo_failsafe_position
                        if self.emergency_stop_latched or self.link_timed_out
                        else self.servo_target_position
                    )
                    current_angle = self.servo_angle_deg
                    elapsed = max(0.0, now - self._servo_last_update)
                    self._servo_last_update = now
                    target_angle = self._servo_position_to_angle(target)
                    max_step = self.servo_slew_deg_per_sec * elapsed
                    if abs(target_angle - current_angle) <= max_step:
                        next_angle = target_angle
                    else:
                        next_angle = current_angle + (
                            max_step if target_angle > current_angle else -max_step
                        )
                    next_position = self._servo_angle_to_position(next_angle)
                    if abs(next_position - self.servo_position) > 1e-6:
                        # Match the control-output lock order: state first, then
                        # allocator/I2C. A concurrent safety transition therefore
                        # always writes the failsafe pulse last.
                        self._write_servo_position(next_position)
                time.sleep(interval)
        except Exception as exc:
            self._on_worker_error("servo", exc)

    def _clear_output_state(self):
        with self.state_lock:
            self._last_tau = [0.0] * 6
            self.force_state = {key: 0.0 for key in self.force_state}
            self.pwm_state = {}
            self.thrust_state = {}
            self.depth_manual_override = False

    def _neutralize(self, reason, *, center_servo=True):
        """Best-effort neutralization; servo may remain active while thrusters lock."""
        self._disable_attitude_hold(reason)
        if center_servo:
            try:
                self._set_servo_target(self.servo_failsafe_position)
                self._write_servo_position(self.servo_failsafe_position)
            except Exception:
                logging.exception("failed to center servo while handling %s", reason)
        try:
            if self.allocator and self.motion_output_enabled:
                self.allocator.stop_all()
        except Exception:
            logging.exception("failed to write neutral thruster PWM while handling %s", reason)
        finally:
            self._clear_output_state()

    def _lock_for_rearm(
        self,
        reason,
        *,
        link_timed_out=False,
        control_packet_time=None,
    ):
        """Neutralize without latching ESTOP; a fresh GUI click is required."""
        with self.state_lock:
            self.thrusters_armed = False
            self.control_enabled = False
            self.rearm_required = True
            self._arm_release_seen = False
            self.link_timed_out = bool(link_timed_out)
            if control_packet_time is not None:
                # A newly authenticated session is itself a fresh control
                # packet. Update its timestamp in the same critical section as
                # the forced lock so the watchdog cannot observe an old packet
                # time together with link_timed_out=False.
                self.last_control_packet_time = float(control_packet_time)
            self.joy_active = False
            self.last_joy_time = 0.0
            self._joystick_centered_since = None
            self.joy_force = dict(NEUTRAL_JOYSTICK)
        self._neutralize(reason)
        self._set_ack(False, "locked_rearm_required", reason=str(reason))
        logging.warning("thrusters locked; GUI must click unlock again: %s", reason)

    def _register_control_packet(self, msg, metadata):
        """Record one authenticated packet and enforce session-change locking."""
        if not metadata:
            return
        if not metadata.get("authenticated", False) and not self.allow_legacy_unsigned:
            raise ValueError("unauthenticated command reached safety state")

        now = time.monotonic()
        if metadata.get("session_changed", False):
            self._lock_for_rearm(
                "authenticated_session_changed",
                control_packet_time=now,
            )

        with self.state_lock:
            self.active_control_session = metadata.get("session")
            self.last_control_packet_time = now
            self.link_timed_out = False
            # A packet without an arm request is the required release edge.
            # Therefore a client cannot recover from a timeout by endlessly
            # retransmitting arm=true.
            if not bool(msg.get("arm", False)):
                self._arm_release_seen = True

    def _check_link_timeout(self, now=None):
        """Lock once when authenticated control traffic exceeds the deadline."""
        now = time.monotonic() if now is None else float(now)
        with self.state_lock:
            last = self.last_control_packet_time
            already_timed_out = self.link_timed_out
        if last and not already_timed_out and now - last > self.cmd_timeout:
            self._lock_for_rearm("control_link_timeout", link_timed_out=True)
            return True
        return False

    def trigger_emergency_stop(self, source):
        """Latch emergency stop. Only an explicit reset command can clear it."""
        with self.state_lock:
            self.emergency_stop_latched = True
            self.emergency_stop_source = str(source)
            self.thrusters_armed = False
            self.control_enabled = False
            self.rearm_required = True
            self._arm_release_seen = False
            self.joy_active = False
            self.last_joy_time = 0.0
            self._joystick_centered_since = None
            self.joy_force = dict(NEUTRAL_JOYSTICK)
        self._neutralize(f"emergency_stop:{source}")
        self._set_ack(False, "emergency_stop_latched", source=str(source))
        logging.critical("EMERGENCY STOP LATCHED: %s", source)

    def reset_emergency_stop(self):
        """Clear the latch but deliberately keep arm/control disabled."""
        self._neutralize("emergency_stop_reset")
        with self.state_lock:
            self.emergency_stop_latched = False
            self.emergency_stop_source = ""
            self.thrusters_armed = False
            self.control_enabled = False
            self.rearm_required = True
            # The explicit reset packet itself contains arm=false, so it also
            # provides the release edge required before the next click.
            self._arm_release_seen = True
            self.last_cmd_time = 0.0
            self.last_joy_time = 0.0
            self._joystick_centered_since = None
            self.joy_active = False
            self.joy_force = dict(NEUTRAL_JOYSTICK)
        self._set_ack(True, "emergency_stop_reset_keep_locked")
        logging.warning("emergency stop reset; thrusters remain locked")

    def _on_worker_error(self, worker, exc):
        logging.critical(
            "%s failed: %s", worker, exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        self.trigger_emergency_stop(f"worker_exception:{worker}")

    # ----------------------------- hardware -----------------------------
    def setup_hardware(self):
        print("=" * 60)
        print("ROV control system - fail-safe build")
        print("=" * 60)
        self.imu = IMUReader(self.imu_port, self.imu_baud, timeout=0.05)
        calibration = resolve_path(self.config, "hardware", "mag_calibration_file")
        self.imu.load_mag_calibration(str(calibration))
        self.imu.configure(report_tag=0x004E, report_rate=self.imu_report_rate)

        self.depth_sensor = DepthSensor(
            self.depth_port,
            self.depth_baud,
            self.depth_addr,
            timeout=0.3,
            depth_raw_zero_mpa=self.depth_raw_zero_mpa,
            depth_m_per_mpa=self.depth_m_per_mpa,
        )
        hardware = self.config["hardware"]
        self.allocator = ThrustAllocator(
            pwm_address=self.pwm_address,
            channels_dict=self.thruster_channels,
            motor_signs=self.thruster_signs,
            # Servo output can be bench-tested independently while propulsion
            # PWM remains disabled.
            enable_hardware=(self.motion_output_enabled or self.servo_output_enabled),
            **self.thruster_geometry,
            pwm_min_us=hardware.getint("pwm_min_us"),
            pwm_neutral_us=self.pwm_neutral_us,
            pwm_max_us=hardware.getint("pwm_max_us"),
            pwm_output_min_us=hardware.getint("pwm_output_min_us"),
            pwm_output_max_us=hardware.getint("pwm_output_max_us"),
            reverse_start_us=self.config.getint(
                "thrusters", "reverse_start_us", fallback=None
            ),
            forward_start_us=self.config.getint(
                "thrusters", "forward_start_us", fallback=None
            ),
            horizontal_limit_fraction=self.horizontal_limit_fraction,
            vertical_limit_fraction=self.vertical_limit_fraction,
            vertical_correction_limit_fraction=self.max_command_fraction,
            forward_front_rear_trim_percent=(
                self.forward_front_rear_trim_percent
            ),
            backward_front_rear_trim_percent=(
                self.backward_front_rear_trim_percent
            ),
            left_lateral_surge_trim_percent=(
                self.left_lateral_surge_trim_percent
            ),
            right_lateral_surge_trim_percent=(
                self.right_lateral_surge_trim_percent
            ),
            debug=self.config.getboolean("logging", "thrust_debug"),
        )
        # The active LB profile is authoritative after the allocator exists;
        # this also makes a profile edited while locked survive a service restart.
        self._activate_performance_profile(self.performance_profiles[self.performance_mode])

        channels = list(self.thruster_channels.values())
        # No interactive input: this startup path is safe under systemd.
        # Propulsion writes remain gated independently from optional servo PWM.
        if self.motion_output_enabled:
            self.allocator.pwm.set_motor_speed(channels, self.pwm_neutral_us)
        self._write_servo_position(self.servo_failsafe_position)
        if self.motion_output_enabled:
            logging.warning(
                "REAL PWM OUTPUT ENABLED; holding all channels at neutral during ESC initialization"
            )
            self.allocator.pwm.initialize_esc(
                channels,
                neutral_pulse=self.pwm_neutral_us,
                duration=self.config.getfloat("hardware", "esc_initialization_seconds"),
            )
        elif self.servo_output_enabled:
            logging.warning(
                "SERVO-ONLY MODE: channel %s enabled; propulsion PWM remains disabled",
                self.servo_channel,
            )
        else:
            logging.warning(
                "BENCH MODE: propulsion and servo PWM are disabled; PCA9685 is not opened"
            )
        if self.motion_output_enabled:
            self.allocator.stop_all()
        time.sleep(0.5)
        self.imu.ser.reset_input_buffer()
        for _ in range(10):
            self.imu.read_pending()
            time.sleep(0.01)

        self.controller.update_controller_pids()
        print(f"Pi {self.pi_ip} -> PC {self.pc_ip}:{self.pc_port} (direct Ethernet)")
        print("Thrusters remain LOCKED until an explicit arm command.")
        print(f"Propulsion PWM: {'ENABLED' if self.motion_output_enabled else 'DISABLED'}")
        print(f"Servo PWM (CH{self.servo_channel}): {'ENABLED' if self.servo_output_enabled else 'DISABLED'}")

    # ----------------------------- sensors -----------------------------
    def _sensor_health(self, now=None):
        now = time.monotonic() if now is None else float(now)
        with self.state_lock:
            imu_age = math.inf if not self._last_imu_update else now - self._last_imu_update
            heading_age = (
                math.inf
                if not self._last_heading_update
                else now - self._last_heading_update
            )
            depth_age = math.inf if not self._last_depth_update else now - self._last_depth_update
            pressure_update = getattr(self, "_last_depth_pressure_update", 0.0)
            pressure_age = math.inf if not pressure_update else now - pressure_update
        return {
            "imu_valid": imu_age <= self.imu_freshness,
            # A rejected outlier does not invalidate the IMU stream or manual
            # control; it only reports that the previous valid pose is being
            # held for this sample.
            "attitude_valid": not bool(self._attitude_fault_reason),
            "attitude_fault": self._attitude_fault_reason,
            "attitude_rejected_samples": self._attitude_rejected_samples,
            "heading_valid": heading_age <= self.imu_freshness,
            "depth_valid": depth_age <= self.depth_freshness,
            "pressure_valid": pressure_age <= self.depth_freshness,
            "depth_fault": self._depth_fault_reason,
            "depth_rejected_samples": self._depth_rejected_samples,
            "depth_frames_received": int(
                getattr(self.depth_sensor, "depth_frames_received", 0)
            ),
            "depth_invalid_frames": int(
                getattr(self.depth_sensor, "depth_invalid_frames", 0)
            ),
            "depth_last_error": str(
                getattr(self.depth_sensor, "depth_last_error", "sensor_unavailable")
            ),
            "imu_age_ms": None if math.isinf(imu_age) else round(imu_age * 1000.0, 1),
            "heading_age_ms": (
                None if math.isinf(heading_age) else round(heading_age * 1000.0, 1)
            ),
            "depth_age_ms": None if math.isinf(depth_age) else round(depth_age * 1000.0, 1),
            "pressure_age_ms": (
                None if math.isinf(pressure_age) else round(pressure_age * 1000.0, 1)
            ),
        }

    @staticmethod
    def _wrap_angle_180(angle):
        value = float(angle)
        if not math.isfinite(value):
            raise ValueError("attitude angle must be finite")
        return (value + 180.0) % 360.0 - 180.0

    def _vehicle_heading(self, raw_heading):
        """Convert raw magnetic heading to the operator-visible vehicle course."""
        heading = float(raw_heading)
        if not math.isfinite(heading):
            raise ValueError("magnetic heading must be finite")
        signed = -heading if self.imu_heading_inverted else heading
        return (signed + self.imu_heading_offset_deg) % 360.0

    def _vehicle_magnetic_heading(self, magnetic_field, attitude):
        """Compute physical ROV course from calibrated sensor-frame magnetics."""
        body_magnetic = vehicle_vector_from_raw(
            magnetic_field,
            self.imu_mount_quaternion,
            self._imu_axis_signs(),
            mount_calibrated=self.imu_3d_mount_calibrated,
            label="calibrated magnetic field",
        )
        raw_heading = tilt_compensated_magnetic_heading_deg(
            body_magnetic, attitude["roll"], attitude["pitch"]
        )
        return raw_heading, self._vehicle_heading(raw_heading)

    def _quaternion_from_config(self, prefix):
        hardware = self.config["hardware"]
        return normalize_quaternion(tuple(
            hardware.getfloat(f"{prefix}_{component}", fallback=default)
            for component, default in zip(("w", "x", "y", "z"), IDENTITY_QUATERNION)
        ))

    def _imu_axis_signs(self):
        """Return the sensor-report-to-ROV axis convention established on bench."""
        return (
            -1.0 if self.imu_roll_inverted else 1.0,
            -1.0 if self.imu_pitch_inverted else 1.0,
            -1.0 if self.imu_yaw_inverted else 1.0,
        )

    def _raw_imu_quaternion(self, raw_euler):
        return raw_euler_to_vehicle_axes(raw_euler, self._imu_axis_signs())

    def _stable_raw_imu_quaternion(self, raw_euler, now):
        """Average a short held-pose window and reject motion during capture."""
        fallback = self._raw_imu_quaternion(raw_euler)
        history = getattr(self, "_recent_raw_quaternions", None)
        if history is None:
            # Lightweight unit-test/maintenance controllers created with
            # ``__new__`` predate the live sample buffer.  The production
            # constructor always creates it.
            return fallback, 1, 0.0
        recent = [
            quaternion
            for timestamp, quaternion in history
            if now - timestamp <= MOUNT_CAPTURE_SAMPLE_WINDOW_SECONDS
        ]
        if len(recent) < MOUNT_CAPTURE_MIN_SAMPLES:
            raise ValueError("attitude_mount_insufficient_stable_samples")
        averaged = average_quaternions(recent)
        spread = max(
            quaternion_distance_deg(averaged, quaternion) for quaternion in recent
        )
        if spread > MOUNT_CAPTURE_MAX_SPREAD_DEG:
            raise ValueError("attitude_mount_pose_not_stable")
        return averaged, len(recent), spread

    def _attitude_mount_calibration_snapshot(self):
        """Expose calibration progress without publishing captured quaternions."""
        with self.state_lock:
            capture = self._attitude_mount_capture
            snapshot = {
                "active": isinstance(capture, dict),
                "workflow": "pitch_yaw_v2",
                "baseline_captured": isinstance(capture, dict),
                "pitch_captured": bool(
                    isinstance(capture, dict) and capture.get("pitch_delta")
                ),
                "yaw_captured": bool(
                    isinstance(capture, dict) and capture.get("yaw_delta")
                ),
                "legacy_roll_captured": bool(
                    isinstance(capture, dict) and capture.get("roll_delta")
                ),
                "required_angle_min_deg": MOUNT_CAPTURE_MIN_ANGLE_DEG,
                "required_angle_max_deg": MOUNT_CAPTURE_MAX_ANGLE_DEG,
                "last_completed_workflow": getattr(
                    self, "_last_attitude_mount_workflow", "none"
                ),
            }
        return snapshot

    @staticmethod
    def _quaternion_config_values(prefix, quaternion):
        return {
            f"{prefix}_{component}": f"{value:.12f}"
            for component, value in zip(("w", "x", "y", "z"), normalize_quaternion(quaternion))
        }

    def _referenced_attitude(self, raw_euler):
        """Return ROV-frame attitude; legacy offsets remain pre-calibration only."""
        roll = float(raw_euler["roll"])
        pitch = float(raw_euler["pitch"])
        yaw = float(raw_euler["yaw"])
        if not all(math.isfinite(value) for value in (roll, pitch, yaw)):
            raise ValueError("raw attitude must be finite")
        if self.imu_3d_mount_calibrated:
            attitude, _vehicle_quaternion = attitude_from_raw_euler(
                raw_euler,
                self.imu_mount_quaternion,
                self.imu_reference_quaternion,
                self._imu_axis_signs(),
            )
            return {
                axis: self._wrap_angle_180(attitude[axis])
                for axis in ("roll", "pitch", "yaw")
            }
        return {
            "roll": self._wrap_angle_180(
                (-roll if self.imu_roll_inverted else roll) + self.imu_roll_offset_deg
            ),
            "pitch": self._wrap_angle_180(
                (-pitch if self.imu_pitch_inverted else pitch) + self.imu_pitch_offset_deg
            ),
            "yaw": self._wrap_angle_180(
                (-yaw if self.imu_yaw_inverted else yaw) + self.imu_yaw_offset_deg
            ),
        }

    def _filter_attitude_sample(self, candidate, body_rate, now):
        """Reject only physically implausible single-frame attitude jumps."""
        candidate = {
            axis: self._wrap_angle_180(candidate[axis])
            for axis in ("roll", "pitch", "yaw")
        }
        previous = self._last_accepted_attitude
        if previous is None or not self._last_accepted_attitude_time:
            self._last_accepted_attitude = dict(candidate)
            self._last_accepted_attitude_time = float(now)
            self._attitude_fault_reason = ""
            return candidate, True

        dt = max(
            0.0,
            min(float(now) - self._last_accepted_attitude_time, 0.5),
        )
        delta = max(
            abs(self._wrap_angle_180(candidate[axis] - previous[axis]))
            for axis in ("roll", "pitch", "yaw")
        )
        rate = math.sqrt(sum(float(value) ** 2 for value in body_rate))
        allowed = max(
            ATTITUDE_MIN_PLAUSIBLE_STEP_DEG,
            min(
                ATTITUDE_MAX_PLAUSIBLE_STEP_DEG,
                ATTITUDE_MIN_PLAUSIBLE_STEP_DEG
                + ATTITUDE_STEP_RATE_SCALE * rate * max(dt, 0.02),
            ),
        )
        if not math.isfinite(delta) or delta > allowed:
            self._attitude_rejected_samples += 1
            self._attitude_fault_reason = "attitude_step_implausible"
            return dict(previous), False

        self._last_accepted_attitude = dict(candidate)
        self._last_accepted_attitude_time = float(now)
        self._attitude_fault_reason = ""
        return candidate, True

    def _update_imu_state(self, latest, now):
        with self.state_lock:
            euler = latest.get("euler_deg", {})
            gyro = latest.get("gyro_dps", {})
            acc = latest.get("acc_mps2", {})
            raw_euler = {
                "roll": float(euler.get("roll", self._latest_raw_euler["roll"])),
                "pitch": float(euler.get("pitch", self._latest_raw_euler["pitch"])),
                "yaw": float(euler.get("yaw", self._latest_raw_euler["yaw"])),
            }
            absolute_attitude = self._referenced_attitude(raw_euler)
            if gyro:
                body_rate = vehicle_body_rate_from_raw(
                    (
                        float(gyro.get("x", 0.0)),
                        float(gyro.get("y", 0.0)),
                        float(gyro.get("z", 0.0)),
                    ),
                    self.imu_mount_quaternion,
                    self._imu_axis_signs(),
                    mount_calibrated=self.imu_3d_mount_calibrated,
                )
            else:
                body_rate = (
                    self.sensor_state["gyro_x"],
                    self.sensor_state["gyro_y"],
                    self.sensor_state["gyro_z"],
                )
            absolute_attitude, attitude_sample_accepted = (
                self._filter_attitude_sample(absolute_attitude, body_rate, now)
            )
            yaw_estimate, yaw_fallback = self._yaw_feedback_estimator.update(
                absolute_attitude["yaw"], body_rate[2], now
            )
            attitude = dict(absolute_attitude)
            attitude["yaw"] = yaw_estimate
            if attitude_sample_accepted:
                self._latest_raw_euler = raw_euler
            history = getattr(self, "_recent_raw_quaternions", None)
            if history is not None:
                history.append((now, self._raw_imu_quaternion(raw_euler)))
            heading_update = {}
            magnetic = latest.get("mag_cal")
            if isinstance(magnetic, dict):
                try:
                    heading_raw, heading_vehicle = self._vehicle_magnetic_heading(
                        (
                            float(magnetic["x"]),
                            float(magnetic["y"]),
                            float(magnetic["z"]),
                        ),
                        attitude,
                    )
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    heading_update = {
                        "heading": heading_raw % 360.0,
                        "heading_vehicle": heading_vehicle,
                    }
                    self._last_heading_update = now
            elif "heading_mag_360" in latest and not self.imu_3d_mount_calibrated:
                # Legacy, unmounted fallback is display-only: yaw hold already
                # requires a completed 3-D mount calibration.
                try:
                    heading_raw = float(latest["heading_mag_360"])
                except (TypeError, ValueError):
                    heading_raw = math.nan
                if math.isfinite(heading_raw):
                    heading_update = {
                        "heading": heading_raw % 360.0,
                        "heading_vehicle": self._vehicle_heading(heading_raw),
                    }
                    self._last_heading_update = now
            self.sensor_state.update({
                "roll": attitude["roll"],
                "pitch": attitude["pitch"],
                "yaw": attitude["yaw"],
                "yaw_control": attitude["yaw"],
                "yaw_feedback_fallback": bool(yaw_fallback),
                "attitude_sample_rejected": not attitude_sample_accepted,
                "gyro_x": body_rate[0],
                "gyro_y": body_rate[1],
                "gyro_z": body_rate[2],
                "acc_x": float(acc.get("x", self.sensor_state["acc_x"])),
                "acc_y": float(acc.get("y", self.sensor_state["acc_y"])),
                "acc_z": float(acc.get("z", self.sensor_state["acc_z"])),
                **heading_update,
            })
            roll = math.radians(self.sensor_state["roll"])
            pitch = math.radians(self.sensor_state["pitch"])
            ax_h = self.sensor_state["acc_x"] * math.cos(pitch) + self.sensor_state["acc_z"] * math.sin(pitch)
            ay_h = self.sensor_state["acc_y"] * math.cos(roll) - self.sensor_state["acc_z"] * math.sin(roll)
            dt = 1.0 / max(1, self.imu_report_rate)
            self._est_vx = self.velocity_leak * self._est_vx + ax_h * dt
            self._est_vy = self.velocity_leak * self._est_vy + ay_h * dt
            self._last_imu_update = now

    def _calibrate_attitude_zero(self):
        """Persist the current *level/forward* ROV pose as the Options datum."""
        now = time.monotonic()
        with self.state_lock:
            if self.thrusters_armed or self.control_enabled:
                raise ValueError("attitude_zero_rejected_thrusters_must_be_locked")
            if any(self.attitude_hold.values()):
                raise ValueError("attitude_zero_rejected_holds_must_be_off")
            if now - self._last_imu_update > self.imu_freshness:
                raise ValueError("attitude_zero_rejected_imu_not_fresh")
            if not self.imu_3d_mount_calibrated:
                raise ValueError("attitude_zero_rejected_mount_uncalibrated")
            raw = dict(self._latest_raw_euler)

        _attitude, vehicle_quaternion = attitude_from_raw_euler(
            raw,
            self.imu_mount_quaternion,
            IDENTITY_QUATERNION,
            self._imu_axis_signs(),
        )
        values = self._quaternion_config_values(
            "imu_reference_quat", vehicle_quaternion
        )
        update_config_sections(self.config_path, {
            "hardware": values,
            "control": {"attitude_control_mapping_validated": "false"},
        })
        with self.state_lock:
            self.imu_reference_quaternion = normalize_quaternion(vehicle_quaternion)
            self.attitude_control_mapping_validated = False
            self.attitude_hold_reason = "mapping_unvalidated"
            for key, value in values.items():
                self.config.set("hardware", key, value)
            self.config.set(
                "control", "attitude_control_mapping_validated", "false"
            )
            self.sensor_state.update(self._referenced_attitude(raw))
        self._set_ack(
            True,
            "attitude_zero_calibrated_3d",
            persisted=True,
            mount_calibrated=True,
            datum="physical_level_forward",
            mapping_validated=False,
        )

    def _calibrate_attitude_mount(self):
        """Start a pitch+yaw mount calibration from a level/forward reference."""
        now = time.monotonic()
        with self.state_lock:
            if self.thrusters_armed or self.control_enabled:
                raise ValueError("attitude_mount_rejected_thrusters_must_be_locked")
            if any(self.attitude_hold.values()):
                raise ValueError("attitude_mount_rejected_holds_must_be_off")
            if now - self._last_imu_update > self.imu_freshness:
                raise ValueError("attitude_mount_rejected_imu_not_fresh")
            raw = dict(self._latest_raw_euler)
        baseline, sample_count, spread_deg = self._stable_raw_imu_quaternion(raw, now)

        # A new axis calibration invalidates the previous physical direction
        # acceptance immediately, even if the workflow is later abandoned.
        self._set_attitude_mapping_validated(False)

        with self.state_lock:
            self._attitude_mount_capture = {
                "workflow": "pitch_yaw_v2",
                "baseline": baseline,
                "roll_delta": None,
                "pitch_delta": None,
                "yaw_delta": None,
                "capture_details": {},
            }
        self._set_ack(
            True,
            "attitude_mount_baseline_captured",
            persisted=False,
            workflow="pitch_yaw_v2",
            sample_count=sample_count,
            sample_spread_deg=round(spread_deg, 3),
        )

    def _capture_attitude_mount_axis(self, axis):
        """Capture one pure ROV axis rotation relative to the level baseline."""
        if axis not in ("roll", "pitch", "yaw"):
            raise ValueError("attitude_mount_unknown_axis")
        now = time.monotonic()
        with self.state_lock:
            if self.thrusters_armed or self.control_enabled:
                raise ValueError("attitude_mount_rejected_thrusters_must_be_locked")
            if any(self.attitude_hold.values()):
                raise ValueError("attitude_mount_rejected_holds_must_be_off")
            if now - self._last_imu_update > self.imu_freshness:
                raise ValueError("attitude_mount_rejected_imu_not_fresh")
            capture = self._attitude_mount_capture
            raw = dict(self._latest_raw_euler)
        if not isinstance(capture, dict):
            raise ValueError("attitude_mount_baseline_required")
        pose, sample_count, spread_deg = self._stable_raw_imu_quaternion(raw, now)
        delta = quaternion_multiply(
            quaternion_inverse(capture["baseline"]), pose
        )
        # Validate motion amplitude and axis now, so a missed/too-small hand
        # motion never becomes persistent calibration data.
        rotation_axis(
            delta,
            minimum_angle_deg=MOUNT_CAPTURE_MIN_ANGLE_DEG,
            maximum_angle_deg=MOUNT_CAPTURE_MAX_ANGLE_DEG,
        )
        angle_deg = rotation_angle_deg(delta)
        with self.state_lock:
            self._attitude_mount_capture[f"{axis}_delta"] = delta
            self._attitude_mount_capture["capture_details"][axis] = {
                "angle_deg": angle_deg,
                "sample_count": sample_count,
                "sample_spread_deg": spread_deg,
            }
        self._set_ack(
            True,
            f"attitude_mount_{axis}_captured",
            persisted=False,
            angle_deg=round(angle_deg, 3),
            sample_count=sample_count,
            sample_spread_deg=round(spread_deg, 3),
        )

    def _finalize_attitude_mount(self):
        """Persist a right-handed 3-D mount, preferring pitch+yaw captures."""
        now = time.monotonic()
        with self.state_lock:
            if self.thrusters_armed or self.control_enabled:
                raise ValueError("attitude_mount_rejected_thrusters_must_be_locked")
            if any(self.attitude_hold.values()):
                raise ValueError("attitude_mount_rejected_holds_must_be_off")
            if now - self._last_imu_update > self.imu_freshness:
                raise ValueError("attitude_mount_rejected_imu_not_fresh")
            capture = self._attitude_mount_capture
            raw = dict(self._latest_raw_euler)
        if not isinstance(capture, dict) or not capture.get("pitch_delta"):
            raise ValueError("attitude_mount_pitch_and_yaw_captures_required")

        returned_pose, sample_count, spread_deg = self._stable_raw_imu_quaternion(raw, now)
        return_error_deg = quaternion_distance_deg(
            capture["baseline"], returned_pose
        )
        if return_error_deg > MOUNT_FINALIZE_MAX_RETURN_ERROR_DEG:
            raise ValueError("attitude_mount_finalize_requires_level_baseline_pose")

        if capture.get("yaw_delta"):
            mount = mount_from_pitch_yaw_captures(
                capture["pitch_delta"],
                capture["yaw_delta"],
                minimum_angle_deg=MOUNT_CAPTURE_MIN_ANGLE_DEG,
                maximum_angle_deg=MOUNT_CAPTURE_MAX_ANGLE_DEG,
                minimum_separation_deg=MOUNT_CAPTURE_MIN_AXIS_SEPARATION_DEG,
            )
            workflow = "pitch_yaw_v2"
            success_message = "attitude_mount_calibrated_pitch_yaw"
        elif capture.get("roll_delta"):
            # Backward compatibility for an older GUI that already captured a
            # positive right-side-down roll.  New operators only see pitch+yaw.
            mount = mount_from_axis_captures(
                capture["roll_delta"],
                capture["pitch_delta"],
                minimum_angle_deg=MOUNT_CAPTURE_MIN_ANGLE_DEG,
                maximum_angle_deg=MOUNT_CAPTURE_MAX_ANGLE_DEG,
                minimum_separation_deg=MOUNT_CAPTURE_MIN_AXIS_SEPARATION_DEG,
            )
            workflow = "legacy_roll_pitch_v1"
            success_message = "attitude_mount_calibrated_3d"
        else:
            raise ValueError("attitude_mount_pitch_and_yaw_captures_required")
        reference = quaternion_multiply(capture["baseline"], mount)
        values = {
            "imu_3d_mount_calibrated": "true",
            **self._quaternion_config_values("imu_mount_quat", mount),
            **self._quaternion_config_values("imu_reference_quat", reference),
        }
        update_config_sections(self.config_path, {"hardware": values})
        with self.state_lock:
            self.imu_3d_mount_calibrated = True
            self.imu_mount_quaternion = mount
            self.imu_reference_quaternion = reference
            self._attitude_mount_capture = None
            self._last_attitude_mount_workflow = workflow
            for key, value in values.items():
                self.config.set("hardware", key, value)
            self.sensor_state.update(self._referenced_attitude(raw))
        self._set_ack(
            True,
            success_message,
            persisted=True,
            workflow=workflow,
            return_error_deg=round(return_error_deg, 3),
            sample_count=sample_count,
            sample_spread_deg=round(spread_deg, 3),
        )

    def _update_depth_state(self, now):
        pressure = float(self.depth_sensor.get_pressure())
        temperature = float(self.depth_sensor.get_temperature())
        with self.state_lock:
            self.sensor_state["pressure_mpa"] = pressure
            self.sensor_state["temperature"] = temperature
            if math.isfinite(pressure):
                self._last_depth_pressure_update = now

        depth = float(self.depth_sensor.get_depth())
        # Publish the direct sensor conversion even when it is outside the
        # safe control envelope. Health remains false, so PID cannot use it.
        with self.state_lock:
            self.sensor_state["depth"] = depth
        if not math.isfinite(depth) or not -0.5 <= depth <= 10.0:
            with self.state_lock:
                self._depth_rejected_samples += 1
                self._depth_fault_reason = "depth_out_of_range"
                self._last_depth_update = 0.0
            return
        if self._prev_depth is not None:
            dt = now - self._prev_depth_time
            if dt > 0.01 and abs(depth - self._prev_depth) / dt > 1.5:
                with self.state_lock:
                    self._depth_rejected_samples += 1
                    self._depth_fault_reason = "depth_rate_implausible"
                    self._last_depth_update = 0.0
                return

        with self.state_lock:
            if self._prev_depth is not None:
                dt = now - self._prev_depth_time
                if dt > 0.01:
                    raw_speed = (depth - self._prev_depth) / dt
                    self._vert_speed = 0.7 * self._vert_speed + 0.3 * raw_speed
            self._prev_depth = depth
            self._prev_depth_time = now
            self._last_depth_update = now
            self._depth_fault_reason = ""

    def _calibrate_depth_zero(self):
        """Set any fresh pressure sample as session zero while propulsion is locked."""
        now = time.monotonic()
        with self.state_lock:
            if self.thrusters_armed or self.control_enabled:
                raise ValueError("depth_zero_rejected_thrusters_must_be_locked")
            pressure_update = getattr(self, "_last_depth_pressure_update", 0.0)
            if not pressure_update or now - pressure_update > self.depth_freshness:
                raise ValueError("depth_zero_rejected_pressure_not_fresh")
            pressure = float(self.sensor_state["pressure_mpa"])
        if not math.isfinite(pressure):
            raise ValueError("depth_zero_rejected_pressure_not_finite")
        if self.depth_sensor is None:
            raise ValueError("depth_zero_rejected_sensor_unavailable")

        self.depth_sensor.set_surface_reference(pressure)
        with self.state_lock:
            self.depth_raw_zero_mpa = pressure
            self.sensor_state["depth"] = 0.0
            self._prev_depth = 0.0
            self._prev_depth_time = now
            self._vert_speed = 0.0
            self._last_depth_update = now
            self._depth_fault_reason = ""
        self._set_ack(
            True,
            "depth_zero_calibrated",
            depth_raw_zero_mpa=pressure,
            persisted=False,
        )

    def _imu_worker(self):
        """Keep IMU freshness independent of a blocking depth query."""
        try:
            while self.is_running:
                now = time.monotonic()
                if self.imu.read_pending() and self.imu.latest:
                    self._update_imu_state(self.imu.latest, now)
                time.sleep(0.02)
        except Exception as exc:
            self._on_worker_error("imu", exc)

    def _depth_worker(self):
        """Query depth separately so a serial timeout cannot starve the IMU."""
        while self.is_running:
            started = time.monotonic()
            try:
                result = self.depth_sensor.query()
                if result is not None:
                    self._update_depth_state(time.monotonic())
            except Exception as exc:
                # Depth is optional for manual control. A transient parser or
                # device error must not trigger the vehicle-wide emergency stop.
                logging.exception("depth sample failed; retrying")
                with self.state_lock:
                    self._depth_fault_reason = (
                        f"depth_worker_exception:{type(exc).__name__}"
                    )
                    self._last_depth_update = 0.0
                    self._last_depth_pressure_update = 0.0
            remaining = self.depth_query_interval - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)

    # ----------------------------- control -----------------------------
    @staticmethod
    def _manual_or_auto(manual_value, auto_value, active, deadband):
        if active and abs(manual_value) > deadband:
            return float(manual_value)
        return float(auto_value)

    @staticmethod
    def _depth_manual_override(mode, joy, joy_active, deadband):
        """Return true only for a deliberate vertical stick displacement."""
        return bool(
            mode.get("depth_hold", False)
            and joy_active
            and abs(float(joy["tau_z"])) > float(deadband)
        )

    def _track_depth_target_during_manual_override(
        self, target, current_depth, *, depth_valid, manual_override
    ):
        """Capture the Pi's current depth while manual heave owns the axis."""
        updated = dict(target)
        if not manual_override or not depth_valid:
            return updated
        depth = float(current_depth)
        if not math.isfinite(depth):
            return updated
        updated["depth_m"] = depth
        with self.state_lock:
            if self.mode.get("depth_hold", False):
                self.target["depth_m"] = depth
        return updated

    @staticmethod
    def _effective_vertical_hover(
        configured_hover, depth_hold_active, depth_manual_override
    ):
        """Return the optional always-on vertical feed-forward contribution.

        Depth and attitude loops never depend on this value. A configured zero
        contributes exactly zero while every closed-loop calculation continues.
        """
        return float(configured_hover)

    @staticmethod
    def _slew_axis(previous, requested, max_step, deadband):
        """Move toward a target by a bounded step, including release to zero."""
        previous = float(previous)
        requested = float(requested)
        max_step = float(max_step)
        if max_step <= 0.0:
            raise ValueError("slew max_step must be > 0")
        target = 0.0 if abs(requested) <= float(deadband) else requested
        delta = target - previous
        if delta > max_step:
            return previous + max_step
        if delta < -max_step:
            return previous - max_step
        return target

    @staticmethod
    def _attitude_manual_overrides(attitude_hold, joy, joy_active, deadband):
        """Return per-axis overrides only when an enabled hold is manually displaced."""
        return {
            "roll": bool(
                attitude_hold.get("roll", False)
                and joy_active
                and abs(float(joy["M_x"])) > float(deadband)
            ),
            "pitch": bool(
                attitude_hold.get("pitch", False)
                and joy_active
                and abs(float(joy["M_y"])) > float(deadband)
            ),
            "yaw": bool(
                attitude_hold.get("yaw", False)
                and joy_active
                and abs(float(joy["M_z"])) > float(deadband)
            ),
        }

    @staticmethod
    def _attitude_hold_for_control(
        attitude_hold, manual_override, yaw_release_settling=False
    ):
        """Pause an overridden PID so release re-enters through its prime edge."""
        return {
            axis: bool(
                attitude_hold.get(axis, False)
                and not manual_override.get(axis, False)
                and not (axis == "yaw" and yaw_release_settling)
            )
            for axis in ("roll", "pitch", "yaw")
        }

    def _track_yaw_target_during_manual_override(
        self, current_yaw, *, yaw_rate=0.0, yaw_hold, imu_valid,
        manual_override, now=None,
    ):
        """Track yaw through manual input and the post-release inertial coast.

        The authenticated joystick callback starts the release state.  The yaw
        PID remains paused while corrected gyro Z is above the operator-tuned
        threshold, so it cannot pull the vehicle back toward the stick-release
        angle.  A short below-threshold dwell rejects one noisy gyro sample.
        """
        now = time.monotonic() if now is None else float(now)
        with self.state_lock:
            target = float(self.attitude_hold_target_deg.get("yaw", 0.0))
            active = bool(yaw_hold and imu_valid and manual_override)
            self._yaw_manual_override_active = active
            if not yaw_hold or not imu_valid:
                self._yaw_release_pending = False
                self._yaw_release_started_at = None
                self._yaw_release_below_threshold_since = None
                return target
            yaw = float(current_yaw)
            if not math.isfinite(yaw):
                return target
            if active:
                self._yaw_release_pending = False
                self._yaw_release_started_at = None
                self._yaw_release_below_threshold_since = None
                target = self._wrap_angle_180(yaw)
                self.attitude_hold_target_deg["yaw"] = target
                return target
            if not self._yaw_release_pending:
                return target

            rate = float(yaw_rate)
            if not math.isfinite(rate):
                return target
            target = self._wrap_angle_180(yaw)
            self.attitude_hold_target_deg["yaw"] = target
            if self._yaw_release_started_at is None:
                self._yaw_release_started_at = now
            if abs(rate) <= self.yaw_release_rate_threshold_dps:
                if self._yaw_release_below_threshold_since is None:
                    self._yaw_release_below_threshold_since = now
            else:
                self._yaw_release_below_threshold_since = None

            below_since = self._yaw_release_below_threshold_since
            settled = (
                below_since is not None
                and now - below_since >= YAW_RELEASE_SETTLE_DWELL_SECONDS
            )
            timed_out = (
                now - self._yaw_release_started_at
                >= YAW_RELEASE_MAX_COAST_SECONDS
            )
            if settled or timed_out:
                self._yaw_release_pending = False
                self._yaw_release_started_at = None
                self._yaw_release_below_threshold_since = None
            return target

    def _write_control_wrench(
        self, limited, attitude_hold, auto, manual_override,
        priority_moments=(0.0, 0.0),
    ):
        """Write one still-safe control result and publish its authoritative state.

        Lock order is always ``state_lock`` then the allocator's private lock.
        Safety transitions first change the armed/E-stop/link state under
        ``state_lock`` and then neutralize through the allocator. Therefore a
        control write either completes before the safety transition (whose
        neutral write is last), or observes the unsafe state and is rejected;
        an old cycle cannot write non-zero PWM after neutralization.
        """
        with self.state_lock:
            if (
                not self.thrusters_armed
                or self.emergency_stop_latched
                or self.link_timed_out
            ):
                return None

            tau_x, tau_y, tau_z, M_x, M_y, M_z = limited
            priority_M_x, priority_M_y = map(float, priority_moments)
            force_dict, pwm_dict = self.allocator.set_wrench(
                tau_x=tau_x, tau_y=tau_y, tau_z=tau_z,
                M_x=M_x, M_y=M_y, M_z=M_z, verbose=False,
                priority_M_x=priority_M_x,
                priority_M_y=priority_M_y,
                write_output=self.motion_output_enabled,
            )
            # Continue the next slew cycle from what the limited allocator
            # actually achieved. This prevents hidden state accumulating beyond
            # the configured motor limit.
            allocation = getattr(self.allocator, "last_result", None)
            if allocation is not None:
                achieved = [float(value) for value in allocation.achieved_tau]
            else:
                achieved = list(limited)
            self._last_tau = achieved
            self.force_state = dict(zip(
                ("tau_x", "tau_y", "tau_z", "M_x", "M_y", "M_z"), achieved
            ))
            self.pwm_state = dict(pwm_dict)
            self.thrust_state = dict(force_dict)
            # These fields are PID contributions, not the total achieved moment.
            # ``forces.M_x/M_y/M_z`` remains the authoritative total including manual
            # input and any slew-down tail after an override is released.
            self.attitude_hold_output = {
                "roll_moment_nm": float(auto[3])
                if attitude_hold.get("roll", False)
                and not manual_override.get("roll", False) else 0.0,
                "pitch_moment_nm": float(auto[4])
                if attitude_hold.get("pitch", False)
                and not manual_override.get("pitch", False) else 0.0,
                "yaw_moment_nm": float(auto[5])
                if attitude_hold.get("yaw", False)
                and not manual_override.get("yaw", False) else 0.0,
                "roll_manual_override": bool(manual_override.get("roll", False)),
                "pitch_manual_override": bool(manual_override.get("pitch", False)),
                "yaw_manual_override": bool(manual_override.get("yaw", False)),
            }
            return force_dict, pwm_dict, achieved

    def _control_worker(self):
        try:
            while self.is_running:
                started = time.monotonic()
                with self.state_lock:
                    cmd_enabled = self.control_enabled
                    armed = self.thrusters_armed
                    estop = self.emergency_stop_latched
                    mode = dict(self.mode)
                    target = dict(self.target)
                    sensor = dict(self.sensor_state)
                    joy = dict(self.joy_force)
                    attitude_hold = dict(self.attitude_hold)
                    attitude_targets = dict(self.attitude_hold_target_deg)
                    joy_active = (started - self.last_joy_time) <= self.joy_timeout
                    self.joy_active = joy_active

                if estop or not armed:
                    # A locked propulsion system must not repeatedly override an
                    # independently enabled auxiliary servo. E-stop still centers it.
                    self._neutralize("latched_or_locked", center_servo=estop)
                    self._sleep_control_cycle(started)
                    continue
                if not joy_active and not cmd_enabled and not any(attitude_hold.values()):
                    self._neutralize("no_active_command")
                    self._sleep_control_cycle(started)
                    continue

                health = self._sensor_health(started)
                current = {
                    "heading": sensor["heading"], "depth": sensor["depth"],
                    "vertical_speed": self._vert_speed,
                    "roll": sensor["roll"], "pitch": sensor["pitch"],
                    "yaw": sensor["yaw"],
                    "gyro_x": sensor["gyro_x"], "gyro_y": sensor["gyro_y"],
                    "gyro_z": sensor["gyro_z"],
                    "est_vx": self._est_vx,
                }
                attitude_inhibited = []
                if any(attitude_hold.values()) and not health["imu_valid"]:
                    self._disable_attitude_hold("imu_invalid")
                    attitude_hold = {
                        "roll": False, "pitch": False, "yaw": False
                    }
                    attitude_inhibited.append("attitude_imu_invalid")
                manual_override = self._attitude_manual_overrides(
                    attitude_hold, joy, joy_active, self.manual_deadband
                )
                attitude_targets["yaw"] = (
                    self._track_yaw_target_during_manual_override(
                        sensor["yaw"],
                        yaw_rate=sensor["gyro_z"],
                        yaw_hold=attitude_hold.get("yaw", False),
                        imu_valid=health["imu_valid"],
                        manual_override=manual_override["yaw"],
                    )
                )
                with self.state_lock:
                    yaw_release_settling = bool(self._yaw_release_pending)
                depth_manual_override = self._depth_manual_override(
                    mode, joy, joy_active, self.manual_deadband
                )
                target = self._track_depth_target_during_manual_override(
                    target,
                    sensor["depth"],
                    depth_valid=health["depth_valid"],
                    manual_override=depth_manual_override,
                )
                control_attitude_hold = self._attitude_hold_for_control(
                    attitude_hold, manual_override,
                    yaw_release_settling=yaw_release_settling,
                )
                attitude_inhibited.extend(
                    f"{axis}_manual_override"
                    for axis in ("roll", "pitch", "yaw")
                    if manual_override[axis]
                )
                if yaw_release_settling:
                    attitude_inhibited.append("yaw_release_settling")
                if cmd_enabled or any(attitude_hold.values()):
                    active_mode = mode if cmd_enabled else {
                        "heading_hold": False,
                        "depth_hold": False,
                        "speed_hold": False,
                    }
                    auto, inhibited = self.controller.compute_control(
                        current, active_mode, target,
                        {"imu": health["imu_valid"], "depth": health["depth_valid"]},
                        attitude_hold=control_attitude_hold,
                        depth_manual_override=depth_manual_override,
                        attitude_targets=attitude_targets,
                    )
                else:
                    auto, inhibited = (0.0,) * 6, []
                inhibited.extend(attitude_inhibited)
                with self.state_lock:
                    self._auto_inhibited = list(inhibited)
                    self.depth_manual_override = bool(depth_manual_override)

                tau_x = self._manual_or_auto(joy["tau_x"], auto[0], joy_active, self.manual_deadband)
                tau_y = self._manual_or_auto(joy["tau_y"], auto[1], joy_active, self.manual_deadband)
                tau_z = self._manual_or_auto(joy["tau_z"], auto[2], joy_active, self.manual_deadband)
                M_x = self._manual_or_auto(joy["M_x"], auto[3], joy_active, self.manual_deadband)
                M_y = self._manual_or_auto(joy["M_y"], auto[4], joy_active, self.manual_deadband)
                M_z = self._manual_or_auto(joy["M_z"], auto[5], joy_active, self.manual_deadband)
                priority_M_x = (
                    float(auto[3]) if control_attitude_hold["roll"] else 0.0
                )
                priority_M_y = (
                    float(auto[4]) if control_attitude_hold["pitch"] else 0.0
                )

                # Equal fixed vertical compensation. Safety gates above and the
                # final state-locked write guarantee neutral output on lock,
                # E-stop, link loss, or when no live command source remains.
                vertical_correction = tau_z
                effective_hover = self._effective_vertical_hover(
                    self.vertical_hover_force_per_thruster_n,
                    bool(cmd_enabled and mode.get("depth_hold", False)),
                    depth_manual_override,
                )
                if (
                    attitude_hold.get("roll", False)
                    or attitude_hold.get("pitch", False)
                ) and health["imu_valid"]:
                    effective_hover = tilt_compensated_hover_per_thruster(
                        effective_hover,
                        sensor["roll"],
                        sensor["pitch"],
                        self.attitude_hover_tilt_max_scale,
                        self.thruster_geometry.get(
                            "vertical_directions_body", {}
                        ).values() or None,
                    )
                tau_z = vertical_command_with_hover(
                    vertical_correction, effective_hover
                )
                vertical_trim = select_vertical_front_rear_trim(
                    vertical_correction,
                    self.vertical_ascent_front_rear_trim_percent,
                    self.vertical_descent_front_rear_trim_percent,
                    self.manual_deadband,
                    pitch_hold=attitude_hold.get("pitch", False),
                )
                vertical_trim_moment = vertical_front_rear_trim_moment(
                    tau_z,
                    vertical_trim,
                    self.thruster_geometry["vertical_x_m"],
                )
                M_y += vertical_trim_moment
                priority_M_y += vertical_trim_moment
                requested = [tau_x, tau_y, tau_z, M_x, M_y, M_z]
                steps = [self.max_force_step] * 3 + [self.max_moment_step] * 3
                limited = [
                    self._slew_axis(previous, value, step, self.manual_deadband)
                    for previous, value, step in zip(self._last_tau, requested, steps)
                ]
                if (
                    control_attitude_hold["pitch"]
                    and self.pitch_heave_feedforward_scale > 0.0
                ):
                    limited[4], _pitch_feedforward = (
                        fuse_pitch_heave_feedforward(
                            limited[4],
                            limited[2],
                            effective_hover,
                            self.vertical_ascent_front_rear_trim_percent,
                            self.vertical_descent_front_rear_trim_percent,
                            self.manual_deadband,
                            self.thruster_geometry["vertical_x_m"],
                            self.pitch_heave_feedforward_scale,
                            self.controller.pitch_max_moment_nm,
                        )
                    )
                    # Treat the fused result as one protected attitude request.
                    # The allocator may scale it for a saturated corner, but it
                    # cannot exceed the configured total pitch-moment limit.
                    priority_M_y = limited[4]
                write_result = self._write_control_wrench(
                    limited, attitude_hold, auto, manual_override,
                    priority_moments=(priority_M_x, priority_M_y),
                )
                if write_result is None:
                    self._sleep_control_cycle(started)
                    continue
                force_dict, pwm_dict, achieved = write_result

                if self.save_data_flag:
                    status = {
                        **health,
                        "armed": armed,
                        "emergency_stop": estop,
                        "cmd_enabled": cmd_enabled,
                        "joy_active": joy_active,
                        "depth_manual_override": depth_manual_override,
                        "auto_inhibited": "|".join(inhibited),
                    "roll_hold": attitude_hold.get("roll", False),
                    "pitch_hold": attitude_hold.get("pitch", False),
                    "yaw_hold": attitude_hold.get("yaw", False),
                    "roll_manual_override": manual_override.get("roll", False),
                    "pitch_manual_override": manual_override.get("pitch", False),
                    "yaw_manual_override": manual_override.get("yaw", False),
                    }
                    save.start_save(
                        sensor, {**self.force_state, "pwm": pwm_dict}, target,
                        status=status, config=self.config,
                    )
                self._sleep_control_cycle(started)
        except Exception as exc:
            self._on_worker_error("control", exc)

    def _safety_worker(self):
        """Independent 50 Hz link watchdog; control-loop load cannot defer it."""
        try:
            while self.is_running:
                self._check_link_timeout()
                time.sleep(0.02)
        except Exception as exc:
            self._on_worker_error("safety_watchdog", exc)

    def _sleep_control_cycle(self, started):
        remaining = self.control_interval - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)

    # ----------------------------- telemetry -----------------------------
    def build_telemetry_packet(self):
        health = self._sensor_health()
        with self.state_lock:
            return {
                "protocol_version": self.protocol_version,
                "config_fingerprint": config_fingerprint(self.config),
                "imu": {key: self.sensor_state[key] for key in
                        ("roll", "pitch", "yaw", "heading", "heading_vehicle",
                         "yaw_control", "attitude_sample_rejected",
                         "gyro_x", "gyro_y", "gyro_z")},
                "attitude_3d_mount_calibrated": self.imu_3d_mount_calibrated,
                "attitude_mount_calibration": self._attitude_mount_calibration_snapshot(),
                "depth": self.sensor_state["depth"],
                "pressure_mpa": self.sensor_state["pressure_mpa"],
                "depth_raw_zero_mpa": self.depth_raw_zero_mpa,
                "temperature": self.sensor_state["temperature"],
                "vert_speed": round(self._vert_speed, 4),
                "est_vx": round(self._est_vx, 4),
                "est_vy": round(self._est_vy, 4),
                "sensor_health": health,
                "mode": dict(self.mode),
                "target": dict(self.target),
                "forces": dict(self.force_state),
                "thrust": dict(self.thrust_state),
                "pwm": dict(self.pwm_state),
                "joy": dict(self.joy_force),
                "joy_active": self.joy_active,
                "depth_manual_override": self.depth_manual_override,
                "attitude_hold": {
                    "roll": bool(self.attitude_hold.get("roll", False)),
                    "pitch": bool(self.attitude_hold.get("pitch", False)),
                    "yaw": bool(self.attitude_hold.get("yaw", False)),
                    "roll_target_deg": float(
                        self.attitude_hold_target_deg.get("roll", 0.0)
                    ),
                    "pitch_target_deg": float(
                        self.attitude_hold_target_deg.get("pitch", 0.0)
                    ),
                    "yaw_target_deg": float(
                        self.attitude_hold_target_deg.get("yaw", 0.0)
                    ),
                    "yaw_control_deg": float(
                        self.sensor_state.get("yaw", 0.0)
                    ),
                    "yaw_control_source": (
                        "gyro_z_integrated_fallback"
                        if self.sensor_state.get("yaw_feedback_fallback", False)
                        else "referenced_euler_yaw"
                    ),
                    "yaw_feedback_fallback": bool(
                        self.sensor_state.get("yaw_feedback_fallback", False)
                    ),
                    "yaw_release_settling": bool(self._yaw_release_pending),
                    "yaw_rate_dps": float(
                        self.sensor_state.get("gyro_z", 0.0)
                    ),
                    "mapping_validated": bool(
                        self.attitude_control_mapping_validated
                    ),
                    **self.attitude_hold_output,
                    "reason": self.attitude_hold_reason,
                },
                "attitude_calibration_config": (
                    self._attitude_calibration_config_snapshot()
                ),
                "attitude_tuning": self._attitude_tuning_snapshot(),
                "motion_tuning": self._motion_tuning_snapshot(),
                "performance_mode": self.performance_mode,
                "performance_profiles": self._performance_profiles_snapshot(),
                "armed": self.thrusters_armed,
                "cmd_enabled": self.control_enabled,
                "emergency_stop": self.emergency_stop_latched,
                "emergency_stop_source": self.emergency_stop_source,
                "rearm_required": self.rearm_required,
                "control_link_ok": bool(
                    self.last_control_packet_time and not self.link_timed_out
                ),
                "control_link_age_ms": (
                    None if not self.last_control_packet_time else
                    round((time.monotonic() - self.last_control_packet_time) * 1000.0, 1)
                ),
                "motion_output_enabled": self.motion_output_enabled,
                "servo": {
                    "output_enabled": self.servo_output_enabled,
                    "channel": self.servo_channel,
                    "position": self.servo_position,
                    "target_position": self.servo_target_position,
                    "min_position": self.servo_min_position,
                    "angle_deg": self.servo_angle_deg,
                    "pwm_us": self.servo_pwm_us,
                },
                "auto_inhibited": list(self._auto_inhibited),
                "features": {
                    name: self.config.getboolean("features", name)
                    for name in ("speed_closed_loop",)
                },
                # Readback is in-memory; config.ini is not re-read every telemetry cycle.
                "pid": self.controller.get_pid_snapshot(),
                "command_ack": dict(self.last_command_ack),
                "servo_ack": dict(self.last_servo_ack),
                "save_data": self.save_data_flag,
                "timestamp": time.time(),
            }

    def _udp_worker(self):
        try:
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            last_send_warning = 0.0
            while self.is_running:
                started = time.monotonic()
                payload = encode_telemetry_packet(
                    self.build_telemetry_packet(),
                    self.shared_key,
                    self.telemetry_session,
                    self.telemetry_seq,
                    protocol_version=self.protocol_version,
                )
                self.telemetry_seq += 1
                try:
                    self.udp_sock.sendto(payload, (self.pc_ip, self.pc_port))
                except OSError as exc:
                    # The PC or direct-Ethernet route can appear after the Pi
                    # service starts. Missing telemetry transport is handled by
                    # the command-link watchdog and must not kill this worker or
                    # latch E-stop; keep retrying while propulsion stays locked.
                    now = time.monotonic()
                    if now - last_send_warning >= 2.0:
                        logging.warning(
                            "telemetry send unavailable; waiting for PC link: %s",
                            exc,
                        )
                        last_send_warning = now
                remaining = self.telemetry_interval - (time.monotonic() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
        except OSError as exc:
            if self.is_running:
                self._on_worker_error("telemetry", exc)
        except Exception as exc:
            self._on_worker_error("telemetry", exc)

    # ----------------------------- callbacks -----------------------------
    def _on_joystick(self, joy_force):
        if joy_force.get("emergency_stop", False):
            self.trigger_emergency_stop("joystick_udp")
            return
        with self.state_lock:
            if self.emergency_stop_latched:
                return
            frame = {**NEUTRAL_JOYSTICK, **joy_force, "emergency_stop": False}
            now = time.monotonic()
            yaw_input_active = bool(
                self.attitude_hold.get("yaw", False)
                and abs(float(frame["M_z"])) > self.manual_deadband
            )
            yaw_was_active = bool(self._yaw_input_active)
            self._yaw_input_active = yaw_input_active
            yaw_fresh = now - self._last_imu_update <= self.imu_freshness
            if self.attitude_hold.get("yaw", False) and yaw_fresh:
                yaw = float(self.sensor_state.get("yaw", 0.0))
                if yaw_input_active and math.isfinite(yaw):
                    self.attitude_hold_target_deg["yaw"] = self._wrap_angle_180(yaw)
                    self._yaw_release_pending = False
                    self._yaw_release_started_at = None
                    self._yaw_release_below_threshold_since = None
                elif yaw_was_active:
                    # Do not capture the release-edge angle.  Keep the yaw PID
                    # paused until corrected gyro Z shows that inertia has
                    # decayed, then the control loop captures the final angle.
                    self._yaw_release_pending = True
                    self._yaw_release_started_at = now
                    self._yaw_release_below_threshold_since = None
            elif not self.attitude_hold.get("yaw", False):
                self._yaw_release_pending = False
                self._yaw_release_started_at = None
                self._yaw_release_below_threshold_since = None
            self.joy_force = frame
            self.last_joy_time = now
            axes = ("tau_x", "tau_y", "tau_z", "M_x", "M_y", "M_z")
            if all(abs(float(frame[axis])) <= self.arm_neutral_threshold for axis in axes):
                if self._joystick_centered_since is None:
                    self._joystick_centered_since = now
            else:
                self._joystick_centered_since = None

    def _joystick_is_centered_and_settled(self, now=None):
        """True only after authenticated neutral input has remained stable."""
        now = time.monotonic() if now is None else float(now)
        with self.state_lock:
            centered_since = self._joystick_centered_since
        return centered_since is not None and now - centered_since >= self.arm_neutral_dwell

    def _on_command(self, msg, metadata=None):
        try:
            self._register_control_packet(msg, metadata)

            if bool(msg.get("emergency_stop", False)):
                self.trigger_emergency_stop("command_udp")
                return

            if bool(msg.get("reset_emergency_stop", False)):
                if bool(msg.get("arm", False)):
                    self._set_ack(False, "reset_rejected_arm_must_be_false")
                    return
                self.reset_emergency_stop()
                return

            if bool(msg.get("depth_zero_calibrate", False)):
                if set(msg) != {"depth_zero_calibrate"}:
                    raise ValueError("depth_zero_must_be_a_standalone_command")
                self._calibrate_depth_zero()
                return

            if bool(msg.get("attitude_zero_calibrate", False)):
                if set(msg) != {"attitude_zero_calibrate"}:
                    raise ValueError("attitude_zero_must_be_a_standalone_command")
                self._calibrate_attitude_zero()
                return

            if bool(msg.get("attitude_mount_calibrate", False)):
                if set(msg) != {"attitude_mount_calibrate"}:
                    raise ValueError("attitude_mount_must_be_a_standalone_command")
                self._calibrate_attitude_mount()
                return

            if bool(msg.get("attitude_mount_capture_roll", False)):
                if set(msg) != {"attitude_mount_capture_roll"}:
                    raise ValueError("attitude_mount_roll_must_be_a_standalone_command")
                self._capture_attitude_mount_axis("roll")
                return

            if bool(msg.get("attitude_mount_capture_pitch", False)):
                if set(msg) != {"attitude_mount_capture_pitch"}:
                    raise ValueError("attitude_mount_pitch_must_be_a_standalone_command")
                self._capture_attitude_mount_axis("pitch")
                return

            if bool(msg.get("attitude_mount_capture_yaw", False)):
                if set(msg) != {"attitude_mount_capture_yaw"}:
                    raise ValueError("attitude_mount_yaw_must_be_a_standalone_command")
                self._capture_attitude_mount_axis("yaw")
                return

            if bool(msg.get("attitude_mount_finalize", False)):
                if set(msg) != {"attitude_mount_finalize"}:
                    raise ValueError("attitude_mount_finalize_must_be_a_standalone_command")
                self._finalize_attitude_mount()
                return

            if "attitude_hold" in msg:
                if set(msg) != {"attitude_hold"}:
                    raise ValueError("attitude_hold_must_be_a_standalone_command")
                applied = self._set_attitude_hold(msg["attitude_hold"])
                self._set_ack(
                    True, "attitude_hold_updated", attitude_hold=applied,
                    targets={
                        "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0
                    },
                )
                return

            if "attitude_mapping_validated" in msg:
                if set(msg) != {"attitude_mapping_validated"}:
                    raise ValueError(
                        "attitude_mapping_validated_must_be_a_standalone_command"
                    )
                applied = self._set_attitude_mapping_validated(
                    msg["attitude_mapping_validated"]
                )
                self._set_ack(
                    True,
                    "attitude_mapping_updated",
                    mapping_validated=applied,
                    attitude_calibration_config=(
                        self._attitude_calibration_config_snapshot()
                    ),
                )
                return

            if "attitude_tuning" in msg:
                if set(msg) != {"attitude_tuning"}:
                    raise ValueError("attitude_tuning_must_be_a_standalone_command")
                with self.state_lock:
                    if self.thrusters_armed or self.control_enabled:
                        raise ValueError("attitude_tuning_rejected_thrusters_must_be_locked")
                applied = self._apply_attitude_tuning(msg["attitude_tuning"])
                self._set_ack(
                    True, "attitude_tuning_saved", attitude_tuning=applied,
                )
                return

            if "performance_profile" in msg:
                profile = self._apply_performance_profile(msg["performance_profile"])
                self._set_ack(
                    True,
                    "performance_profile_saved",
                    performance_mode=profile["mode"],
                    performance_profile=profile,
                )

            if "performance_mode" in msg:
                mode = self._apply_performance_mode(msg["performance_mode"])
                self._set_ack(
                    True,
                    "performance_mode_updated",
                    performance_mode=mode,
                    performance_profiles=self._performance_profiles_snapshot(),
                )

            if "joystick" in msg:
                self._on_joystick({**NEUTRAL_JOYSTICK, **msg["joystick"]})

            # Validate all non-state inputs first. The complete state below is
            # then committed under one lock, so the control thread cannot see a
            # new arm flag together with stale autonomous-control state.
            normalized_mode = (
                self.controller.normalize_mode(msg["mode"])
                if "mode" in msg else None
            )
            normalized_target = (
                self.controller.normalize_target(msg["target"])
                if "target" in msg else None
            )
            if normalized_target is not None and not all(
                math.isfinite(value) for value in normalized_target.values()
            ):
                raise ValueError("target values must be finite")

            request_arm = bool(msg["arm"]) if "arm" in msg else None
            request_enable = bool(msg["enable"]) if "enable" in msg else None
            should_neutralize = False
            ack = None
            with self.state_lock:
                if request_arm is True:
                    if self.emergency_stop_latched:
                        ack = (False, "arm_rejected_emergency_stop_latched")
                    elif self.rearm_required and not self._arm_release_seen:
                        ack = (False, "arm_rejected_release_then_click_again")
                    elif not self._joystick_is_centered_and_settled():
                        ack = (False, "arm_rejected_joystick_not_centered")
                    else:
                        self.thrusters_armed = True
                        self.rearm_required = False
                        # An unlock never resumes an old autonomous command.
                        self.control_enabled = bool(request_enable)
                        if self.control_enabled:
                            self.last_cmd_time = time.monotonic()
                        ack = (True, "armed")
                elif request_arm is False:
                    self.thrusters_armed = False
                    self.control_enabled = False
                    self.rearm_required = True
                    self._arm_release_seen = True
                    # Each LOCKED -> ARMED transition needs a fresh neutral
                    # observation, not a neutral frame retained from earlier.
                    self._joystick_centered_since = None
                    should_neutralize = True
                    ack = (True, "locked")
                elif request_enable is not None:
                    if request_enable and self.emergency_stop_latched:
                        self.control_enabled = False
                        ack = (False, "enable_rejected_emergency_stop_latched")
                    elif request_enable and not self.thrusters_armed:
                        # Autonomous enable is never an implicit arm operation.
                        # Reject it at the vehicle even if a buggy or stale GUI
                        # sends enable=True while propulsion is locked.
                        self.control_enabled = False
                        should_neutralize = True
                        ack = (False, "enable_rejected_thrusters_locked")
                    else:
                        self.control_enabled = request_enable
                        if request_enable:
                            self.last_cmd_time = time.monotonic()
                        elif not self.thrusters_armed:
                            should_neutralize = True

                if normalized_mode is not None:
                    self.mode = normalized_mode
                if normalized_target is not None:
                    self.target = normalized_target
                if request_enable is False:
                    self._last_tau = [0.0] * 6

            if should_neutralize:
                self._neutralize("manual_lock")
            if ack is not None:
                self._set_ack(*ack)

            if "servo_min_position" in msg:
                requested_min = max(
                    0.0, min(1.0, float(msg["servo_min_position"]))
                )
                minimum_changed = False
                with self.state_lock:
                    minimum_changed = not math.isclose(
                        self.servo_min_position,
                        requested_min,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    self.servo_min_position = requested_min
                    self.servo_target_position = max(
                        requested_min, self.servo_target_position
                    )
                if minimum_changed:
                    try:
                        update_config_sections(
                            self.config_path,
                            {
                                "servo": {
                                    "closed_position": requested_min,
                                    "failsafe_position": requested_min,
                                }
                            },
                        )
                    except OSError:
                        logging.exception(
                            "failed to persist dynamic servo lower limit"
                        )

            if "servo_position" in msg:
                requested_servo = float(msg["servo_position"])
                requested_servo = max(
                    self.servo_min_position, min(1.0, requested_servo)
                )
                immediate = bool(msg.get("servo_immediate", False))
                with self.state_lock:
                    estop = self.emergency_stop_latched
                if estop:
                    self._set_servo_ack(False, "servo_rejected_emergency_stop_latched")
                elif not self.servo_output_enabled:
                    self._set_servo_ack(False, "servo_rejected_output_disabled")
                else:
                    self._set_servo_target(requested_servo)
                    if immediate:
                        self._write_servo_position(requested_servo)
                    self._set_servo_ack(
                        True,
                        "servo_position_written" if immediate else "servo_target_updated",
                        servo_target_position=requested_servo,
                    )

            if "motion_tuning" in msg:
                applied = self._apply_motion_tuning(msg["motion_tuning"])
                self._set_ack(True, "motion_tuning_saved", motion_tuning=applied)

            if "save_data" in msg:
                save_flag = bool(msg["save_data"])
                with self.state_lock:
                    self.save_data_flag = save_flag
                if not save_flag and save.is_saving():
                    save.close_session()

            if "pid_set" in msg:
                pid = msg["pid_set"]
                values = (float(pid["kp"]), float(pid["ki"]), float(pid["kd"]))
                self.controller.set_pid_runtime(int(pid["type"]), values)
                self._set_ack(True, "pid_applied", pid=self.controller.get_pid_snapshot())
            elif "pid_save" in msg:
                pid = msg["pid_save"]
                values = (float(pid["kp"]), float(pid["ki"]), float(pid["kd"]))
                self.controller.write_pid_to_config(int(pid["type"]), values)
                self._set_ack(True, "pid_saved", pid=self.controller.get_pid_snapshot())
            elif "pid_get" in msg:
                # Legacy query is acknowledged, but telemetry already carries the snapshot.
                self._set_ack(True, "pid_readback_in_telemetry", pid=self.controller.get_pid_snapshot())
        except Exception as exc:
            logging.exception("command rejected")
            self._set_ack(False, "command_rejected", error=str(exc))

    # ----------------------------- lifecycle -----------------------------
    def start(self):
        self.is_running = True
        self.cmd_handler = CommandHandler(
            host=self.bind_host, port=self.cmd_port,
            command_callback=self._on_command, error_callback=self._on_worker_error,
            timeout=self.cmd_timeout,
            shared_key=self.shared_key,
            allowed_ip=self.allowed_ground_ip,
            protocol_version=self.protocol_version,
            max_packet_bytes=self.max_command_packet_bytes,
            allow_legacy_unsigned=self.allow_legacy_unsigned,
        )
        self.cmd_handler.start()
        if self.legacy_joystick_enabled:
            logging.warning(
                "UNSAFE COMPATIBILITY MODE: unauthenticated legacy joystick UDP is enabled"
            )
            self.joy_handler = JoystickHandler(
                host=self.bind_host, port=self.joy_port,
                data_callback=self._on_joystick, error_callback=self._on_worker_error,
                debug=self.config.getboolean("logging", "joystick_debug"),
            )
            self.joy_handler.start()
        for target, name in (
            (self._safety_worker, "SafetyWatchdog"),
            (self._imu_worker, "IMU"),
            (self._depth_worker, "Depth"),
            (self._servo_worker, "Servo"),
            (self._control_worker, "Control"),
            (self._udp_worker, "Telemetry"),
        ):
            threading.Thread(target=target, daemon=True, name=name).start()

    def stop(self):
        self.is_running = False
        with self.state_lock:
            self.thrusters_armed = False
            self.control_enabled = False
            self.rearm_required = True
        # Repeat neutral on normal exit.  This is best-effort software safety;
        # a physical power cut/watchdog is still required for a frozen OS.
        for _ in range(3):
            self._neutralize("shutdown")
            time.sleep(0.02)
        if self.cmd_handler:
            self.cmd_handler.stop()
        if self.joy_handler:
            self.joy_handler.stop()
        if save.is_saving():
            save.close_session()
        for device in (self.imu, self.depth_sensor):
            try:
                if device:
                    device.close()
            except Exception:
                logging.exception("device close failed")
        try:
            if self.udp_sock:
                self.udp_sock.close()
        except Exception:
            logging.exception("UDP close failed")
        try:
            if self.allocator:
                self.allocator.close(stop_thrusters=self.motion_output_enabled)
        except Exception:
            logging.exception("allocator close failed")

    def get_sensor_state(self):
        with self.state_lock:
            return dict(self.sensor_state)

    def get_force_state(self):
        with self.state_lock:
            return dict(self.force_state)


if __name__ == "__main__":
    rov = ROVController()
    shutdown_requested = threading.Event()

    def _request_shutdown(signum, _frame):
        logging.warning("shutdown signal received: %s", signum)
        shutdown_requested.set()

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)
    try:
        rov.setup_hardware()
        rov.start()
        while not shutdown_requested.is_set():
            state = rov.get_sensor_state()
            forces = rov.get_force_state()
            health = rov._sensor_health()
            print(
                f"R={state['roll']:+6.1f} P={state['pitch']:+6.1f} "
                f"H={state['heading']:6.1f} D={state['depth']:6.3f} "
                f"Fz={forces['tau_z']:+7.2f} "
                f"Mx={forces['M_x']:+6.2f} My={forces['M_y']:+6.2f} "
                f"Mz={forces['M_z']:+6.2f} "
                f"ARM={rov.thrusters_armed} ESTOP={rov.emergency_stop_latched} "
                f"IMU={health['imu_valid']} DEPTH={health['depth_valid']}"
            )
            shutdown_requested.wait(0.5)
    except Exception:
        logging.exception("main program failed")
        rov.trigger_emergency_stop("main_exception")
    finally:
        rov.stop()
