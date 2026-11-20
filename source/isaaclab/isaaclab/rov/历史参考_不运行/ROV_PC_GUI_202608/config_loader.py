#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-source runtime configuration helpers for the ROV programs."""

from __future__ import annotations

import configparser
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping



PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = Path(
    os.environ.get("ROV_CONFIG", str(PROJECT_DIR / "config.ini"))
).expanduser()


def load_config(path=None):
    """Load the canonical INI file and fail closed when it is unavailable."""
    config_path = Path(path or DEFAULT_CONFIG_PATH).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"ROV configuration file not found: {config_path}")

    config = configparser.ConfigParser(interpolation=None)
    loaded = config.read(config_path, encoding="utf-8")
    if not loaded:
        raise RuntimeError(f"Unable to read ROV configuration: {config_path}")

    required_sections = {
        "features", "network", "safety", "hardware", "servo", "thrusters",
        "timeouts", "control", "pid", "logging",
    }
    missing = sorted(required_sections.difference(config.sections()))
    if missing:
        raise ValueError(f"config.ini missing sections: {', '.join(missing)}")

    # Fixed direct-Ethernet topology is intentional for competition use.
    if config.get("network", "pi_ip") != "192.168.1.3":
        raise ValueError("network.pi_ip must remain 192.168.1.3 for the direct cable link")
    if config.get("network", "pc_ip") != "192.168.1.4":
        raise ValueError("network.pc_ip must remain 192.168.1.4 for the direct cable link")
    if config.get("safety", "allowed_ground_ip") != config.get("network", "pc_ip"):
        raise ValueError("safety.allowed_ground_ip must match network.pc_ip")
    timeout_ms = config.getint("safety", "command_timeout_ms")
    if not 100 <= timeout_ms <= 2000:
        raise ValueError("safety.command_timeout_ms must be in 100..2000")
    neutral_threshold = config.getfloat("safety", "arm_neutral_threshold")
    if not 0.0 <= neutral_threshold <= 5.0:
        raise ValueError("safety.arm_neutral_threshold must be in 0..5")
    neutral_dwell_ms = config.getint("safety", "arm_neutral_dwell_ms")
    if not 100 <= neutral_dwell_ms <= 2000:
        raise ValueError("safety.arm_neutral_dwell_ms must be in 100..2000")
    shared_key = config.get("safety", "shared_key").strip()
    if shared_key.upper().startswith(("CHANGE", "REPLACE", "TODO")):
        raise ValueError("safety.shared_key is still a placeholder")
    if len(shared_key.encode("utf-8")) < 32:
        raise ValueError("safety.shared_key must contain at least 32 UTF-8 bytes")
    if config.getint("safety", "protocol_version") != 6:
        raise ValueError("safety.protocol_version must be 6 for this build")
    packet_bytes = config.getint("safety", "max_command_packet_bytes")
    if not 256 <= packet_bytes <= 65507:
        raise ValueError("safety.max_command_packet_bytes must be in 256..65507")
    config.getboolean("safety", "motion_output_enabled")
    config.getboolean("safety", "allow_legacy_unsigned")
    config.getboolean("safety", "legacy_joystick_enabled")
    config.getboolean("hardware", "imu_roll_inverted")
    config.getboolean("hardware", "imu_pitch_inverted")
    config.getboolean("hardware", "imu_yaw_inverted")
    config.getboolean("hardware", "imu_heading_inverted")
    for name in (
        "imu_roll_offset_deg",
        "imu_pitch_offset_deg",
        "imu_yaw_offset_deg",
        "imu_heading_offset_deg",
    ):
        offset = config.getfloat("hardware", name)
        if not -360.0 <= offset <= 360.0:
            raise ValueError(f"hardware.{name} must be in -360..360")
    config.getboolean("hardware", "imu_3d_mount_calibrated", fallback=False)
    for prefix in ("imu_mount_quat", "imu_reference_quat"):
        values = tuple(
            config.getfloat("hardware", f"{prefix}_{component}", fallback=default)
            for component, default in zip(("w", "x", "y", "z"), (1.0, 0.0, 0.0, 0.0))
        )
        if not all(math.isfinite(value) for value in values) or sum(value * value for value in values) < 1e-18:
            raise ValueError(f"hardware.{prefix} must be a non-zero finite quaternion")
    servo_channel = config.getint("servo", "channel")
    if not 0 <= servo_channel <= 15:
        raise ValueError("servo.channel must be in 0..15")
    thruster_channels = [
        int(value.strip()) for value in config.get("thrusters", "channels").split(",")
    ]
    if servo_channel in thruster_channels:
        raise ValueError("servo.channel must not overlap a thruster PWM channel")
    if len(thruster_channels) != 8 or len(set(thruster_channels)) != 8:
        raise ValueError("thrusters.channels must contain eight distinct channels")
    servo_low = config.getint("servo", "pulse_min_us")
    servo_center = config.getint("servo", "pulse_center_us")
    servo_high = config.getint("servo", "pulse_max_us")
    if not 500 <= servo_low < servo_center < servo_high <= 2500:
        raise ValueError("servo pulse range must satisfy 500 <= min < center < max <= 2500")
    servo_angle_min = config.getfloat("servo", "angle_min_deg")
    servo_angle_max = config.getfloat("servo", "angle_max_deg")
    if not 0.0 <= servo_angle_min < servo_angle_max <= 180.0:
        raise ValueError("servo angle range must stay within 0..180 degrees")
    if not 1.0 <= config.getfloat("servo", "slew_deg_per_sec") <= 360.0:
        raise ValueError("servo.slew_deg_per_sec must be in 1..360")
    servo_toggle_button = config.getint("servo", "toggle_button")
    if not 0 <= servo_toggle_button <= 31:
        raise ValueError("servo.toggle_button must be in 0..31")
    if servo_toggle_button in (0, 1, 6, 7, 8, 9):
        raise ValueError(
            "servo.toggle_button conflicts with an arm, E-stop, Options, L3, R3, or LB button"
        )
    servo_open = config.getfloat("servo", "open_position")
    servo_closed = config.getfloat("servo", "closed_position")
    if not 0.0 <= servo_closed < servo_open <= 1.0:
        raise ValueError(
            "servo positions must satisfy 0 <= closed_position < open_position <= 1"
        )
    servo_jog_open = config.getfloat(
        "servo", "jog_open_position", fallback=1.0
    )
    servo_jog_step = config.getfloat(
        "servo", "jog_step_per_refresh", fallback=0.05
    )
    if not servo_closed < servo_jog_open <= 1.0:
        raise ValueError(
            "servo.jog_open_position must be above closed_position and <= 1"
        )
    if not 0.005 <= servo_jog_step <= 0.2:
        raise ValueError("servo.jog_step_per_refresh must be in 0.005..0.2")
    servo_failsafe = config.getfloat("servo", "failsafe_position")
    if not 0.0 <= servo_failsafe <= 1.0:
        raise ValueError("servo.failsafe_position must be in 0..1")
    config.getboolean("servo", "output_enabled")

    hardware_pwm_min = config.getint("hardware", "pwm_min_us")
    hardware_pwm_neutral = config.getint("hardware", "pwm_neutral_us")
    hardware_pwm_max = config.getint("hardware", "pwm_max_us")
    if not 500 <= hardware_pwm_min < hardware_pwm_neutral < hardware_pwm_max <= 2500:
        raise ValueError("hardware PWM range must satisfy 500 <= min < neutral < max <= 2500")
    command_fraction = config.getfloat("thrusters", "max_command_fraction")
    if not 0.0 <= command_fraction <= 0.5:
        raise ValueError("thrusters.max_command_fraction must be in 0..0.5")
    horizontal_fraction = config.getfloat(
        "thrusters", "horizontal_limit_fraction", fallback=command_fraction
    )
    vertical_fraction = config.getfloat(
        "thrusters", "vertical_limit_fraction", fallback=command_fraction
    )
    for option, value in (
        ("horizontal_limit_fraction", horizontal_fraction),
        ("vertical_limit_fraction", vertical_fraction),
    ):
        if not 0.0 <= value <= command_fraction:
            raise ValueError(
                f"thrusters.{option} must be in 0..max_command_fraction"
            )
    hover_per_thruster = config.getfloat(
        "control", "vertical_hover_force_per_thruster_n", fallback=0.0
    )
    if not 0.0 <= hover_per_thruster <= 10.0:
        raise ValueError(
            "control.vertical_hover_force_per_thruster_n must be in 0..10 N"
        )
    if hover_per_thruster > 55.0 * vertical_fraction + 1e-9:
        raise ValueError(
            "vertical hover force per thruster exceeds the vertical thrust limit"
        )
    trim_fields = (
        "vertical_ascent_front_rear_trim_percent",
        "vertical_descent_front_rear_trim_percent",
        "forward_front_rear_trim_percent",
        "backward_front_rear_trim_percent",
        "left_lateral_surge_trim_percent",
        "right_lateral_surge_trim_percent",
    )
    for field in trim_fields:
        trim = config.getfloat("control", field, fallback=0.0)
        if not -100.0 <= trim <= 100.0:
            raise ValueError(f"control.{field} must be in -100..100")
    hardware_output_min = config.getint("hardware", "pwm_output_min_us")
    hardware_output_max = config.getint("hardware", "pwm_output_max_us")
    if not (
        hardware_pwm_min <= hardware_output_min < hardware_pwm_neutral
        < hardware_output_max <= hardware_pwm_max
    ):
        raise ValueError("hardware PWM output window must contain neutral")
    for option, comparator in (("pwm_min_us", lambda value: value < hardware_pwm_neutral),
                               ("pwm_max_us", lambda value: value > hardware_pwm_neutral)):
        values = [int(value.strip()) for value in config.get("thrusters", option).split(",")]
        if len(values) != len(thruster_channels):
            raise ValueError(f"thrusters.{option} must have one value per channel")
        for value in values:
            if not comparator(value) or not hardware_output_min <= value <= hardware_output_max:
                raise ValueError(
                    f"thrusters.{option} must stay inside the hardware output window"
                )
    reverse_start = config.getint("thrusters", "reverse_start_us", fallback=None)
    forward_start = config.getint("thrusters", "forward_start_us", fallback=None)
    if reverse_start is not None and not all(
        low < reverse_start < hardware_pwm_neutral
        for low in [int(value.strip()) for value in config.get("thrusters", "pwm_min_us").split(",")]
    ):
        raise ValueError(
            "thrusters.reverse_start_us must stay above every reverse limit and below neutral"
        )
    if forward_start is not None and not all(
        hardware_pwm_neutral < forward_start < high
        for high in [int(value.strip()) for value in config.get("thrusters", "pwm_max_us").split(",")]
    ):
        raise ValueError(
            "thrusters.forward_start_us must stay above neutral and below every forward limit"
        )

    for option in ("lateral_x_m", "lateral_y_m", "vertical_x_m", "vertical_y_m"):
        dimension = config.getfloat("thrusters", option)
        if not math.isfinite(dimension) or not 0.001 <= dimension <= 2.0:
            raise ValueError(f"thrusters.{option} must be in 0.001..2.0 m")

    vertical_names = (
        "vertical_fl", "vertical_fr", "vertical_rl", "vertical_rr"
    )
    geometry_options = tuple(
        f"{name}_{suffix}"
        for name in vertical_names
        for suffix in ("position_m", "direction_body")
    )
    present_geometry = [
        option for option in geometry_options
        if config.has_option("thrusters", option)
    ]
    if present_geometry and len(present_geometry) != len(geometry_options):
        raise ValueError(
            "explicit vertical geometry must define position_m and direction_body for all four thrusters"
        )
    if present_geometry:
        for name in vertical_names:
            position = tuple(
                float(value.strip()) for value in config.get(
                    "thrusters", f"{name}_position_m"
                ).split(",")
            )
            direction = tuple(
                float(value.strip()) for value in config.get(
                    "thrusters", f"{name}_direction_body"
                ).split(",")
            )
            if len(position) != 3 or not all(
                math.isfinite(value) and abs(value) <= 2.0 for value in position
            ):
                raise ValueError(f"thrusters.{name}_position_m must be three finite metres")
            magnitude = math.sqrt(sum(value * value for value in direction))
            if (
                len(direction) != 3
                or not all(math.isfinite(value) for value in direction)
                or abs(magnitude - 1.0) > 1e-3
                or direction[2] < 0.5
            ):
                raise ValueError(
                    f"thrusters.{name}_direction_body must be a unit vector with positive body-z"
                )

    ports = [
        config.getint("network", name)
        for name in ("command_port", "telemetry_port", "joystick_port")
    ]
    if any(port < 1 or port > 65535 for port in ports) or len(set(ports)) != len(ports):
        raise ValueError("network ports must be distinct values in 1..65535")

    force_step = config.getfloat("control", "max_force_step_per_cycle")
    moment_step = config.getfloat("control", "max_moment_step_per_cycle")
    depth_query_hz = config.getfloat("control", "depth_query_hz")
    if not 0.0 < force_step <= 200.0:
        raise ValueError("control.max_force_step_per_cycle must be in (0, 200]")
    if not 0.0 < moment_step <= 40.0:
        raise ValueError("control.max_moment_step_per_cycle must be in (0, 40]")
    if not 1.0 <= depth_query_hz <= 10.0:
        raise ValueError("control.depth_query_hz must be in 1..10 Hz")
    for option in ("joystick_response_exponent", "trigger_response_exponent"):
        exponent = config.getfloat("control", option)
        if not 1.0 <= exponent <= 4.0:
            raise ValueError(f"control.{option} must be in 1.0..4.0")
    depth_min = config.getfloat("control", "depth_target_min_m")
    depth_max = config.getfloat("control", "depth_target_max_m")
    depth_step = config.getfloat("control", "depth_target_max_step_m")
    if not -2.0 <= depth_min < depth_max <= 100.0:
        raise ValueError(
            "control depth target range must satisfy -2 <= min < max <= 100 m"
        )
    if not 0.05 <= depth_step <= 5.0:
        raise ValueError("control.depth_target_max_step_m must be in 0.05..5.0")
    depth_deadband = config.getfloat("control", "depth_hold_deadband_m", fallback=0.0)
    depth_filter_alpha = config.getfloat(
        "control", "depth_hold_filter_alpha", fallback=1.0
    )
    depth_max_force = config.getfloat(
        "control", "depth_max_force_n", fallback=60.0
    )
    depth_release_brake_exit_speed = config.getfloat(
        "control", "depth_release_brake_exit_speed_mps", fallback=0.03
    )
    if not 0.0 <= depth_deadband <= 0.20:
        raise ValueError("control.depth_hold_deadband_m must be in 0..0.20 m")
    if not 0.05 <= depth_filter_alpha <= 1.0:
        raise ValueError("control.depth_hold_filter_alpha must be in 0.05..1.0")
    if not 1.0 <= depth_max_force <= 200.0:
        raise ValueError("control.depth_max_force_n must be in 1..200 N")
    if not 0.01 <= depth_release_brake_exit_speed <= 0.30:
        raise ValueError(
            "control.depth_release_brake_exit_speed_mps must be in 0.01..0.30 m/s"
        )
    for option in (
        "roll_max_moment_nm", "pitch_max_moment_nm", "yaw_max_moment_nm"
    ):
        limit = config.getfloat("control", option)
        if not 0.1 <= limit <= 20.0:
            raise ValueError(f"control.{option} must be in 0.1..20 Nm")
    yaw_release_threshold = config.getfloat(
        "control", "yaw_release_rate_threshold_dps", fallback=3.0
    )
    if not 0.5 <= yaw_release_threshold <= 30.0:
        raise ValueError(
            "control.yaw_release_rate_threshold_dps must be in 0.5..30 deg/s"
        )
    config.getboolean(
        "control", "attitude_control_mapping_validated", fallback=False
    )
    hover_tilt_scale = config.getfloat(
        "control", "attitude_hover_tilt_max_scale", fallback=1.15
    )
    if not 1.0 <= hover_tilt_scale <= 1.5:
        raise ValueError(
            "control.attitude_hover_tilt_max_scale must be in 1.0..1.5"
        )
    pitch_heave_feedforward_scale = config.getfloat(
        "control", "pitch_heave_feedforward_scale", fallback=0.0
    )
    if not 0.0 <= pitch_heave_feedforward_scale <= 1.0:
        raise ValueError(
            "control.pitch_heave_feedforward_scale must be in 0..1"
        )
    for axis in ("roll", "pitch", "yaw"):
        gains = tuple(
            config.getfloat("pid", f"{axis}_{suffix}")
            for suffix in ("kp", "ki", "kd")
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in gains):
            raise ValueError(f"pid.{axis}_kp/ki/kd must be finite and non-negative")

    # Performance profiles are optional for backward-compatible bench configs,
    # but when present they must stay inside the same global safety envelope.
    for section_name in ("mode_balance", "mode_rage"):
        if not config.has_section(section_name):
            continue
        horizontal = config.getfloat(section_name, "horizontal_limit_fraction")
        vertical = config.getfloat(section_name, "vertical_limit_fraction")
        if not 0.0 <= horizontal <= command_fraction or not 0.0 <= vertical <= command_fraction:
            raise ValueError(f"{section_name} thrust fractions exceed the global limit")
        hover = config.getfloat(
            section_name, "hover_force_per_thruster_n", fallback=hover_per_thruster
        )
        if not 0.0 <= hover <= 10.0 or hover > 55.0 * vertical + 1e-9:
            raise ValueError(f"{section_name} fixed vertical compensation is invalid")
        for field in trim_fields:
            trim = config.getfloat(
                section_name,
                field,
                fallback=config.getfloat("control", field, fallback=0.0),
            )
            if not -100.0 <= trim <= 100.0:
                raise ValueError(f"{section_name}.{field} must be in -100..100")
        for option in (
            "roll_max_moment_nm", "pitch_max_moment_nm", "yaw_max_moment_nm"
        ):
            limit = config.getfloat(
                section_name,
                option,
                fallback=config.getfloat("control", option),
            )
            if not 0.1 <= limit <= 20.0:
                raise ValueError(f"{section_name}.{option} must be in 0.1..20 Nm")
        for axis in ("roll", "pitch", "yaw"):
            gains = tuple(
                config.getfloat(section_name, f"{axis}_{suffix}")
                for suffix in ("kp", "ki", "kd")
            )
            if not all(math.isfinite(value) and value >= 0.0 for value in gains):
                raise ValueError(f"{section_name}.{axis} PID values must be finite and non-negative")

    if not 1 <= config.getint("logging", "max_session_files") <= 100:
        raise ValueError("logging.max_session_files must be in 1..100")
    if not 1 <= config.getint("logging", "log_backup_count") <= 100:
        raise ValueError("logging.log_backup_count must be in 1..100")
    if not 65536 <= config.getint("logging", "log_max_bytes") <= 104857600:
        raise ValueError("logging.log_max_bytes must be in 65536..104857600")

    config.config_path = config_path
    return config


def get_int(config, section, option):
    """Read decimal or 0x-prefixed integer values."""
    return int(config.get(section, option).strip(), 0)


def config_fingerprint(config):
    """Return a stable non-secret fingerprint for PC/Pi drift detection."""
    canonical = []
    for section in sorted(config.sections()):
        for option, value in sorted(config.items(section, raw=True)):
            if (section.lower(), option.lower()) == ("safety", "shared_key"):
                continue
            canonical.append((section.lower(), option.lower(), value.strip()))
    payload = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_path(config, section, option):
    """Resolve a configured relative path beside config.ini."""
    value = Path(config.get(section, option)).expanduser()
    if value.is_absolute():
        return value
    base = Path(getattr(config, "config_path", DEFAULT_CONFIG_PATH)).parent
    return (base / value).resolve()


def _replace_section_values(text, section, values: Mapping[str, object]):
    """Return INI text with one section updated while retaining comments."""
    original = text
    lines = original.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in original else "\n"
    section_pattern = re.compile(r"^\s*\[([^]]+)\]\s*(?:[#;].*)?$")
    key_pattern = re.compile(r"^\s*([^#;=\s]+)\s*=.*$")
    wanted = {str(key).lower(): (str(key), str(value)) for key, value in values.items()}

    section_start = None
    section_end = len(lines)
    for index, line in enumerate(lines):
        match = section_pattern.match(line.rstrip("\r\n"))
        if match and match.group(1).strip().lower() == section.lower():
            section_start = index
            for next_index in range(index + 1, len(lines)):
                if section_pattern.match(lines[next_index].rstrip("\r\n")):
                    section_end = next_index
                    break
            break

    if section_start is None:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += newline
        if lines and lines[-1].strip():
            lines.append(newline)
        lines.append(f"[{section}]{newline}")
        lines.extend(f"{key} = {value}{newline}" for key, value in values.items())
    else:
        found = set()
        for index in range(section_start + 1, section_end):
            match = key_pattern.match(lines[index].rstrip("\r\n"))
            if not match:
                continue
            lower_key = match.group(1).lower()
            if lower_key in wanted:
                key, value = wanted[lower_key]
                lines[index] = f"{key} = {value}{newline}"
                found.add(lower_key)
        missing = [wanted[key] for key in wanted if key not in found]
        if missing:
            lines[section_end:section_end] = [
                f"{key} = {value}{newline}" for key, value in missing
            ]
    return "".join(lines)


def update_config_sections(path, section_values: Mapping[str, Mapping[str, object]]):
    """Atomically update multiple INI sections in one filesystem replacement."""
    config_path = Path(path).resolve()
    updated = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    for section, values in section_values.items():
        updated = _replace_section_values(updated, section, values)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False,
            dir=str(config_path.parent), prefix=config_path.name + ".", suffix=".tmp"
        ) as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
            temp_name = stream.name
        os.replace(temp_name, config_path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def update_config_values(path, section, values: Mapping[str, object]):
    """Atomically update one INI section while retaining comments and switches."""
    update_config_sections(path, {section: values})
