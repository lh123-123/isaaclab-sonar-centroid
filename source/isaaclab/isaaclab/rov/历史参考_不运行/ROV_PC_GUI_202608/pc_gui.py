#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BlueROV 上位机 GUI
功能: 遥测显示 / 模式指令 / 虚拟手柄 / PID调参 / 数据保存 / 锁存急停
通信: HMAC认证统一控制UDP 5000 | 遥测UDP 5001
"""

import socket
import json
import math
import os
import queue
from collections import deque
import threading
import time
import sys
import struct
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener
from config_loader import (
    config_fingerprint,
    load_config,
    update_config_sections,
    update_config_values,
)
from input_shaping import (
    apply_response_curve,
    remap_deadzone,
    select_cardinal_axis,
)
from secure_protocol import (
    TelemetryReplayGuard,
    decode_telemetry_packet,
    encode_control_packet,
    new_session_id,
)


_DIRECT_HTTP_OPENER = build_opener(ProxyHandler({}))
CAMERA_PROFILE_LABELS = {
    "水下（已标定）": "water",
    "岸上（已标定）": "shore",
    "自动对焦（默认）": "autofocus",
}


def open_direct_http_stream(source, timeout=5):
    """Open the direct-Ethernet ROV video URL without any system proxy."""
    request = source if isinstance(source, Request) else Request(
        str(source), headers={"User-Agent": "USST-ROV-GUI/1.0"}
    )
    return _DIRECT_HTTP_OPENER.open(request, timeout=timeout)


def camera_profile_endpoint(source):
    """Return the camera-only profile endpoint on the stream's HTTP origin."""
    parsed = urlsplit(str(source))
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("画质预设仅支持树莓派 HTTP 摄像头")
    return urlunsplit((parsed.scheme, parsed.netloc, "/camera/profile", "", ""))


def build_camera_profile_request(source, profile):
    payload = json.dumps({"profile": str(profile)}).encode("utf-8")
    return Request(
        camera_profile_endpoint(source),
        data=payload,
        headers={
            "User-Agent": "USST-ROV-GUI/1.0",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )


APP_CONFIG = load_config()
VIDEO_ANNOTATIONS_PATH = Path(APP_CONFIG.config_path).with_name(
    "video_annotations.json"
)


def _validated_video_annotation_lines(value, *, maximum=500):
    """Return finite normalized line coordinates; reject a corrupt payload."""
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError("invalid video annotation list")
    validated = []
    for line in value:
        if not isinstance(line, (list, tuple)) or len(line) != 2:
            raise ValueError("invalid video annotation line")
        points = []
        for point in line:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("invalid video annotation point")
            coordinates = tuple(float(coordinate) for coordinate in point)
            if not all(math.isfinite(coordinate) for coordinate in coordinates):
                raise ValueError("non-finite video annotation point")
            if not all(0.0 <= coordinate <= 1.0 for coordinate in coordinates):
                raise ValueError("video annotation point outside frame")
            points.append(coordinates)
        validated.append(tuple(points))
    return validated


def load_video_annotations(path=VIDEO_ANNOTATIONS_PATH):
    """Load persistent lines without allowing a damaged file to stop the GUI."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return []
        return _validated_video_annotation_lines(payload.get("lines"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []


def save_video_annotations(path, lines):
    """Atomically persist normalized video lines beside config.ini."""
    path = Path(path)
    validated = _validated_video_annotation_lines(list(lines))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(
                {"version": 1, "lines": validated},
                temporary,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


# ================================================================
#  视觉主题（仅影响上位机显示，不参与任何控制计算）
# ================================================================
UI = {
    "canvas": "#F3F6FA",
    "surface": "#FFFFFF",
    "surface_alt": "#F7F9FC",
    "border": "#DCE4EE",
    "navy": "#0B2035",
    "navy_soft": "#173A5E",
    "text": "#172B3A",
    "muted": "#6B7C8F",
    "blue": "#1677FF",
    "cyan": "#009FB0",
    "green": "#159A67",
    "amber": "#E89B16",
    "orange": "#E56A24",
    "red": "#D92D20",
    "red_dark": "#A81712",
}

_INSTANCE_LOCK_PORT = 49741
JOY_SEND_INTERVAL = 1.0 / 40.0
SERVO_ACK_RETRY_DELAYS_MS = (120, 300)
FAST_LABEL_REFRESH_INTERVAL_MS = 50
DASHBOARD_REFRESH_INTERVAL_MS = 200
POSE_RENDER_INTERVAL_MS = 125


def fit_main_window_geometry(screen_width, screen_height):
    """Leave room for window borders/taskbar so the right column is visible."""
    width = min(1560, max(1, int(screen_width) - 60))
    height = min(900, max(1, int(screen_height) - 100))
    return width, height


def acquire_single_instance_lock():
    """Hold a loopback TCP port so two GUIs cannot compete for UDP state."""
    lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            lock_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
            )
        lock_socket.bind(("127.0.0.1", _INSTANCE_LOCK_PORT))
        lock_socket.listen(1)
    except OSError:
        lock_socket.close()
        return None
    return lock_socket


def telemetry_is_after(marker, reference):
    """Return True only for telemetry received after a local command."""
    if marker is None:
        return False
    if reference is None:
        return True
    marker_session, marker_seq = marker
    reference_session, reference_seq = reference
    if marker_session != reference_session:
        return True
    return marker_seq > reference_seq


class TelemetryLossWindow:
    """Track authenticated sequence gaps over a bounded recent window."""

    def __init__(self, capacity=200):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("telemetry loss window capacity must be positive")
        self.capacity = capacity
        self._samples = deque(maxlen=capacity)
        self._session = None
        self._last_seq = None

    def observe(self, session, seq):
        seq = int(seq)
        if session != self._session:
            self._session = session
            self._last_seq = seq
            self._samples.clear()
            self._samples.append(1)
            return
        if self._last_seq is not None and seq <= self._last_seq:
            raise ValueError("telemetry sequence must increase")
        missing = 0 if self._last_seq is None else seq - self._last_seq - 1
        if missing >= self.capacity:
            self._samples.clear()
            self._samples.extend([0] * (self.capacity - 1))
        elif missing > 0:
            self._samples.extend([0] * missing)
        self._samples.append(1)
        self._last_seq = seq

    def loss_percent(self):
        if not self._samples:
            return None
        received = sum(self._samples)
        return 100.0 * (len(self._samples) - received) / len(self._samples)


def reconcile_estop_state(
    local_latched,
    pending_target,
    pending_until,
    remote_latched,
    now,
):
    """Merge a local E-stop transition with the Pi's authoritative state.

    UDP telemetry can still contain the pre-transition state briefly after the
    operator asserts or resets E-stop.  Ignore that opposite state only while
    the requested transition is pending.  A matching packet acknowledges the
    transition immediately; after the deadline, the Pi always wins.
    """
    remote_latched = bool(remote_latched)
    if pending_target is not None:
        pending_target = bool(pending_target)
        if remote_latched == pending_target:
            return remote_latched, None, 0.0
        # An unacknowledged E-stop assertion must remain locally latched and
        # continue retransmitting forever.  Only a reset request may time out
        # and fall back to the Pi's still-latched state.
        if pending_target:
            return True, True, pending_until
        if now < pending_until:
            return bool(local_latched), pending_target, pending_until
    return remote_latched, None, 0.0


def parse_finite_number(value, label):
    """Parse one operator-entered number and reject NaN/Inf explicitly."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是数字") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}必须是有限数字")
    return result


def servo_position_to_angle(position, angle_min_deg, angle_max_deg):
    """Convert the normalized GUI/Pi servo position to the configured angle."""
    angle_min = float(angle_min_deg)
    angle_max = float(angle_max_deg)
    if not math.isfinite(angle_min) or not math.isfinite(angle_max):
        raise ValueError("舵机角度范围必须是有限数字")
    if not angle_min < angle_max:
        raise ValueError("舵机角度范围无效")
    position = max(0.0, min(1.0, float(position)))
    return angle_min + position * (angle_max - angle_min)


def servo_angle_to_position(angle_deg, angle_min_deg, angle_max_deg):
    """Convert an angle entry to the normalized command sent to the Pi."""
    angle_min = float(angle_min_deg)
    angle_max = float(angle_max_deg)
    angle = max(angle_min, min(angle_max, float(angle_deg)))
    return (angle - angle_min) / (angle_max - angle_min)


def validate_motion_tuning(
    horizontal_fraction, vertical_fraction, hover_force_per_thruster_n,
    vertical_ascent_front_rear_trim_percent,
    vertical_descent_front_rear_trim_percent,
    forward_front_rear_trim_percent,
    backward_front_rear_trim_percent,
):
    """Validate all operator fields sent and persisted as one setting."""
    horizontal = parse_finite_number(horizontal_fraction, "水平推力上限")
    vertical = parse_finite_number(vertical_fraction, "垂直推力上限")
    hover = parse_finite_number(hover_force_per_thruster_n, "每台垂推固定补偿")
    ascent_trim = parse_finite_number(
        vertical_ascent_front_rear_trim_percent, "上升前后差"
    )
    descent_trim = parse_finite_number(
        vertical_descent_front_rear_trim_percent, "下降前后差"
    )
    forward_trim = parse_finite_number(
        forward_front_rear_trim_percent, "前进前后差"
    )
    backward_trim = parse_finite_number(
        backward_front_rear_trim_percent, "后退前后差"
    )
    if not 0.0 <= horizontal <= 0.5:
        raise ValueError("水平推力上限必须在 0～0.5")
    if not 0.0 <= vertical <= 0.5:
        raise ValueError("垂直推力上限必须在 0～0.5")
    if not 0.0 <= hover <= 10.0:
        raise ValueError("每台垂推固定补偿必须在 0～10 N")
    if hover > 55.0 * vertical + 1e-9:
        raise ValueError("每台垂推固定补偿不能超过当前垂直推力上限")
    if not all(
        -100.0 <= value <= 100.0
        for value in (ascent_trim, descent_trim, forward_trim, backward_trim)
    ):
        raise ValueError("四项前后差必须在 -100～100%")
    return {
        "horizontal_limit_fraction": horizontal,
        "vertical_limit_fraction": vertical,
        "hover_force_per_thruster_n": hover,
        "vertical_ascent_front_rear_trim_percent": ascent_trim,
        "vertical_descent_front_rear_trim_percent": descent_trim,
        "forward_front_rear_trim_percent": forward_trim,
        "backward_front_rear_trim_percent": backward_trim,
    }


def validate_performance_profile(
    mode, horizontal_percent, vertical_percent, pid_values,
    motion_values=None, moment_values=None,
):
    """Validate one complete operator-facing balance/rage profile."""
    mode = str(mode).strip().lower()
    if mode not in ("balance", "rage"):
        raise ValueError("模式必须是 balance 或 rage")
    horizontal = parse_finite_number(horizontal_percent, f"{mode} 水平推力上限")
    vertical = parse_finite_number(vertical_percent, f"{mode} 垂直推力上限")
    if not 0.0 <= horizontal <= 50.0 or not 0.0 <= vertical <= 50.0:
        raise ValueError(f"{mode} 推力上限必须在 0～50%")
    profile = {
        "mode": mode,
        "horizontal_limit_fraction": horizontal / 100.0,
        "vertical_limit_fraction": vertical / 100.0,
    }
    motion_values = dict(motion_values or {})
    motion_defaults = {
        "hover_force_per_thruster_n": 0.0,
        "vertical_ascent_front_rear_trim_percent": 0.0,
        "vertical_descent_front_rear_trim_percent": 0.0,
        "forward_front_rear_trim_percent": 0.0,
        "backward_front_rear_trim_percent": 0.0,
        "left_lateral_surge_trim_percent": 0.0,
        "right_lateral_surge_trim_percent": 0.0,
    }
    labels = {
        "hover_force_per_thruster_n": f"{mode} 每台垂推固定补偿",
        "vertical_ascent_front_rear_trim_percent": f"{mode} 上升前后差",
        "vertical_descent_front_rear_trim_percent": f"{mode} 下降前后差",
        "forward_front_rear_trim_percent": f"{mode} 前进前后差",
        "backward_front_rear_trim_percent": f"{mode} 后退前后差",
        "left_lateral_surge_trim_percent": f"{mode} 左横移前后补偿",
        "right_lateral_surge_trim_percent": f"{mode} 右横移前后补偿",
    }
    for field, default in motion_defaults.items():
        profile[field] = parse_finite_number(
            motion_values.get(field, default), labels[field]
        )
    hover = profile["hover_force_per_thruster_n"]
    if not 0.0 <= hover <= 10.0:
        raise ValueError(f"{mode} 每台垂推固定补偿必须在 0～10 N")
    if hover > 55.0 * profile["vertical_limit_fraction"] + 1e-9:
        raise ValueError(f"{mode} 固定垂推补偿不能超过垂直推力上限")
    trim_fields = tuple(field for field in motion_defaults if field != "hover_force_per_thruster_n")
    if not all(-100.0 <= profile[field] <= 100.0 for field in trim_fields):
        raise ValueError(f"{mode} 方向补偿必须在 -100～100%")
    moment_values = dict(moment_values or {})
    for axis, label in (("roll", "横滚"), ("pitch", "俯仰"), ("yaw", "偏转")):
        field = f"{axis}_max_moment_nm"
        profile[field] = parse_finite_number(
            moment_values.get(field, 4.0), f"{mode} {label}力矩上限"
        )
        if not 0.1 <= profile[field] <= 20.0:
            raise ValueError(f"{mode} {label}力矩上限必须在 0.1～20 Nm")
    for axis in ("roll", "pitch", "yaw"):
        profile[axis + "_pid"] = dict(
            zip(("kp", "ki", "kd"), validate_pid_triplet(
                *pid_values[axis], label=f"{mode} {axis} PID"
            ))
        )
    return profile


def validate_attitude_tuning(
    roll_max_moment_nm, pitch_max_moment_nm, yaw_max_moment_nm,
    yaw_release_rate_threshold_dps,
):
    """Validate persistent closed-loop moment limits; these are not trims."""
    values = {
        "roll_max_moment_nm": parse_finite_number(
            roll_max_moment_nm, "横滚力矩上限"
        ),
        "pitch_max_moment_nm": parse_finite_number(
            pitch_max_moment_nm, "俯仰力矩上限"
        ),
        "yaw_max_moment_nm": parse_finite_number(
            yaw_max_moment_nm, "偏转力矩上限"
        ),
    }
    if not all(0.1 <= value <= 20.0 for value in values.values()):
        raise ValueError("横滚/俯仰/偏转力矩上限必须在 0.1～20 Nm")
    release_threshold = parse_finite_number(
        yaw_release_rate_threshold_dps, "偏转松手锁定角速度"
    )
    if not 0.5 <= release_threshold <= 30.0:
        raise ValueError("偏转松手锁定角速度必须在 0.5～30 °/s")
    values["yaw_release_rate_threshold_dps"] = release_threshold
    return values


def step_gripper_jog_position(
    current_position, direction, closed_position, open_position, step=0.05
):
    """Advance one bounded L3/R3 gripper-jog step.

    ``direction`` is +1 for opening, -1 for closing, and 0 for hold.  The
    operator-tested fixed open/closed positions remain the hard travel bounds.
    """
    current = float(current_position)
    lower = float(closed_position)
    upper = float(open_position)
    increment = float(step)
    if not all(math.isfinite(value) for value in (current, lower, upper, increment)):
        raise ValueError("gripper jog values must be finite")
    if lower > upper or increment <= 0.0:
        raise ValueError("gripper jog bounds/step are invalid")
    bounded = max(lower, min(upper, current))
    if direction > 0:
        return min(upper, bounded + increment)
    if direction < 0:
        return max(lower, bounded - increment)
    return bounded


def format_motion_command(telemetry, deadband=0.5):
    """Describe Pi-received software motion intent; this is not measured motion."""
    if not isinstance(telemetry, dict) or not telemetry:
        return "运动指令：控制失联"
    if telemetry.get("emergency_stop"):
        return "运动指令：急停"
    if not telemetry.get("control_link_ok", False):
        return "运动指令：控制失联"
    if not telemetry.get("armed", False):
        return "运动指令：未解锁"

    joy = telemetry.get("joy", {})
    actions = []
    axes = {
        key: float(joy.get(key, 0.0))
        for key in ("tau_x", "tau_y", "tau_z", "M_z")
    }
    if axes["tau_x"] > deadband:
        actions.append("前进")
    elif axes["tau_x"] < -deadband:
        actions.append("后退")
    if axes["tau_y"] > deadband:
        actions.append("右平移")
    elif axes["tau_y"] < -deadband:
        actions.append("左平移")
    if axes["M_z"] > deadband:
        actions.append("右转")
    elif axes["M_z"] < -deadband:
        actions.append("左转")
    if axes["tau_z"] < -deadband:
        actions.append("上升")
    elif axes["tau_z"] > deadband:
        actions.append("下潜")
    if not actions:
        mode = telemetry.get("mode", {})
        if telemetry.get("cmd_enabled") and mode.get("depth_hold"):
            actions.append("定深保持")
        else:
            actions.append("静止")
    return "运动指令：" + " + ".join(actions)


def split_motion_command(motion_text):
    """Return the expanded-view motion label, action, and action colour."""
    prefix = "运动指令："
    text = str(motion_text)
    action = text[len(prefix):].strip() if text.startswith(prefix) else text.strip()
    if not action:
        action = "未知"
    action_color = "#FF4D4F" if action == "未解锁" else "#FFD166"
    return prefix, action, action_color


def expanded_attitude_hold_texts(telemetry):
    """Return concise Pi-authoritative pitch/roll/yaw video-overlay text."""
    hold = telemetry.get("attitude_hold") if isinstance(telemetry, dict) else None
    if not isinstance(hold, dict):
        return "锁俯仰：未知", "锁横滚：未知", "锁偏转：未知"

    def row(axis, label):
        if axis not in hold:
            return f"锁{label}：未知"
        return f"锁{label}：{'开启' if bool(hold[axis]) else '关闭'}"

    return (
        row("pitch", "俯仰"),
        row("roll", "横滚"),
        row("yaw", "偏转"),
    )


def expanded_attitude_value_texts(telemetry):
    """Return Pi-authoritative target depth and live/target attitude values."""
    if not isinstance(telemetry, dict):
        return (
            "目标深度 --", "实时 Yaw --°", "目标 Yaw --°",
            "俯仰 --°", "横滚 --°",
        )
    target = telemetry.get("target")
    imu = telemetry.get("imu")
    hold = telemetry.get("attitude_hold")

    def finite(mapping, key):
        if not isinstance(mapping, dict):
            return None
        try:
            value = float(mapping[key])
        except (KeyError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    target_depth = finite(target, "depth_m")
    yaw = finite(hold, "yaw_control_deg")
    yaw_target = finite(hold, "yaw_target_deg")
    pitch = finite(imu, "pitch")
    roll = finite(imu, "roll")
    return (
        "目标深度 --" if target_depth is None else f"目标深度 {target_depth:.3f} m",
        "实时 Yaw --°" if yaw is None else f"实时 Yaw {yaw:+.1f}°",
        "目标 Yaw --°" if yaw_target is None else f"目标 Yaw {yaw_target:+.1f}°",
        "俯仰 --°" if pitch is None else f"俯仰 {pitch:+.1f}°",
        "横滚 --°" if roll is None else f"横滚 {roll:+.1f}°",
    )


def expanded_video_layout(width, height, frame_width, frame_height):
    """Return a full-canvas video layout; source cropping is handled separately."""
    dimensions = (width, height, frame_width, frame_height)
    if not all(isinstance(value, (int, float)) and value > 0 for value in dimensions):
        raise ValueError("video layout dimensions must be positive")

    width = int(width)
    height = int(height)
    frame_width = int(frame_width)
    frame_height = int(frame_height)

    return 0, 0, width, height


def video_cover_crop(width, height, frame_width, frame_height):
    """Return a centered source crop that fills the target without distortion."""
    dimensions = (width, height, frame_width, frame_height)
    if not all(isinstance(value, (int, float)) and value > 0 for value in dimensions):
        raise ValueError("video crop dimensions must be positive")

    width = int(width)
    height = int(height)
    frame_width = int(frame_width)
    frame_height = int(frame_height)
    target_aspect = width / height
    frame_aspect = frame_width / frame_height
    if frame_aspect > target_aspect:
        crop_height = frame_height
        crop_width = max(1, min(frame_width, int(round(frame_height * target_aspect))))
        crop_left = (frame_width - crop_width) // 2
        crop_top = 0
    else:
        crop_width = frame_width
        crop_height = max(1, min(frame_height, int(round(frame_width / target_aspect))))
        crop_left = 0
        crop_top = (frame_height - crop_height) // 2
    return crop_left, crop_top, crop_width, crop_height


def expanded_canvas_to_video_point(
    x, y, canvas_width, canvas_height, frame_width, frame_height,
):
    """Map one expanded-canvas click to a normalized source-frame point."""
    video_left, video_top, video_width, video_height = expanded_video_layout(
        canvas_width, canvas_height, frame_width, frame_height
    )
    crop_left, crop_top, crop_width, crop_height = video_cover_crop(
        video_width, video_height, frame_width, frame_height
    )
    relative_x = min(1.0, max(0.0, (float(x) - video_left) / video_width))
    relative_y = min(1.0, max(0.0, (float(y) - video_top) / video_height))
    source_x = crop_left + relative_x * crop_width
    source_y = crop_top + relative_y * crop_height
    return source_x / frame_width, source_y / frame_height


def video_point_to_expanded_canvas(
    point, canvas_width, canvas_height, frame_width, frame_height,
):
    """Map a normalized source-frame point back onto the expanded canvas."""
    video_left, video_top, video_width, video_height = expanded_video_layout(
        canvas_width, canvas_height, frame_width, frame_height
    )
    crop_left, crop_top, crop_width, crop_height = video_cover_crop(
        video_width, video_height, frame_width, frame_height
    )
    source_x = float(point[0]) * frame_width
    source_y = float(point[1]) * frame_height
    x = video_left + ((source_x - crop_left) / crop_width) * video_width
    y = video_top + ((source_y - crop_top) / crop_height) * video_height
    return x, y


def expanded_mission_map_layout(
    canvas_width, canvas_height, image_width, image_height,
):
    """Return a legible bottom-left map box that avoids the central gripper."""
    dimensions = (canvas_width, canvas_height, image_width, image_height)
    if not all(isinstance(value, (int, float)) and value > 0 for value in dimensions):
        raise ValueError("mission map dimensions must be positive")
    canvas_width = int(canvas_width)
    canvas_height = int(canvas_height)
    image_width = int(image_width)
    image_height = int(image_height)
    target_width = min(460, max(240, int(round(canvas_width * 0.28))))
    target_width = min(target_width, max(1, canvas_width - 32))
    target_height = max(1, int(round(target_width * image_height / image_width)))
    if target_height > canvas_height * 0.32:
        target_height = max(1, int(round(canvas_height * 0.32)))
        target_width = max(1, int(round(target_height * image_width / image_height)))
    return 16, max(0, canvas_height - target_height - 16), target_width, target_height


def expanded_status_side_width(width):
    """Preserve the existing expanded-HUD text anchors over full-screen video."""
    if not isinstance(width, (int, float)) or width <= 0:
        raise ValueError("expanded status width must be positive")
    return min(180, max(0, (int(width) - 320) // 2))


class DpadAttitudeToggleController:
    """Map D-pad press edges to non-propulsive GUI actions."""

    def __init__(self):
        self._last_direction = (0.0, 0.0)

    @staticmethod
    def _normalized_direction(direction):
        x, y = select_cardinal_axis(*direction)
        return (
            1.0 if x > 0.0 else -1.0 if x < 0.0 else 0.0,
            1.0 if y > 0.0 else -1.0 if y < 0.0 else 0.0,
        )

    def update(self, direction, *, allow_trigger):
        """Return one attitude-toggle action; held directions never repeat."""
        direction = self._normalized_direction(direction)
        action = None
        if allow_trigger and self._last_direction == (0.0, 0.0):
            if direction == (0.0, 1.0):
                action = "pitch_hold_toggle"
            elif direction == (1.0, 0.0):
                action = "roll_hold_toggle"
            elif direction == (-1.0, 0.0):
                action = "yaw_hold_toggle"
            elif direction == (0.0, -1.0):
                action = "timer_toggle"
        self._last_direction = direction
        return action


_ATTITUDE_HOLD_REJECTION_TEXT = {
    "attitude_hold_rejected_emergency_stop": "急停已锁存",
    "attitude_hold_rejected_thrusters_locked": "推进器未解锁",
    "attitude_hold_rejected_control_link": "控制链路未确认",
    "attitude_hold_rejected_imu_invalid": "IMU 数据无效或超时",
    "attitude_hold_rejected_mount_uncalibrated": "IMU 三维安装方向尚未标定",
    "attitude_hold_rejected_mapping_unvalidated": "姿态推进器/IMU 方向映射未验收（mapping_validated=false）",
}


def attitude_hold_rejection_text(value):
    """Translate a Pi rejection code (or an exception containing it)."""
    text = str(value or "").strip()
    for code, translated in _ATTITUDE_HOLD_REJECTION_TEXT.items():
        if code in text:
            return translated
    return text or "Pi 未报告具体原因"


def attitude_hold_enable_block_reason(telemetry):
    """Return one fail-closed preflight reason, with the Pi still authoritative."""
    tele = telemetry if isinstance(telemetry, dict) else {}
    health = tele.get("sensor_health", {})
    hold = tele.get("attitude_hold", {})
    if not tele.get("control_link_ok", False):
        return "控制链路未确认"
    if not tele.get("armed", False):
        return "推进器未解锁"
    if tele.get("emergency_stop", False):
        return "急停已锁存"
    if not isinstance(health, dict) or not health.get("imu_valid", False):
        return "IMU 数据无效或超时"
    if not tele.get("attitude_3d_mount_calibrated", False):
        return "IMU 三维安装方向尚未标定"
    if not isinstance(hold, dict) or not hold.get("mapping_validated", False):
        return "姿态推进器/IMU 方向映射未验收（mapping_validated=false）"
    return None


def validate_attitude_calibration_snapshot(snapshot):
    """Return the only Pi calibration fields that may update local config.ini."""
    if not isinstance(snapshot, dict):
        raise ValueError("Pi 姿态标定配置不是对象")
    boolean_fields = ("imu_3d_mount_calibrated", "mapping_validated")
    for field in boolean_fields:
        if not isinstance(snapshot.get(field), bool):
            raise ValueError(f"Pi 姿态标定配置 {field} 不是布尔值")

    quaternions = {}
    for prefix in ("imu_mount_quat", "imu_reference_quat"):
        values = []
        for component in ("w", "x", "y", "z"):
            field = f"{prefix}_{component}"
            value = snapshot.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Pi 姿态标定配置 {field} 不是数字")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Pi 姿态标定配置 {field} 不是有限值")
            values.append(value)
        norm = math.sqrt(sum(value * value for value in values))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"Pi 姿态标定配置 {prefix} 未归一化")
        quaternions[prefix] = values

    hardware = {
        "imu_3d_mount_calibrated": (
            "true" if snapshot["imu_3d_mount_calibrated"] else "false"
        ),
    }
    for prefix, values in quaternions.items():
        for component, value in zip(("w", "x", "y", "z"), values):
            hardware[f"{prefix}_{component}"] = f"{value:.12f}"
    return {
        "hardware": hardware,
        "control": {
            "attitude_control_mapping_validated": (
                "true" if snapshot["mapping_validated"] else "false"
            ),
        },
    }


def validate_pid_triplet(kp, ki, kd, *, label="PID"):
    """Return non-negative finite PID gains within conservative GUI limits."""
    values = (
        parse_finite_number(kp, f"{label} Kp"),
        parse_finite_number(ki, f"{label} Ki"),
        parse_finite_number(kd, f"{label} Kd"),
    )
    limits = (1000.0, 200.0, 500.0)
    for name, value, limit in zip(("Kp", "Ki", "Kd"), values, limits):
        if value < 0.0:
            raise ValueError(f"{label} {name}不能为负数")
        if value > limit:
            raise ValueError(f"{label} {name}不能超过 {limit:g}")
    return values


def valid_depth_measurement(telemetry):
    """Read one depth value only when Pi marks the sensor sample as valid."""
    if not isinstance(telemetry, dict):
        raise ValueError("没有可用的 Pi 遥测")
    health = telemetry.get("sensor_health")
    if not isinstance(health, dict) or not bool(health.get("depth_valid", False)):
        raise ValueError("深度传感器数据已超时，禁止启动定深")
    if "depth" not in telemetry:
        raise ValueError("Pi 遥测缺少深度数据")
    return parse_finite_number(telemetry["depth"], "当前深度")


def format_depth_display(telemetry):
    """Display the finite direct sensor conversion; health gates PID separately."""
    if not isinstance(telemetry, dict) or "depth" not in telemetry:
        return "--"
    try:
        return f"{parse_finite_number(telemetry['depth'], '当前深度'):.3f} m"
    except ValueError:
        return "--"


def format_pressure_status_text(telemetry):
    """Show raw pressure only when Pi marks the pressure sample fresh."""
    if not isinstance(telemetry, dict):
        return ""
    health = telemetry.get("sensor_health", {})
    if not isinstance(health, dict) or not health.get("pressure_valid", False):
        return " · 原始压力 --"
    raw_pressure = telemetry.get("pressure_mpa")
    if not isinstance(raw_pressure, (int, float)):
        return " · 原始压力 --"
    try:
        return f" · 原始压力 {parse_finite_number(raw_pressure, '深度计原始压力'):.6f} MPa"
    except ValueError:
        return " · 原始压力 --"


def validate_depth_request(
    telemetry,
    target_depth,
    kp,
    ki,
    kd,
    *,
    minimum_depth,
    maximum_depth,
    maximum_step,
):
    """Validate one depth-hold request against fresh Pi telemetry."""
    current_depth = valid_depth_measurement(telemetry)
    target = parse_finite_number(target_depth, "目标深度")
    minimum = float(minimum_depth)
    maximum = float(maximum_depth)
    max_step = float(maximum_step)
    if not minimum <= target <= maximum:
        raise ValueError(
            f"目标深度必须在 {minimum:.2f}～{maximum:.2f} m 之间"
        )
    delta = target - current_depth
    if abs(delta) > max_step:
        raise ValueError(
            f"目标与当前深度相差 {abs(delta):.2f} m，超过单次允许的 "
            f"{max_step:.2f} m；请分段调整"
        )
    gains = validate_pid_triplet(kp, ki, kd, label="定深 PID")
    return {
        "current_depth": current_depth,
        "target_depth": target,
        "depth_error": delta,
        "pid": gains,
    }


# ================================================================
#  生成菠萝图标 (32x32, 内嵌 PhotoImage)
# ================================================================
def _make_pineapple_icon(root):
    """用 tkinter Canvas 画菠萝, 转 PhotoImage 设为图标"""
    size = 32
    img = tk.PhotoImage(width=size, height=size)

    # 菠萝身体: 黄色椭圆
    for y in range(size):
        for x in range(size):
            dx = (x - 16) / 12.0
            dy = (y - 20) / 11.0
            in_body = (dx * dx + dy * dy <= 1.0) and y >= 7 and y < 30

            # 绿色叶子区
            dx_center = abs(x - 16)
            in_leaf = (y < 10 and dx_center < 3 + (y - 2) * 2.5
                       and y < 9 + abs(x - 16) * 0.2)

            if in_body:
                # 菠萝纹: 斜线纹理
                if (x + y) % 5 < 3:
                    img.put("#f0b800", (x, y))  # 亮黄
                else:
                    img.put("#d4940a", (x, y))  # 暗黄
            elif in_leaf:
                img.put("#2ca02c", (x, y))  # 绿色
            else:
                img.put("#000000", (x, y))  # 暂存, 稍后透明

    # 重绘透明区域 (黑色→透明)
    for y in range(size):
        for x in range(size):
            dx = (x - 16) / 12.0
            dy = (y - 20) / 11.0
            in_body = (dx * dx + dy * dy <= 1.0) and y >= 7 and y < 30
            dx_center = abs(x - 16)
            in_leaf = (y < 10 and dx_center < 3 + (y - 2) * 2.5
                       and y < 9 + abs(x - 16) * 0.2)

            if not in_body and not in_leaf:
                # 透明: 用系统背景色近似
                img.put("#f0f0f0", (x, y))

    # 设置图标
    root.iconphoto(True, img)
    return img


# ================================================================
#  推进器俯视图（纯显示组件，不发送任何控制指令）
# ================================================================
class ThrusterDiagram(tk.Frame):
    """在 ROV 俯视图上显示八推进器推力，并动画化四个蓝色旋叶。"""

    IMAGE_SOURCE_SIZE = 1256.0
    THRUSTER_LABELS = {
        "lateral_fl": "左前平",
        "lateral_fr": "右前平",
        "lateral_rl": "左后平",
        "lateral_rr": "右后平",
        "vertical_fl": "左前垂",
        "vertical_fr": "右前垂",
        "vertical_rl": "左后垂",
        "vertical_rr": "右后垂",
    }
    # 位置使用图片归一化坐标；图片顶部为艇艏。
    LABEL_POSITIONS = {
        "lateral_fl": (0.15, 0.09),
        "lateral_fr": (0.85, 0.09),
        "lateral_rl": (0.15, 0.91),
        "lateral_rr": (0.85, 0.91),
        "vertical_fl": (0.13, 0.37),
        "vertical_fr": (0.87, 0.37),
        "vertical_rl": (0.13, 0.64),
        "vertical_rr": (0.87, 0.64),
    }
    ROTOR_CENTERS = {
        "vertical_fl": (0.236, 0.400),
        "vertical_fr": (0.795, 0.400),
        "vertical_rl": (0.236, 0.668),
        "vertical_rr": (0.795, 0.668),
    }

    def __init__(self, parent, image_path, display_size=314):
        super().__init__(parent, bg=UI["surface"])
        self._size = display_size
        self._thrust = {key: 0.0 for key in self.THRUSTER_LABELS}
        self._connected = False
        self._angles = {key: 0.0 for key in self.ROTOR_CENTERS}
        self._last_frame_time = time.monotonic()

        self.canvas = tk.Canvas(
            self,
            width=display_size,
            height=display_size,
            bg=UI["surface"],
            highlightthickness=0,
        )
        self.canvas.pack(expand=True)

        self._source_image = tk.PhotoImage(file=str(image_path))
        factor = max(1, round(self._source_image.width() / display_size))
        self._image = self._source_image.subsample(factor, factor)
        self._image_x = (display_size - self._image.width()) / 2
        self._image_y = (display_size - self._image.height()) / 2
        self.canvas.create_image(
            display_size / 2,
            display_size / 2,
            image=self._image,
        )

        self.canvas.create_text(
            display_size / 2,
            10,
            text="艇艏  ↑",
            fill=UI["navy"],
            font=("Microsoft YaHei UI", 8, "bold"),
        )

        self._rotor_items = {}
        for key in self.ROTOR_CENTERS:
            items = []
            for _ in range(3):
                items.append(
                    self.canvas.create_polygon(
                        0, 0, 0, 0,
                        fill="#188DFF",
                        outline="#8AD8FF",
                        width=1,
                        state=tk.HIDDEN,
                    )
                )
            self._rotor_items[key] = items

        self._label_items = {}
        for key, (nx, ny) in self.LABEL_POSITIONS.items():
            x, y = nx * display_size, ny * display_size
            rect = self.canvas.create_rectangle(
                x - 34, y - 15, x + 34, y + 15,
                fill=UI["navy"], outline="", width=0,
            )
            text_item = self.canvas.create_text(
                x, y,
                text=f"{self.THRUSTER_LABELS[key]}\n--",
                fill="white",
                font=("Microsoft YaHei UI", 7, "bold"),
                justify=tk.CENTER,
            )
            self._label_items[key] = (rect, text_item)

        self.after(40, self._animate)

    @staticmethod
    def _blade_points(cx, cy, angle_deg, inner_radius, outer_radius):
        """生成一片略带后掠角的旋叶多边形。"""
        points = []
        for radius, offset in (
            (inner_radius, -14),
            (outer_radius, -30),
            (outer_radius, 10),
            (inner_radius, 20),
        ):
            angle = math.radians(angle_deg + offset)
            points.extend((cx + radius * math.cos(angle),
                           cy + radius * math.sin(angle)))
        return points

    def set_thrust(self, thrust, connected=True):
        """接收遥测推力快照；仅缓存并刷新标签。"""
        self._connected = bool(connected)
        for key in self.THRUSTER_LABELS:
            try:
                value = float(thrust.get(key, 0.0)) if connected else 0.0
                if not math.isfinite(value):
                    value = 0.0
            except (TypeError, ValueError):
                value = 0.0
            self._thrust[key] = value

            rect, text_item = self._label_items[key]
            if not connected:
                text = f"{self.THRUSTER_LABELS[key]}\n--"
                fill = UI["muted"]
            else:
                text = f"{self.THRUSTER_LABELS[key]}\n{value:+.1f} N"
                if value > 0.5:
                    fill = UI["blue"]
                elif value < -0.5:
                    fill = UI["orange"]
                else:
                    fill = UI["navy"]
            self.canvas.itemconfigure(rect, fill=fill)
            self.canvas.itemconfigure(text_item, text=text)

    def _animate(self):
        now = time.monotonic()
        dt = min(now - self._last_frame_time, 0.1)
        self._last_frame_time = now

        image_width = self._image.width()
        image_height = self._image.height()
        inner_radius = image_width * 0.025
        outer_radius = image_width * 0.073

        for key, (nx, ny) in self.ROTOR_CENTERS.items():
            value = self._thrust[key] if self._connected else 0.0
            magnitude = abs(value)
            active = magnitude > 0.05
            if active:
                # 低速也保持可见运动；约 65N 时达到约 2.2 圈/秒。
                degrees_per_second = min(800.0, 70.0 + magnitude * 11.0)
                direction = 1.0 if value >= 0.0 else -1.0
                self._angles[key] = (
                    self._angles[key] + direction * degrees_per_second * dt
                ) % 360.0

            cx = self._image_x + nx * image_width
            cy = self._image_y + ny * image_height
            color = "#0079F2" if magnitude >= 30.0 else "#279BFF"
            for index, item in enumerate(self._rotor_items[key]):
                if not active:
                    self.canvas.itemconfigure(item, state=tk.HIDDEN)
                    continue
                angle = self._angles[key] + index * 120.0
                self.canvas.coords(
                    item,
                    *self._blade_points(
                        cx, cy, angle, inner_radius, outer_radius
                    ),
                )
                self.canvas.itemconfigure(item, fill=color, state=tk.NORMAL)

        self.after(40, self._animate)


# ================================================================
#  IMU驱动的ROV三维姿态（纯Tkinter轻量渲染）
# ================================================================
class ROVPoseView(tk.Frame):
    """Render the real ROV mesh with live roll/pitch/yaw telemetry."""

    BACKGROUND = "#0A2233"

    def __init__(self, parent, mesh_path):
        super().__init__(parent, bg=self.BACKGROUND)
        self.canvas = tk.Canvas(
            self,
            bg=self.BACKGROUND,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._vertices = []
        self._faces = []
        self._mesh_error = ""
        self._connected = False
        self._target_angles = [0.0, 0.0, 0.0]
        self._display_angles = [0.0, 0.0, 0.0]
        self._last_size = (0, 0)
        self._dirty = True
        self._load_mesh(Path(mesh_path))
        self.canvas.bind("<Configure>", self._on_resize)
        self.after(POSE_RENDER_INTERVAL_MS, self._animate)

    def _load_mesh(self, path):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            vertices = payload.get("vertices", [])
            faces = payload.get("faces", [])
            if not vertices or not faces:
                raise ValueError("mesh has no vertices or faces")
            self._vertices = [tuple(map(float, point)) for point in vertices]
            self._faces = [tuple(map(int, face)) for face in faces]
        except Exception as exc:
            self._mesh_error = str(exc)
            self._vertices = []
            self._faces = []

    def _on_resize(self, _event=None):
        self._dirty = True

    def set_pose(self, roll, pitch, yaw, *, connected=True):
        values = []
        for index, value in enumerate((roll, pitch, yaw)):
            try:
                value = float(value)
                if not math.isfinite(value):
                    value = 0.0
            except (TypeError, ValueError):
                value = 0.0
            # Roll and yaw are signed half-turn angles.  Pitch currently comes
            # from a quaternion decomposition (normally +/-90), but bounding
            # all three here guarantees the view can never display a stale or
            # out-of-range value from any telemetry path.
            values.append((value + 180.0) % 360.0 - 180.0)
        # The telemetry labels, mesh, and pose overlay must show one exact
        # authoritative triplet.  A second smoothed angle state previously
        # lagged the labels and crossed 180 degrees during wrap-around.
        self._target_angles = list(values)
        self._display_angles = list(values)
        self._connected = bool(connected)
        self._dirty = True

    def set_connected(self, connected):
        self._connected = bool(connected)
        self._dirty = True

    @staticmethod
    def _shortest_angle_delta(target, current):
        return (float(target) - float(current) + 180.0) % 360.0 - 180.0

    def _animate(self):
        size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        if self._dirty or size != self._last_size:
            self._render()
            self._last_size = size
            self._dirty = False
        self.after(POSE_RENDER_INTERVAL_MS, self._animate)

    @staticmethod
    def _rotate(point, roll_deg, pitch_deg, yaw_deg):
        x, y, z = point
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)

        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # Body -> world: roll(X), pitch(Y), then yaw(Z).
        y, z = y * cr - z * sr, y * sr + z * cr
        x, z = x * cp + z * sp, -x * sp + z * cp
        x, y = x * cy - y * sy, x * sy + y * cy
        return x, y, z

    @staticmethod
    def _dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    @classmethod
    def _unit(cls, value):
        length = math.sqrt(cls._dot(value, value)) or 1.0
        return tuple(component / length for component in value)

    @staticmethod
    def _shade_blue(intensity, connected):
        if connected:
            base = (24, 126, 197)
        else:
            base = (76, 101, 116)
        factor = max(0.35, min(1.05, float(intensity)))
        rgb = tuple(max(0, min(255, round(channel * factor))) for channel in base)
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def _render(self):
        canvas = self.canvas
        width = max(320, canvas.winfo_width())
        height = max(260, canvas.winfo_height())
        canvas.delete("all")

        if not self._vertices:
            canvas.create_text(
                width / 2,
                height / 2,
                text=f"ROV模型载入失败\n{self._mesh_error}",
                fill="#F5B942",
                font=("Microsoft YaHei UI", 12, "bold"),
                justify=tk.CENTER,
            )
            return

        camera = self._unit((3.2, -4.0, -2.6))
        world_up = (0.0, 0.0, -1.0)
        screen_right = self._unit(self._cross(world_up, camera))
        screen_up = self._unit(self._cross(camera, screen_right))
        scale = min(width, height) * 0.40
        center_x = width * 0.51
        center_y = height * 0.54

        def project(point):
            return (
                center_x + self._dot(point, screen_right) * scale,
                center_y - self._dot(point, screen_up) * scale,
                self._dot(point, camera),
            )

        # Fixed world reference grid.  The model rotates; the grid does not.
        grid_color = "#15394C"
        for step in range(-4, 5):
            offset = step * 0.25
            for start, end in (
                ((-1.15, offset, 0.48), (1.15, offset, 0.48)),
                ((offset, -1.15, 0.48), (offset, 1.15, 0.48)),
            ):
                a = project(start)
                b = project(end)
                canvas.create_line(a[0], a[1], b[0], b[1], fill=grid_color)

        roll, pitch, yaw = self._display_angles
        rotated = [self._rotate(point, roll, pitch, yaw) for point in self._vertices]
        projected = [project(point) for point in rotated]

        visible_faces = []
        for face in self._faces:
            try:
                pa, pb, pc = (rotated[index] for index in face)
            except IndexError:
                continue
            ab = tuple(pb[axis] - pa[axis] for axis in range(3))
            ac = tuple(pc[axis] - pa[axis] for axis in range(3))
            normal = self._unit(self._cross(ab, ac))
            facing = self._dot(normal, camera)
            if facing <= 0.015:
                continue
            depth = sum(projected[index][2] for index in face) / 3.0
            visible_faces.append((depth, facing, face))

        # Painter's algorithm: far surfaces first, near surfaces last.
        visible_faces.sort(key=lambda item: item[0])
        outline = "#0A2A3D" if self._connected else "#263D4A"
        for _depth, facing, face in visible_faces:
            coords = []
            for index in face:
                coords.extend(projected[index][:2])
            fill = self._shade_blue(0.48 + 0.56 * facing, self._connected)
            canvas.create_polygon(coords, fill=fill, outline=outline, width=1)

        # Vehicle-forward indicator rotates with the model.
        bow_start = project(self._rotate((0.0, 0.0, 0.0), roll, pitch, yaw))
        bow_end = project(self._rotate((1.15, 0.0, 0.0), roll, pitch, yaw))
        canvas.create_line(
            bow_start[0], bow_start[1], bow_end[0], bow_end[1],
            fill="#4FD7FF", width=3, arrow=tk.LAST,
        )
        canvas.create_text(
            bow_end[0] + 18, bow_end[1], text="艏 +X",
            fill="#82E8FF", font=("Microsoft YaHei UI", 9, "bold"),
        )

        status = "IMU实时姿态" if self._connected else "等待IMU遥测"
        status_color = "#36D89A" if self._connected else "#F5B942"
        canvas.create_text(
            18, 16, anchor="nw", text=status,
            fill=status_color, font=("Microsoft YaHei UI", 11, "bold"),
        )
        canvas.create_text(
            18, 43, anchor="nw",
            text=f"ROLL  {roll:+07.2f}°\nPITCH {pitch:+07.2f}°\nYAW   {yaw:+07.2f}°",
            fill="#D6E8F2", font=("Consolas", 11, "bold"), justify=tk.LEFT,
        )
        canvas.create_text(
            width - 16, height - 13, anchor="se",
            text="Robocup-ROV.STEP · +X前  +Y右  +Z下",
            fill="#7192A4", font=("Microsoft YaHei UI", 8),
        )


# ================================================================
#  比赛计时器
# ================================================================
class CompetitionTimer(tk.Frame):
    """比赛计时器：支持正计时和倒计时。"""

    def __init__(self, parent):
        super().__init__(parent, bg=UI["surface"], bd=0, highlightthickness=0)
        self._running = False
        self._elapsed = 0.0
        self._countdown_target = 0.0
        self._mode = "up"  # "up" | "down"
        self._last_tick = 0.0

        # 标题
        header = ttk.Frame(self, style="App.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(
            header, text="⏱ 比赛计时", foreground=UI["navy"],
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side=tk.LEFT)
        self._mode_label = ttk.Label(
            header, text="正计时", foreground=UI["green"],
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        self._mode_label.pack(side=tk.RIGHT)

        # 大字时间
        self._time_var = tk.StringVar(value="00:00")
        tk.Label(
            self, textvariable=self._time_var,
            bg=UI["surface"], fg=UI["navy"],
            font=("Consolas", 28, "bold"),
        ).pack(pady=(2, 0))

        # 按钮行
        btn_row = ttk.Frame(self, style="App.TFrame")
        btn_row.pack(fill=tk.X, pady=(2, 0))
        for column in range(3):
            btn_row.grid_columnconfigure(column, weight=1)
        self._start_btn = tk.Button(
            btn_row, text="▶ 开始", command=self._toggle_run,
            font=("Microsoft YaHei UI", 9, "bold"),
            fg="white", bg=UI["green"], relief=tk.FLAT,
            activeforeground="white", activebackground="#117B54",
            width=6, cursor="hand2",
        )
        self._start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=(0, 2))
        tk.Button(
            btn_row, text="↺ 复位", command=self._reset,
            font=("Microsoft YaHei UI", 9, "bold"),
            fg=UI["text"], bg="#E8EEF5", relief=tk.FLAT,
            width=6, cursor="hand2",
        ).grid(row=0, column=1, sticky="ew", padx=2, pady=(0, 2))
        tk.Button(
            btn_row, text="⏬ 5min", command=lambda: self._set_countdown(300),
            font=("Microsoft YaHei UI", 8),
            fg=UI["text"], bg="#E8EEF5", relief=tk.FLAT,
            width=5, cursor="hand2",
        ).grid(row=1, column=0, sticky="ew", padx=(0, 2))
        tk.Button(
            btn_row, text="10min", command=lambda: self._set_countdown(600),
            font=("Microsoft YaHei UI", 8),
            fg=UI["text"], bg="#E8EEF5", relief=tk.FLAT,
            width=5, cursor="hand2",
        ).grid(row=1, column=1, sticky="ew", padx=2)
        tk.Button(
            btn_row, text="15min", command=lambda: self._set_countdown(900),
            font=("Microsoft YaHei UI", 8),
            fg=UI["text"], bg="#E8EEF5", relief=tk.FLAT,
            width=5, cursor="hand2",
        ).grid(row=1, column=2, sticky="ew", padx=(2, 0))

    def _set_countdown(self, seconds):
        self._running = False
        self._mode = "down"
        self._countdown_target = float(seconds)
        self._elapsed = 0.0
        self._update_display()
        self._start_btn.configure(text="▶ 开始", bg=UI["green"])
        self._mode_label.configure(text=f"倒计时 {seconds//60}min", foreground=UI["amber"])

    def _toggle_run(self):
        if (
            not self._running
            and self._mode == "down"
            and self._countdown_target > 0
            and self._elapsed >= self._countdown_target
        ):
            self._elapsed = 0.0
        self._running = not self._running
        if self._running:
            self._last_tick = time.monotonic()
            self._start_btn.configure(text="⏸ 暂停", bg=UI["amber"])
        else:
            self._start_btn.configure(text="▶ 开始", bg=UI["green"])

    def _reset(self):
        self._running = False
        self._mode = "up"
        self._elapsed = 0.0
        self._countdown_target = 0.0
        self._update_display()
        self._start_btn.configure(text="▶ 开始", bg=UI["green"])
        self._mode_label.configure(text="正计时", foreground=UI["green"])

    def _update_display(self):
        if self._mode == "down":
            remaining = max(0.0, self._countdown_target - self._elapsed)
            total_sec = int(math.ceil(remaining))
        else:
            total_sec = int(self._elapsed)
        minutes = total_sec // 60
        seconds = total_sec % 60
        self._time_var.set(f"{minutes:02d}:{seconds:02d}")

        # 倒计时最后30秒变红
        if self._mode == "down" and remaining <= 30 and self._running:
            # flash effect via color change handled in refresh
            pass

    def expanded_status(self):
        """Return integer match time and a concise state for expanded video."""
        if self._mode == "down":
            mode_text = f"倒计时 {int(self._countdown_target // 60)} min"
            finished = self._countdown_target > 0 and self._elapsed >= self._countdown_target
        else:
            mode_text = "正计时"
            finished = False
        if finished:
            state_text = "已结束"
        elif self._running:
            state_text = "进行中"
        elif self._elapsed > 0:
            state_text = "已暂停"
        else:
            state_text = "待开始"
        return self._time_var.get(), f"{mode_text} · {state_text}"

    def tick(self):
        """主循环调用，更新计时器状态。"""
        if not self._running:
            return
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now
        if self._mode == "up":
            self._elapsed += dt
        else:
            self._elapsed += dt
            if self._elapsed >= self._countdown_target:
                self._elapsed = self._countdown_target
                self._running = False
                self._start_btn.configure(text="⏰ 时间到!", bg=UI["red"])
        self._update_display()


# ================================================================
#  深度/航向趋势图（轻量 Canvas 折线）
# ================================================================
class TrendChart(tk.Frame):
    """Canvas 小型趋势图，追踪最近 N 秒的数据。"""

    def __init__(
        self, parent, title, unit, color, window_seconds=30, *, minimum, maximum,
    ):
        super().__init__(parent, bg=UI["surface"], bd=0, highlightthickness=0)
        minimum = float(minimum)
        maximum = float(maximum)
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
            raise ValueError("trend chart minimum must be below maximum")
        self._title = title
        self._unit = unit
        self._color = color
        self._value_min = minimum
        self._value_max = maximum
        self._fill_color = self._blend_hex(
            color, UI["surface_alt"], alpha=0.12
        )
        self._window = window_seconds
        self._data = []  # [(timestamp, value), ...]

        ttk.Label(
            self, text=title, foreground=UI["navy_soft"],
            font=("Microsoft YaHei UI", 8, "bold"),
        ).pack(anchor="w")

        self.canvas = tk.Canvas(
            self, width=1, bg=UI["surface_alt"], height=48,
            highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._value_var = tk.StringVar(value="--")
        ttk.Label(
            self, textvariable=self._value_var,
            foreground=color, font=("Consolas", 10, "bold"),
        ).pack(anchor="e")

    @staticmethod
    def _blend_hex(foreground, background, alpha):
        """Pre-blend a Tk-compatible six-digit fill color (Canvas has no alpha)."""
        try:
            fg = tuple(int(foreground[index:index + 2], 16) for index in (1, 3, 5))
            bg = tuple(int(background[index:index + 2], 16) for index in (1, 3, 5))
            mixed = tuple(
                round(alpha * fg_value + (1.0 - alpha) * bg_value)
                for fg_value, bg_value in zip(fg, bg)
            )
            return "#" + "".join(f"{value:02X}" for value in mixed)
        except (TypeError, ValueError):
            return background

    def push(self, value):
        """添加一个数据点。"""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        self._data.append((time.monotonic(), value))
        self._value_var.set(f"{value:.1f} {self._unit}")
        self._prune()

    def _prune(self):
        cutoff = time.monotonic() - self._window
        self._data = [(t, v) for t, v in self._data if t >= cutoff]

    def redraw(self):
        """重绘折线图。"""
        self._prune()
        canvas = self.canvas
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 20 or h < 20:
            return
        canvas.delete("all")
        if len(self._data) < 2:
            canvas.create_text(
                w // 2, h // 2, text="等待数据…", fill=UI["muted"],
                font=("Microsoft YaHei UI", 8),
            )
            return

        elapsed = [t - self._data[0][0] for t, _ in self._data]
        total_t = max(elapsed[-1], 0.001)
        values = [v for _, v in self._data]
        v_min = self._value_min
        v_max = self._value_max

        def _map(t, v):
            x = 8 + (t / total_t) * (w - 16)
            bounded = min(v_max, max(v_min, v))
            y = h - 8 - ((bounded - v_min) / (v_max - v_min)) * (h - 16)
            return x, y

        points = [_map(elapsed[i], values[i]) for i in range(len(self._data))]
        # 填充区域
        if len(points) >= 2:
            fill_points = list(points) + [(points[-1][0], h - 8), (points[0][0], h - 8)]
            flat = [coord for pt in fill_points for coord in pt]
            canvas.create_polygon(
                flat, fill=self._fill_color, outline="", width=0,
            )
        # One Canvas polyline keeps the 20 Hz telemetry display responsive.
        # Creating one item per segment grows to tens of thousands of Tk calls
        # per second once five 30-second charts have filled their history.
        line_points = [coordinate for point in points for coordinate in point]
        canvas.create_line(*line_points, fill=self._color, width=2)

        # 标签
        canvas.create_text(
            8, 6, anchor="nw",
            text=f"{v_max:.1f}", fill=UI["muted"],
            font=("Consolas", 7),
        )
        canvas.create_text(
            8, h - 8, anchor="sw",
            text=f"{v_min:.1f}", fill=UI["muted"],
            font=("Consolas", 7),
        )


# ================================================================
#  摄像头视频流（OpenCV + Tkinter）
# ================================================================
class VideoStream(tk.Frame):
    """摄像头视频流面板，支持 USB 摄像头 / RTSP / HTTP MJPEG 流。"""

    def __init__(
        self, parent, source="0", depth_provider=None, status_provider=None,
        timer_provider=None, mode_provider=None, gripper_next_provider=None,
    ):
        super().__init__(parent, bg="#000000")
        self.source = None
        self.cap = None
        self.running = False
        self._frame = None
        self._last_frame = None
        self._frame_resolution = None
        self._fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.monotonic()
        self._lock = threading.Lock()
        self._http_mjpeg = False
        self._capture_generation = 0
        self._depth_provider = depth_provider
        self._status_provider = status_provider
        self._timer_provider = timer_provider
        self._mode_provider = mode_provider
        self._gripper_next_provider = gripper_next_provider
        self._expanded_window = None
        self._expanded_canvas = None
        self._annotation_mode = False
        self._annotation_pending_point = None
        self._annotation_storage_path = VIDEO_ANNOTATIONS_PATH
        self._annotation_lines = load_video_annotations(
            self._annotation_storage_path
        )
        self._annotation_pressed_keys = set()
        self._mission_map_source = None
        self._mission_map_render_cache = {}
        mission_map_path = (
            Path(__file__).resolve().parent / "assets" / "mission_map_overlay.png"
        )
        try:
            self._mission_map_source = Image.open(mission_map_path).convert("RGBA")
            self._mission_map_source.load()
        except (OSError, ValueError):
            self._mission_map_source = None
        self._profile_request_generation = 0
        self._profile_apply_lock = threading.Lock()
        # Video is intentionally decoupled from the 5 Hz telemetry/dashboard
        # refresh.  Rendering at about 30 Hz keeps a 30 FPS MJPEG stream fluid.
        self._display_interval_ms = 33
        self._placeholder_drawn = False

        self.canvas = tk.Canvas(self, bg="#000000", highlightthickness=0, bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 控件覆盖层
        self._overlay_frame = ttk.Frame(self.canvas, style="App.TFrame")
        self._overlay_id = None

        controls = ttk.Frame(self, style="App.TFrame", padding=(4, 2))
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="视频源", style="MetricLabel.TLabel").pack(side=tk.LEFT)
        self._source_var = tk.StringVar(value=str(source))
        ttk.Entry(controls, textvariable=self._source_var, width=28).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="连接", command=self._on_connect,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="断开", command=self._on_disconnect).pack(side=tk.LEFT)
        ttk.Separator(controls, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=1
        )
        ttk.Label(
            controls, text="画质预设", style="MetricLabel.TLabel"
        ).pack(side=tk.LEFT)
        self._profile_var = tk.StringVar(value="水下（已标定）")
        self._profile_combo = ttk.Combobox(
            controls,
            textvariable=self._profile_var,
            values=tuple(CAMERA_PROFILE_LABELS),
            state="readonly",
            width=14,
        )
        self._profile_combo.pack(side=tk.LEFT, padx=(4, 2))
        self._profile_combo.bind("<<ComboboxSelected>>", self._on_camera_profile_selected)
        self._profile_status_label = ttk.Label(
            controls, text="水下已应用", foreground=UI["green"],
            font=("Microsoft YaHei UI", 8, "bold"),
        )
        self._profile_status_label.pack(side=tk.LEFT, padx=(2, 0))
        self._fps_label = ttk.Label(controls, text="-- FPS", style="Muted.TLabel")
        self._fps_label.pack(side=tk.RIGHT, padx=4)
        self._status_label = ttk.Label(controls, text="未连接",
                                        foreground=UI["amber"],
                                        font=("Microsoft YaHei UI", 9, "bold"))
        self._status_label.pack(side=tk.RIGHT, padx=8)

        self.canvas.bind("<Configure>", lambda e: self._dirty)
        self.canvas.bind("<Button-1>", self._open_expanded_view, add="+")
        self.canvas.configure(cursor="hand2")
        # Connect to the configured Pi stream as soon as this panel has joined
        # the Tk event loop.  The button remains available for reconnecting or
        # changing source, but normal ROV startup should show video directly.
        if source:
            self.after_idle(lambda: self.set_source(source))
        self.after(self._display_interval_ms, self._display_loop)

    @property
    def _dirty(self):
        pass

    def _display_loop(self):
        """Render video independently of the slower telemetry dashboard loop."""
        self.update_display()
        if self.winfo_exists():
            self.after(self._display_interval_ms, self._display_loop)

    def _on_connect(self):
        src = self._source_var.get().strip()
        if not src:
            return
        self.set_source(src)

    def _on_disconnect(self):
        self.stop()
        self._status_label.configure(text="已断开", foreground=UI["muted"])
        self._fps_label.configure(text="-- FPS")
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width() // 2 if self.canvas.winfo_width() > 10 else 160,
            self.canvas.winfo_height() // 2 if self.canvas.winfo_height() > 10 else 120,
            text="视频已断开", fill="#666666",
            font=("Microsoft YaHei UI", 14, "bold"),
        )

    def _on_camera_profile_selected(self, _event=None):
        label = self._profile_var.get()
        profile = CAMERA_PROFILE_LABELS.get(label)
        if profile is None:
            return
        source = self.source or self._source_var.get().strip()
        self._profile_request_generation += 1
        generation = self._profile_request_generation
        self._profile_status_label.configure(
            text="应用中…", foreground=UI["amber"]
        )
        threading.Thread(
            target=self._apply_camera_profile_request,
            args=(source, profile, label, generation),
            daemon=True,
        ).start()

    def _apply_camera_profile_request(self, source, profile, label, generation):
        ok = False
        detail = ""
        try:
            request = build_camera_profile_request(source, profile)
            # Serialize rapid operator changes so the last selection is also
            # the last profile applied on the camera.
            with self._profile_apply_lock:
                with open_direct_http_stream(request, timeout=3) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            ok = bool(payload.get("ok")) and payload.get("profile") == profile
            if not ok:
                detail = str(payload.get("error") or "返回状态无效")
        except Exception as exc:
            detail = str(exc)
        try:
            self.after(
                0,
                lambda: self._finish_camera_profile_request(
                    label, generation, ok, detail
                ),
            )
        except tk.TclError:
            pass

    def _finish_camera_profile_request(self, label, generation, ok, detail):
        if generation != self._profile_request_generation:
            return
        if ok:
            short_label = label.split("（", 1)[0]
            self._profile_status_label.configure(
                text=f"{short_label}已应用", foreground=UI["green"]
            )
        else:
            message = detail[:22] if detail else "应用失败"
            self._profile_status_label.configure(
                text=f"失败：{message}", foreground=UI["red"]
            )

    def _open_expanded_view(self, _event=None):
        """Open a normal maximized window so the Windows taskbar remains usable."""
        if self._expanded_window is not None and self._expanded_window.winfo_exists():
            self._expanded_window.deiconify()
            self._expanded_window.lift()
            self._expanded_window.focus_force()
            return

        window = tk.Toplevel(self.winfo_toplevel())
        window.title("ROV 实时视频")
        window.configure(bg="#000000")
        window.minsize(640, 480)
        window.protocol("WM_DELETE_WINDOW", self._close_expanded_view)
        canvas = tk.Canvas(window, bg="#000000", highlightthickness=0, bd=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.bind("<Button-1>", self._on_expanded_annotation_click)
        canvas.bind("<Double-Button-1>", self._on_expanded_double_click)
        window.bind("<KeyPress>", self._on_expanded_annotation_key_press, add="+")
        window.bind("<KeyRelease>", self._on_expanded_annotation_key_release, add="+")
        self._expanded_window = window
        self._expanded_canvas = canvas
        # ``zoomed`` gives a full bordered Windows window while preserving the
        # taskbar, so the operator can still switch back to the control GUI.
        window.after_idle(lambda: window.state("zoomed"))
        self._render_expanded_frame()

    def toggle_expanded_view(self):
        """Toggle the expanded camera window for a controller press edge."""
        window = self._expanded_window
        if window is not None:
            try:
                if window.winfo_exists():
                    self._close_expanded_view()
                    return
            except tk.TclError:
                self._expanded_window = None
                self._expanded_canvas = None
        self._open_expanded_view()

    def _on_expanded_annotation_key_press(self, event):
        key = str(getattr(event, "keysym", "")).lower()
        if key not in {"r", "t"}:
            return None
        pressed = getattr(self, "_annotation_pressed_keys", None)
        if pressed is None:
            pressed = self._annotation_pressed_keys = set()
        if key in pressed:
            return "break"
        pressed.add(key)
        if key == "r":
            self._annotation_mode = not getattr(self, "_annotation_mode", False)
            self._annotation_pending_point = None
            canvas = getattr(self, "_expanded_canvas", None)
            if canvas is not None:
                canvas.configure(cursor="crosshair" if self._annotation_mode else "")
        else:
            self._annotation_lines = []
            self._annotation_pending_point = None
            self._persist_annotations()
        self._render_expanded_frame()
        return "break"

    def _on_expanded_annotation_key_release(self, event):
        key = str(getattr(event, "keysym", "")).lower()
        if key not in {"r", "t"}:
            return None
        getattr(self, "_annotation_pressed_keys", set()).discard(key)
        return "break"

    def _on_expanded_annotation_click(self, event):
        if not getattr(self, "_annotation_mode", False):
            return None
        frame = getattr(self, "_last_frame", None)
        canvas = getattr(self, "_expanded_canvas", None)
        if frame is None or canvas is None:
            return "break"
        frame_height, frame_width = frame.shape[:2]
        point = expanded_canvas_to_video_point(
            event.x,
            event.y,
            canvas.winfo_width(),
            canvas.winfo_height(),
            frame_width,
            frame_height,
        )
        pending = getattr(self, "_annotation_pending_point", None)
        if pending is None:
            self._annotation_pending_point = point
        else:
            self._annotation_lines.append((pending, point))
            self._annotation_pending_point = None
            self._persist_annotations()
        self._render_expanded_frame()
        return "break"

    def _persist_annotations(self):
        path = getattr(self, "_annotation_storage_path", VIDEO_ANNOTATIONS_PATH)
        try:
            save_video_annotations(path, self._annotation_lines)
            self._annotation_persist_error = ""
        except (OSError, TypeError, ValueError) as exc:
            self._annotation_persist_error = str(exc)

    def _on_expanded_double_click(self, event=None):
        if getattr(self, "_annotation_mode", False):
            # Tk reports the second press as a double-click event instead of a
            # normal Button-1 event, so count it as the requested second point.
            return self._on_expanded_annotation_click(event)
        self._close_expanded_view()
        return "break"

    def _close_expanded_view(self):
        window = self._expanded_window
        self._expanded_window = None
        self._expanded_canvas = None
        self._annotation_mode = False
        self._annotation_pending_point = None
        self._annotation_pressed_keys.clear()
        if window is not None and window.winfo_exists():
            window.destroy()

    def _expanded_depth_text(self):
        if self._depth_provider is None:
            return "深度 --"
        try:
            return str(self._depth_provider())
        except Exception:
            return "深度 --"

    def _expanded_status_texts(self):
        if self._status_provider is not None:
            try:
                values = tuple(str(value) for value in self._status_provider())
                if len(values) == 11:
                    return values
                if len(values) == 6:
                    depth_text, depth_hold_text, pitch_text, roll_text, yaw_text, motion_text = values
                    return (
                        depth_text, "目标深度 --", depth_hold_text,
                        "实时 Yaw --°", "目标 Yaw --°",
                        "俯仰 --°", "横滚 --°",
                        pitch_text, roll_text, yaw_text, motion_text,
                    )
                if len(values) == 5:
                    depth_text, depth_hold_text, pitch_text, roll_text, motion_text = values
                    return (
                        depth_text, "目标深度 --", depth_hold_text,
                        "实时 Yaw --°", "目标 Yaw --°",
                        "俯仰 --°", "横滚 --°",
                        pitch_text, roll_text, "锁偏转：未知", motion_text,
                    )
                if len(values) == 3:
                    depth_text, depth_hold_text, motion_text = values
                    return (
                        depth_text, "目标深度 --", depth_hold_text,
                        "实时 Yaw --°", "目标 Yaw --°",
                        "俯仰 --°", "横滚 --°",
                        "锁俯仰：未知", "锁横滚：未知", "锁偏转：未知",
                        motion_text,
                    )
            except Exception:
                pass
        return (
            self._expanded_depth_text(), "目标深度 --", "定深：未知",
            "实时 Yaw --°", "目标 Yaw --°",
            "俯仰 --°", "横滚 --°",
            "锁俯仰：未知", "锁横滚：未知", "锁偏转：未知",
            "运动指令：未知",
        )

    def _expanded_timer_texts(self):
        timer_provider = getattr(self, "_timer_provider", None)
        if timer_provider is not None:
            try:
                values = tuple(str(value) for value in timer_provider())
                if len(values) == 2:
                    return values
            except Exception:
                pass
        return "00:00", "正计时 · 待开始"

    def _expanded_mode_text(self):
        provider = getattr(self, "_mode_provider", None)
        if provider is None:
            return "模式：--"
        try:
            return str(provider())
        except Exception:
            return "模式：--"

    def _expanded_gripper_next_text(self):
        provider = getattr(self, "_gripper_next_provider", None)
        if provider is None:
            return "下次--"
        try:
            return str(provider())
        except Exception:
            return "下次--"

    def _render_frame(self, canvas, frame, *, draw_depth=False):
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width < 10 or height < 10 or frame is None:
            return
        frame_height, frame_width = frame.shape[:2]
        video_left, video_top, new_width, new_height = expanded_video_layout(
            width, height, frame_width, frame_height
        )
        if new_width < 4 or new_height < 4:
            return
        crop_left, crop_top, crop_width, crop_height = video_cover_crop(
            new_width, new_height, frame_width, frame_height
        )
        visible_frame = frame[
            crop_top:crop_top + crop_height,
            crop_left:crop_left + crop_width,
        ]
        scale = max(new_width / frame_width, new_height / frame_height)
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(
            visible_frame, (new_width, new_height), interpolation=interpolation
        )
        image = ImageTk.PhotoImage(Image.fromarray(resized))
        canvas.delete("all")
        canvas.create_image(
            video_left,
            video_top,
            anchor=tk.NW,
            image=image,
        )
        canvas._video_image_ref = image

        if draw_depth:
            self._draw_expanded_mission_map(canvas, width, height)
            self._draw_expanded_annotations(
                canvas, width, height, frame_width, frame_height
            )
            (
                depth_text, target_depth_text, depth_hold_text,
                yaw_value_text, yaw_target_text,
                pitch_value_text, roll_value_text,
                pitch_hold_text, roll_hold_text, yaw_hold_text, motion_text,
            ) = self._expanded_status_texts()
            timer_text, timer_state_text = self._expanded_timer_texts()
            motion_label, motion_action, motion_color = split_motion_command(motion_text)
            resolution_text = f"视频 {frame_width}×{frame_height}"
            status_side = expanded_status_side_width(width)
            left_bar_width = status_side
            right_bar_width = status_side
            left_x = 18
            right_x = width - status_side + 18
            left_text_width = max(40, left_bar_width - 30)
            right_text_width = max(40, right_bar_width - 30)
            status_y = 28

            # Keep the established HUD anchors and draw text directly over
            # the full-width video without any side-panel background.
            # Left HUD: match clock. Every row shares one left edge.
            canvas.create_text(
                left_x, status_y, anchor=tk.NW, text="比赛计时",
                fill="#8CC9D8", font=("Microsoft YaHei UI", 13, "bold"),
                width=left_text_width, justify=tk.LEFT,
            )
            canvas.create_text(
                left_x, status_y + 34, anchor=tk.NW, text=timer_text,
                fill="#FFFFFF", font=("Consolas", 27, "bold"),
                width=left_text_width, justify=tk.LEFT,
            )
            canvas.create_text(
                left_x, status_y + 84, anchor=tk.NW, text=timer_state_text,
                fill="#A8BAC5", font=("Microsoft YaHei UI", 11, "bold"),
                width=left_text_width, justify=tk.LEFT,
            )
            canvas.create_text(
                left_x, status_y + 142, anchor=tk.NW, text="实时姿态",
                fill="#8CC9D8", font=("Microsoft YaHei UI", 13, "bold"),
                width=left_text_width, justify=tk.LEFT,
            )
            canvas.create_text(
                left_x, status_y + 178, anchor=tk.NW, text=pitch_value_text,
                fill="#FFFFFF", font=("Microsoft YaHei UI", 15, "bold"),
                width=0, justify=tk.LEFT,
            )
            canvas.create_text(
                left_x, status_y + 214, anchor=tk.NW, text=roll_value_text,
                fill="#FFFFFF", font=("Microsoft YaHei UI", 15, "bold"),
                width=0, justify=tk.LEFT,
            )
            canvas.create_text(
                left_x,
                status_y + 254,
                anchor=tk.NW,
                text=self._expanded_gripper_next_text(),
                fill="#FFD166",
                font=("Microsoft YaHei UI", 14, "bold"),
                width=left_text_width,
                justify=tk.LEFT,
            )

            # Right overlay: vehicle status. Every row shares one left edge.
            canvas.create_text(
                right_x, status_y, anchor=tk.NW, text=depth_text,
                fill="#EAF8FF", font=("Microsoft YaHei UI", 17, "bold"),
                width=0, justify=tk.LEFT,
            )
            canvas.create_text(
                right_x, status_y + 42, anchor=tk.NW, text=target_depth_text,
                fill="#8CC9D8", font=("Microsoft YaHei UI", 13, "bold"),
                width=0, justify=tk.LEFT,
            )
            canvas.create_text(
                right_x, status_y + 190, anchor=tk.NW, text=depth_hold_text,
                fill="#4FE09B", font=("Microsoft YaHei UI", 14, "bold"),
                width=right_text_width, justify=tk.LEFT,
            )
            canvas.create_text(
                right_x, status_y + 112, anchor=tk.NW, text=yaw_value_text,
                fill="#FFFFFF", font=("Microsoft YaHei UI", 14, "bold"),
                width=0, justify=tk.LEFT,
            )
            canvas.create_text(
                right_x, status_y + 146, anchor=tk.NW, text=yaw_target_text,
                fill="#8CC9D8", font=("Microsoft YaHei UI", 13, "bold"),
                width=0, justify=tk.LEFT,
            )
            canvas.create_text(
                right_x, status_y + 222, anchor=tk.NW, text=pitch_hold_text,
                fill="#4FE09B", font=("Microsoft YaHei UI", 14, "bold"),
                width=right_text_width, justify=tk.LEFT,
            )
            canvas.create_text(
                right_x, status_y + 254, anchor=tk.NW, text=roll_hold_text,
                fill="#4FE09B", font=("Microsoft YaHei UI", 14, "bold"),
                width=right_text_width, justify=tk.LEFT,
            )
            canvas.create_text(
                right_x, status_y + 286, anchor=tk.NW, text=yaw_hold_text,
                fill="#4FE09B", font=("Microsoft YaHei UI", 14, "bold"),
                width=right_text_width, justify=tk.LEFT,
            )
            canvas.create_text(
                right_x, status_y + 332, anchor=tk.NW, text=motion_label,
                fill=motion_color, font=("Microsoft YaHei UI", 16, "bold"),
                width=right_text_width, justify=tk.LEFT,
            )
            canvas.create_text(
                right_x, status_y + 370, anchor=tk.NW, text=motion_action,
                fill=motion_color, font=("Microsoft YaHei UI", 18, "bold"),
                width=right_text_width, justify=tk.LEFT,
            )
            canvas.create_text(
                right_x, status_y + 408, anchor=tk.NW,
                text=self._expanded_mode_text(),
                fill="#FFD166", font=("Microsoft YaHei UI", 15, "bold"),
                width=right_text_width, justify=tk.LEFT,
            )
            canvas.create_text(
                right_x, status_y + 446, anchor=tk.NW, text=resolution_text,
                fill="#8CC9D8", font=("Consolas", 10, "bold"),
                width=right_text_width, justify=tk.LEFT,
            )

    def _draw_expanded_mission_map(self, canvas, canvas_width, canvas_height):
        source = getattr(self, "_mission_map_source", None)
        if source is None:
            return
        left, top, width, height = expanded_mission_map_layout(
            canvas_width, canvas_height, source.width, source.height
        )
        cache_key = (width, height)
        cache = getattr(self, "_mission_map_render_cache", {})
        photo = cache.get(cache_key)
        if photo is None:
            resampling = getattr(Image, "Resampling", Image)
            resized = source.resize((width, height), resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            self._mission_map_render_cache = {cache_key: photo}
        canvas.create_image(left, top, anchor=tk.NW, image=photo)
        canvas._mission_map_image_ref = photo

    def _draw_expanded_annotations(
        self, canvas, canvas_width, canvas_height, frame_width, frame_height,
    ):
        try:
            line_width = max(1.0, float(canvas.winfo_fpixels("1m")))
        except (AttributeError, TypeError, ValueError, tk.TclError):
            line_width = 4.0
        for start, end in getattr(self, "_annotation_lines", ()):
            start_x, start_y = video_point_to_expanded_canvas(
                start, canvas_width, canvas_height, frame_width, frame_height
            )
            end_x, end_y = video_point_to_expanded_canvas(
                end, canvas_width, canvas_height, frame_width, frame_height
            )
            canvas.create_line(
                start_x,
                start_y,
                end_x,
                end_y,
                fill="#FF8F8F",
                width=line_width,
                capstyle=tk.ROUND,
            )

        pending = getattr(self, "_annotation_pending_point", None)
        mode = getattr(self, "_annotation_mode", False)
        if pending is not None:
            x, y = video_point_to_expanded_canvas(
                pending, canvas_width, canvas_height, frame_width, frame_height
            )
            radius = max(3.0, line_width)
            canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                outline="#FF8F8F",
                width=max(1.0, line_width / 2.0),
            )
        if mode:
            prompt = "划线模式：点击终点 · R退出 · T清空" if pending else (
                "划线模式：点击起点 · R退出 · T清空"
            )
            canvas.create_text(
                canvas_width // 2,
                18,
                anchor=tk.N,
                text=prompt,
                fill="#FFB0B0",
                font=("Microsoft YaHei UI", 11, "bold"),
            )

    def _render_expanded_frame(self):
        if (
            self._expanded_window is None
            or not self._expanded_window.winfo_exists()
            or self._expanded_canvas is None
        ):
            return
        self._render_frame(self._expanded_canvas, self._last_frame, draw_depth=True)

    def set_source(self, source):
        """切换视频源：整数=USB摄像头，字符串=RTSP/HTTP URL。"""
        self.stop()
        try:
            if isinstance(source, str) and source.isdigit():
                src = int(source)
            elif isinstance(source, int):
                src = source
            else:
                src = source

            self._http_mjpeg = (
                isinstance(src, str)
                and src.lower().startswith(("http://", "https://"))
            )
            if not self._http_mjpeg:
                self.cap = cv2.VideoCapture(src)
                if not self.cap.isOpened():
                    self._status_label.configure(text=f"无法打开: {source}", foreground=UI["red"])
                    self.cap = None
                    return

            self.source = source
            self.running = True
            self._capture_generation += 1
            generation = self._capture_generation
            self._frame_count = 0
            self._last_fps_time = time.monotonic()
            self._placeholder_drawn = False
            self._status_label.configure(text=f"● 在线", foreground=UI["green"])
            threading.Thread(target=self._capture_loop, args=(generation,), daemon=True).start()
            if self._http_mjpeg:
                # Reapply the selected preset after reconnects; a new GUI
                # session intentionally starts from the calibrated water mode.
                self.after_idle(self._on_camera_profile_selected)
        except Exception as exc:
            self._status_label.configure(text=f"错误: {exc}", foreground=UI["red"])
            self.cap = None

    def _capture_loop(self, generation):
        """后台线程持续抓帧。"""
        if self._http_mjpeg:
            self._capture_http_mjpeg(generation)
            return
        while (self.running and generation == self._capture_generation
               and self.cap and self.cap.isOpened()):
            try:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with self._lock:
                        self._frame = frame
                        self._frame_count += 1
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.05)
        if generation == self._capture_generation:
            self.running = False

    def _capture_http_mjpeg(self, generation):
        """Read MJPEG frames directly, avoiding OpenCV's FFmpeg multipart parser."""
        while self.running and generation == self._capture_generation:
            try:
                request = Request(str(self.source), headers={"User-Agent": "USST-ROV-GUI/1.0"})
                with open_direct_http_stream(request, timeout=5) as response:
                    buffer = bytearray()
                    while self.running and generation == self._capture_generation:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        buffer.extend(chunk)
                        while True:
                            start = buffer.find(b"\xff\xd8")
                            end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                            if start < 0:
                                if len(buffer) > 4 * 1024 * 1024:
                                    buffer.clear()
                                break
                            if end < 0:
                                if start:
                                    del buffer[:start]
                                break
                            encoded = np.frombuffer(buffer[start:end + 2], dtype=np.uint8)
                            del buffer[:end + 2]
                            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                            if frame is None:
                                continue
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            with self._lock:
                                self._frame = frame
                                self._frame_count += 1
            except Exception:
                if self.running and generation == self._capture_generation:
                    time.sleep(0.25)

    def update_display(self):
        """主线程调用，将最新帧渲染到 Canvas。"""
        now = time.monotonic()
        with self._lock:
            frame = self._frame
            self._frame = None
            if frame is not None:
                self._last_frame = frame
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / max(0.001, elapsed)
                self._frame_count = 0
                self._last_fps_time = now
            else:
                elapsed = None
        if elapsed is not None:
            self._fps_label.configure(text=f"{self._fps:.0f} FPS")

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        if frame is None:
            if not self.running and not self._placeholder_drawn:
                self.canvas.delete("all")
                self.canvas.create_text(
                    w // 2, h // 2,
                    text="等待视频流…" if self.cap else "未连接",
                    fill="#555555",
                    font=("Microsoft YaHei UI", 14, "bold"),
                )
                self._placeholder_drawn = True
            return

        try:
            fh, fw = frame.shape[:2]
            self._frame_resolution = (fw, fh)
            self._placeholder_drawn = False
            self._render_frame(self.canvas, frame)
            if (
                self._expanded_window is not None
                and self._expanded_window.winfo_exists()
                and self._expanded_canvas is not None
            ):
                self._render_frame(self._expanded_canvas, frame, draw_depth=True)
        except Exception:
            pass

    def stop(self):
        self.running = False
        self._capture_generation += 1
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        with self._lock:
            self._frame = None
            self._last_frame = None
        self._close_expanded_view()


# ================================================================
#  网络通信层
# ================================================================
class ROVClient:
    def __init__(self, rov_ip=None, cmd_port=None, joy_port=None, telem_port=None):
        self.rov_ip = rov_ip or APP_CONFIG.get("network", "pi_ip")
        self.cmd_port = cmd_port or APP_CONFIG.getint("network", "command_port")
        self.joy_port = joy_port or APP_CONFIG.getint("network", "joystick_port")
        self.telem_port = telem_port or APP_CONFIG.getint("network", "telemetry_port")
        self.shared_key = APP_CONFIG.get("safety", "shared_key")
        self.protocol_version = APP_CONFIG.getint("safety", "protocol_version")
        self.max_packet_bytes = APP_CONFIG.getint("safety", "max_command_packet_bytes")
        self.session = new_session_id()
        self._seq = 0
        self._tx_lock = threading.Lock()

        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.telem_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.telem_sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
            )
        else:
            self.telem_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.telem_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.telem_sock.bind(("0.0.0.0", self.telem_port))
        self.telem_sock.settimeout(0.3)

        self.running = False
        self.latest = {}
        self._lock = threading.Lock()
        self._last_telemetry_time = 0.0
        self._telemetry_guard = TelemetryReplayGuard()
        self._telemetry_session = None
        self._telemetry_last_seq = None
        self._telemetry_received = 0
        self._telemetry_missing = 0
        self._telemetry_loss_window = TelemetryLossWindow(capacity=200)
        self._telemetry_decode_errors = 0
        self._telemetry_socket_errors = 0
        self._last_telemetry_error = ""
        self._last_telemetry_error_report = 0.0

    def _send_payload(self, payload):
        """Serialize seq allocation and socket send across GUI/joystick threads."""
        with self._tx_lock:
            packet = encode_control_packet(
                payload,
                self.shared_key,
                self.session,
                self._seq,
                protocol_version=self.protocol_version,
                max_packet_bytes=self.max_packet_bytes,
            )
            self._seq += 1
            self.cmd_sock.sendto(packet, (self.rov_ip, self.cmd_port))

    def send_cmd(self, enable=None, mode=None, target=None, save_data=None,
                 pid_set=None, pid_save=None, pid_get=0, arm=None, emergency_stop=None,
                 reset_emergency_stop=None, servo_position=None,
                 servo_min_position=None, servo_immediate=None,
                 depth_zero_calibrate=None, attitude_zero_calibrate=None,
                 attitude_mount_calibrate=None,
                 attitude_mount_capture_roll=None,
                 attitude_mount_capture_pitch=None,
                 attitude_mount_capture_yaw=None,
                 attitude_mount_finalize=None,
                 motion_tuning=None, attitude_hold=None,
                 attitude_tuning=None, attitude_mapping_validated=None,
                 performance_mode=None, performance_profile=None):
        msg = {}
        if enable is not None:
            msg["enable"] = bool(enable)
        if mode is not None:
            msg["mode"] = mode
        if target is not None:
            msg["target"] = target
        if save_data is not None:
            msg["save_data"] = bool(save_data)
        if pid_get:
            msg["pid_get"] = int(pid_get)  # legacy; normal readback is telemetry
        if pid_set:
            msg["pid_set"] = pid_set
        if pid_save:
            msg["pid_save"] = pid_save
        if arm is not None:
            msg["arm"] = arm
        if emergency_stop is not None:
            msg["emergency_stop"] = bool(emergency_stop)
        if reset_emergency_stop is not None:
            msg["reset_emergency_stop"] = bool(reset_emergency_stop)
        if servo_position is not None:
            msg["servo_position"] = float(servo_position)
        if servo_min_position is not None:
            msg["servo_min_position"] = float(servo_min_position)
        if servo_immediate is not None:
            msg["servo_immediate"] = bool(servo_immediate)
        if depth_zero_calibrate is not None:
            msg["depth_zero_calibrate"] = bool(depth_zero_calibrate)
        if attitude_zero_calibrate is not None:
            msg["attitude_zero_calibrate"] = bool(attitude_zero_calibrate)
        if attitude_mount_calibrate is not None:
            msg["attitude_mount_calibrate"] = bool(attitude_mount_calibrate)
        if attitude_mount_capture_roll is not None:
            msg["attitude_mount_capture_roll"] = bool(attitude_mount_capture_roll)
        if attitude_mount_capture_pitch is not None:
            msg["attitude_mount_capture_pitch"] = bool(attitude_mount_capture_pitch)
        if attitude_mount_capture_yaw is not None:
            msg["attitude_mount_capture_yaw"] = bool(attitude_mount_capture_yaw)
        if attitude_mount_finalize is not None:
            msg["attitude_mount_finalize"] = bool(attitude_mount_finalize)
        if motion_tuning is not None:
            msg["motion_tuning"] = dict(motion_tuning)
        if attitude_hold is not None:
            msg["attitude_hold"] = dict(attitude_hold)
        if attitude_tuning is not None:
            msg["attitude_tuning"] = dict(attitude_tuning)
        if attitude_mapping_validated is not None:
            msg["attitude_mapping_validated"] = bool(
                attitude_mapping_validated
            )
        if performance_mode is not None:
            msg["performance_mode"] = str(performance_mode)
        if performance_profile is not None:
            msg["performance_profile"] = dict(performance_profile)
        if msg:
            self._send_payload(msg)

    def send_joy(self, Fx=0, Fy=0, Fz=0, N=0, M=0):
        self._send_payload({
            "joystick": {
                "tau_x": round(float(Fx), 3),
                "tau_y": round(float(Fy), 3),
                "tau_z": round(float(Fz), 3),
                "M_x": 0.0,
                "M_y": round(float(M), 3),
                "M_z": round(float(N), 3),
            }
        })

    def _record_telemetry_error(self, kind, exc):
        now = time.monotonic()
        with self._lock:
            if kind == "socket":
                self._telemetry_socket_errors += 1
            else:
                self._telemetry_decode_errors += 1
            self._last_telemetry_error = f"{type(exc).__name__}: {exc}"
            should_report = now - self._last_telemetry_error_report >= 1.0
            if should_report:
                self._last_telemetry_error_report = now
        if should_report:
            print(f"[遥测接收异常/{kind}] {self._last_telemetry_error}")

    def _listen(self):
        while self.running:
            try:
                data, addr = self.telem_sock.recvfrom(16385)
                if addr[0] != self.rov_ip:
                    continue
                verified = decode_telemetry_packet(
                    data,
                    self.shared_key,
                    protocol_version=self.protocol_version,
                )
                telemetry = self._telemetry_guard.accept(verified)
                with self._lock:
                    if verified.session != self._telemetry_session:
                        self._telemetry_session = verified.session
                        self._telemetry_last_seq = verified.seq
                    else:
                        previous = self._telemetry_last_seq
                        if previous is not None and verified.seq > previous + 1:
                            self._telemetry_missing += verified.seq - previous - 1
                        self._telemetry_last_seq = verified.seq
                    self._telemetry_loss_window.observe(
                        verified.session, verified.seq
                    )
                    self._telemetry_received += 1
                    self.latest = telemetry
                    self._last_telemetry_time = time.monotonic()
            except socket.timeout:
                pass
            except OSError as exc:
                if self.running:
                    self._record_telemetry_error("socket", exc)
            except Exception as exc:
                self._record_telemetry_error("decode", exc)

    def start(self):
        self.running = True
        threading.Thread(target=self._listen, daemon=True).start()

    def stop(self):
        self.running = False
        # 关闭上位机也按急停处理，避免最后一帧PWM继续保持。
        for _ in range(3):
            self.send_cmd(enable=False, arm=False)
            self.send_joy(0, 0, 0, 0, 0)
            time.sleep(0.03)
        self.cmd_sock.close()
        self.telem_sock.close()

    def get_telemetry(self):
        with self._lock:
            if time.monotonic() - self._last_telemetry_time > 0.6:
                return {}
            snapshot = dict(self.latest)
            snapshot["_rx_marker"] = (
                self._telemetry_session,
                self._telemetry_last_seq,
            )
            return snapshot

    def get_telemetry_marker(self):
        with self._lock:
            if self._telemetry_session is None or self._telemetry_last_seq is None:
                return None
            return self._telemetry_session, self._telemetry_last_seq

    def get_telemetry_loss_percent(self):
        """Return recent loss from authenticated telemetry sequence gaps."""
        with self._lock:
            return self._telemetry_loss_window.loss_percent()

    def get_telemetry_diagnostics(self):
        with self._lock:
            return {
                "received": self._telemetry_received,
                "missing": self._telemetry_missing,
                "decode_errors": self._telemetry_decode_errors,
                "socket_errors": self._telemetry_socket_errors,
                "last_error": self._last_telemetry_error,
                "receive_buffer_bytes": self.telem_sock.getsockopt(
                    socket.SOL_SOCKET, socket.SO_RCVBUF
                ),
            }


# ================================================================
#  物理手柄读取 (pygame)
# ================================================================
class PhysicalJoystick:
    """使用 pygame 读取真实物理手柄（Xbox / PS / 兼容手柄）"""
    # SDL/pygame-ce DualSense layout: the D-pad is four buttons, not a hat.
    _DPAD_UP_BUTTON = 11
    _DPAD_DOWN_BUTTON = 12
    _DPAD_LEFT_BUTTON = 13
    _DPAD_RIGHT_BUTTON = 14

    def __init__(self, fx_scale=10.0, fy_scale=10.0, fz_scale=10.0,
                 n_scale=5.0, m_scale=5.0, stick_response_exponent=2.2,
                 trigger_response_exponent=2.2):
        self._available = False
        self._joy = None
        self._fx_scale = fx_scale
        self._fy_scale = fy_scale
        self._fz_scale = fz_scale
        self._n_scale = n_scale
        self._m_scale = m_scale
        self._stick_response_exponent = stick_response_exponent
        self._trigger_response_exponent = trigger_response_exponent

        if not HAS_PYGAME:
            return

        try:
            pygame.init()
            count = pygame.joystick.get_count()
            if count > 0:
                self._joy = pygame.joystick.Joystick(0)
                self._joy.init()
                self._available = True
        except Exception:
            self._available = False

    @property
    def available(self):
        return self._available

    @property
    def name(self):
        return self._joy.get_name() if self._joy else "无"

    def read(self):
        """读取手柄状态，返回 (Fx, Fy, Fz, N, M)；无手柄时返回 None。"""
        if not self._available or not self._joy:
            return None

        # 处理 pygame 事件队列（避免卡死）
        for _ in range(8):
            try:
                pygame.event.get()
            except Exception:
                break

        try:
            # --- 轴映射 (Mode2 RC布局) ---
            # 左摇杆四分区：↑↓ 前进/后退，←→ 左移/右移；一次只输出一个动作。
            left_y = -self._get_axis(1)
            left_x = self._get_axis(0)
            left_x, left_y = select_cardinal_axis(left_x, left_y)

            # 右摇杆四分区：↑↓ 上浮/下潜，←→ 左转/右转；一次只输出一个动作。
            right_y = -self._get_axis(3)
            right_x = self._get_axis(2)
            right_x, right_y = select_cardinal_axis(right_x, right_y)

            # LT(R2,axis4) 和 RT(R2,axis5): 从[-1,1] 映射到 [0,1]
            lt = (self._get_axis(4) + 1) / 2.0
            rt = (self._get_axis(5) + 1) / 2.0

            # LT - RT → M (俯仰力矩)
            # 无LT/RT触发轴时 axis4/5 返回0，映射后 lt=rt=0.5，差值为0
            pitch = (lt - rt)

            # Keep the existing deadzone, then use a progressive response so
            # half travel is gentle and only full travel reaches full command.
            stick = lambda value: apply_response_curve(
                remap_deadzone(value), self._stick_response_exponent
            )
            trigger = lambda value: apply_response_curve(
                remap_deadzone(value), self._trigger_response_exponent
            )
            Fx = stick(left_y) * self._fx_scale
            Fy = stick(left_x) * self._fy_scale             # 左摇杆左右 → 横移
            # Vehicle body +Z is down.  Stick-up is ascent, hence negative Fz.
            Fz = -stick(right_y) * self._fz_scale
            N_val = stick(right_x) * self._n_scale          # 右摇杆左右 → 偏航
            M_val = trigger(pitch) * self._m_scale

            return (round(Fx, 1), round(Fy, 1), round(Fz, 1),
                    round(N_val, 1), round(M_val, 1))

        except Exception:
            return None

    def _get_axis(self, idx):
        try:
            return self._joy.get_axis(idx)
        except Exception:
            return 0.0

    def _get_button(self, idx):
        try:
            return self._joy.get_button(idx)
        except Exception:
            return 0

    def is_button_pressed(self, idx):
        """Return one physical button state without exposing pygame to the GUI."""
        return bool(self._get_button(idx))

    def _read_dpad(self):
        """Return the D-pad as (x, y), with positive y meaning Up."""
        try:
            if self._joy.get_numhats() > 0:
                hat_x, hat_y = self._joy.get_hat(0)
                return float(hat_x), float(hat_y)
        except Exception:
            pass

        dpad_x = (
            float(self._get_button(self._DPAD_RIGHT_BUTTON))
            - float(self._get_button(self._DPAD_LEFT_BUTTON))
        )
        dpad_y = (
            float(self._get_button(self._DPAD_UP_BUTTON))
            - float(self._get_button(self._DPAD_DOWN_BUTTON))
        )
        return dpad_x, dpad_y

    def read_dpad(self):
        """Return one cardinal D-pad direction without generating thrust."""
        if not self._available or not self._joy:
            return 0.0, 0.0
        return select_cardinal_axis(*self._read_dpad())

    def close(self):
        if not HAS_PYGAME:
            return
        if self._joy:
            try:
                self._joy.quit()
            except Exception:
                pass
        try:
            pygame.quit()
        except Exception:
            pass


# ================================================================
#  GUI 主界面
# ================================================================
class ROVGui:
    # SDL/pygame-ce DualSense face-button layout, plus the Options button.
    _FACE_BUTTON_ACTIONS = {
        0: "arm",       # Cross / X: request propulsion arm or lock.
        1: "estop",     # Circle: latch E-stop only; it never resets one.
        2: "depth_toggle",  # Square: toggle one-key depth hold.
        3: "video_toggle",  # Triangle: expand/close the live camera window.
        4: "surface_zero",  # Create (upper-left three-bars): current-depth zero.
        6: "attitude_zero",  # Options: persist the current roll/pitch/yaw datum.
    }
    # SDL/pygame-ce DualSense stick-click buttons. The newest operator mapping
    # uses R3 to open and L3 to close.
    _GRIPPER_OPEN_JOG_BUTTON = 8
    _GRIPPER_CLOSE_JOG_BUTTON = 7
    _PERFORMANCE_MODE_BUTTON = 9  # DualSense L1/LB in the tested SDL mapping

    def __init__(self, root, client):
        self.root = root
        self.client = client
        self.root.title("USST RoboCup ROV 综合控制站")
        window_width, window_height = fit_main_window_geometry(
            self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        )
        window_x = max(0, (self.root.winfo_screenwidth() - window_width) // 2)
        window_y = max(0, (self.root.winfo_screenheight() - window_height) // 2)
        self.root.geometry(
            f"{window_width}x{window_height}+{window_x}+{window_y}"
        )
        self.root.minsize(
            min(1280, window_width), min(860, window_height)
        )
        self.root.configure(bg=UI["canvas"])
        self.root.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._configure_styles()

        # 手柄状态（五个运动轴；夹爪命令独立走 servo_position）。
        self._joy = {"Fx": 0, "Fy": 0, "Fz": 0, "N": 0, "M": 0}
        self._joy_keys = set()
        self._emergency_stop_latched = False
        self._local_estop_hold = False
        self._last_estop_tx = 0.0
        self._estop_pending_target = None
        self._estop_pending_until = 0.0
        self._estop_request_marker = None
        self._remote_armed = False
        # A manual unlock is a short, bounded handshake.  UDP has no delivery
        # acknowledgement, so one click sends a few neutral arm requests, all
        # before the Pi's 500 ms command-link timeout.  The generation token
        # cancels stale callbacks after another user action.
        self._arm_request_generation = 0
        self._last_refresh_error = None
        self._last_refresh_error_time = 0.0
        self._last_attitude_hold_rejection_ack = None
        self._last_attitude_calibration_sync = None
        self._last_attitude_calibration_sync_error = None
        self._last_pi_telemetry_session = None
        self._pid_synced_from_pi = False
        self._applied_pid_values = {1: None, 4: None, 5: None, 6: None}
        self._require_joystick_centered = False
        self._servo_angle_min = APP_CONFIG.getfloat("servo", "angle_min_deg")
        self._servo_angle_max = APP_CONFIG.getfloat("servo", "angle_max_deg")
        self._gripper_toggle_button = APP_CONFIG.getint("servo", "toggle_button")
        self._gripper_open_position = APP_CONFIG.getfloat("servo", "open_position")
        self._gripper_closed_position = APP_CONFIG.getfloat(
            "servo", "closed_position"
        )
        self._gripper_jog_open_position = APP_CONFIG.getfloat(
            "servo", "jog_open_position", fallback=1.0
        )
        self._gripper_jog_step = APP_CONFIG.getfloat(
            "servo", "jog_step_per_refresh", fallback=0.05
        )
        self._gripper_button_down = False
        self._gripper_close_on_next_press = False
        self._servo_command_generation = 0
        self._gripper_toggle_requests = queue.SimpleQueue()
        self._gripper_jog_direction = 0
        self._performance_button_down = False
        self._performance_mode_requests = queue.SimpleQueue()
        self._face_button_down = {
            button: False for button in self._FACE_BUTTON_ACTIONS
        }
        self._face_button_requests = queue.SimpleQueue()
        self._dpad_attitude = DpadAttitudeToggleController()
        # A physical vertical-stick release is handled on Tk's main thread so
        # it can safely update the target entry and send one fresh target.
        self._vertical_manual_active = False
        self._vertical_target_update_requests = queue.SimpleQueue()
        # Pi owns and persists the roll/pitch/yaw datum.  Reconnecting or
        # reopening this GUI therefore cannot silently change attitude zero.
        self._imu_roll_inverted = APP_CONFIG.getboolean("hardware", "imu_roll_inverted")
        self._imu_pitch_inverted = APP_CONFIG.getboolean("hardware", "imu_pitch_inverted")
        self._imu_yaw_inverted = APP_CONFIG.getboolean("hardware", "imu_yaw_inverted")
        self._imu_heading_inverted = APP_CONFIG.getboolean(
            "hardware", "imu_heading_inverted"
        )
        self._imu_yaw_offset_deg = APP_CONFIG.getfloat("hardware", "imu_yaw_offset_deg")
        self._imu_heading_offset_deg = APP_CONFIG.getfloat(
            "hardware", "imu_heading_offset_deg"
        )

        # 手柄力缩放 (满杆 = 推进器额定推力)
        #   侧推: 4×65N@45°, Fx_max=184N, Fy_max=170N, Mz_max=43Nm
        #   垂推: 4×65N@90°,  Fz_max=260N
        self._joy_fx = 150.0     # 前进力 N (保守, 极限184)
        self._joy_fy = 150.0     # 侧移力 N
        self._joy_fz = 200.0     # 垂向力 N (保守, 极限260)
        self._joy_yaw = 40.0     # 偏航力矩 Nm (极限43)
        self._joy_pitch = 30.0   # 俯仰力矩 Nm
        self._joy_response_exponent = APP_CONFIG.getfloat(
            "control", "joystick_response_exponent"
        )
        self._trigger_response_exponent = APP_CONFIG.getfloat(
            "control", "trigger_response_exponent"
        )
        self._depth_target_min_m = APP_CONFIG.getfloat(
            "control", "depth_target_min_m"
        )
        self._depth_target_max_m = APP_CONFIG.getfloat(
            "control", "depth_target_max_m"
        )
        self._depth_target_max_step_m = APP_CONFIG.getfloat(
            "control", "depth_target_max_step_m"
        )
        legacy_vertical_trim = APP_CONFIG.getfloat(
            "control", "vertical_front_rear_trim_percent", fallback=0.0
        )
        legacy_horizontal_trim = APP_CONFIG.getfloat(
            "control", "lateral_front_rear_trim_percent", fallback=0.0
        )
        self._motion_tuning = validate_motion_tuning(
            APP_CONFIG.getfloat("thrusters", "horizontal_limit_fraction"),
            APP_CONFIG.getfloat("thrusters", "vertical_limit_fraction"),
            APP_CONFIG.getfloat(
                "control", "vertical_hover_force_per_thruster_n", fallback=0.0
            ),
            APP_CONFIG.getfloat(
                "control", "vertical_ascent_front_rear_trim_percent", fallback=0.0
            ),
            APP_CONFIG.getfloat(
                "control", "vertical_descent_front_rear_trim_percent",
                fallback=legacy_vertical_trim,
            ),
            APP_CONFIG.getfloat(
                "control", "forward_front_rear_trim_percent",
                fallback=legacy_horizontal_trim,
            ),
            APP_CONFIG.getfloat(
                "control", "backward_front_rear_trim_percent",
                fallback=legacy_horizontal_trim,
            ),
        )
        self._motion_tuning_synced_from_pi = False
        # Retained for legacy motion_tuning telemetry/commands. The visible
        # editor below uses complete balance/rage profiles instead.
        self._horizontal_limit_var = tk.StringVar(
            value=f"{self._motion_tuning['horizontal_limit_fraction']:.2f}"
        )
        self._vertical_limit_var = tk.StringVar(
            value=f"{self._motion_tuning['vertical_limit_fraction']:.2f}"
        )
        self._vertical_hover_var = tk.StringVar(
            value=f"{self._motion_tuning['hover_force_per_thruster_n']:.2f}"
        )
        self._vertical_ascent_front_rear_trim_var = tk.StringVar(
            value=f"{self._motion_tuning['vertical_ascent_front_rear_trim_percent']:g}"
        )
        self._vertical_descent_front_rear_trim_var = tk.StringVar(
            value=f"{self._motion_tuning['vertical_descent_front_rear_trim_percent']:g}"
        )
        self._forward_front_rear_trim_var = tk.StringVar(
            value=f"{self._motion_tuning['forward_front_rear_trim_percent']:g}"
        )
        self._backward_front_rear_trim_var = tk.StringVar(
            value=f"{self._motion_tuning['backward_front_rear_trim_percent']:g}"
        )
        self._motion_tuning_status_var = tk.StringVar(value="等待 Pi 参数回读")
        self._attitude_tuning = validate_attitude_tuning(
            APP_CONFIG.getfloat("control", "roll_max_moment_nm"),
            APP_CONFIG.getfloat("control", "pitch_max_moment_nm"),
            APP_CONFIG.getfloat("control", "yaw_max_moment_nm"),
            APP_CONFIG.getfloat(
                "control", "yaw_release_rate_threshold_dps", fallback=3.0
            ),
        )
        self._attitude_tuning_synced_from_pi = False
        self._attitude_tuning_dirty = False
        self._roll_moment_limit_var = tk.StringVar(
            value=f"{self._attitude_tuning['roll_max_moment_nm']:g}"
        )
        self._pitch_moment_limit_var = tk.StringVar(
            value=f"{self._attitude_tuning['pitch_max_moment_nm']:g}"
        )
        self._yaw_moment_limit_var = tk.StringVar(
            value=f"{self._attitude_tuning['yaw_max_moment_nm']:g}"
        )
        self._yaw_release_rate_threshold_var = tk.StringVar(
            value=f"{self._attitude_tuning['yaw_release_rate_threshold_dps']:g}"
        )
        # 物理手柄
        self._physical_joy = PhysicalJoystick(
            fx_scale=self._joy_fx, fy_scale=self._joy_fy,
            fz_scale=self._joy_fz, n_scale=self._joy_yaw,
            m_scale=self._joy_pitch,
            stick_response_exponent=self._joy_response_exponent,
            trigger_response_exponent=self._trigger_response_exponent,
        )
        self._phys_display = None  # 物理手柄实时值
        # Main-bar quick controls are initialized before the detached settings
        # pages so both surfaces share one authoritative Tk variable set.
        self._depth_en = tk.BooleanVar(value=False)
        self._tgt_d_var = tk.StringVar(value="0.5")
        self._servo_position_var = tk.DoubleVar(
            value=self._gripper_closed_position
        )
        closed_angle = servo_position_to_angle(
            self._gripper_closed_position,
            self._servo_angle_min,
            self._servo_angle_max,
        )
        open_angle = servo_position_to_angle(
            self._gripper_open_position,
            self._servo_angle_min,
            self._servo_angle_max,
        )
        self._servo_angle_var = tk.StringVar(value=f"{closed_angle:.1f}°")
        # Internal names are retained for config compatibility; values shown
        # in the GUI are degrees rather than normalized percentages.
        self._gripper_open_angle_var = tk.StringVar(
            value=f"{open_angle:g}"
        )
        self._gripper_closed_angle_var = tk.StringVar(
            value=f"{closed_angle:g}"
        )
        self._servo_lower_angle_var = tk.StringVar(value=f"{closed_angle:g}°")
        self._servo_status_var = tk.StringVar(value="等待 Pi 遥测")
        self._performance_mode_var = tk.StringVar(value="当前：平衡模式")
        self._performance_mode = "balance"
        self._performance_profiles = self._load_local_performance_profiles()
        self._performance_profile_dirty = {"balance": False, "rage": False}

        # 构建界面
        self._content = ttk.Frame(
            self.root, style="App.TFrame", padding=(12, 8, 12, 10)
        )
        self._content.pack(fill=tk.BOTH, expand=True)
        self._build_connection_bar()
        # Detailed autonomous and joystick controls live in non-modal tool
        # windows.  They stay initialized while withdrawn, so telemetry and
        # controller state keep updating without compressing the dashboard.
        self._build_settings_windows()
        self._build_main_action_bar(self._content)

        self._dashboard = ttk.Frame(self._content, style="App.TFrame")
        self._dashboard.pack(fill=tk.BOTH, expand=True)
        self._dashboard.grid_rowconfigure(0, weight=1)
        # Keep all three columns visible on Windows high-DPI displays.  Tk
        # scales these logical pixel minima, so the old 285/680/340 values
        # exceeded a 1707 px desktop at 125-150% scaling.
        self._dashboard.grid_columnconfigure(0, weight=3, minsize=210)
        self._dashboard.grid_columnconfigure(1, weight=10, minsize=520)
        self._dashboard.grid_columnconfigure(2, weight=1, minsize=280)

        self._left_column = ttk.Frame(self._dashboard, style="App.TFrame")
        self._center_column = ttk.Frame(self._dashboard, style="App.TFrame")
        self._right_column = ttk.Frame(self._dashboard, style="App.TFrame")
        self._left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self._center_column.grid(row=0, column=1, sticky="nsew", padx=5)
        self._right_column.grid(row=0, column=2, sticky="nsew", padx=(5, 0))

        self._left_column.grid_rowconfigure(0, weight=0)
        self._left_column.grid_rowconfigure(1, weight=1)
        self._left_column.grid_columnconfigure(0, weight=1)

        self._center_column.grid_rowconfigure(0, weight=1)
        self._center_column.grid_rowconfigure(1, weight=1)
        self._center_column.grid_columnconfigure(0, weight=1)

        self._build_left_panels(self._left_column)
        self._build_center_panels(self._center_column)
        self._build_thruster_panel(self._right_column)
        self._build_telemetry_panel(self._right_column)

        # 手柄持续发送 (40Hz)，降低手柄采样到控制包的等待时间。
        self._joy_running = True
        self._joy_thread = threading.Thread(target=self._joy_loop, daemon=True)
        self._joy_thread.start()

        # Lightweight labels stay at 20 Hz; the canvas-heavy dashboard is 5 Hz.
        self._refresh_display()
        self._refresh_fast_yaw_display()

        # 绑定键盘
        self.root.bind("<KeyPress>", self._on_key_down)
        self.root.bind("<KeyRelease>", self._on_key_up)
        self.root.bind_all("<Control-s>", self._save_current_parameters, add="+")
        self.root.bind_all("<Button-1>", self._clear_text_focus_on_background_click, add="+")
        self.root.bind("<FocusOut>", self._on_focus_out)

    def _load_local_performance_profiles(self):
        """Build editable defaults before the first Pi telemetry arrives."""
        profiles = {}
        for mode in ("balance", "rage"):
            section = f"mode_{mode}"
            fallback = {
                "balance": (20.0, 30.0),
                "rage": (30.0, 40.0),
            }[mode]
            gains = {}
            for axis in ("roll", "pitch", "yaw"):
                gains[axis] = tuple(
                    APP_CONFIG.getfloat(
                        section, f"{axis}_{field}",
                        fallback=APP_CONFIG.getfloat("pid", f"{axis}_{field}"),
                    )
                    for field in ("kp", "ki", "kd")
                )
            profiles[mode] = {
                "mode": mode,
                "horizontal_limit_fraction": APP_CONFIG.getfloat(
                    section, "horizontal_limit_fraction", fallback=fallback[0] / 100.0
                ),
                "vertical_limit_fraction": APP_CONFIG.getfloat(
                    section, "vertical_limit_fraction", fallback=fallback[1] / 100.0
                ),
                "hover_force_per_thruster_n": APP_CONFIG.getfloat(
                    section, "hover_force_per_thruster_n",
                    fallback=self._motion_tuning["hover_force_per_thruster_n"],
                ),
                **{
                    field: APP_CONFIG.getfloat(
                        section, field,
                        fallback=(
                            (8.0 if field == "forward_front_rear_trim_percent" else -8.0)
                            if mode == "rage" and field in (
                                "forward_front_rear_trim_percent",
                                "backward_front_rear_trim_percent",
                            )
                            else (
                                0.0
                                if field in (
                                    "left_lateral_surge_trim_percent",
                                    "right_lateral_surge_trim_percent",
                                )
                                else self._motion_tuning[field]
                            )
                        ),
                    )
                    for field in (
                        "vertical_ascent_front_rear_trim_percent",
                        "vertical_descent_front_rear_trim_percent",
                        "forward_front_rear_trim_percent",
                        "backward_front_rear_trim_percent",
                        "left_lateral_surge_trim_percent",
                        "right_lateral_surge_trim_percent",
                    )
                },
                **{
                    f"{axis}_max_moment_nm": APP_CONFIG.getfloat(
                        section, f"{axis}_max_moment_nm",
                        fallback=(
                            max(3.5, self._attitude_tuning["pitch_max_moment_nm"])
                            if mode == "rage" and axis == "pitch"
                            else self._attitude_tuning[f"{axis}_max_moment_nm"]
                        ),
                    )
                    for axis in ("roll", "pitch", "yaw")
                },
                "roll_pid": dict(zip(("kp", "ki", "kd"), gains["roll"])),
                "pitch_pid": dict(zip(("kp", "ki", "kd"), gains["pitch"])),
                "yaw_pid": dict(zip(("kp", "ki", "kd"), gains["yaw"])),
            }
        return profiles

    def _configure_styles(self):
        """统一颜色、字体和控件密度；不改变任何控制行为。"""
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=UI["surface"])
        style.configure("App.TFrame", background=UI["canvas"])
        style.configure("TLabel", background=UI["surface"], foreground=UI["text"])
        style.configure(
            "Muted.TLabel", background=UI["surface"], foreground=UI["muted"]
        )
        style.configure(
            "Card.TLabelframe",
            background=UI["surface"],
            bordercolor=UI["border"],
            lightcolor=UI["border"],
            darkcolor=UI["border"],
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=UI["surface"],
            foreground=UI["navy"],
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        style.configure(
            "Metric.TFrame",
            background=UI["surface_alt"],
            bordercolor=UI["border"],
            lightcolor=UI["border"],
            darkcolor=UI["border"],
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "MetricLabel.TLabel",
            background=UI["surface_alt"],
            foreground=UI["muted"],
            font=("Microsoft YaHei UI", 7),
        )
        style.configure(
            "MetricValue.TLabel",
            background=UI["surface_alt"],
            foreground=UI["text"],
            font=("Consolas", 11, "bold"),
        )
        style.configure(
            "SmallMetric.TLabel",
            background=UI["surface_alt"],
            foreground=UI["orange"],
            font=("Consolas", 10, "bold"),
        )
        style.configure(
            "TEntry",
            fieldbackground=UI["surface_alt"],
            foreground=UI["text"],
            bordercolor=UI["border"],
            lightcolor=UI["border"],
            darkcolor=UI["border"],
            padding=(7, 2),
        )
        style.map("TEntry", bordercolor=[("focus", UI["blue"])])
        style.configure(
            "TCheckbutton",
            background=UI["surface"],
            foreground=UI["text"],
            font=("Microsoft YaHei UI", 10),
        )
        style.map(
            "TCheckbutton",
            background=[("active", UI["surface"])],
            indicatorcolor=[("selected", UI["blue"])],
        )
        style.configure(
            "TButton",
            font=("Microsoft YaHei UI", 10, "bold"),
            foreground=UI["navy"],
            background="#E8EEF5",
            bordercolor="#E8EEF5",
            padding=(12, 5),
        )
        style.map(
            "TButton",
            background=[("active", "#DCE6F1"), ("pressed", "#CFDCE9")],
        )
        style.configure(
            "Primary.TButton",
            foreground="white",
            background=UI["blue"],
            bordercolor=UI["blue"],
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#0E67DD"), ("pressed", "#0958BF")],
        )
        style.configure("Horizontal.TSeparator", background=UI["border"])

    # ================================================================
    #  连接栏
    # ================================================================
    def _build_connection_bar(self):
        frame = ttk.LabelFrame(
            self._content, text="  连接与通信  ", style="Card.TLabelframe", padding=(14, 3)
        )
        frame.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(frame, text="ROV 地址", font=("Microsoft YaHei UI", 9, "bold")).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self._ip_var = tk.StringVar(value=self.client.rov_ip)
        ttk.Entry(frame, textvariable=self._ip_var, width=17).pack(side=tk.LEFT)

        ttk.Button(
            frame, text="应用地址", command=self._update_ip, style="Primary.TButton"
        ).pack(side=tk.LEFT, padx=8)

        ttk.Label(
            frame,
            text="遥测 UDP 5001  ·  控制 UDP 5000",
            style="Muted.TLabel",
        ).pack(side=tk.LEFT, padx=10)

        self._conn_label = ttk.Label(
            frame,
            text="● 等待遥测",
            foreground=UI["amber"],
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self._conn_label.pack(side=tk.RIGHT, padx=(8, 2))

        self._delay_label = ttk.Label(
            frame,
            text="延迟 -- ms",
            foreground=UI["muted"],
            font=("Consolas", 9, "bold"),
        )
        self._delay_label.pack(side=tk.RIGHT, padx=8)
        self._pkt_loss_label = ttk.Label(
            frame,
            text="丢包 --%",
            foreground=UI["muted"],
            font=("Consolas", 9, "bold"),
        )
        self._pkt_loss_label.pack(side=tk.RIGHT, padx=8)

        # 遥测统计
        self._telem_seq_last = -1
        self._telem_seq_gaps = 0
        self._telem_seq_total = 0

    def _update_ip(self):
        self.client.rov_ip = self._ip_var.get()

    def _build_settings_windows(self):
        """Build persistent, non-modal pages for detailed control settings."""
        self._autonomy_window = tk.Toplevel(self.root)
        self._autonomy_window.withdraw()
        self._autonomy_window.title("自主控制")
        self._autonomy_window.geometry("1220x640")
        self._autonomy_window.minsize(1080, 640)
        self._autonomy_window.configure(bg=UI["canvas"])
        self._autonomy_window.protocol(
            "WM_DELETE_WINDOW", self._autonomy_window.withdraw
        )
        autonomy_body = ttk.Frame(
            self._autonomy_window,
            style="App.TFrame",
            padding=(12, 10, 12, 12),
        )
        autonomy_body.pack(fill=tk.BOTH, expand=True)
        self._build_command_panel(autonomy_body)
        self._autonomy_window.update_idletasks()

        self._joystick_window = tk.Toplevel(self.root)
        self._joystick_window.withdraw()
        self._joystick_window.title("手柄控制说明与实时状态")
        self._joystick_window.geometry("960x135")
        self._joystick_window.minsize(760, 125)
        self._joystick_window.configure(bg=UI["canvas"])
        self._joystick_window.protocol(
            "WM_DELETE_WINDOW", self._joystick_window.withdraw
        )
        joystick_body = ttk.Frame(
            self._joystick_window,
            style="App.TFrame",
            padding=(12, 10, 12, 12),
        )
        joystick_body.pack(fill=tk.BOTH, expand=True)
        self._build_joystick_panel(joystick_body)
        self._joystick_window.update_idletasks()

    @staticmethod
    def _show_tool_window(window):
        # Resolve the hidden-window layout before Windows paints it.
        window.update_idletasks()
        window.deiconify()
        window.lift()
        window.focus_set()

    def _build_main_action_bar(self, parent):
        """Keep all safety-critical actions visible on the main dashboard."""
        frame = ttk.LabelFrame(
            parent,
            text="  操作与安全  ",
            style="Card.TLabelframe",
            padding=(10, 4, 10, 5),
        )
        frame.pack(fill=tk.X, pady=(0, 6))
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X)

        ttk.Button(
            btn_row,
            text="⚙ 自主控制",
            command=lambda: self._show_tool_window(self._autonomy_window),
            width=14,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            btn_row,
            text="🎮 手柄控制",
            command=lambda: self._show_tool_window(self._joystick_window),
            width=14,
        ).pack(side=tk.LEFT, padx=(0, 12))

        self._arm_btn = tk.Button(
            btn_row, text="🔒 推进器锁定", command=self._toggle_arm,
            font=("Microsoft YaHei UI", 10, "bold"),
            fg="white", bg=UI["red_dark"],
            activeforeground="white", activebackground="#86110D",
            width=14, height=1, relief=tk.FLAT, borderwidth=0,
            cursor="hand2",
        )
        self._arm_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._estop_btn = tk.Button(
            btn_row, text="⛔ 紧急停止", command=self._toggle_emergency_stop,
            font=("Microsoft YaHei UI", 10, "bold"), fg="white", bg=UI["red"],
            activeforeground="white", activebackground=UI["red_dark"],
            width=12, height=1, relief=tk.FLAT, borderwidth=0,
            cursor="hand2",
        )
        self._estop_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._start_btn = tk.Button(
            btn_row, text="▶ 自主启动", command=self._cmd_start,
            font=("Microsoft YaHei UI", 10, "bold"), fg="white", bg=UI["green"],
            activeforeground="white", activebackground="#117B54",
            disabledforeground="#D8DEE8", width=10, height=1,
            relief=tk.FLAT, borderwidth=0, cursor="hand2", state=tk.DISABLED,
        )
        self._start_btn.pack(side=tk.LEFT, padx=4)
        tk.Button(
            btn_row, text="■ 自主停止", command=self._cmd_stop,
            font=("Microsoft YaHei UI", 10, "bold"), fg="white", bg=UI["navy_soft"],
            activeforeground="white", activebackground=UI["navy"],
            width=10, height=1, relief=tk.FLAT, borderwidth=0,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=4)
        self._hover_btn = tk.Button(
            btn_row, text="🎯 一键定深", command=self._cmd_hover,
            font=("Microsoft YaHei UI", 10, "bold"), fg="white", bg=UI["blue"],
            activeforeground="white", activebackground="#0E67DD",
            disabledforeground="#D8DEE8", width=10, height=1,
            relief=tk.FLAT, borderwidth=0, cursor="hand2", state=tk.DISABLED,
        )
        self._hover_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._save_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btn_row, text="保存CSV", variable=self._save_var).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        self._arm_light = ttk.Label(
            btn_row, text="● 锁定", foreground=UI["red"],
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self._arm_light.pack(side=tk.RIGHT, padx=6)
        self._cmd_light = ttk.Label(
            btn_row, text="● 自主", foreground=UI["muted"],
            font=("Microsoft YaHei UI", 10),
        )
        self._cmd_light.pack(side=tk.RIGHT, padx=6)
        self._joy_light = ttk.Label(
            btn_row, text="● 手柄", foreground=UI["muted"],
            font=("Microsoft YaHei UI", 10),
        )
        self._joy_light.pack(side=tk.RIGHT, padx=6)
        self._performance_mode_label = ttk.Label(
            btn_row, textvariable=self._performance_mode_var,
            foreground=UI["blue"], font=("Microsoft YaHei UI", 10, "bold"),
        )
        self._performance_mode_label.pack(side=tk.RIGHT, padx=8)
        self._yaw_hold_quick_light = ttk.Label(
            btn_row, text="● 偏转锁：未知", foreground=UI["muted"],
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self._yaw_hold_quick_light.pack(side=tk.RIGHT, padx=6)
        self._roll_hold_quick_light = ttk.Label(
            btn_row, text="● 横滚锁：未知", foreground=UI["muted"],
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self._roll_hold_quick_light.pack(side=tk.RIGHT, padx=6)
        self._pitch_hold_quick_light = ttk.Label(
            btn_row, text="● 俯仰锁：未知", foreground=UI["muted"],
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self._pitch_hold_quick_light.pack(side=tk.RIGHT, padx=6)

        quick_row = ttk.Frame(frame)
        quick_row.pack(fill=tk.X, pady=(5, 0))
        depth_controls = ttk.Frame(quick_row)
        depth_controls.pack(side=tk.LEFT)
        ttk.Checkbutton(
            depth_controls, text="深度保持", variable=self._depth_en,
        ).pack(side=tk.LEFT)
        ttk.Label(depth_controls, text="目标深度 m").pack(
            side=tk.LEFT, padx=(12, 5)
        )
        ttk.Entry(
            depth_controls, textvariable=self._tgt_d_var, width=7,
        ).pack(side=tk.LEFT)
        ttk.Button(
            depth_controls, text="取当前值", command=self._capture_current_depth,
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(
            depth_controls, text="当前位置置零", command=self._calibrate_depth_zero,
        ).pack(side=tk.LEFT, padx=(6, 0))
        self._main_yaw_target_var = tk.StringVar(value="目标Yaw：--°")
        ttk.Label(
            depth_controls,
            textvariable=self._main_yaw_target_var,
            foreground=UI["blue"],
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side=tk.LEFT, padx=(16, 0))

        ttk.Separator(quick_row, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=12
        )
        servo_controls = ttk.Frame(quick_row)
        servo_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(servo_controls, text="舵机 CH8 当前").pack(side=tk.LEFT)
        ttk.Label(
            servo_controls, textvariable=self._servo_angle_var,
            font=("Consolas", 10, "bold"), foreground=UI["blue"],
        ).pack(side=tk.LEFT, padx=(5, 10))
        ttk.Label(servo_controls, text="下限").pack(side=tk.LEFT)
        ttk.Label(
            servo_controls, textvariable=self._servo_lower_angle_var
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(servo_controls, text="RB开").pack(side=tk.LEFT, padx=(8, 2))
        open_entry = ttk.Entry(
            servo_controls,
            textvariable=self._gripper_open_angle_var,
            width=6,
        )
        open_entry.pack(side=tk.LEFT)
        open_entry.bind("<Return>", self._apply_gripper_positions)
        ttk.Label(servo_controls, text="°").pack(side=tk.LEFT, padx=(1, 5))
        ttk.Label(servo_controls, text="RB关").pack(side=tk.LEFT, padx=(0, 2))
        closed_entry = ttk.Entry(
            servo_controls,
            textvariable=self._gripper_closed_angle_var,
            width=6,
        )
        closed_entry.pack(side=tk.LEFT)
        closed_entry.bind("<Return>", self._apply_gripper_positions)
        ttk.Label(servo_controls, text="°").pack(side=tk.LEFT, padx=(1, 0))
        ttk.Label(
            servo_controls, textvariable=self._servo_status_var,
            foreground=UI["muted"],
        ).pack(side=tk.LEFT, padx=(10, 0))

        self._safety_detail = ttk.Label(
            frame, text="安全状态：等待 Pi 遥测", foreground=UI["muted"],
            font=("Microsoft YaHei UI", 9), wraplength=1450, justify=tk.LEFT,
        )
        self._safety_detail.pack(anchor=tk.W, pady=(4, 0))

    # ================================================================
    #  左栏组合：计时器 + 趋势图
    # ================================================================
    def _build_left_panels(self, parent):
        # 比赛计时器 (row=0)
        self._timer = CompetitionTimer(parent)
        self._timer.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        # 深度与三轴姿态趋势 (row=1)
        trends_frame = ttk.LabelFrame(
            parent, text="  实时趋势  ", style="Card.TLabelframe", padding=4,
        )
        trends_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        for row in range(5):
            trends_frame.grid_rowconfigure(row, weight=1)
        trends_frame.grid_columnconfigure(0, weight=1)
        self._depth_trend = TrendChart(
            trends_frame, "深度 Depth", "m", UI["blue"],
            minimum=self._depth_target_min_m, maximum=self._depth_target_max_m,
        )
        self._depth_trend.grid(row=0, column=0, sticky="nsew", pady=(0, 2))
        self._roll_trend = TrendChart(
            trends_frame, "横滚 Roll", "°", UI["green"],
            minimum=-180.0, maximum=180.0,
        )
        self._roll_trend.grid(row=1, column=0, sticky="nsew", pady=2)
        self._pitch_trend = TrendChart(
            trends_frame, "俯仰 Pitch", "°", UI["orange"],
            minimum=-90.0, maximum=90.0,
        )
        self._pitch_trend.grid(row=2, column=0, sticky="nsew", pady=2)
        self._yaw_trend = TrendChart(
            trends_frame, "偏转 Yaw", "°", UI["red"],
            minimum=-180.0, maximum=180.0,
        )
        self._yaw_trend.grid(row=3, column=0, sticky="nsew", pady=2)
        self._heading_trend = TrendChart(
            trends_frame, "航向 Heading", "°", UI["cyan"],
            minimum=0.0, maximum=360.0,
        )
        self._heading_trend.grid(row=4, column=0, sticky="nsew", pady=(2, 0))

    def _build_telemetry_panel(self, parent, row=0):
        frame = ttk.LabelFrame(
            parent, text="  实时遥测  ", style="Card.TLabelframe", padding=(4, 1, 4, 2)
        )
        if row == 0:
            frame.pack(fill=tk.BOTH, expand=True)
        else:
            frame.grid(row=row, column=0, sticky="nsew")

        def metric_group(title, definitions):
            ttk.Label(
                frame, text=title, foreground=UI["navy_soft"],
                font=("Microsoft YaHei UI", 8, "bold"),
            ).pack(anchor="w", padx=4, pady=(1, 0))
            grid = ttk.Frame(frame)
            grid.pack(fill=tk.X)
            result = []
            for index, (label, initial, color) in enumerate(definitions):
                result.append(
                    self._make_digit(
                        grid, label, initial, index % 2,
                        row=index // 2, fg=color,
                    )
                )
            return result

        (
            self._lbl_roll, self._lbl_pitch, self._lbl_yaw, self._lbl_head,
        ) = metric_group("姿态角", [
            ("Roll 横滚", "0.0°", UI["text"]),
            ("Pitch 俯仰", "0.0°", UI["text"]),
            ("Yaw 偏航", "0.0°", UI["text"]),
            ("Heading 航向", "0.0°", UI["blue"]),
        ])
        attitude_actions = ttk.Frame(frame)
        attitude_actions.pack(fill=tk.X, padx=4, pady=(1, 2))
        for column in range(2):
            attitude_actions.grid_columnconfigure(column, weight=1)
        ttk.Button(
            attitude_actions, text="三维安装标定",
            command=self._calibrate_attitude_mount, width=11,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=(0, 2))
        ttk.Button(
            attitude_actions, text="②上仰采集",
            command=self._capture_attitude_mount_pitch, width=9,
        ).grid(row=0, column=1, sticky="ew", padx=(2, 0), pady=(0, 2))
        ttk.Button(
            attitude_actions, text="③水平右偏采集",
            command=self._capture_attitude_mount_yaw, width=12,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(
            attitude_actions, text="④完成",
            command=self._finalize_attitude_mount, width=7,
        ).grid(row=1, column=1, sticky="ew", padx=(2, 0))
        ttk.Label(
            frame,
            text="锁定推进器：水平→上仰20–45°→水平右偏20–45°",
            foreground=UI["muted"], font=("Microsoft YaHei UI", 7),
            wraplength=300, justify=tk.LEFT,
        ).pack(anchor="w", padx=4, pady=(0, 1))
        (
            self._lbl_depth, self._lbl_vz, self._lbl_vx, self._lbl_vy,
        ) = metric_group("深度与速度", [
            ("Depth 水深", "0.000 m", UI["blue"]),
            ("Vz 垂速", "0.000 m/s", UI["orange"]),
            ("Vx 前速（估）", "0.000 m/s", UI["orange"]),
            ("Vy 侧速（估）", "0.000 m/s", UI["orange"]),
        ])
        self._build_event_log(frame)

    def _build_center_panels(self, parent):
        """中心栏：3D姿态 + 视频流，上下排列。"""
        # 3D姿态 (row=0)
        pose_frame = ttk.LabelFrame(
            parent,
            text="  ROV三维姿态 · IMU实时驱动  ",
            style="Card.TLabelframe",
            padding=6,
        )
        pose_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        mesh_path = Path(__file__).resolve().parent / "assets" / "robocup_rov_mesh.json"
        self._pose_view = ROVPoseView(pose_frame, mesh_path)
        self._pose_view.pack(fill=tk.BOTH, expand=True)

        # 视频流 (row=1)
        video_frame = ttk.LabelFrame(
            parent,
            text="  实时视频  ",
            style="Card.TLabelframe",
            padding=4,
        )
        video_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        if HAS_OPENCV:
            self._video_view = VideoStream(
                video_frame,
                source=APP_CONFIG.get("video", "mjpeg_url"),
                depth_provider=self._expanded_video_depth_text,
                status_provider=self._expanded_video_status,
                timer_provider=self._timer.expanded_status,
                mode_provider=self._expanded_video_mode_text,
                gripper_next_provider=self._expanded_gripper_next_text,
            )
            self._video_view.pack(fill=tk.BOTH, expand=True)
        else:
            placeholder = ttk.Frame(video_frame)
            placeholder.pack(fill=tk.BOTH, expand=True)
            ttk.Label(
                placeholder,
                text="⚠ OpenCV 未安装\n请运行 pip install opencv-python Pillow",
                foreground=UI["amber"],
                font=("Microsoft YaHei UI", 12, "bold"),
                justify=tk.CENTER,
            ).pack(expand=True)

    def _expanded_video_depth_text(self):
        """Return Pi-provided depth for the maximized video overlay."""
        telemetry = self.client.get_telemetry()
        try:
            depth = valid_depth_measurement(telemetry)
        except ValueError:
            return "深度 --"
        return f"深度 {depth:.3f} m"

    def _expanded_video_status(self):
        """Return Pi-authoritative status text for the reserved black video bar."""
        telemetry = self.client.get_telemetry()
        depth_text = self._expanded_video_depth_text()
        if not telemetry:
            pitch_text, roll_text, yaw_text = expanded_attitude_hold_texts(telemetry)
            target_depth_text, yaw_value_text, yaw_target_text, pitch_value_text, roll_value_text = expanded_attitude_value_texts(telemetry)
            return (
                depth_text, target_depth_text, "定深：未知",
                yaw_value_text, yaw_target_text, pitch_value_text, roll_value_text,
                pitch_text, roll_text, yaw_text,
                "运动指令：控制失联",
            )
        mode = telemetry.get("mode", {})
        depth_on = bool(
            telemetry.get("cmd_enabled")
            and isinstance(mode, dict)
            and mode.get("depth_hold")
        )
        pitch_text, roll_text, yaw_text = expanded_attitude_hold_texts(telemetry)
        target_depth_text, yaw_value_text, yaw_target_text, pitch_value_text, roll_value_text = expanded_attitude_value_texts(telemetry)
        return (
            depth_text,
            target_depth_text,
            "定深：已开启" if depth_on else "定深：未开启",
            yaw_value_text,
            yaw_target_text,
            pitch_value_text,
            roll_value_text,
            pitch_text,
            roll_text,
            yaw_text,
            format_motion_command(telemetry),
        )

    def _expanded_video_mode_text(self):
        telemetry = self.client.get_telemetry() or {}
        mode = str(telemetry.get("performance_mode", self._performance_mode)).lower()
        return "模式：狂暴" if mode == "rage" else "模式：平衡"

    def _expanded_gripper_next_text(self):
        return "下次关" if self._gripper_close_on_next_press else "下次开"

    def _make_digit(self, parent, label, init, col, row=0, fg="black"):
        f = ttk.Frame(parent, style="Metric.TFrame", padding=(4, 1))
        f.grid(row=row, column=col, padx=2, pady=0, sticky="ew")
        parent.grid_columnconfigure(col, weight=1)
        ttk.Label(f, text=label, style="MetricLabel.TLabel").pack()
        var = tk.StringVar(value=init)
        ttk.Label(
            f,
            textvariable=var,
            style="MetricValue.TLabel",
            foreground=fg,
        ).pack()
        return var

    def _make_pwm(self, parent, name, col):
        f = ttk.Frame(parent, relief=tk.RIDGE, borderwidth=1)
        f.grid(row=0, column=col, padx=2, sticky="ew")
        parent.grid_columnconfigure(col, weight=1)
        ttk.Label(f, text=name, font=("", 7), foreground="gray").pack()
        var = tk.StringVar(value="1500")
        ttk.Label(f, textvariable=var, font=("Consolas", 9)).pack()
        return var

    # ================================================================
    #  指令面板 (定向/定深独立勾选 + PID参数)
    # ================================================================
    def _build_command_panel(self, parent):
        frame = ttk.LabelFrame(
            parent,
            text="  自主控制  ",
            style="Card.TLabelframe",
            padding=(10, 1),
        )
        frame.pack(fill=tk.X, pady=(0, 4))

        body = ttk.Frame(frame)
        body.pack(fill=tk.X)
        controls = ttk.Frame(body)
        controls.pack(fill=tk.X)

        # 六自由度输出仍保留在Pi遥测中，但按操作界面要求不再单独显示。
        self._force_vars = {}

        # ----- Runtime roll/pitch/yaw stabilization -----
        # PID gains and moment limits belong to the two LB profiles below.
        # The only shared attitude tuning exposed here is yaw-release speed.
        attitude_rows = ttk.Frame(controls)
        attitude_rows.pack(fill=tk.X)
        for row_index, (axis, title, dpad_text) in enumerate((
            ("pitch", "俯仰自动", "十字键↑"),
            ("roll", "横滚自动", "十字键→"),
            ("yaw", "偏转自动", "十字键←"),
        )):
            row_attitude = ttk.Frame(attitude_rows)
            row_attitude.grid(row=row_index, column=0, sticky=tk.W, pady=(0, 2))
            button = tk.Button(
                row_attitude,
                text=f"{dpad_text} {title}：关",
                command=lambda selected=axis: self._toggle_attitude_hold(selected),
                # Keep the longest legacy-compatible state
                # ("偏转自动：状态未知") fully visible at Windows DPI scaling.
                width=20,
                fg="#000000",
                bg="#FFFFFF",
                activeforeground="#000000",
                activebackground="#FFFFFF",
                disabledforeground="#000000",
                relief=tk.SOLID,
                borderwidth=1,
            )
            button.grid(row=0, column=0, sticky=tk.W)
            setattr(self, f"_{axis}_hold_btn", button)
            ttk.Label(
                row_attitude,
                text="目标 0°" if axis != "yaw" else "",
                width=8,
                anchor=tk.E,
            ).grid(row=0, column=1, padx=(12, 4), sticky=tk.E)
            for gain in ("kp", "ki", "kd"):
                variable = tk.StringVar(
                    value=f"{APP_CONFIG.getfloat('pid', f'{axis}_{gain}'):g}"
                )
                setattr(self, f"_{axis[0]}_{gain}", variable)
            if axis == "yaw":
                ttk.Label(row_attitude, text="松手锁定≤").grid(
                    row=0, column=2, padx=(14, 4), sticky=tk.E
                )
                release_entry = ttk.Entry(
                    row_attitude,
                    textvariable=self._yaw_release_rate_threshold_var,
                    width=6,
                )
                release_entry.grid(row=0, column=3, sticky=tk.W)
                release_entry.bind(
                    "<Return>", self._apply_attitude_tuning_from_entry
                )
                self._bind_attitude_tuning_entry(release_entry)
                ttk.Label(row_attitude, text="°/s").grid(
                    row=0, column=4, padx=(3, 0), sticky=tk.W
                )

        # ----- LB performance profiles: two complete independent sets -----
        profile_frame = ttk.LabelFrame(
            controls, text="LB性能模式参数（推力与方向补偿为百分比）",
            style="Card.TLabelframe", padding=(6, 3),
        )
        profile_frame.pack(fill=tk.X, pady=(3, 2))
        self._performance_profile_entries = {}
        for mode, title in (("balance", "平衡模式"), ("rage", "狂暴模式")):
            profile = self._performance_profiles[mode]
            vars_by_key = {
                "horizontal": tk.StringVar(
                    value=f"{profile['horizontal_limit_fraction'] * 100.0:.1f}"
                ),
                "vertical": tk.StringVar(
                    value=f"{profile['vertical_limit_fraction'] * 100.0:.1f}"
                ),
                "hover": tk.StringVar(
                    value=f"{profile['hover_force_per_thruster_n']:g}"
                ),
                "ascent_trim": tk.StringVar(
                    value=f"{profile['vertical_ascent_front_rear_trim_percent']:g}"
                ),
                "descent_trim": tk.StringVar(
                    value=f"{profile['vertical_descent_front_rear_trim_percent']:g}"
                ),
                "forward_trim": tk.StringVar(
                    value=f"{profile['forward_front_rear_trim_percent']:g}"
                ),
                "backward_trim": tk.StringVar(
                    value=f"{profile['backward_front_rear_trim_percent']:g}"
                ),
                "left_lateral": tk.StringVar(
                    value=f"{profile['left_lateral_surge_trim_percent']:g}"
                ),
                "right_lateral": tk.StringVar(
                    value=f"{profile['right_lateral_surge_trim_percent']:g}"
                ),
                "roll_moment": tk.StringVar(
                    value=f"{profile['roll_max_moment_nm']:g}"
                ),
                "pitch_moment": tk.StringVar(
                    value=f"{profile['pitch_max_moment_nm']:g}"
                ),
                "yaw_moment": tk.StringVar(
                    value=f"{profile['yaw_max_moment_nm']:g}"
                ),
            }
            for axis in ("roll", "pitch", "yaw"):
                for field in ("kp", "ki", "kd"):
                    vars_by_key[f"{axis}_{field}"] = tk.StringVar(
                        value=f"{profile[axis + '_pid'][field]:g}"
                    )
            self._performance_profile_entries[mode] = vars_by_key
            mode_frame = ttk.Frame(profile_frame)
            mode_frame.pack(fill=tk.X, pady=(2, 1))
            ttk.Label(
                mode_frame, text=title, width=9,
                font=("Microsoft YaHei UI", 9, "bold"),
                foreground=UI["blue"] if mode == "balance" else UI["orange"],
            ).grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
            scalar_specs = (
                ("水平%", "horizontal", 5),
                ("垂直%", "vertical", 5),
                ("垂补N", "hover", 5),
                ("上升差%", "ascent_trim", 6),
                ("下降差%", "descent_trim", 6),
                ("前进差%", "forward_trim", 6),
                ("后退差%", "backward_trim", 6),
            )
            for column, (label, key, width) in enumerate(scalar_specs, start=1):
                cell = ttk.Frame(mode_frame)
                cell.grid(row=0, column=column, padx=(3, 2), sticky=tk.W)
                ttk.Label(cell, text=label).pack(side=tk.LEFT, padx=(0, 2))
                entry = ttk.Entry(
                    cell, textvariable=vars_by_key[key], width=width
                )
                entry.pack(side=tk.LEFT)
                entry.bind(
                    "<Return>",
                    lambda event, selected=mode: self._apply_performance_profile_from_entries(
                        selected, event
                    ),
                )
                self._bind_performance_profile_entry(entry, mode)
            ttk.Label(
                mode_frame, text="横移补偿", width=9,
                foreground=UI["muted"],
            ).grid(row=1, column=0, padx=(0, 5), sticky=tk.W)
            lateral_specs = (
                ("左移前后%", "left_lateral"),
                ("右移前后%", "right_lateral"),
            )
            for column, (label, key) in enumerate(lateral_specs, start=1):
                cell = ttk.Frame(mode_frame)
                cell.grid(row=1, column=column, padx=(3, 2), sticky=tk.W)
                ttk.Label(cell, text=label).pack(side=tk.LEFT, padx=(0, 2))
                entry = ttk.Entry(
                    cell, textvariable=vars_by_key[key], width=6
                )
                entry.pack(side=tk.LEFT)
                entry.bind(
                    "<Return>",
                    lambda event, selected=mode: self._apply_performance_profile_from_entries(
                        selected, event
                    ),
                )
                self._bind_performance_profile_entry(entry, mode)
            ttk.Label(mode_frame, text="负值=后退", foreground=UI["muted"]).grid(
                row=1, column=3, padx=(8, 0), sticky=tk.W
            )
            for row_index, (axis, axis_title) in enumerate((
                ("roll", "横滚"), ("pitch", "俯仰"), ("yaw", "偏转")
            ), start=2):
                row = ttk.Frame(mode_frame)
                row.grid(
                    row=row_index, column=0, columnspan=len(scalar_specs) + 1,
                    sticky=tk.W, pady=(1, 0),
                )
                ttk.Label(row, text=axis_title, width=9).pack(side=tk.LEFT)
                ttk.Label(row, text="力矩Nm").pack(side=tk.LEFT, padx=(3, 2))
                moment_entry = ttk.Entry(
                    row, textvariable=vars_by_key[f"{axis}_moment"], width=6
                )
                moment_entry.pack(side=tk.LEFT, padx=(0, 8))
                moment_entry.bind(
                    "<Return>",
                    lambda event, selected=mode: self._apply_performance_profile_from_entries(
                        selected, event
                    ),
                )
                self._bind_performance_profile_entry(moment_entry, mode)
                for field in ("kp", "ki", "kd"):
                    ttk.Label(row, text=field.capitalize()).pack(
                        side=tk.LEFT, padx=(5, 2)
                    )
                    entry = ttk.Entry(
                        row, textvariable=vars_by_key[f"{axis}_{field}"], width=6
                    )
                    entry.pack(side=tk.LEFT)
                    entry.bind(
                        "<Return>",
                        lambda event, selected=mode: self._apply_performance_profile_from_entries(
                            selected, event
                        ),
                    )
                    self._bind_performance_profile_entry(entry, mode)

        self._performance_status_var = tk.StringVar(value="LB模式：等待 Pi 参数回读")
        ttk.Label(
            controls, textvariable=self._performance_status_var,
            foreground=UI["muted"], font=("Microsoft YaHei UI", 8),
        ).pack(fill=tk.X, padx=(2, 0), pady=(0, 2))

        self._attitude_status_var = tk.StringVar(value="姿态自动：等待 Pi 遥测")
        ttk.Label(
            controls,
            textvariable=self._attitude_status_var,
            foreground="#000000",
            font=("Microsoft YaHei UI", 8),
        ).pack(fill=tk.X, padx=(2, 0), pady=(0, 1))

        row_mapping = ttk.Frame(controls)
        row_mapping.pack(fill=tk.X, pady=(1, 2))
        self._attitude_mapping_btn = tk.Button(
            row_mapping,
            text="姿态安全门：等待 Pi 遥测",
            command=self._toggle_attitude_mapping_gate,
            width=28,
            fg="#000000",
            bg="#FFFFFF",
            activeforeground="#000000",
            activebackground="#FFFFFF",
            disabledforeground="#000000",
            relief=tk.SOLID,
            borderwidth=1,
            state=tk.DISABLED,
        )
        self._attitude_mapping_btn.pack(side=tk.LEFT)
        ttk.Label(
            row_mapping,
            text="标定/置零后自动关闭；确认正反馈方向无误再验收",
            foreground=UI["amber"],
            font=("Microsoft YaHei UI", 8),
        ).pack(side=tk.LEFT, padx=(10, 0))

        # ----- 定深 PID -----
        row_d = ttk.Frame(controls)
        row_d.pack(fill=tk.X)
        ttk.Label(
            row_d, text="定深 PID", font=("Microsoft YaHei UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(row_d, text="Kp").pack(side=tk.LEFT, padx=(0, 6))
        self._d_kp = tk.StringVar(value="120.0")
        d_kp_entry = ttk.Entry(row_d, textvariable=self._d_kp, width=7)
        d_kp_entry.pack(side=tk.LEFT)
        ttk.Label(row_d, text="Ki").pack(side=tk.LEFT, padx=(10, 6))
        self._d_ki = tk.StringVar(value="10.0")
        d_ki_entry = ttk.Entry(row_d, textvariable=self._d_ki, width=7)
        d_ki_entry.pack(side=tk.LEFT)
        ttk.Label(row_d, text="Kd").pack(side=tk.LEFT, padx=(10, 6))
        self._d_kd = tk.StringVar(value="0.0")
        d_kd_entry = ttk.Entry(row_d, textvariable=self._d_kd, width=7)
        d_kd_entry.pack(side=tk.LEFT)
        for entry in (d_kp_entry, d_ki_entry, d_kd_entry):
            entry.bind("<Return>", lambda event: self._apply_pid_entry(1, event))
        self._depth_status_var = tk.StringVar(value="深度闭环待机")
        ttk.Label(
            row_d, textvariable=self._depth_status_var, foreground=UI["muted"]
        ).pack(side=tk.LEFT, padx=(12, 0))

        # ----- 实验性速度闭环（下位机 config.ini 开关默认关闭） -----
        row_s = ttk.Frame(controls)
        row_s.pack(fill=tk.X)
        self._speed_en = tk.BooleanVar(value=False)
        ttk.Checkbutton(row_s, text="速度闭环（实验）", variable=self._speed_en).pack(side=tk.LEFT)
        ttk.Label(row_s, text="前向速度 m/s").pack(side=tk.LEFT, padx=(16, 6))
        self._tgt_s_var = tk.StringVar(value="0.0")
        ttk.Entry(row_s, textvariable=self._tgt_s_var, width=7).pack(side=tk.LEFT)
        ttk.Label(
            row_s,
            text="需同时开启 Pi 端 speed_closed_loop",
            foreground=UI["amber"],
        ).pack(side=tk.LEFT, padx=10)

    def _confirmed_armed_telemetry(self):
        """Return fresh Pi telemetry only when propulsion is truly unlocked."""
        tele = self.client.get_telemetry()
        if not tele:
            return None
        if not bool(tele.get("control_link_ok", False)):
            return None
        if not bool(tele.get("armed", False)):
            return None
        if bool(tele.get("rearm_required", True)):
            return None
        if bool(tele.get("emergency_stop", False)):
            return None
        if self._emergency_stop_latched or self._estop_pending_target is not None:
            return None
        return tele

    def _wait_for_pid_ack(self, requested_pid, baseline_marker, timeout=1.0,
                          success_message="pid_applied"):
        """Require a fresh, matching Pi PID readback after apply or save."""
        deadline = time.monotonic() + timeout
        prefix_by_type = {
            1: "depth", 2: "heading", 3: "speed", 4: "roll", 5: "pitch",
            6: "yaw",
        }
        prefix = prefix_by_type[int(requested_pid["type"])]
        while time.monotonic() < deadline:
            tele = self.client.get_telemetry()
            marker = tele.get("_rx_marker") if tele else None
            if tele and marker is not None and marker != baseline_marker:
                ack = tele.get("command_ack", {})
                if ack and not bool(ack.get("ok", False)):
                    raise OSError(
                        f"Pi 拒绝 PID 更新：{ack.get('error') or ack.get('message') or '未知错误'}"
                    )
                if ack.get("message") == success_message:
                    actual = ack.get("pid", {}).get(prefix, {})
                    matches = all(
                        math.isclose(
                            float(actual.get(field)), float(requested_pid[field]),
                            rel_tol=1e-9, abs_tol=1e-9,
                        )
                        for field in ("kp", "ki", "kd")
                    )
                    if not matches:
                        raise OSError(f"Pi {prefix} PID 回读与界面输入不一致")
                    return
            time.sleep(0.01)
        raise OSError("Pi PID 更新确认超时")

    def _wait_for_motion_tuning_ack(self, requested, baseline_marker, timeout=1.5):
        """Require fresh Pi telemetry proving runtime apply and disk persistence."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            tele = self.client.get_telemetry()
            marker = tele.get("_rx_marker") if tele else None
            if tele and marker is not None and marker != baseline_marker:
                ack = tele.get("command_ack", {})
                if ack and not bool(ack.get("ok", False)):
                    raise OSError(
                        f"Pi 拒绝推力设置：{ack.get('error') or ack.get('message') or '未知错误'}"
                    )
                if ack.get("message") == "motion_tuning_saved":
                    actual = ack.get("motion_tuning", {})
                    if all(
                        math.isclose(
                            float(actual.get(field)), float(requested[field]),
                            rel_tol=1e-9, abs_tol=1e-9,
                        )
                        for field in requested
                    ):
                        return dict(actual)
            time.sleep(0.01)
        raise OSError("Pi 推力设置保存确认超时")

    def _wait_for_attitude_tuning_ack(
        self, requested, baseline_marker, baseline_ack=None, timeout=1.5
    ):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            tele = self.client.get_telemetry()
            marker = tele.get("_rx_marker") if tele else None
            if tele and marker is not None and marker != baseline_marker:
                ack = tele.get("command_ack", {})
                ack_changed = bool(ack) and ack != (baseline_ack or {})
                if ack_changed and not bool(ack.get("ok", False)):
                    raise OSError(
                        f"Pi 拒绝姿态限幅设置：{ack.get('error') or ack.get('message') or '未知错误'}"
                    )
                if ack_changed and ack.get("message") == "attitude_tuning_saved":
                    actual = ack.get("attitude_tuning", {})
                    if all(math.isclose(
                        float(actual.get(field)), float(requested[field]),
                        rel_tol=1e-9, abs_tol=1e-9,
                    ) for field in requested):
                        return dict(actual)
            time.sleep(0.01)
        raise OSError("Pi 姿态限幅保存确认超时")

    def _wait_for_attitude_hold_ack(
        self, requested, baseline_marker, baseline_ack=None, timeout=1.0
    ):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            tele = self.client.get_telemetry()
            marker = tele.get("_rx_marker") if tele else None
            if tele and marker is not None and marker != baseline_marker:
                ack = tele.get("command_ack", {})
                ack_changed = bool(ack) and ack != (baseline_ack or {})
                if ack_changed and not bool(ack.get("ok", False)):
                    raise OSError(
                        f"Pi 拒绝姿态自动调整：{ack.get('error') or ack.get('message') or '未知错误'}"
                    )
                hold = tele.get("attitude_hold", {})
                if ack_changed and ack.get("message") == "attitude_hold_updated" and all(
                    bool(hold.get(axis, False)) == bool(requested[axis])
                    for axis in requested
                ):
                    return dict(hold)
            time.sleep(0.01)
        raise OSError("Pi 姿态自动调整确认超时")

    def _wait_for_attitude_mapping_ack(
        self, requested, baseline_marker, baseline_ack=None, timeout=1.5
    ):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            tele = self.client.get_telemetry()
            marker = tele.get("_rx_marker") if tele else None
            if tele and marker is not None and marker != baseline_marker:
                ack = tele.get("command_ack", {})
                ack_changed = bool(ack) and ack != (baseline_ack or {})
                if ack_changed and not bool(ack.get("ok", False)):
                    raise OSError(
                        "Pi 拒绝切换姿态安全门："
                        f"{ack.get('error') or ack.get('message') or '未知错误'}"
                    )
                hold = tele.get("attitude_hold", {})
                if (
                    ack_changed
                    and ack.get("message") == "attitude_mapping_updated"
                    and bool(hold.get("mapping_validated", False))
                    == bool(requested)
                ):
                    return bool(requested)
            time.sleep(0.01)
        raise OSError("Pi 姿态安全门切换确认超时")

    def _wait_for_autonomous_enabled(self, baseline_marker, timeout=1.0):
        """Require fresh Pi telemetry proving autonomous output is enabled."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            tele = self.client.get_telemetry()
            marker = tele.get("_rx_marker") if tele else None
            if tele and marker is not None and marker != baseline_marker:
                ack = tele.get("command_ack", {})
                if ack and not bool(ack.get("ok", False)):
                    raise OSError(
                        f"Pi 拒绝自主启动：{ack.get('error') or ack.get('message') or '未知错误'}"
                    )
                if bool(tele.get("cmd_enabled", False)):
                    return
            time.sleep(0.01)
        raise OSError("Pi 未确认自主启动，已回到停止状态")

    def _cmd_start(self):
        """启动自主控制 (定向/定深)"""
        if self._emergency_stop_latched:
            messagebox.showwarning("急停锁存", "请先复位急停。")
            return
        tele = self._confirmed_armed_telemetry()
        if tele is None:
            messagebox.showwarning(
                "推进器未解锁",
                "Pi 尚未确认推进器解锁。请保持手柄回中，先点击“推进器锁定”等待按钮变绿。",
            )
            return

        # 构建模式码
        mode = {
            "heading_hold": False,
            "depth_hold": self._depth_en.get(),
            "speed_hold": self._speed_en.get(),
        }
        if not any(mode.values()):
            messagebox.showwarning("未选择模式", "请至少勾选一种自主控制模式。")
            return
        pid_list = []
        try:
            depth_target = 0.0
            speed_target = (
                parse_finite_number(self._tgt_s_var.get() or 0, "目标速度")
                if mode["speed_hold"] else 0.0
            )
            if abs(speed_target) > 5.0:
                raise ValueError("目标速度必须在 -5～5 m/s 之间")

            if mode["depth_hold"]:
                depth_request = validate_depth_request(
                    tele,
                    self._tgt_d_var.get(),
                    self._d_kp.get(), self._d_ki.get(), self._d_kd.get(),
                    minimum_depth=self._depth_target_min_m,
                    maximum_depth=self._depth_target_max_m,
                    maximum_step=self._depth_target_max_step_m,
                )
                depth_target = depth_request["target_depth"]
                gains = depth_request["pid"]
                pid_list.append(dict(zip(("type", "kp", "ki", "kd"), (1, *gains))))

            target = {
                "heading_deg": 0.0,
                "depth_m": depth_target,
                "speed_mps": speed_target,
            }

            # Stop the previous autonomous request first. Send validated gains
            # before the final enable packet. Each update must be acknowledged
            # and read back by the Pi, so a failed write can never fall through
            # into an autonomous start with stale PID values.
            self.client.send_cmd(enable=False)
            for pid in pid_list:
                baseline_marker = self.client.get_telemetry_marker()
                self.client.send_cmd(
                    mode=mode, target=target, pid_set=pid,
                    save_data=self._save_var.get(),
                )
                self._wait_for_pid_ack(pid, baseline_marker)
                self._record_applied_pid(pid["type"], (pid["kp"], pid["ki"], pid["kd"]))
            baseline_marker = self.client.get_telemetry_marker()
            self.client.send_cmd(
                enable=True, mode=mode, target=target,
                save_data=self._save_var.get(),
            )
            try:
                self._wait_for_autonomous_enabled(baseline_marker)
            except OSError:
                self.client.send_cmd(enable=False)
                raise
        except (OSError, ValueError) as exc:
            try:
                self.client.send_cmd(enable=False)
            except OSError:
                pass
            self._log_event(f"自主启动被拒绝：{exc}")
            messagebox.showwarning("自主启动被拒绝", str(exc))
            return

        self._log_event(f"自主启动: mode={mode} target={target}")
        print(f"[GUI启动] mode={mode} target={target}")

    def _servo_target_is_confirmed(self, position, immediate, baseline_marker):
        telemetry = self.client.get_telemetry() or {}
        if not telemetry_is_after(telemetry.get("_rx_marker"), baseline_marker):
            return False
        servo = telemetry.get("servo", {})
        ack = telemetry.get("servo_ack", {})
        try:
            confirmed = (
                bool(ack.get("ok", False))
                and ack.get("message") in (
                    "servo_target_updated", "servo_position_written"
                )
                and math.isclose(
                    float(ack.get("servo_target_position")),
                    float(position),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                and math.isclose(
                    float(servo.get("target_position")),
                    float(position),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            )
        except (TypeError, ValueError):
            return False
        if not confirmed:
            return False
        if immediate:
            try:
                return math.isclose(
                    float(servo.get("position")), float(position),
                    rel_tol=0.0, abs_tol=1e-6
                )
            except (TypeError, ValueError):
                return False
        return True

    def _retry_servo_target(self, position, immediate, generation, baseline_marker):
        if generation != self._servo_command_generation:
            return
        if self._emergency_stop_latched:
            return
        if self._servo_target_is_confirmed(position, immediate, baseline_marker):
            return
        try:
            self.client.send_cmd(
                servo_position=position,
                servo_min_position=self._gripper_closed_position,
                servo_immediate=immediate,
            )
        except OSError:
            self._servo_status_var.set("夹爪指令补发失败")

    def _send_servo_target(self, position, *, immediate=False):
        """Send once, then perform only two ACK-gated loss-recovery retries."""
        self._servo_command_generation += 1
        generation = self._servo_command_generation
        baseline_marker = self.client.get_telemetry_marker()
        self.client.send_cmd(
            servo_position=position,
            servo_min_position=self._gripper_closed_position,
            servo_immediate=immediate,
        )
        for delay_ms in SERVO_ACK_RETRY_DELAYS_MS:
            self.root.after(
                delay_ms,
                lambda p=position, i=immediate, g=generation, marker=baseline_marker: (
                    self._retry_servo_target(p, i, g, marker)
                ),
            )

    def _apply_gripper_positions(self, _event=None):
        """Apply RB targets and make the close angle the live software minimum."""
        try:
            open_angle = float(
                str(self._gripper_open_angle_var.get()).strip().rstrip("°")
            )
            closed_angle = float(
                str(self._gripper_closed_angle_var.get()).strip().rstrip("°")
            )
            if not math.isfinite(open_angle) or not math.isfinite(closed_angle):
                raise ValueError("RB开/关角度必须是有限数字")
            if not self._servo_angle_min <= open_angle <= self._servo_angle_max:
                raise ValueError(
                    f"RB开角度必须在 {self._servo_angle_min:g}～"
                    f"{self._servo_angle_max:g}° 之间"
                )
            if not self._servo_angle_min <= closed_angle <= self._servo_angle_max:
                raise ValueError(
                    f"RB关角度必须在 {self._servo_angle_min:g}～"
                    f"{self._servo_angle_max:g}° 之间"
                )
            if closed_angle >= open_angle:
                raise ValueError("RB关的闭合角度必须小于RB开的打开角度")
        except (TypeError, ValueError) as exc:
            self._servo_status_var.set(f"RB开/关输入无效：{exc}")
            return "break"

        self._gripper_open_position = servo_angle_to_position(
            open_angle, self._servo_angle_min, self._servo_angle_max
        )
        self._gripper_closed_position = servo_angle_to_position(
            closed_angle, self._servo_angle_min, self._servo_angle_max
        )
        try:
            update_config_values(
                APP_CONFIG.config_path,
                "servo",
                {
                    "open_position": self._gripper_open_position,
                    "closed_position": self._gripper_closed_position,
                    "failsafe_position": self._gripper_closed_position,
                },
            )
        except OSError as exc:
            self._servo_status_var.set(f"RB参数已应用，但本机保存失败：{exc}")
            return "break"
        APP_CONFIG.set("servo", "open_position", str(self._gripper_open_position))
        APP_CONFIG.set("servo", "closed_position", str(self._gripper_closed_position))
        APP_CONFIG.set("servo", "failsafe_position", str(self._gripper_closed_position))
        self._servo_lower_angle_var.set(f"{closed_angle:g}°")
        current = max(
            self._gripper_closed_position,
            min(1.0, float(self._servo_position_var.get())),
        )
        self._servo_position_var.set(current)
        self._servo_angle_var.set(
            f"{servo_position_to_angle(current, self._servo_angle_min, self._servo_angle_max):.1f}°"
        )
        try:
            self._send_servo_target(current, immediate=False)
        except OSError:
            self._servo_status_var.set("RB参数已保存，但下限同步到 Pi 失败")
            return "break"
        self._servo_status_var.set(
            f"下限/RB关 {closed_angle:.1f}°，RB开 {open_angle:.1f}°，上限 180°"
        )
        self._finish_parameter_entry(_event)
        return "break"

    def _toggle_gripper(self):
        """Toggle the gripper once for each rising edge of the configured button."""
        if self._emergency_stop_latched:
            self._servo_status_var.set("急停锁存：夹爪指令已忽略")
            return

        closing = self._gripper_close_on_next_press
        position = (
            self._gripper_closed_position
            if closing
            else self._gripper_open_position
        )
        self._servo_position_var.set(position)
        try:
            self._send_servo_target(position, immediate=True)
        except OSError:
            self._servo_status_var.set("夹爪指令发送失败")
            return

        action = "合拢" if closing else "打开"
        self._gripper_close_on_next_press = not closing
        angle = servo_position_to_angle(
            position, self._servo_angle_min, self._servo_angle_max
        )
        self._servo_angle_var.set(f"{angle:.1f}°")
        self._servo_status_var.set(f"手柄夹爪：已发送{action}指令")
        self._log_event(f"手柄夹爪{action}: position={position:+.2f}")

    def _apply_gripper_jog_step(self):
        """Apply one slow, bounded L3/R3 step on Tk's main thread."""
        direction = self._gripper_jog_direction
        if direction == 0 or self._emergency_stop_latched:
            return
        current = float(self._servo_position_var.get())
        position = step_gripper_jog_position(
            current,
            direction,
            self._gripper_closed_position,
            self._gripper_jog_open_position,
            self._gripper_jog_step,
        )
        if math.isclose(position, current, rel_tol=0.0, abs_tol=1e-9):
            return
        self._servo_position_var.set(position)
        # Fine movement must never rewrite RB's last completed open/close state.
        # Only a successful RB toggle advances the next full action.
        action = "R3 平缓打开" if direction > 0 else "L3 平缓合拢"
        angle = servo_position_to_angle(
            position, self._servo_angle_min, self._servo_angle_max
        )
        self._servo_angle_var.set(f"{angle:.1f}°")
        try:
            self._send_servo_target(position, immediate=False)
        except OSError:
            self._servo_status_var.set(f"{action}发送失败")
            return
        self._servo_status_var.set(f"{action}：{angle:.1f}°")

    def _center_servo_ui(self):
        self._servo_position_var.set(self._gripper_closed_position)
        angle = servo_position_to_angle(
            self._gripper_closed_position,
            self._servo_angle_min,
            self._servo_angle_max,
        )
        self._servo_angle_var.set(f"{angle:.1f}°")
        # The safety/start position is closed, so the next deliberate press opens.
        self._gripper_close_on_next_press = False

    def _sample_gripper_button(self, *, allow_toggle):
        """Track RB state and queue only a permitted released-to-pressed edge."""
        button_down = self._physical_joy.is_button_pressed(
            self._gripper_toggle_button
        )
        if allow_toggle and button_down and not self._gripper_button_down:
            self._gripper_toggle_requests.put(None)
        self._gripper_button_down = button_down

    def _sample_performance_mode_button(self, *, allow_toggle):
        button_down = self._physical_joy.is_button_pressed(
            self._PERFORMANCE_MODE_BUTTON
        )
        if allow_toggle and button_down and not self._performance_button_down:
            self._performance_mode_requests.put(None)
        self._performance_button_down = button_down

    def _sample_gripper_jog(self, *, allow_jog):
        """Track held stick-click buttons; both pressed or inhibited means stop."""
        open_down = self._physical_joy.is_button_pressed(
            self._GRIPPER_OPEN_JOG_BUTTON
        )
        close_down = self._physical_joy.is_button_pressed(
            self._GRIPPER_CLOSE_JOG_BUTTON
        )
        self._gripper_jog_direction = (
            int(open_down) - int(close_down) if allow_jog else 0
        )

    def _sample_face_buttons(self, *, allow_actions):
        """Queue each DualSense face-button action only on a press edge."""
        for button, action in self._FACE_BUTTON_ACTIONS.items():
            button_down = self._physical_joy.is_button_pressed(button)
            if (
                allow_actions
                and button_down
                and not self._face_button_down[button]
            ):
                self._face_button_requests.put(action)
            self._face_button_down[button] = button_down

    def _sample_dpad_attitude(self, *, allow_actions):
        """Queue one D-pad action per press; D-pad never modifies force axes."""
        action = self._dpad_attitude.update(
            self._physical_joy.read_dpad(), allow_trigger=allow_actions
        )
        if action is not None:
            self._face_button_requests.put(action)

    def _handle_face_button_action(self, action):
        """Run a queued face-button action on Tk's main thread."""
        if action == "timer_toggle":
            if hasattr(self, "_timer"):
                self._timer._toggle_run()
            return

        if action == "video_toggle":
            if hasattr(self, "_video_view"):
                self._video_view.toggle_expanded_view()
            return

        if action == "attitude_zero":
            self._calibrate_attitude_zero()
            return

        if action == "surface_zero":
            self._calibrate_depth_zero()
            return

        if action == "estop":
            # A controller press can make the system safe, never reset it.
            if not self._emergency_stop_latched:
                self._toggle_emergency_stop()
            return

        if self._emergency_stop_latched:
            return

        if action == "pitch_hold_toggle":
            self._toggle_attitude_hold("pitch")
        elif action == "roll_hold_toggle":
            self._toggle_attitude_hold("roll")
        elif action == "yaw_hold_toggle":
            self._toggle_attitude_hold("yaw")
        elif action == "arm":
            self._toggle_arm()
        elif action == "depth_toggle":
            self._toggle_depth_hold()
        elif action == "performance_mode_toggle":
            self._toggle_performance_mode()

    def _toggle_attitude_hold(self, axis):
        """Request one Pi-authoritative zero-angle attitude-loop toggle."""
        if axis not in ("roll", "pitch", "yaw"):
            raise ValueError("unknown attitude axis")
        name = {"roll": "横滚", "pitch": "俯仰", "yaw": "偏转"}[axis]
        tele = self.client.get_telemetry() or {}
        hold = tele.get("attitude_hold", {})
        requested = {
            "roll": bool(hold.get("roll", False)),
            "pitch": bool(hold.get("pitch", False)),
            "yaw": bool(hold.get("yaw", False)),
        }
        requested[axis] = not requested[axis]
        if requested[axis]:
            blocked = attitude_hold_enable_block_reason(tele)
            if blocked:
                message = f"{name}锁定未开启：{blocked}"
                status = getattr(self, "_attitude_status_var", None)
                if status is not None:
                    status.set(message)
                self._log_event(message)
                return
        try:
            baseline_marker = self.client.get_telemetry_marker()
            baseline_ack = dict(tele.get("command_ack", {}))
            self.client.send_cmd(attitude_hold=requested)
            actual = self._wait_for_attitude_hold_ack(
                requested, baseline_marker, baseline_ack=baseline_ack
            )
        except (OSError, ValueError) as exc:
            reason = attitude_hold_rejection_text(exc)
            message = f"{name}锁定切换失败：{reason}"
            self._set_parameter_status(message)
            status = getattr(self, "_attitude_status_var", None)
            if status is not None:
                status.set(message)
            latest = self.client.get_telemetry() or {}
            ack = latest.get("command_ack", {})
            ack_error = str(ack.get("error") or "") if isinstance(ack, dict) else ""
            if "attitude_hold_rejected_" in ack_error:
                self._last_attitude_hold_rejection_ack = (
                    ack.get("timestamp"), ack.get("message"), ack_error
                )
            self._log_event(message)
            return
        state = "开" if actual[axis] else "关"
        self._set_parameter_status(f"{name}自动调整：{state}（Pi 已确认，目标 0°）")
        self._log_event(f"{name}自动调整已{state}；目标为标定后的 0°")

    def _toggle_attitude_mapping_gate(self):
        """Toggle the Pi-persisted acceptance gate, never a local-only flag."""
        tele = self.client.get_telemetry() or {}
        hold = tele.get("attitude_hold")
        if not isinstance(hold, dict) or "mapping_validated" not in hold:
            self._attitude_status_var.set("姿态安全门切换失败：Pi 状态未知")
            self._log_event("姿态安全门切换失败：没有新鲜的 Pi 状态")
            return
        requested = not bool(hold["mapping_validated"])
        if requested:
            health = tele.get("sensor_health", {})
            if tele.get("armed", False):
                reason = "请先锁定推进器"
            elif tele.get("cmd_enabled", False):
                reason = "请先停止自主控制"
            elif any(
                bool(hold.get(axis, False))
                for axis in ("roll", "pitch", "yaw")
            ):
                reason = "请先关闭横滚/俯仰/偏转锁定"
            elif not isinstance(health, dict) or not health.get("imu_valid", False):
                reason = "IMU 数据无效或超时"
            elif not tele.get("attitude_3d_mount_calibrated", False):
                reason = "IMU 三维安装方向尚未标定"
            else:
                reason = None
            if reason:
                message = f"姿态安全门未开启：{reason}"
                self._attitude_status_var.set(message)
                self._log_event(message)
                return
            if not messagebox.askyesno(
                "验收姿态安全门",
                "确认已检查：右侧向下时 Roll 为正、机头上仰时 Pitch 为正，"
                "右转时 Yaw 为正，且垂推/斜推的通道和出力方向正确。\n\n"
                "开启后，十字键↑/→/←可以启动真实姿态闭环输出。",
            ):
                return

        try:
            baseline_marker = self.client.get_telemetry_marker()
            baseline_ack = dict(tele.get("command_ack", {}))
            self.client.send_cmd(attitude_mapping_validated=requested)
            actual = self._wait_for_attitude_mapping_ack(
                requested, baseline_marker, baseline_ack=baseline_ack
            )
        except (OSError, ValueError) as exc:
            message = f"姿态安全门切换失败：{exc}"
            self._attitude_status_var.set(message)
            self._log_event(message)
            return
        state = "已验收" if actual else "已关闭"
        self._attitude_status_var.set(f"姿态安全门：{state}（Pi 已保存）")
        self._log_event(f"姿态安全门{state}；Pi 已持久化")

    def _sync_attitude_calibration_from_pi(self, telemetry):
        """Mirror only Pi-owned posture calibration fields into local config."""
        snapshot = (
            telemetry.get("attitude_calibration_config")
            if isinstance(telemetry, dict) else None
        )
        if snapshot is None:
            return False
        try:
            sections = validate_attitude_calibration_snapshot(snapshot)
        except ValueError as exc:
            error = str(exc)
            if error != self._last_attitude_calibration_sync_error:
                self._last_attitude_calibration_sync_error = error
                self._log_event(f"拒绝同步 Pi 姿态标定配置：{error}")
            return False

        fingerprint = tuple(
            (section, key, value)
            for section, values in sections.items()
            for key, value in sorted(values.items())
        )
        if fingerprint == self._last_attitude_calibration_sync:
            return False

        changed = any(
            APP_CONFIG.get(section, key, fallback=None) != value
            for section, values in sections.items()
            for key, value in values.items()
        )
        if changed:
            update_config_sections(APP_CONFIG.config_path, sections)
            for section, values in sections.items():
                for key, value in values.items():
                    APP_CONFIG.set(section, key, value)
            self._log_event("已将 Pi 姿态标定和安全门状态同步到本地配置")
        self._last_attitude_calibration_sync = fingerprint
        self._last_attitude_calibration_sync_error = None
        return changed

    @staticmethod
    def _wrap_angle_180(angle):
        value = float(angle)
        if not math.isfinite(value):
            raise ValueError("yaw must be finite")
        return (value + 180.0) % 360.0 - 180.0

    def _vehicle_imu(self, imu):
        """Prefer Pi-authoritative control heading, with legacy fallback."""
        def angle(name):
            try:
                value = float(imu.get(name, 0.0))
            except (AttributeError, TypeError, ValueError):
                return 0.0
            return value if math.isfinite(value) else 0.0

        roll = self._wrap_angle_180(angle("roll"))
        pitch = self._wrap_angle_180(angle("pitch"))
        yaw = angle("yaw_control") if "yaw_control" in imu else angle("yaw")
        heading = angle("heading")
        # The Pi owns roll/pitch/yaw mounting signs and persistent zeroing, so
        # the GUI never applies a second offset after reconnecting. Heading is
        # intentionally independent: it remains the magnetic 0..360 course.
        return {
            "roll": roll,
            "pitch": pitch,
            "yaw": self._wrap_angle_180(yaw),
            "heading": (
                angle("heading_vehicle") % 360.0
                if "heading_vehicle" in imu
                else ((-heading if self._imu_heading_inverted else heading)
                      + self._imu_heading_offset_deg) % 360.0
            ),
        }

    def _raw_heading_from_vehicle(self, vehicle_heading):
        """Convert an operator-visible vehicle heading back to Pi raw heading."""
        heading = float(vehicle_heading)
        if not math.isfinite(heading):
            raise ValueError("目标航向必须是有限数字")
        if self._imu_heading_inverted:
            return (self._imu_heading_offset_deg - heading) % 360.0
        return (heading - self._imu_heading_offset_deg) % 360.0

    def _calibrate_attitude_zero(self):
        """Persist the current physical level/forward pose as the 3-D datum."""
        telemetry = self.client.get_telemetry() or {}
        imu = telemetry.get("imu")
        health = telemetry.get("sensor_health", {})
        hold = telemetry.get("attitude_hold")
        if telemetry.get("armed") or telemetry.get("cmd_enabled"):
            self._log_event("姿态置零被拒绝：请先锁定推进器并停止自动控制")
            return
        if not isinstance(imu, dict) or not health.get("imu_valid", False):
            self._log_event("姿态置零失败：没有新鲜的 Pi IMU 遥测")
            return
        if not telemetry.get("attitude_3d_mount_calibrated", False):
            self._log_event("姿态置零被拒绝：请先完成 IMU 三维安装标定")
            return
        if not isinstance(hold, dict) or any(
            bool(hold.get(axis, False)) for axis in ("roll", "pitch")
        ):
            self._log_event("姿态置零被拒绝：必须先关闭横滚/俯仰姿态锁定")
            return
        if not messagebox.askyesno(
            "Options 姿态置零",
            "请确认 ROV 已物理水平放稳，机头指向你要定义的正前方。\n\n"
            "这会把当前姿态定义为横滚/俯仰/偏航 0° 并永久保存。"
            "机体倾斜时禁止执行。",
        ):
            return
        self.client.send_cmd(attitude_zero_calibrate=True)
        self._log_event("已请求 Pi 全姿态置零；当前物理水平/正前为新基准，等待成功回执")

    def _mount_calibration_is_safe(self):
        telemetry = self.client.get_telemetry() or {}
        health = telemetry.get("sensor_health", {})
        hold = telemetry.get("attitude_hold")
        if telemetry.get("armed") or telemetry.get("cmd_enabled"):
            self._log_event("三维安装标定被拒绝：请先锁定推进器并停止自动控制")
            return False
        if not isinstance(telemetry.get("imu"), dict) or not health.get("imu_valid", False):
            self._log_event("三维安装标定失败：没有新鲜的 Pi IMU 遥测")
            return False
        if not isinstance(hold, dict) or any(
            bool(hold.get(axis, False)) for axis in ("roll", "pitch")
        ):
            self._log_event("三维安装标定被拒绝：必须先关闭横滚/俯仰姿态锁定")
            return False
        return True

    def _calibrate_attitude_mount(self):
        """Step 1: record the level ROV reference, without persisting it yet."""
        if not self._mount_calibration_is_safe():
            return
        if not messagebox.askyesno(
            "三维安装标定：步骤 1/4",
            "确认 ROV 已水平放稳、机头朝正前方、推进器锁定、两轴姿态锁定已关闭。\n\n"
            "保持静止至少 1 秒后点击“是”。下一步将让机头正向上仰 20–45°。",
        ):
            return
        self.client.send_cmd(attitude_mount_calibrate=True)
        self._log_event("三维安装标定：已请求采集水平/正前基准；请机头正向上仰 20–45°，稳定 1 秒后点击②")

    def _capture_attitude_mount_pitch(self):
        """Step 2: capture a pure nose-up rotation as the ROV pitch axis."""
        if not self._mount_calibration_is_safe():
            return
        self.client.send_cmd(attitude_mount_capture_pitch=True)
        self._log_event("三维安装标定：已请求上仰轴采集；请先回到水平，再从上方看向右/顺时针偏转 20–45°，稳定 1 秒后点击③")

    def _capture_attitude_mount_yaw(self):
        """Step 3: capture positive yaw while the ROV remains level."""
        if not self._mount_calibration_is_safe():
            return
        self.client.send_cmd(attitude_mount_capture_yaw=True)
        self._log_event("三维安装标定：已请求水平向右偏轴采集；请回到水平/正前基准，稳定 1 秒后点击④完成")

    def _capture_attitude_mount_roll(self):
        """Retain the old wire command for compatibility; it is not in the UI."""
        if not self._mount_calibration_is_safe():
            return
        self.client.send_cmd(attitude_mount_capture_roll=True)
        self._log_event("旧版横滚轴采集已发送；新标定流程请使用水平向右偏采集")

    def _finalize_attitude_mount(self):
        """Step 4: build and persist the complete right-handed mount rotation."""
        if not self._mount_calibration_is_safe():
            return
        self.client.send_cmd(attitude_mount_finalize=True)
        self._log_event("三维安装标定：已请求完成并永久保存；等待 Pi 成功回执")

    def _cmd_stop(self):
        """停止自主控制"""
        self.client.send_cmd(enable=False)

    def _capture_current_depth(self):
        """Copy a fresh Pi depth sample into the target field without enabling."""
        tele = self.client.get_telemetry()
        try:
            current_depth = valid_depth_measurement(tele)
        except ValueError as exc:
            self._log_event(f"读取当前深度失败：{exc}")
            messagebox.showwarning("无法读取当前深度", str(exc))
            return
        self._tgt_d_var.set(f"{current_depth:.3f}")
        self._depth_status_var.set(f"已取当前值 {current_depth:.3f} m（尚未启动）")
        self._log_event(f"目标深度已取当前值：{current_depth:.3f}m")

    def _toggle_depth_hold(self):
        """Toggle depth hold from the Square button without changing other modes."""
        if self._emergency_stop_latched:
            messagebox.showwarning("急停锁存", "请先复位急停。")
            return

        telemetry = self.client.get_telemetry() or {}
        remote_mode = telemetry.get("mode")
        remote_target = telemetry.get("target")
        depth_is_running = (
            bool(telemetry.get("cmd_enabled", False))
            and isinstance(remote_mode, dict)
            and bool(remote_mode.get("depth_hold", False))
        )
        if not depth_is_running:
            self._cmd_hover()
            return

        # Keep other autonomous modes intact; the Pi normalizes target as a
        # complete snapshot, so send all three target fields together.
        mode = {
            "heading_hold": False,
            "depth_hold": False,
            "speed_hold": bool(remote_mode.get("speed_hold", False)),
        }
        target = {
            "heading_deg": float(remote_target.get("heading_deg", 0.0))
            if isinstance(remote_target, dict) else 0.0,
            "depth_m": float(remote_target.get("depth_m", 0.0))
            if isinstance(remote_target, dict) else 0.0,
            "speed_mps": float(remote_target.get("speed_mps", 0.0))
            if isinstance(remote_target, dict) else 0.0,
        }
        try:
            self.client.send_cmd(
                enable=any(mode.values()), mode=mode, target=target,
                save_data=self._save_var.get(),
            )
        except (OSError, TypeError, ValueError) as exc:
            self._log_event(f"手柄关闭定深失败：{exc}")
            messagebox.showwarning("关闭定深失败", str(exc))
            return
        self._depth_en.set(False)
        self._depth_status_var.set("手柄已关闭定深")
        self._log_event("手柄 □：已关闭定深")

    def _track_manual_vertical_motion(self, vertical_force):
        """Queue exactly one target refresh when a physical vertical move ends."""
        try:
            active = abs(float(vertical_force)) > 0.01
        except (TypeError, ValueError):
            active = False
        if self._vertical_manual_active and not active:
            self._vertical_target_update_requests.put(None)
        self._vertical_manual_active = active

    def _reset_manual_vertical_motion(self):
        """Discard an unfinished manual move while output is inhibited."""
        self._vertical_manual_active = False

    def _update_depth_target_after_manual_vertical_move(self):
        """Reflect the Pi-captured release depth without sending a late target."""
        telemetry = self.client.get_telemetry() or {}
        try:
            current_depth = valid_depth_measurement(telemetry)
        except ValueError as exc:
            self._log_event(f"手动上下结束，未更新目标深度：{exc}")
            return

        self._tgt_d_var.set(f"{current_depth:.3f}")
        remote_mode = telemetry.get("mode")
        remote_target = telemetry.get("target")
        depth_is_running = (
            bool(telemetry.get("cmd_enabled", False))
            and isinstance(remote_mode, dict)
            and bool(remote_mode.get("depth_hold", False))
        )
        if not depth_is_running:
            self._depth_status_var.set(
                f"手动上下结束：目标已更新为 {current_depth:.3f} m（定深未运行）"
            )
            self._log_event(f"手动上下结束：目标深度={current_depth:.3f}m")
            return
        self._depth_status_var.set(
            f"手动上下结束：Pi 已捕获定深目标 {current_depth:.3f} m"
        )
        self._log_event(f"手动上下结束：Pi 已捕获定深目标≈{current_depth:.3f}m")

    def _calibrate_depth_zero(self):
        """Use any fresh raw pressure sample as the session depth zero."""
        telemetry = self.client.get_telemetry()
        try:
            if not isinstance(telemetry, dict):
                raise ValueError("遥测未连接，无法读取当前压力")
            health = telemetry.get("sensor_health", {})
            if not isinstance(health, dict) or not health.get("pressure_valid"):
                raise ValueError("深度计原始压力数据已超时，无法置零")
            parse_finite_number(telemetry.get("pressure_mpa"), "深度计原始压力")
        except ValueError as exc:
            self._log_event(f"当前位置置零被拒绝：{exc}")
            messagebox.showwarning("当前位置置零不可用", str(exc))
            return
        if bool(telemetry.get("armed", False)) or bool(telemetry.get("cmd_enabled", False)):
            messagebox.showwarning(
                "请先锁定推进器",
                "当前位置置零只能在推进器锁定、未启动自主控制时执行。",
            )
            return
        try:
            self.client.send_cmd(depth_zero_calibrate=True)
        except OSError as exc:
            self._log_event(f"当前位置置零命令发送失败：{exc}")
            messagebox.showerror("当前位置置零失败", f"命令未送达 Pi：{exc}")
            return
        self._depth_status_var.set("当前位置置零请求已发送，等待 Pi 确认")
        self._log_event("已请求 Pi 以当前压力作为本次深度零点")

    def _cmd_hover(self):
        """Safe one-key hold: capture current depth and enable depth hold only."""
        if self._emergency_stop_latched:
            messagebox.showwarning("急停锁存", "请先复位急停。")
            return
        tele = self._confirmed_armed_telemetry()
        if tele is None:
            messagebox.showwarning(
                "推进器未解锁",
                "Pi 尚未确认推进器解锁，不能启动一键悬停。",
            )
            return
        try:
            current_depth = valid_depth_measurement(tele)
        except ValueError as exc:
            self._log_event(f"一键悬停被拒绝：{exc}")
            messagebox.showwarning("一键悬停被拒绝", str(exc))
            return

        self._tgt_d_var.set(f"{current_depth:.3f}")
        self._depth_en.set(True)
        self._speed_en.set(False)

        self._log_event(f"一键定深: 深度={current_depth:.3f}m")
        self._cmd_start()

    def _set_arm_ui_locked(self):
        self._arm_btn.configure(text="🔒 推进器锁定", bg=UI["red_dark"])
        self._arm_light.configure(text="● 锁定", foreground=UI["red"])
        if hasattr(self, "_start_btn"):
            self._start_btn.configure(state=tk.DISABLED)
        if hasattr(self, "_hover_btn"):
            self._hover_btn.configure(state=tk.DISABLED)

    def _zero_local_joystick(self):
        """清除物理/键盘历史值，避免急停复位后恢复上一帧满舵。"""
        self._joy_keys.clear()
        self._joy.update({
            "Fx": 0.0, "Fy": 0.0, "Fz": 0.0, "N": 0.0, "M": 0.0,
        })
        self._phys_display = (0.0, 0.0, 0.0, 0.0, 0.0)

    def _send_emergency_stop(self):
        """发送一次急停帧；锁存期间由手柄线程周期重发。"""
        if not self._emergency_stop_latched:
            return
        self.client.send_cmd(
            enable=False,
            arm=False,
            emergency_stop=True,
        )
        self.client.send_joy(0, 0, 0, 0, 0)
        self._last_estop_tx = time.monotonic()

    def _send_remote_lock(self):
        """Best-effort zero/lock when the GUI loses its telemetry evidence."""
        self._zero_local_joystick()
        try:
            self.client.send_cmd(enable=False, arm=False)
            self.client.send_joy(0, 0, 0, 0, 0)
        except OSError:
            # Pi link timeout remains the independent fail-closed backstop.
            pass

    def _toggle_emergency_stop(self):
        """
        第一次点击触发锁存急停；再次点击仅复位急停。
        复位后推进器仍锁定，必须单独点击解锁按钮。
        """
        # Cancel every unlock retry created by an earlier click before changing
        # the emergency-stop state in either direction.
        self._arm_request_generation += 1
        if not self._emergency_stop_latched:
            self._emergency_stop_latched = True
            self._local_estop_hold = True
            self._estop_pending_target = True
            self._estop_pending_until = time.monotonic() + 1.0
            self._estop_request_marker = self.client.get_telemetry_marker()
            self._require_joystick_centered = True
            self._zero_local_joystick()
            self._center_servo_ui()
            self._set_arm_ui_locked()
            self._estop_btn.configure(
                text="⚠ 急停已锁存\n点击复位",
                bg=UI["red_dark"],
                height=2,
            )
            self._send_emergency_stop()
            # UDP无确认机制，短时间重复发送以降低单包丢失风险。
            for delay_ms in (50, 100, 150, 200):
                self.root.after(delay_ms, self._send_emergency_stop)
            return

        if not messagebox.askyesno(
            "复位急停",
            "确认复位急停？\n复位后推进器仍保持锁定，需要再次手动解锁。",
        ):
            return

        for _ in range(3):
            self.client.send_cmd(
                enable=False,
                arm=False,
                reset_emergency_stop=True,
            )
        self.client.send_joy(0, 0, 0, 0, 0)
        self._emergency_stop_latched = False
        self._local_estop_hold = False
        self._estop_pending_target = False
        self._estop_pending_until = time.monotonic() + 1.0
        self._estop_request_marker = self.client.get_telemetry_marker()
        self._require_joystick_centered = True
        self._zero_local_joystick()
        self._center_servo_ui()
        self._set_arm_ui_locked()
        self._estop_btn.configure(
            text="⛔ 紧急停止",
            bg=UI["red"],
            height=1,
        )
        messagebox.showinfo(
            "急停已复位",
            "推进器仍处于锁定和1540us中位状态；确认安全后再单独解锁。",
        )

    def _toggle_arm(self):
        """解锁/锁定推进器"""
        if self._emergency_stop_latched:
            messagebox.showwarning("急停锁存", "请先点击红色急停按钮并确认复位。")
            return
        if self._estop_pending_target is not None:
            messagebox.showwarning(
                "等待 Pi 确认",
                "急停状态仍在同步，请保持摇杆回中并等待 Pi 最新遥测确认。",
            )
            return
        if not self._remote_armed:
            # Pi is the authority: require a fresh physical neutral period
            # there. Locally send neutral and wait for telemetry confirmation;
            # never paint an unacknowledged unlock as successful.
            self._require_joystick_centered = True
            self._zero_local_joystick()
            self._arm_request_generation += 1
            generation = self._arm_request_generation
            for delay_ms in (0, 150, 300):
                self.root.after(
                    delay_ms,
                    lambda token=generation: self._send_arm_attempt(token),
                )
            self._arm_btn.configure(
                text="⏳ 等待 Pi 解锁",
                bg=UI["amber"],
            )
            self._arm_light.configure(text="● 等待 Pi 确认", foreground=UI["amber"])
            self._log_event("已请求解锁：Pi 将在摇杆连续回中后确认")
        else:
            self._arm_request_generation += 1
            self._zero_local_joystick()
            self._remote_armed = False
            self.client.send_joy(0, 0, 0, 0, 0)
            self.client.send_cmd(arm=False, enable=False)
            self._set_arm_ui_locked()

    def _send_arm_attempt(self, generation):
        """Send one bounded neutral arm attempt for the latest manual click."""
        if generation != self._arm_request_generation:
            return
        if (
            self._remote_armed
            or self._emergency_stop_latched
            or self._estop_pending_target is not None
        ):
            return
        telemetry = self.client.get_telemetry()
        if not telemetry or not bool(telemetry.get("control_link_ok", False)):
            self._arm_request_generation += 1
            self._set_arm_ui_locked()
            self._arm_light.configure(
                text="● 解锁取消：链路未确认", foreground=UI["red"]
            )
            return
        self.client.send_joy(0, 0, 0, 0, 0)
        self.client.send_cmd(arm=True, enable=False)


    # ================================================================
    #  手柄面板
    # ================================================================
    def _build_joystick_panel(self, parent):
        frame = ttk.LabelFrame(
            parent,
            text="  手柄控制  ",
            style="Card.TLabelframe",
            padding=(10, 1),
        )
        frame.pack(fill=tk.X, pady=(0, 0))

        # --- 物理手柄状态 + 实时发送数据 ---
        bar = ttk.Frame(frame)
        bar.pack(fill=tk.X, pady=(0, 2))

        joy_name = self._physical_joy.name if self._physical_joy.available else "未检测到"
        joy_status = "🟢 已连接" if self._physical_joy.available else "🔴 未检测到"
        self._joy_status_label = ttk.Label(bar,
                                           text=f"物理手柄: {joy_status} ({joy_name})",
                                           font=("Microsoft YaHei UI", 9, "bold"),
                                           foreground=UI["green"] if self._physical_joy.available else UI["muted"])
        self._joy_status_label.pack(side=tk.LEFT, padx=2)
        ttk.Label(
            bar,
            text=(
                f"RB(button {self._gripper_toggle_button}): 夹爪开/合"
                "  |  LB(button 9): 平衡/狂暴"
                "  |  R3按住:平缓打开至最大角度  L3按住:平缓合拢"
                "  |  □: 定深开/关  |  Create(button 4): 水面置零"
            ),
            font=("Microsoft YaHei UI", 8, "bold"),
            foreground=UI["blue"],
        ).pack(side=tk.RIGHT, padx=2)

        # 大字: 实时发送数据
        self._joy_send_var = tk.StringVar(value="等待数据...")
        ttk.Label(
            frame,
            textvariable=self._joy_send_var,
            font=("Consolas", 9, "bold"),
            foreground=UI["orange"],
            wraplength=470,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=2, pady=(0, 2))

        # D-pad actions never generate movement force.
        self._parameter_status_var = tk.StringVar(
            value="十字键↑：俯仰自动 · →：横滚自动 · ←：偏转自动 · ↓：计时开始/暂停/继续"
        )
        ttk.Label(
            frame,
            textvariable=self._parameter_status_var,
            font=("Microsoft YaHei UI", 7),
            foreground=UI["muted"],
        ).pack(fill=tk.X, padx=2, pady=(0, 1))

        # --- 键盘快捷键 ---
        kb = ttk.Frame(frame)
        kb.pack(fill=tk.X, pady=(2, 0))

        ttk.Label(
            kb, text="键盘快捷键", font=("Microsoft YaHei UI", 8, "bold"),
            foreground=UI["muted"]
        ).pack(side=tk.LEFT, padx=(0, 8))
        shortcuts = ("W/S:前后", "A/D:左右", "Q/E:上下", "J/L:偏航", "X:回零")
        for s in shortcuts:
            tk.Label(
                kb,
                text=s,
                bg="#EAF0F6",
                fg=UI["navy_soft"],
                font=("Microsoft YaHei UI", 7),
                padx=8,
                pady=0,
            ).pack(side=tk.LEFT, padx=3)

    def _build_thruster_panel(self, parent):
        frame = ttk.LabelFrame(
            parent,
            text="  推进器实时状态  ",
            style="Card.TLabelframe",
            padding=(8, 5),
        )
        frame.pack(fill=tk.X, expand=False, pady=(0, 5))

        image_path = Path(__file__).resolve().parent / "assets" / "rov_top_view.png"
        # Keep the lower event log readable at 150% Windows DPI.
        self._thruster_diagram = ThrusterDiagram(frame, image_path, display_size=190)
        self._thruster_diagram.pack(anchor=tk.CENTER)
        self._thruster_diagram.set_thrust({}, connected=False)

        ttk.Label(
            frame,
            text="蓝色旋叶转速 ∝ |推力|  ·  正反推旋转方向相反",
            style="Muted.TLabel",
            font=("Microsoft YaHei UI", 8),
        ).pack(pady=(1, 0))

    def _on_key_down(self, event):
        key = event.keysym.lower()
        if not self._keyboard_control_allowed(event) or key not in self._movement_keys:
            return
        self._joy_keys.add(key)
        self._update_joy()

    def _set_parameter_status(self, message):
        status = getattr(self, "_parameter_status_var", None)
        if status is not None:
            status.set(message)

    @staticmethod
    def _finish_parameter_entry(event):
        """Successful Enter commits the value and removes the edit focus ring."""
        if event is None:
            return
        try:
            event.widget.selection_clear()
            focus_target = getattr(event.widget, "master", None)
            (focus_target or event.widget.winfo_toplevel()).focus_set()
        except tk.TclError:
            pass

    def _bind_performance_profile_entry(self, entry, mode):
        """Pause Pi readback painting while one performance profile is edited."""
        for sequence in ("<FocusIn>", "<KeyRelease>", "<<Paste>>", "<<Cut>>"):
            entry.bind(
                sequence,
                lambda event, selected=mode: self._mark_performance_profile_dirty(
                    selected, event
                ),
                add="+",
            )
        entry.bind(
            "<Escape>",
            lambda event, selected=mode: self._cancel_performance_profile_edit(
                selected, event
            ),
            add="+",
        )

    def _mark_performance_profile_dirty(self, mode, event=None):
        if str(getattr(event, "keysym", "")).lower() in {"return", "enter", "escape"}:
            return
        self._performance_profile_dirty[mode] = True

    def _normalize_performance_profile(self, mode, values, fallback=None):
        """Validate a Pi profile and expand an old six-field profile locally."""
        if not isinstance(values, dict):
            raise ValueError("Pi 模式参数不是对象")
        legacy_fields = {
            "mode", "horizontal_limit_fraction", "vertical_limit_fraction",
            "roll_pid", "pitch_pid", "yaw_pid",
        }
        extended_fields = {
            "hover_force_per_thruster_n",
            "vertical_ascent_front_rear_trim_percent",
            "vertical_descent_front_rear_trim_percent",
            "forward_front_rear_trim_percent",
            "backward_front_rear_trim_percent",
            "roll_max_moment_nm", "pitch_max_moment_nm", "yaw_max_moment_nm",
        }
        lateral_fields = {
            "left_lateral_surge_trim_percent",
            "right_lateral_surge_trim_percent",
        }
        if set(values) not in (
            legacy_fields,
            legacy_fields | extended_fields,
            legacy_fields | extended_fields | lateral_fields,
        ):
            raise ValueError("Pi 模式参数字段不完整或未知")
        if values.get("mode") != mode:
            raise ValueError("Pi 模式参数名称不匹配")
        base = dict(fallback or self._performance_profiles.get(mode, {}))
        merged = {**base, **values}
        return validate_performance_profile(
            mode,
            float(merged["horizontal_limit_fraction"]) * 100.0,
            float(merged["vertical_limit_fraction"]) * 100.0,
            {
                axis: tuple(
                    merged[f"{axis}_pid"][field]
                    for field in ("kp", "ki", "kd")
                )
                for axis in ("roll", "pitch", "yaw")
            },
            {field: merged[field] for field in (
                "hover_force_per_thruster_n",
                "vertical_ascent_front_rear_trim_percent",
                "vertical_descent_front_rear_trim_percent",
                "forward_front_rear_trim_percent",
                "backward_front_rear_trim_percent",
                "left_lateral_surge_trim_percent",
                "right_lateral_surge_trim_percent",
            )},
            {field: merged[field] for field in (
                "roll_max_moment_nm", "pitch_max_moment_nm", "yaw_max_moment_nm"
            )},
        )

    def _cancel_performance_profile_edit(self, mode, event=None):
        self._set_performance_profile_entries(
            mode, self._performance_profiles[mode]
        )
        self._performance_profile_dirty[mode] = False
        self._performance_status_var.set(
            f"{('平衡' if mode == 'balance' else '狂暴')}模式编辑已取消"
        )
        self._finish_parameter_entry(event)
        return "break"

    def _apply_performance_profile_from_entries(self, mode, event=None):
        entries = self._performance_profile_entries[mode]
        try:
            requested = validate_performance_profile(
                mode,
                entries["horizontal"].get(),
                entries["vertical"].get(),
                {
                    axis: tuple(
                        entries[f"{axis}_{field}"].get()
                        for field in ("kp", "ki", "kd")
                    )
                    for axis in ("roll", "pitch", "yaw")
                },
                {
                    "hover_force_per_thruster_n": entries["hover"].get(),
                    "vertical_ascent_front_rear_trim_percent": (
                        entries["ascent_trim"].get()
                    ),
                    "vertical_descent_front_rear_trim_percent": (
                        entries["descent_trim"].get()
                    ),
                    "forward_front_rear_trim_percent": entries[
                        "forward_trim"
                    ].get(),
                    "backward_front_rear_trim_percent": entries[
                        "backward_trim"
                    ].get(),
                    "left_lateral_surge_trim_percent": entries[
                        "left_lateral"
                    ].get(),
                    "right_lateral_surge_trim_percent": entries[
                        "right_lateral"
                    ].get(),
                },
                {
                    f"{axis}_max_moment_nm": entries[f"{axis}_moment"].get()
                    for axis in ("roll", "pitch", "yaw")
                },
            )
            baseline_marker = self.client.get_telemetry_marker()
            self.client.send_cmd(performance_profile=requested)
            readback = self._wait_for_performance_profile_ack(
                requested, baseline_marker
            )
            actual = self._normalize_performance_profile(
                mode, readback, fallback=requested
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            self._performance_status_var.set(f"{mode}参数未应用：{exc}")
            return "break" if event is not None else False
        self._performance_profiles[mode] = actual
        self._performance_profile_dirty[mode] = False
        self._set_performance_profile_entries(mode, actual)
        profile_values = {
            "horizontal_limit_fraction": actual["horizontal_limit_fraction"],
            "vertical_limit_fraction": actual["vertical_limit_fraction"],
            **{
                field: actual[field]
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
                f"{axis}_{field}": actual[axis + "_pid"][field]
                for axis in ("roll", "pitch", "yaw")
                for field in ("kp", "ki", "kd")
            },
        }
        update_config_sections(APP_CONFIG.config_path, {f"mode_{mode}": profile_values})
        for key, value in profile_values.items():
            APP_CONFIG.set(f"mode_{mode}", key, str(value))
        self._performance_status_var.set(
            f"{('平衡' if mode == 'balance' else '狂暴')}模式参数已应用 · Ctrl+S 可再次保存"
        )
        self._finish_parameter_entry(event)
        return "break" if event is not None else True

    def _set_performance_profile_entries(self, mode, profile):
        entries = self._performance_profile_entries.get(mode, {})
        if not entries:
            return
        entries["horizontal"].set(f"{float(profile['horizontal_limit_fraction']) * 100.0:.1f}")
        entries["vertical"].set(f"{float(profile['vertical_limit_fraction']) * 100.0:.1f}")
        for entry_key, field in (
            ("hover", "hover_force_per_thruster_n"),
            ("ascent_trim", "vertical_ascent_front_rear_trim_percent"),
            ("descent_trim", "vertical_descent_front_rear_trim_percent"),
            ("forward_trim", "forward_front_rear_trim_percent"),
            ("backward_trim", "backward_front_rear_trim_percent"),
            ("left_lateral", "left_lateral_surge_trim_percent"),
            ("right_lateral", "right_lateral_surge_trim_percent"),
            ("roll_moment", "roll_max_moment_nm"),
            ("pitch_moment", "pitch_max_moment_nm"),
            ("yaw_moment", "yaw_max_moment_nm"),
        ):
            entries[entry_key].set(f"{float(profile[field]):g}")
        for axis in ("roll", "pitch", "yaw"):
            for field in ("kp", "ki", "kd"):
                entries[f"{axis}_{field}"].set(
                    f"{float(profile[axis + '_pid'][field]):g}"
                )

    def _sync_performance_profiles_from_pi(self, remote_profiles):
        """Refresh clean profile fields without overwriting operator input."""
        if not isinstance(remote_profiles, dict):
            return
        for mode in ("balance", "rage"):
            profile = remote_profiles.get(mode)
            if not isinstance(profile, dict):
                continue
            try:
                normalized = self._normalize_performance_profile(mode, profile)
                self._performance_profiles[mode] = normalized
                if not self._performance_profile_dirty.get(mode, False):
                    self._set_performance_profile_entries(mode, normalized)
            except (KeyError, TypeError, ValueError):
                pass

    def _wait_for_performance_profile_ack(self, requested, baseline_marker, timeout=1.5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            telemetry = self.client.get_telemetry()
            marker = telemetry.get("_rx_marker") if telemetry else None
            if telemetry and marker is not None and marker != baseline_marker:
                ack = telemetry.get("command_ack", {})
                if ack and not bool(ack.get("ok", False)):
                    raise OSError(ack.get("error") or ack.get("message") or "Pi 拒绝模式参数")
                if ack.get("message") == "performance_profile_saved":
                    actual = ack.get("performance_profile", {})
                    if actual.get("mode") == requested["mode"]:
                        return dict(actual)
            time.sleep(0.01)
        raise OSError("Pi 模式参数保存确认超时")

    def _toggle_performance_mode(self):
        telemetry = self.client.get_telemetry() or {}
        current = str(telemetry.get("performance_mode", self._performance_mode)).lower()
        requested = "rage" if current == "balance" else "balance"
        try:
            self.client.send_cmd(performance_mode=requested)
        except OSError as exc:
            self._performance_status_var.set(f"LB模式切换失败：{exc}")
            return
        self._performance_mode = requested
        self._performance_mode_var.set(
            "当前：狂暴模式" if requested == "rage" else "当前：平衡模式"
        )
        self._performance_status_var.set(
            "LB切换请求已发送：" + ("狂暴模式" if requested == "rage" else "平衡模式")
        )
        self._log_event(f"LB切换性能模式：{requested}")

    def _pid_values_from_entries(self, pid_type):
        if int(pid_type) == 1:
            return validate_pid_triplet(
                self._d_kp.get(), self._d_ki.get(), self._d_kd.get(),
                label="深度 PID",
            )
        if int(pid_type) == 4:
            return validate_pid_triplet(
                self._r_kp.get(), self._r_ki.get(), self._r_kd.get(),
                label="横滚 PID",
            )
        if int(pid_type) == 5:
            return validate_pid_triplet(
                self._p_kp.get(), self._p_ki.get(), self._p_kd.get(),
                label="俯仰 PID",
            )
        if int(pid_type) == 6:
            return validate_pid_triplet(
                self._y_kp.get(), self._y_ki.get(), self._y_kd.get(),
                label="偏转 PID",
            )
        raise ValueError("不支持的 PID 类型")

    def _record_applied_pid(self, pid_type, values):
        if not hasattr(self, "_applied_pid_values"):
            self._applied_pid_values = {1: None, 4: None, 5: None, 6: None}
        self._applied_pid_values[int(pid_type)] = tuple(map(float, values))

    def _apply_pid_entry(self, pid_type, _event=None):
        """Apply a completed PID edit at runtime; Ctrl+S persists it."""
        try:
            values = self._pid_values_from_entries(pid_type)
            pid = dict(zip(("type", "kp", "ki", "kd"), (int(pid_type), *values)))
            baseline_marker = self.client.get_telemetry_marker()
            self.client.send_cmd(pid_set=pid)
            self._wait_for_pid_ack(pid, baseline_marker, success_message="pid_applied")
            self._record_applied_pid(pid_type, values)
        except (OSError, ValueError) as exc:
            self._set_parameter_status(f"未应用：{exc}")
            return "break"
        name = {1: "深度", 4: "横滚", 5: "俯仰", 6: "偏转"}[int(pid_type)]
        self._set_parameter_status(f"{name} PID 已应用 · Ctrl+S 永久保存")
        self._log_event(f"{name} PID 已临时应用；按 Ctrl+S 可永久保存")
        self._finish_parameter_entry(_event)
        return "break"

    def _apply_motion_tuning_from_entry(self, _event=None):
        """Enter applies all force limits/compensation and persists both ends."""
        try:
            tuning = validate_motion_tuning(
                self._horizontal_limit_var.get(),
                self._vertical_limit_var.get(),
                self._vertical_hover_var.get(),
                self._vertical_ascent_front_rear_trim_var.get(),
                self._vertical_descent_front_rear_trim_var.get(),
                self._forward_front_rear_trim_var.get(),
                self._backward_front_rear_trim_var.get(),
            )
            baseline_marker = self.client.get_telemetry_marker()
            self.client.send_cmd(motion_tuning=tuning)
            actual = self._wait_for_motion_tuning_ack(tuning, baseline_marker)
        except (OSError, ValueError) as exc:
            self._set_motion_tuning_status(f"✗ 未应用：{exc}", ok=False)
            return "break"

        self._motion_tuning = actual
        self._motion_tuning_synced_from_pi = True
        self._horizontal_limit_var.set(
            f"{actual['horizontal_limit_fraction']:.2f}"
        )
        self._vertical_limit_var.set(f"{actual['vertical_limit_fraction']:.2f}")
        self._vertical_hover_var.set(
            f"{actual['hover_force_per_thruster_n']:.2f}"
        )
        self._vertical_ascent_front_rear_trim_var.set(
            actual["vertical_ascent_front_rear_trim_percent"]
        )
        self._vertical_descent_front_rear_trim_var.set(
            actual["vertical_descent_front_rear_trim_percent"]
        )
        self._forward_front_rear_trim_var.set(
            actual["forward_front_rear_trim_percent"]
        )
        self._backward_front_rear_trim_var.set(
            actual["backward_front_rear_trim_percent"]
        )
        try:
            update_config_sections(
                APP_CONFIG.config_path,
                {
                    "thrusters": {
                        "horizontal_limit_fraction": actual[
                            "horizontal_limit_fraction"
                        ],
                        "vertical_limit_fraction": actual[
                            "vertical_limit_fraction"
                        ],
                    },
                    "control": {
                        "vertical_hover_force_per_thruster_n": actual[
                            "hover_force_per_thruster_n"
                        ],
                        "vertical_ascent_front_rear_trim_percent": actual[
                            "vertical_ascent_front_rear_trim_percent"
                        ],
                        "vertical_descent_front_rear_trim_percent": actual[
                            "vertical_descent_front_rear_trim_percent"
                        ],
                        "forward_front_rear_trim_percent": actual[
                            "forward_front_rear_trim_percent"
                        ],
                        "backward_front_rear_trim_percent": actual[
                            "backward_front_rear_trim_percent"
                        ],
                    },
                },
            )
        except OSError as exc:
            self._set_motion_tuning_status(
                f"△ Pi 已确认；本机保存失败：{exc}", ok=False
            )
            self._log_event("推力设置已在 Pi 生效并保存；本机配置保存失败")
            return "break"

        APP_CONFIG.set(
            "thrusters", "horizontal_limit_fraction",
            str(actual["horizontal_limit_fraction"]),
        )
        APP_CONFIG.set(
            "thrusters", "vertical_limit_fraction",
            str(actual["vertical_limit_fraction"]),
        )
        APP_CONFIG.set(
            "control", "vertical_hover_force_per_thruster_n",
            str(actual["hover_force_per_thruster_n"]),
        )
        APP_CONFIG.set(
            "control", "vertical_ascent_front_rear_trim_percent",
            str(actual["vertical_ascent_front_rear_trim_percent"]),
        )
        APP_CONFIG.set(
            "control", "vertical_descent_front_rear_trim_percent",
            str(actual["vertical_descent_front_rear_trim_percent"]),
        )
        APP_CONFIG.set(
            "control", "forward_front_rear_trim_percent",
            str(actual["forward_front_rear_trim_percent"]),
        )
        APP_CONFIG.set(
            "control", "backward_front_rear_trim_percent",
            str(actual["backward_front_rear_trim_percent"]),
        )
        self._set_motion_tuning_status(
            "✓ Pi 已确认：实时生效并永久保存", ok=True
        )
        self._log_event(
            "推力设置已保存：水平={:.2f} 垂直={:.2f} 固定补偿={:.2f}N/台 "
            "上升={:+.1f}% 下降={:+.1f}% 前进={:+.1f}% 后退={:+.1f}%".format(
                actual["horizontal_limit_fraction"],
                actual["vertical_limit_fraction"],
                actual["hover_force_per_thruster_n"],
                actual["vertical_ascent_front_rear_trim_percent"],
                actual["vertical_descent_front_rear_trim_percent"],
                actual["forward_front_rear_trim_percent"],
                actual["backward_front_rear_trim_percent"],
            )
        )
        # A successful Enter visibly completes the edit: the blue focus ring
        # dims, while the green Pi-ACK line remains the authoritative result.
        self.root.focus_set()
        return "break"

    def _set_motion_tuning_status(self, message, ok=None):
        """Make the Pi ACK result unambiguous instead of relying on focus color."""
        self._motion_tuning_status_var.set(str(message))
        if hasattr(self, "_motion_tuning_status_label"):
            color = UI["green"] if ok is True else UI["red"] if ok is False else UI["text"]
            self._motion_tuning_status_label.configure(foreground=color)

    def _bind_attitude_tuning_entry(self, entry):
        """Keep uncommitted operator input out of the periodic Pi readback."""
        entry.bind("<KeyRelease>", self._mark_attitude_tuning_dirty, add="+")
        entry.bind("<<Paste>>", self._mark_attitude_tuning_dirty, add="+")
        entry.bind("<<Cut>>", self._mark_attitude_tuning_dirty, add="+")
        entry.bind("<Escape>", self._cancel_attitude_tuning_edit, add="+")

    def _mark_attitude_tuning_dirty(self, event=None):
        keysym = str(getattr(event, "keysym", "")).lower()
        if keysym in {"return", "enter"}:
            return
        self._attitude_tuning_dirty = True

    def _cancel_attitude_tuning_edit(self, _event=None):
        self._restore_attitude_tuning_entries()
        self._set_parameter_status("已取消姿态参数编辑，恢复 Pi 当前值")
        return "break"

    def _set_attitude_tuning_entries(self, actual):
        self._roll_moment_limit_var.set(f"{actual['roll_max_moment_nm']:g}")
        self._pitch_moment_limit_var.set(f"{actual['pitch_max_moment_nm']:g}")
        self._yaw_moment_limit_var.set(f"{actual['yaw_max_moment_nm']:g}")
        self._yaw_release_rate_threshold_var.set(
            f"{actual['yaw_release_rate_threshold_dps']:g}"
        )

    def _sync_attitude_tuning_from_pi(self, telemetry):
        """Apply one clean Pi readback without overwriting an operator edit."""
        if (
            getattr(self, "_attitude_tuning_synced_from_pi", False)
            or getattr(self, "_attitude_tuning_dirty", False)
            or not isinstance(telemetry, dict)
            or not isinstance(telemetry.get("attitude_tuning"), dict)
        ):
            return False
        remote = telemetry["attitude_tuning"]
        try:
            actual = validate_attitude_tuning(
                remote.get("roll_max_moment_nm"),
                remote.get("pitch_max_moment_nm"),
                remote.get("yaw_max_moment_nm"),
                remote.get("yaw_release_rate_threshold_dps"),
            )
        except ValueError:
            self._set_parameter_status("Pi 姿态力矩限幅回读无效")
            return False
        self._attitude_tuning = actual
        self._set_attitude_tuning_entries(actual)
        self._attitude_tuning_synced_from_pi = True
        return True

    def _apply_attitude_tuning_from_entry(self, _event=None):
        try:
            telemetry = self.client.get_telemetry() or {}
            if telemetry.get("armed") or telemetry.get("cmd_enabled"):
                raise ValueError("请先锁定推进器并关闭自主控制，再按回车保存")
            active_mode = str(
                telemetry.get("performance_mode", self._performance_mode)
            ).strip().lower()
            active_profile = self._performance_profiles.get(active_mode)
            if not isinstance(active_profile, dict):
                raise ValueError("无法确认当前性能模式的力矩上限")
            tuning = validate_attitude_tuning(
                active_profile["roll_max_moment_nm"],
                active_profile["pitch_max_moment_nm"],
                active_profile["yaw_max_moment_nm"],
                self._yaw_release_rate_threshold_var.get(),
            )
            baseline_marker = self.client.get_telemetry_marker()
            baseline_ack = telemetry.get("command_ack", {})
            self.client.send_cmd(attitude_tuning=tuning)
            actual = self._wait_for_attitude_tuning_ack(
                tuning,
                baseline_marker,
                baseline_ack=baseline_ack,
            )
        except (OSError, ValueError) as exc:
            # Keep the typed values visible so the operator can correct or retry
            # after locking the thrusters; Esc remains the explicit cancel path.
            self._attitude_tuning_dirty = True
            message = f"姿态力矩限幅未保存：{exc}"
            self._set_parameter_status(message)
            self._log_event(message)
            return "break" if _event is not None else False
        self._attitude_tuning = actual
        self._attitude_tuning_synced_from_pi = True
        self._set_attitude_tuning_entries(actual)
        self._attitude_tuning_dirty = False
        try:
            update_config_values(APP_CONFIG.config_path, "control", actual)
        except OSError as exc:
            self._set_parameter_status(f"Pi 已保存；本机保存失败：{exc}")
            return "break" if _event is not None else False
        for key, value in actual.items():
            APP_CONFIG.set("control", key, str(value))
        self._set_parameter_status("姿态力矩上限与偏转松手阈值已保存到 Pi")
        self._finish_parameter_entry(_event)
        return "break" if _event is not None else True

    def _restore_attitude_tuning_entries(self):
        """Restore Pi-authoritative values after an explicit operator cancel."""
        self._set_attitude_tuning_entries(self._attitude_tuning)
        self._attitude_tuning_dirty = False

    def _save_current_parameters(self, _event=None):
        """Persist all visible tuning fields without starting any control mode."""
        try:
            for mode in ("balance", "rage"):
                if not self._apply_performance_profile_from_entries(mode):
                    raise ValueError(f"{mode} 性能参数保存失败")
            if not self._apply_attitude_tuning_from_entry():
                raise ValueError("偏转松手锁定角速度保存失败")
            depth_values = self._pid_values_from_entries(1)
            pid = dict(zip(("type", "kp", "ki", "kd"), (1, *depth_values)))
            baseline_marker = self.client.get_telemetry_marker()
            self.client.send_cmd(pid_save=pid)
            self._wait_for_pid_ack(
                pid, baseline_marker, success_message="pid_saved"
            )
            self._record_applied_pid(1, depth_values)

            updates = {
                "depth_kp": depth_values[0],
                "depth_ki": depth_values[1],
                "depth_kd": depth_values[2],
            }
            update_config_values(APP_CONFIG.config_path, "pid", updates)
            for key, value in updates.items():
                APP_CONFIG.set("pid", key, str(value))
        except (OSError, ValueError) as exc:
            self._set_parameter_status(f"保存失败：{exc}")
            return "break"
        self._set_parameter_status(
            "已永久保存：平衡/狂暴完整参数、偏转松手阈值、定深 PID"
        )
        self._log_event("已永久保存两套 LB 性能参数、偏转松手阈值和定深 PID")
        self._finish_parameter_entry(_event)
        return "break"

    def _on_key_up(self, event):
        key = event.keysym.lower()
        if key not in self._movement_keys:
            return
        self._joy_keys.discard(key)
        self._update_joy()

    @property
    def _movement_keys(self):
        return {"w", "a", "s", "d", "q", "e", "j", "l", "u", "i"}

    @staticmethod
    def _keyboard_control_allowed(event):
        """Do not turn text entry (PID/IP/target fields) into propulsion."""
        return event.widget.winfo_class() not in {
            "Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox",
        }

    def _on_focus_out(self, _event):
        """A missed key-release must never leave keyboard thrust active."""
        if self._joy_keys:
            self._zero_local_joystick()
            self.client.send_joy(0, 0, 0, 0, 0)

    def _clear_text_focus_on_background_click(self, event):
        """Clicking non-text UI clears a numeric-entry caret without changing state."""
        text_widgets = {"Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox"}
        if event.widget.winfo_class() not in text_widgets:
            # bind_all also receives clicks made inside detached tool windows.
            # Focus the window that was actually clicked; focusing root here
            # made an autonomy slider/blank-area click push that window behind.
            target = event.widget.winfo_toplevel()
            target.focus_set()

    def _update_joy(self):
        fx = self._joy_fx
        fy = self._joy_fy
        fz = self._joy_fz
        y = self._joy_yaw
        p = self._joy_pitch
        j = self._joy

        # 五个运动通道归零；夹爪命令通过独立的 servo_position 发送。
        j["Fx"] = 0
        j["Fy"] = 0
        j["Fz"] = 0
        j["N"] = 0
        j["M"] = 0

        if "w" in self._joy_keys: j["Fx"] = fx
        if "s" in self._joy_keys: j["Fx"] = -fx
        if "a" in self._joy_keys: j["Fy"] = -fy
        if "d" in self._joy_keys: j["Fy"] = fy
        if "e" in self._joy_keys: j["Fz"] = fz
        if "q" in self._joy_keys: j["Fz"] = -fz
        if "j" in self._joy_keys: j["N"] = y
        if "l" in self._joy_keys: j["N"] = -y
        if "u" in self._joy_keys: j["M"] = p
        if "i" in self._joy_keys: j["M"] = -p

    def _joy_loop(self):
        """40Hz 手柄持续发送, 物理手柄优先, 键盘备用"""
        while self._joy_running:
            if self._emergency_stop_latched:
                # Keep sampling the button while inhibited so a held LB cannot
                # become a false rising edge immediately after E-stop reset.
                inhibited_phys = self._physical_joy.read()
                if inhibited_phys is not None:
                    self._sample_dpad_attitude(allow_actions=False)
                    self._sample_gripper_button(allow_toggle=False)
                    self._sample_performance_mode_button(allow_toggle=False)
                    self._sample_gripper_jog(allow_jog=False)
                    self._sample_face_buttons(allow_actions=False)
                else:
                    self._gripper_jog_direction = 0
                self._phys_display = (0.0, 0.0, 0.0, 0.0, 0.0)
                self._reset_manual_vertical_motion()
                self.client.send_joy(0, 0, 0, 0, 0)
                if time.monotonic() - self._last_estop_tx >= 0.5:
                    self._send_emergency_stop()
                time.sleep(JOY_SEND_INTERVAL)
                continue

            if self._estop_pending_target is False:
                inhibited_phys = self._physical_joy.read()
                if inhibited_phys is not None:
                    self._sample_dpad_attitude(allow_actions=False)
                    self._sample_gripper_button(allow_toggle=False)
                    self._sample_performance_mode_button(allow_toggle=False)
                    self._sample_gripper_jog(allow_jog=False)
                    self._sample_face_buttons(allow_actions=False)
                else:
                    self._gripper_jog_direction = 0
                self._phys_display = (0.0, 0.0, 0.0, 0.0, 0.0)
                self._reset_manual_vertical_motion()
                self.client.send_cmd(enable=False, arm=False)
                self.client.send_joy(0, 0, 0, 0, 0)
                time.sleep(JOY_SEND_INTERVAL)
                continue

            # 优先读取物理手柄
            phys = self._physical_joy.read()
            if phys is not None:
                Fx, Fy, Fz, N, M = phys
                # Holding LB cannot toggle at the 40 Hz loop rate.
                self._sample_gripper_button(allow_toggle=True)
                self._sample_performance_mode_button(allow_toggle=True)
                self._sample_gripper_jog(allow_jog=True)
                self._sample_face_buttons(allow_actions=True)
                self._sample_dpad_attitude(allow_actions=True)

                # 急停复位后，必须先让所有运动轴回中，避免重新解锁时
                # 立即恢复急停前的满舵指令。
                if self._require_joystick_centered:
                    if (
                        any(abs(value) > 0.01 for value in (Fx, Fy, Fz, N, M))
                    ):
                        self._phys_display = (0.0, 0.0, 0.0, 0.0, 0.0)
                        self.client.send_joy(0, 0, 0, 0, 0)
                        time.sleep(JOY_SEND_INTERVAL)
                        continue
                    self._require_joystick_centered = False

                self._track_manual_vertical_motion(Fz)
                self._phys_display = (Fx, Fy, Fz, N, M)
                self.client.send_joy(Fx=Fx, Fy=Fy, Fz=Fz, N=N, M=M)
            else:
                self._phys_display = None
                self._gripper_jog_direction = 0
                self._reset_manual_vertical_motion()
                if self._require_joystick_centered:
                    if any(
                        abs(self._joy[key]) > 0.01
                        for key in ("Fx", "Fy", "Fz", "N", "M")
                    ):
                        self.client.send_joy(0, 0, 0, 0, 0)
                        time.sleep(JOY_SEND_INTERVAL)
                        continue
                    self._require_joystick_centered = False
                # 回退: 使用键盘虚拟手柄
                j = self._joy
                self.client.send_joy(
                    Fx=j["Fx"], Fy=j["Fy"], Fz=j["Fz"], N=j["N"], M=j["M"]
                )
            time.sleep(JOY_SEND_INTERVAL)

    # ================================================================
    #  界面刷新
    # ================================================================
    def _refresh_safety_state(self, tele):
        """Paint the Pi-authoritative safety state before optional UI widgets."""
        marker = tele.get("_rx_marker")
        can_reconcile_estop = (
            self._estop_pending_target is None
            or telemetry_is_after(marker, self._estop_request_marker)
        )
        remote_estop = bool(tele.get("emergency_stop", False))
        preserve_local_estop = (
            self._local_estop_hold
            and self._estop_pending_target is None
            and not remote_estop
        )
        if (
            "emergency_stop" in tele
            and can_reconcile_estop
            and not preserve_local_estop
        ):
            previous_pending_target = self._estop_pending_target
            (
                self._emergency_stop_latched,
                self._estop_pending_target,
                self._estop_pending_until,
            ) = reconcile_estop_state(
                self._emergency_stop_latched,
                self._estop_pending_target,
                self._estop_pending_until,
                remote_estop,
                time.monotonic(),
            )
            if (
                previous_pending_target is not None
                and self._estop_pending_target is None
            ):
                self._estop_request_marker = None

        self._remote_armed = bool(tele.get("armed", False))

        if self._emergency_stop_latched:
            self._require_joystick_centered = True
            self._estop_btn.configure(
                text="⚠ 急停已锁存\n点击复位",
                bg=UI["red_dark"],
                height=2,
            )
        else:
            self._estop_btn.configure(
                text="⛔ 紧急停止",
                bg=UI["red"],
                height=1,
            )

        # The Pi is the authority.  A fresh armed=True packet is the actual
        # propulsion state; local button text must never override it.
        show_armed = (
            self._remote_armed
            and bool(tele.get("control_link_ok", False))
            and not bool(tele.get("rearm_required", True))
            and not remote_estop
            and not self._emergency_stop_latched
            and self._estop_pending_target is None
        )
        if show_armed:
            self._arm_btn.configure(text="🔓 推进器已解锁", bg=UI["green"])
            self._arm_light.configure(text="● 解锁", foreground=UI["green"])
            self._start_btn.configure(state=tk.NORMAL)
            self._hover_btn.configure(state=tk.NORMAL)
        elif self._remote_armed and not remote_estop:
            # Pi reports armed but a local E-stop/reset handshake has not yet
            # reconciled.  Keep the indication explicitly pending, not red.
            self._arm_btn.configure(text="⏳ 等待状态同步", bg=UI["amber"])
            self._arm_light.configure(text="● 等待 Pi 确认", foreground=UI["amber"])
            self._start_btn.configure(state=tk.DISABLED)
            self._hover_btn.configure(state=tk.DISABLED)
        else:
            self._set_arm_ui_locked()

    def _refresh_attitude_hold_state(self, telemetry):
        """Paint roll/pitch/yaw solely from fresh Pi-authenticated telemetry."""
        hold = telemetry.get("attitude_hold") if isinstance(telemetry, dict) else None
        self._refresh_yaw_control_values(telemetry)
        axis_rows = (
            ("roll", "→", "横滚"),
            ("pitch", "↑", "俯仰"),
            ("yaw", "←", "偏转"),
        )
        if not isinstance(hold, dict):
            for axis, dpad, name in axis_rows:
                button = getattr(self, f"_{axis}_hold_btn", None)
                if button is not None:
                    button.configure(
                        text=f"十字键{dpad} {name}自动：状态未知",
                        fg="#000000",
                        bg="#FFFFFF",
                        disabledforeground="#000000",
                        state=tk.DISABLED,
                    )
                quick_light = getattr(self, f"_{axis}_hold_quick_light", None)
                if quick_light is not None:
                    quick_light.configure(
                        text=f"● {name}锁：未知",
                        foreground=UI["muted"],
                    )
            mapping_button = getattr(self, "_attitude_mapping_btn", None)
            if mapping_button is not None:
                mapping_button.configure(
                    text="姿态安全门：状态未知",
                    fg="#000000",
                    bg="#FFFFFF",
                    disabledforeground="#000000",
                    state=tk.DISABLED,
                )
            status = getattr(self, "_attitude_status_var", None)
            if status is not None:
                status.set("姿态自动：遥测中断，Pi 实际状态不可确认")
            return

        mapping_button = getattr(self, "_attitude_mapping_btn", None)
        mapping_known = "mapping_validated" in hold
        if mapping_button is not None:
            if mapping_known:
                mapping_validated = bool(hold["mapping_validated"])
                mapping_button.configure(
                    text=(
                        "姿态安全门：已验收（点击关闭）"
                        if mapping_validated
                        else "姿态安全门：关闭（点击验收）"
                    ),
                    fg="#000000",
                    bg="#E8F7F0" if mapping_validated else "#FFF4DD",
                    activeforeground="#000000",
                    activebackground=(
                        "#E8F7F0" if mapping_validated else "#FFF4DD"
                    ),
                    state=tk.NORMAL,
                )
            else:
                mapping_button.configure(
                    text="姿态安全门：状态未知",
                    fg="#000000",
                    bg="#FFFFFF",
                    disabledforeground="#000000",
                    state=tk.DISABLED,
                )

        axis_states = {}
        for axis, dpad, name in axis_rows:
            if axis not in hold:
                axis_states[axis] = (None, False)
                button = getattr(self, f"_{axis}_hold_btn", None)
                if button is not None:
                    button.configure(
                        text=f"十字键{dpad} {name}自动：状态未知",
                        fg="#000000", bg="#FFFFFF",
                        disabledforeground="#000000", state=tk.DISABLED,
                    )
                quick_light = getattr(self, f"_{axis}_hold_quick_light", None)
                if quick_light is not None:
                    quick_light.configure(
                        text=f"● {name}锁：未知", foreground=UI["muted"]
                    )
                continue

            enabled = bool(hold[axis])
            overridden = enabled and bool(
                hold.get(f"{axis}_manual_override", False)
            )
            release_settling = bool(
                axis == "yaw" and enabled
                and hold.get("yaw_release_settling", False)
            )
            axis_states[axis] = (enabled, overridden)
            button = getattr(self, f"_{axis}_hold_btn", None)
            if button is not None:
                state_text = (
                    "惯性等待" if release_settling
                    else "人工接管" if overridden
                    else "开" if enabled else "关"
                )
                button.configure(
                    text=(
                        f"十字键{dpad} "
                        f"{name}自动："
                        f"{state_text}"
                    ),
                    fg="#000000",
                    bg="#FFFFFF",
                    activeforeground="#000000",
                    activebackground="#FFFFFF",
                    state=tk.NORMAL,
                )
            quick_light = getattr(self, f"_{axis}_hold_quick_light", None)
            if quick_light is not None:
                quick_state = (
                    "惯性等待" if release_settling
                    else "人工接管" if overridden
                    else "开" if enabled else "关"
                )
                quick_light.configure(
                    text=(
                        f"● {name}锁："
                        f"{quick_state}"
                    ),
                    foreground=(
                        UI["amber"] if overridden or release_settling
                        else UI["green"] if enabled
                        else UI["muted"]
                    ),
                )

        status = getattr(self, "_attitude_status_var", None)
        ack = telemetry.get("command_ack", {})
        ack_error = str(ack.get("error") or "") if isinstance(ack, dict) else ""
        if (
            isinstance(ack, dict)
            and not bool(ack.get("ok", False))
            and "attitude_hold_rejected_" in ack_error
        ):
            translated = attitude_hold_rejection_text(ack_error)
            fingerprint = (
                ack.get("timestamp"), ack.get("message"), ack_error
            )
            if fingerprint != getattr(
                self, "_last_attitude_hold_rejection_ack", None
            ):
                self._last_attitude_hold_rejection_ack = fingerprint
                logger = getattr(self, "_log_event", None)
                if callable(logger):
                    logger(f"姿态锁定被 Pi 拒绝：{translated}")
            if status is not None:
                status.set(f"姿态锁定被 Pi 拒绝：{translated}")
            return

        if hold.get("mapping_validated") is False and not any(
            enabled for enabled, _override in axis_states.values()
            if enabled is not None
        ):
            if status is not None:
                status.set(
                    "姿态自动：横滚 关 · 俯仰 关 · 偏转 关 · 门禁："
                    "姿态推进器/IMU 方向映射未验收"
                    "（mapping_validated=false）"
                )
            return

        reason = str(hold.get("reason") or "not_reported")
        reason_text = {
            "startup_disabled": "开机默认关闭",
            "operator_enabled": "操作员已开启",
            "operator_disabled": "操作员已关闭",
            "imu_invalid": "IMU 无效，Pi 已关闭",
            "heading_invalid": "磁航向无效，Pi 已关闭偏转锁",
            "control_link_timeout": "控制链路超时，Pi 已关闭",
            "latched_or_locked": "推进器锁定，Pi 已关闭",
            "manual_lock": "人工锁定，Pi 已关闭",
            "emergency_stop_reset": "急停复位后保持关闭",
            "emergency_stop": "急停已锁存，Pi 已关闭",
            "mount_uncalibrated": "IMU 三维安装方向未标定",
            "mapping_unvalidated": "姿态推进器/IMU 方向映射未验收",
            "not_reported": "Pi 未报告原因",
        }.get(reason, reason)
        if status is not None:
            def state_text(axis):
                enabled, overridden = axis_states.get(axis, (None, False))
                if enabled is None:
                    return "未知"
                if axis == "yaw" and bool(
                    hold.get("yaw_release_settling", False)
                ):
                    return "惯性等待"
                return "人工接管" if overridden else "开" if enabled else "关"

            status.set(
                f"姿态自动：横滚 {state_text('roll')} · "
                f"俯仰 {state_text('pitch')} · 偏转 {state_text('yaw')} · "
                f"Pi 原因：{reason_text}"
            )

    def _refresh_yaw_control_values(self, telemetry):
        """Show the Pi-authoritative Yaw target in the main action bar."""
        target_var = getattr(self, "_main_yaw_target_var", None)
        hold = telemetry.get("attitude_hold") if isinstance(telemetry, dict) else None
        try:
            if not isinstance(hold, dict):
                raise ValueError("missing attitude telemetry")
            if hold.get("yaw_control_source") not in (
                "referenced_euler_yaw",
                "gyro_z_integrated_fallback",
            ):
                raise ValueError("unknown yaw control source")
            target = float(hold["yaw_target_deg"])
            if not math.isfinite(target):
                raise ValueError("non-finite yaw telemetry")
            target = self._wrap_angle_180(target)
        except (KeyError, TypeError, ValueError):
            if target_var is not None:
                target_var.set("目标Yaw：--°")
            return False
        if target_var is not None:
            target_var.set(f"目标Yaw：{target:+.1f}°")
        return True

    def _refresh_fast_yaw_display(self):
        """Refresh light heading labels at telemetry rate without redrawing charts."""
        try:
            telemetry = self.client.get_telemetry()
            self._refresh_yaw_control_values(telemetry)
            if telemetry:
                vehicle_imu = self._vehicle_imu(telemetry.get("imu", {}))
                self._lbl_yaw.set(f"{vehicle_imu['yaw']:.1f}°")
                self._lbl_head.set(f"{vehicle_imu['heading']:.1f}°")
        except Exception:
            # The guarded 5 Hz loop remains responsible for persistent error
            # reporting; a transient 20 Hz label failure must not stop Tk.
            pass
        finally:
            self.root.after(
                FAST_LABEL_REFRESH_INTERVAL_MS,
                self._refresh_fast_yaw_display,
            )

    def _refresh_display(self):
        """Keep the periodic GUI refresh alive after a local widget failure."""
        try:
            self._refresh_display_once()
        except Exception as exc:
            now = time.monotonic()
            error_text = f"{type(exc).__name__}: {exc}"
            if (
                error_text != self._last_refresh_error
                or now - self._last_refresh_error_time >= 5.0
            ):
                self._last_refresh_error = error_text
                self._last_refresh_error_time = now
                self._log_event(f"界面刷新异常，已自动继续: {error_text}")
                try:
                    with Path(__file__).with_name("gui_errors.log").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} {error_text}\n"
                        )
                except OSError:
                    pass
        finally:
            self.root.after(
                DASHBOARD_REFRESH_INTERVAL_MS,
                self._refresh_display,
            )

    def _reset_pi_sync_on_session_change(self, telemetry):
        """Force one fresh parameter readback after a Pi service restart."""
        marker = telemetry.get("_rx_marker") if isinstance(telemetry, dict) else None
        if not isinstance(marker, tuple) or len(marker) != 2:
            return False
        session = marker[0]
        previous = getattr(self, "_last_pi_telemetry_session", None)
        self._last_pi_telemetry_session = session
        if previous is None or session == previous:
            return False
        self._pid_synced_from_pi = False
        self._motion_tuning_synced_from_pi = False
        self._attitude_tuning_synced_from_pi = False
        return True

    def _refresh_display_once(self):
        # Tk widgets stay on the main thread. The joystick worker only raises
        # requests; this 10 Hz UI loop consumes every deliberate button press.
        while not self._gripper_toggle_requests.empty():
            self._gripper_toggle_requests.get_nowait()
            self._toggle_gripper()
        while not self._performance_mode_requests.empty():
            self._performance_mode_requests.get_nowait()
            self._toggle_performance_mode()
        self._apply_gripper_jog_step()

        while not self._vertical_target_update_requests.empty():
            self._vertical_target_update_requests.get_nowait()
            self._update_depth_target_after_manual_vertical_move()

        while not self._face_button_requests.empty():
            action = self._face_button_requests.get_nowait()
            self._handle_face_button_action(action)

        tele = self.client.get_telemetry()

        if tele:
            self._reset_pi_sync_on_session_change(tele)
            remote_performance_mode = str(
                tele.get("performance_mode", self._performance_mode)
            ).lower()
            if remote_performance_mode in ("balance", "rage"):
                self._performance_mode = remote_performance_mode
                self._performance_mode_var.set(
                    "当前：狂暴模式"
                    if remote_performance_mode == "rage"
                    else "当前：平衡模式"
                )
            remote_profiles = tele.get("performance_profiles", {})
            self._sync_performance_profiles_from_pi(remote_profiles)
            # Safety indication is critical and must not wait for optional
            # pose, diagram, camera, or chart widgets to finish updating.
            self._refresh_safety_state(tele)
            imu = tele.get("imu", {})
            vehicle_imu = self._vehicle_imu(imu)
            display_roll = vehicle_imu["roll"]
            self._lbl_roll.set(f"{display_roll:.1f}°")
            self._lbl_pitch.set(f"{vehicle_imu['pitch']:.1f}°")
            self._lbl_yaw.set(f"{vehicle_imu['yaw']:.1f}°")
            self._lbl_head.set(f"{vehicle_imu['heading']:.1f}°")
            self._lbl_depth.set(format_depth_display(tele))
            self._lbl_vz.set(f"{tele.get('vert_speed', 0):+.3f} m/s")
            self._lbl_vx.set(f"{tele.get('est_vx', 0):+.3f} m/s")
            self._lbl_vy.set(f"{tele.get('est_vy', 0):+.3f} m/s")

            sensor_health = tele.get("sensor_health", {})
            current_depth = tele.get("depth")
            pressure_text = format_pressure_status_text(tele)
            remote_mode = tele.get("mode", {})
            remote_target = tele.get("target", {})
            try:
                if (
                    tele.get("cmd_enabled")
                    and isinstance(remote_mode, dict)
                    and remote_mode.get("depth_hold")
                    and sensor_health.get("depth_valid")
                ):
                    depth_error = (
                        parse_finite_number(remote_target.get("depth_m"), "目标深度")
                        - parse_finite_number(current_depth, "当前深度")
                    )
                    self._depth_status_var.set(
                        f"定深运行 · 误差 {depth_error:+.3f} m{pressure_text}"
                    )
                elif not sensor_health.get("depth_valid"):
                    reason = sensor_health.get("depth_fault") or "数据超时"
                    self._depth_status_var.set(
                        f"深度无效({reason}) · 定深禁止{pressure_text}"
                    )
                else:
                    self._depth_status_var.set(f"深度闭环待机{pressure_text}")
            except ValueError:
                self._depth_status_var.set("深度数据异常 · 定深禁止")
            self._pose_view.set_pose(
                display_roll,
                vehicle_imu["pitch"],
                vehicle_imu["yaw"],
                connected=bool(sensor_health.get("imu_valid", True)),
            )

            forces = tele.get("forces", {})
            for k, var in self._force_vars.items():
                var.set(f"{forces.get(k, 0):+.1f}")

            thrust = tele.get("thrust", {})
            self._thruster_diagram.set_thrust(thrust, connected=True)

            # 首次连接时自动读回Pi端实际PID，不需要周期发查询命令。
            if not self._pid_synced_from_pi and isinstance(tele.get("pid"), dict):
                pid = tele["pid"]
                depth_pid = pid.get("depth", {})
                roll_pid = pid.get("roll", {})
                pitch_pid = pid.get("pitch", {})
                yaw_pid = pid.get("yaw", {})
                if depth_pid and roll_pid and pitch_pid and yaw_pid:
                    self._d_kp.set(str(depth_pid.get("kp", 120.0)))
                    self._d_ki.set(str(depth_pid.get("ki", 10.0)))
                    self._d_kd.set(str(depth_pid.get("kd", 0.0)))
                    self._r_kp.set(str(roll_pid.get("kp", 1.5)))
                    self._r_ki.set(str(roll_pid.get("ki", 0.0)))
                    self._r_kd.set(str(roll_pid.get("kd", 0.2)))
                    self._p_kp.set(str(pitch_pid.get("kp", 1.5)))
                    self._p_ki.set(str(pitch_pid.get("ki", 0.0)))
                    self._p_kd.set(str(pitch_pid.get("kd", 0.2)))
                    self._y_kp.set(str(yaw_pid.get("kp", 0.2)))
                    self._y_ki.set(str(yaw_pid.get("ki", 0.01)))
                    self._y_kd.set(str(yaw_pid.get("kd", 0.08)))
                    self._record_applied_pid(1, (
                        depth_pid.get("kp", 120.0), depth_pid.get("ki", 10.0),
                        depth_pid.get("kd", 0.0),
                    ))
                    self._record_applied_pid(4, (
                        roll_pid.get("kp", 1.5), roll_pid.get("ki", 0.0),
                        roll_pid.get("kd", 0.2),
                    ))
                    self._record_applied_pid(5, (
                        pitch_pid.get("kp", 1.5), pitch_pid.get("ki", 0.0),
                        pitch_pid.get("kd", 0.2),
                    ))
                    self._record_applied_pid(6, (
                        yaw_pid.get("kp", 0.2), yaw_pid.get("ki", 0.01),
                        yaw_pid.get("kd", 0.08),
                    ))
                    self._pid_synced_from_pi = True

            if (
                not self._motion_tuning_synced_from_pi
                and isinstance(tele.get("motion_tuning"), dict)
            ):
                remote_tuning = tele["motion_tuning"]
                try:
                    actual = validate_motion_tuning(
                        remote_tuning.get("horizontal_limit_fraction"),
                        remote_tuning.get("vertical_limit_fraction"),
                        remote_tuning.get("hover_force_per_thruster_n"),
                        remote_tuning.get("vertical_ascent_front_rear_trim_percent"),
                        remote_tuning.get("vertical_descent_front_rear_trim_percent"),
                        remote_tuning.get("forward_front_rear_trim_percent"),
                        remote_tuning.get("backward_front_rear_trim_percent"),
                    )
                except ValueError:
                    self._set_motion_tuning_status(
                        "✗ Pi 推力设置回读无效", ok=False
                    )
                else:
                    self._motion_tuning = actual
                    self._horizontal_limit_var.set(
                        f"{actual['horizontal_limit_fraction']:.2f}"
                    )
                    self._vertical_limit_var.set(
                        f"{actual['vertical_limit_fraction']:.2f}"
                    )
                    self._vertical_hover_var.set(
                        f"{actual['hover_force_per_thruster_n']:.2f}"
                    )
                    self._vertical_ascent_front_rear_trim_var.set(
                        actual["vertical_ascent_front_rear_trim_percent"]
                    )
                    self._vertical_descent_front_rear_trim_var.set(
                        actual["vertical_descent_front_rear_trim_percent"]
                    )
                    self._forward_front_rear_trim_var.set(
                        actual["forward_front_rear_trim_percent"]
                    )
                    self._backward_front_rear_trim_var.set(
                        actual["backward_front_rear_trim_percent"]
                    )
                    self._set_motion_tuning_status(
                        "✓ 已同步 Pi 开机保存值", ok=True
                    )
                    self._motion_tuning_synced_from_pi = True

            self._sync_attitude_tuning_from_pi(tele)

            self._sync_attitude_calibration_from_pi(tele)
            self._refresh_attitude_hold_state(tele)

            # 状态灯
            if tele.get("cmd_enabled"):
                self._cmd_light.configure(text="● 自主", foreground=UI["green"])
            else:
                self._cmd_light.configure(text="● 自主", foreground=UI["muted"])

            if tele.get("joy_active"):
                self._joy_light.configure(text="● 手柄", foreground=UI["orange"])
            else:
                self._joy_light.configure(text="● 手柄", foreground=UI["muted"])

            health = tele.get("sensor_health", {})
            ack = tele.get("command_ack", {})
            servo = tele.get("servo", {})
            if servo:
                position = float(
                    servo.get("position", self._servo_position_var.get())
                )
                angle = servo_position_to_angle(
                    position, self._servo_angle_min, self._servo_angle_max
                )
                self._servo_angle_var.set(f"{angle:.1f}°")
                if servo.get("output_enabled"):
                    self._servo_status_var.set(
                        f"Pi CH{servo.get('channel')}  {int(servo.get('pwm_us', 0))}µs"
                    )
                else:
                    self._servo_status_var.set("Pi 舵机输出已禁用")
            output = "REAL PWM" if tele.get("motion_output_enabled") else "BENCH PWM"
            link = "链路正常" if tele.get("control_link_ok") else "链路待确认"
            remote_config = str(tele.get("config_fingerprint", ""))
            local_config = config_fingerprint(APP_CONFIG)
            config_status = (
                "配置一致"
                if remote_config and remote_config == local_config
                else "配置不同"
            )
            imu = "IMU 正常" if health.get("imu_valid") else "IMU 超时"
            if health.get("depth_valid"):
                depth = "深度正常"
            else:
                depth = f"深度无效({health.get('depth_fault') or '超时'})"
            inhibited = ", ".join(tele.get("auto_inhibited", [])) or "无"
            if ack.get("ok", False):
                ack_text = ack.get("message", "等待命令确认")
            else:
                ack_text = ack.get("error") or ack.get("message", "Pi 拒绝命令")
                if "attitude_hold_rejected_" in str(ack_text):
                    ack_text = f"姿态锁定拒绝：{attitude_hold_rejection_text(ack_text)}"
            color = UI["green"] if ack.get("ok", False) else UI["amber"]
            self._safety_detail.configure(
                text=(f"安全状态：{output} · {link} · {config_status} · {imu} · {depth} · "
                      f"自动抑制：{inhibited} · 命令：{ack_text}"),
                foreground=color,
            )

            self._conn_label.configure(text="● 遥测在线", foreground=UI["green"])
            if not getattr(self, '_was_connected', False):
                self._was_connected = True
                self._log_event("遥测连接建立")
        else:
            if getattr(self, '_was_connected', False):
                self._was_connected = False
                self._log_event("遥测连接丢失")
                self._arm_request_generation += 1
                self._send_remote_lock()
                for delay_ms in (50, 100):
                    self.root.after(delay_ms, self._send_remote_lock)
            self._conn_label.configure(text="● 等待遥测", foreground=UI["amber"])
            self._thruster_diagram.set_thrust({}, connected=False)
            self._pose_view.set_connected(False)
            # Never leave a green local unlock indication after telemetry loss.
            self._require_joystick_centered = True
            self._center_servo_ui()
            self._remote_armed = False
            self._set_arm_ui_locked()
            self._servo_status_var.set("遥测中断：请求回到中位")
            self._safety_detail.configure(
                text="安全状态：未收到 Pi 遥测；推进器状态不可确认",
                foreground=UI["red"],
            )
            self._depth_status_var.set("遥测中断 · 定深禁止")
            self._refresh_attitude_hold_state(None)

        # ---- 显示物理手柄实时发送值 ----
        phys = getattr(self, '_phys_display', None)
        if phys is not None:
            Fx, Fy, Fz, N, M = phys
            self._joy_send_var.set(
                f"📤 发送: Fx={Fx:+6.1f}  Fy={Fy:+6.1f}  "
                f"Fz={Fz:+6.1f}  N={N:+5.1f}  M={M:+5.1f}")
        else:
            self._joy_send_var.set("📤 发送: (键盘模式)")

        # ---- 通信质量统计 ----
        if tele:
            link_age = tele.get("control_link_age_ms")
            if link_age is not None:
                age = float(link_age)
                if age < 50:
                    color = UI["green"]
                elif age < 200:
                    color = UI["amber"]
                else:
                    color = UI["red"]
                self._delay_label.configure(text=f"延迟 {age:.0f}ms", foreground=color)
            else:
                self._delay_label.configure(text="延迟 -- ms", foreground=UI["muted"])

            # Count only actual gaps in authenticated telemetry packet sequences.
            loss_pct = self.client.get_telemetry_loss_percent()
            if loss_pct is not None:
                if loss_pct < 1:
                    loss_color = UI["green"]
                elif loss_pct < 5:
                    loss_color = UI["amber"]
                else:
                    loss_color = UI["red"]
                self._pkt_loss_label.configure(
                    text=f"丢包 {loss_pct:.1f}%", foreground=loss_color,
                )
            else:
                self._pkt_loss_label.configure(
                    text="丢包 --%", foreground=UI["muted"],
                )
        else:
            self._delay_label.configure(text="延迟 -- ms", foreground=UI["muted"])
            self._pkt_loss_label.configure(text="丢包 --%", foreground=UI["muted"])

        # ---- 趋势图更新 ----
        if tele:
            if sensor_health.get("depth_valid"):
                self._depth_trend.push(tele.get("depth", 0))
            vehicle_imu = self._vehicle_imu(tele.get("imu", {}))
            self._roll_trend.push(vehicle_imu["roll"])
            self._pitch_trend.push(vehicle_imu["pitch"])
            self._yaw_trend.push(vehicle_imu["yaw"])
            self._heading_trend.push(vehicle_imu["heading"])
        for chart_name in (
            "_depth_trend", "_roll_trend", "_pitch_trend",
            "_yaw_trend", "_heading_trend",
        ):
            chart = getattr(self, chart_name, None)
            if chart is not None:
                chart.redraw()

        # ---- 计时器更新 ----
        if hasattr(self, '_timer'):
            self._timer.tick()

    # ================================================================
    #  事件日志面板
    # ================================================================
    def _build_event_log(self, parent):
        frame = ttk.LabelFrame(
            parent, text="  事件日志  ", style="Card.TLabelframe", padding=(6, 1),
        )
        frame.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        self._log_text = tk.Text(
            text_frame, width=1, height=3, bg=UI["surface_alt"], fg=UI["text"],
            font=("Microsoft YaHei UI", 9), relief=tk.FLAT, borderwidth=0,
            state=tk.DISABLED, wrap=tk.WORD, padx=5, pady=1,
            spacing1=0, spacing3=1,
        )
        scrollbar = ttk.Scrollbar(text_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scrollbar.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._log_event("GUI 启动完成，等待连接 ROV…")

    def _log_event(self, message):
        """向事件日志追加一行带时间戳的消息。"""
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        try:
            self._log_text.configure(state=tk.NORMAL)
            self._log_text.insert(tk.END, line)
            self._log_text.see(tk.END)
            self._log_text.configure(state=tk.DISABLED)
        except Exception:
            pass

    def _on_close(self):
        self._joy_running = False
        self.client.stop()
        self._physical_joy.close()
        if HAS_OPENCV and hasattr(self, '_video_view'):
            self._video_view.stop()
        self.root.destroy()


# ================================================================
#  入口
# ================================================================
if __name__ == "__main__":
    instance_lock = acquire_single_instance_lock()
    if instance_lock is None:
        warning_root = tk.Tk()
        warning_root.withdraw()
        messagebox.showerror(
            "ROV 控制站已运行",
            "检测到另一个 ROV 控制站正在运行。\n"
            "为防止两个窗口争抢遥测和控制会话，本窗口不会启动。",
        )
        warning_root.destroy()
        raise SystemExit(1)

    rov_ip = sys.argv[1] if len(sys.argv) > 1 else APP_CONFIG.get("network", "pi_ip")

    client = ROVClient(rov_ip=rov_ip)
    client.start()

    root = tk.Tk()

    # 设置菠萝图标
    _make_pineapple_icon(root)

    gui = ROVGui(root, client)
    root.mainloop()
