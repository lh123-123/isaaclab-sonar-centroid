#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authenticated PC -> Raspberry Pi control protocol.

Only this module defines the bytes covered by the HMAC.  Both the GUI and the
Pi import it, which prevents the two ends from silently drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import re
import secrets
import zlib
from typing import Any, Dict, Mapping, Optional


PROTOCOL_VERSION = 6
DEFAULT_MAX_PACKET_BYTES = 4096
TELEMETRY_WIRE_COMPRESSION_THRESHOLD = 1200
_TELEMETRY_COMPRESSED_MAGIC = b"RZ1\x00"
_SESSION_RE = re.compile(r"^[0-9a-f]{32}$")
_HMAC_RE = re.compile(r"^[0-9a-f]{64}$")
_ENVELOPE_FIELDS = {"protocol_version", "session", "seq", "payload", "hmac"}
_TELEMETRY_ENVELOPE_FIELDS = {
    "protocol_version", "kind", "session", "seq", "payload", "hmac"
}
_CONTROL_FIELDS = {
    "arm",
    "enable",
    "mode",
    "target",
    "save_data",
    "pid_set",
    "pid_save",
    "pid_get",
    "emergency_stop",
    "reset_emergency_stop",
    "joystick",
    "servo_position",
    "servo_min_position",
    "servo_immediate",
    "motion_tuning",
    "attitude_hold",
    "attitude_mapping_validated",
    "attitude_tuning",
    "attitude_zero_calibrate",
    "attitude_mount_calibrate",
    "attitude_mount_capture_roll",
    "attitude_mount_capture_pitch",
    "attitude_mount_capture_yaw",
    "attitude_mount_finalize",
    "performance_mode",
    "performance_profile",
    "depth_zero_calibrate",
    "heartbeat",
}
_BOOL_FIELDS = {
    "arm", "enable", "save_data", "emergency_stop",
    "reset_emergency_stop", "attitude_zero_calibrate", "attitude_mount_calibrate", "attitude_mount_capture_roll", "attitude_mount_capture_pitch", "attitude_mount_capture_yaw", "attitude_mount_finalize", "attitude_mapping_validated", "depth_zero_calibrate", "heartbeat",
    "servo_immediate",
}
_MODE_FIELDS = {"heading_hold", "depth_hold", "speed_hold"}
_TARGET_FIELDS = {"heading_deg", "depth_m", "speed_mps"}
_JOYSTICK_FIELDS = {"tau_x", "tau_y", "tau_z", "M_x", "M_y", "M_z"}
_MOTION_TUNING_FIELDS = {
    "horizontal_limit_fraction",
    "vertical_limit_fraction",
    "hover_force_per_thruster_n",
    "vertical_ascent_front_rear_trim_percent",
    "vertical_descent_front_rear_trim_percent",
    "forward_front_rear_trim_percent",
    "backward_front_rear_trim_percent",
}
_ATTITUDE_HOLD_FIELDS = {"roll", "pitch", "yaw"}
_ATTITUDE_MOMENT_TUNING_FIELDS = {
    "roll_max_moment_nm", "pitch_max_moment_nm", "yaw_max_moment_nm"
}
_ATTITUDE_TUNING_FIELDS = _ATTITUDE_MOMENT_TUNING_FIELDS | {
    "yaw_release_rate_threshold_dps"
}
_PERFORMANCE_PROFILE_LEGACY_FIELDS = {
    "mode", "horizontal_limit_fraction", "vertical_limit_fraction",
    "roll_pid", "pitch_pid", "yaw_pid",
}
_PERFORMANCE_PROFILE_V1_FIELDS = _PERFORMANCE_PROFILE_LEGACY_FIELDS | {
    "hover_force_per_thruster_n",
    "vertical_ascent_front_rear_trim_percent",
    "vertical_descent_front_rear_trim_percent",
    "forward_front_rear_trim_percent",
    "backward_front_rear_trim_percent",
    "roll_max_moment_nm", "pitch_max_moment_nm", "yaw_max_moment_nm",
}
_PERFORMANCE_PROFILE_FIELDS = _PERFORMANCE_PROFILE_V1_FIELDS | {
    "left_lateral_surge_trim_percent",
    "right_lateral_surge_trim_percent",
}


class ProtocolError(ValueError):
    """A packet failed structural, authentication, or value validation."""


@dataclass(frozen=True)
class VerifiedControlPacket:
    session: str
    seq: int
    payload: Dict[str, Any]


@dataclass(frozen=True)
class VerifiedTelemetryPacket:
    session: str
    seq: int
    payload: Dict[str, Any]


class TelemetryReplayGuard:
    """Accept monotonic telemetry and prevent a retired Pi session returning."""

    def __init__(self) -> None:
        self.active_session: Optional[str] = None
        self.last_seq = -1
        self.retired_sessions: set[str] = set()

    def accept(self, packet: VerifiedTelemetryPacket) -> Dict[str, Any]:
        if packet.session != self.active_session:
            if packet.session in self.retired_sessions:
                raise ProtocolError("retired telemetry session cannot return")
            if self.active_session is not None:
                self.retired_sessions.add(self.active_session)
            self.active_session = packet.session
            self.last_seq = -1
        if packet.seq <= self.last_seq:
            raise ProtocolError(
                f"replayed/out-of-order telemetry seq {packet.seq}; last {self.last_seq}"
            )
        self.last_seq = packet.seq
        return dict(packet.payload)


def new_session_id() -> str:
    return secrets.token_hex(16)


def _key_bytes(shared_key: str) -> bytes:
    if not isinstance(shared_key, str):
        raise ProtocolError("shared key must be text")
    encoded = shared_key.encode("utf-8")
    if len(encoded) < 32:
        raise ProtocolError("shared key must contain at least 32 UTF-8 bytes")
    return encoded


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"packet is not canonical JSON: {exc}") from exc


def _strict_json_loads(data: bytes) -> Any:
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProtocolError(f"non-finite JSON constant is forbidden: {value}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise ProtocolError("packet is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc.msg}") from exc


def _finite_number(
    value: Any, field: str, *, absolute_limit: Optional[float] = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{field} must be finite")
    if absolute_limit is not None and abs(result) > absolute_limit:
        raise ProtocolError(f"{field} exceeds safe protocol limit {absolute_limit}")
    return result


def validate_control_payload(payload: Any) -> Dict[str, Any]:
    """Return a shallow copy after strict field/type/finite validation."""
    if not isinstance(payload, dict):
        raise ProtocolError("payload must be an object")
    unknown = set(payload).difference(_CONTROL_FIELDS)
    if unknown:
        raise ProtocolError(f"unknown control fields: {sorted(unknown)}")
    if not payload:
        raise ProtocolError("empty control payload is forbidden")

    for field in _BOOL_FIELDS.intersection(payload):
        if not isinstance(payload[field], bool):
            raise ProtocolError(f"{field} must be true or false")

    if "mode" in payload:
        mode = payload["mode"]
        if not isinstance(mode, dict) or set(mode).difference(_MODE_FIELDS):
            raise ProtocolError("mode contains unknown fields or is not an object")
        for field, value in mode.items():
            if not isinstance(value, bool):
                raise ProtocolError(f"mode.{field} must be true or false")
            if field == "heading_hold" and value:
                raise ProtocolError(
                    "mode.heading_hold is temporarily disabled and must be false"
                )

    if "target" in payload:
        target = payload["target"]
        if not isinstance(target, dict) or set(target).difference(_TARGET_FIELDS):
            raise ProtocolError("target contains unknown fields or is not an object")
        for field, value in target.items():
            _finite_number(value, f"target.{field}", absolute_limit=100000.0)

    if "joystick" in payload:
        joystick = payload["joystick"]
        if not isinstance(joystick, dict) or set(joystick).difference(_JOYSTICK_FIELDS):
            raise ProtocolError("joystick contains unknown fields or is not an object")
        for field, value in joystick.items():
            _finite_number(value, f"joystick.{field}", absolute_limit=1000.0)

    if "servo_position" in payload:
        value = _finite_number(payload["servo_position"], "servo_position", absolute_limit=1.0)
        if not -1.0 <= value <= 1.0:
            raise ProtocolError("servo_position must be in -1..1")
    if "servo_min_position" in payload:
        value = _finite_number(
            payload["servo_min_position"], "servo_min_position", absolute_limit=1.0
        )
        if not 0.0 <= value < 1.0:
            raise ProtocolError("servo_min_position must be in 0..1")
    if "servo_immediate" in payload and "servo_position" not in payload:
        raise ProtocolError("servo_immediate requires servo_position")

    if "motion_tuning" in payload:
        tuning = payload["motion_tuning"]
        if not isinstance(tuning, dict) or set(tuning) != _MOTION_TUNING_FIELDS:
            raise ProtocolError(
                "motion_tuning must contain exactly horizontal_limit_fraction/"
                "vertical_limit_fraction/hover_force_per_thruster_n/"
                "vertical_ascent_front_rear_trim_percent/"
                "vertical_descent_front_rear_trim_percent/"
                "forward_front_rear_trim_percent/"
                "backward_front_rear_trim_percent"
            )
        horizontal = _finite_number(
            tuning["horizontal_limit_fraction"],
            "motion_tuning.horizontal_limit_fraction",
        )
        vertical = _finite_number(
            tuning["vertical_limit_fraction"],
            "motion_tuning.vertical_limit_fraction",
        )
        hover = _finite_number(
            tuning["hover_force_per_thruster_n"],
            "motion_tuning.hover_force_per_thruster_n",
        )
        ascent_trim = _finite_number(
            tuning["vertical_ascent_front_rear_trim_percent"],
            "motion_tuning.vertical_ascent_front_rear_trim_percent",
        )
        descent_trim = _finite_number(
            tuning["vertical_descent_front_rear_trim_percent"],
            "motion_tuning.vertical_descent_front_rear_trim_percent",
        )
        forward_trim = _finite_number(
            tuning["forward_front_rear_trim_percent"],
            "motion_tuning.forward_front_rear_trim_percent",
        )
        backward_trim = _finite_number(
            tuning["backward_front_rear_trim_percent"],
            "motion_tuning.backward_front_rear_trim_percent",
        )
        if not 0.0 <= horizontal <= 0.5 or not 0.0 <= vertical <= 0.5:
            raise ProtocolError("motion tuning fractions must be in 0..0.5")
        if not 0.0 <= hover <= 10.0:
            raise ProtocolError("hover force per thruster must be in 0..10 N")
        if hover > 55.0 * vertical + 1e-9:
            raise ProtocolError("hover force exceeds the vertical thrust limit")
        if not all(
            -100.0 <= value <= 100.0
            for value in (ascent_trim, descent_trim, forward_trim, backward_trim)
        ):
            raise ProtocolError("motion tuning trims must be in -100..100 percent")

    if "attitude_hold" in payload:
        hold = payload["attitude_hold"]
        if not isinstance(hold, dict) or set(hold) != _ATTITUDE_HOLD_FIELDS:
            raise ProtocolError("attitude_hold must contain exactly roll/pitch/yaw")
        for field, value in hold.items():
            if not isinstance(value, bool):
                raise ProtocolError(f"attitude_hold.{field} must be true or false")

    if "attitude_tuning" in payload:
        tuning = payload["attitude_tuning"]
        if not isinstance(tuning, dict) or set(tuning) != _ATTITUDE_TUNING_FIELDS:
            raise ProtocolError(
                "attitude_tuning must contain exactly roll_max_moment_nm/"
                "pitch_max_moment_nm/yaw_max_moment_nm/"
                "yaw_release_rate_threshold_dps"
            )
        for field in _ATTITUDE_MOMENT_TUNING_FIELDS:
            value = _finite_number(tuning[field], f"attitude_tuning.{field}")
            if not 0.1 <= value <= 20.0:
                raise ProtocolError(f"attitude_tuning.{field} must be in 0.1..20 Nm")
        release_threshold = _finite_number(
            tuning["yaw_release_rate_threshold_dps"],
            "attitude_tuning.yaw_release_rate_threshold_dps",
        )
        if not 0.5 <= release_threshold <= 30.0:
            raise ProtocolError(
                "attitude_tuning.yaw_release_rate_threshold_dps must be in "
                "0.5..30 deg/s"
            )

    if "performance_mode" in payload:
        value = payload["performance_mode"]
        if value not in ("balance", "rage"):
            raise ProtocolError("performance_mode must be balance or rage")

    if "performance_profile" in payload:
        profile = payload["performance_profile"]
        profile_fields = set(profile) if isinstance(profile, dict) else set()
        if profile_fields not in (
            _PERFORMANCE_PROFILE_LEGACY_FIELDS,
            _PERFORMANCE_PROFILE_V1_FIELDS,
            _PERFORMANCE_PROFILE_FIELDS,
        ):
            raise ProtocolError("performance_profile fields are incomplete or unknown")
        if profile["mode"] not in ("balance", "rage"):
            raise ProtocolError("performance_profile.mode must be balance or rage")
        for field in ("horizontal_limit_fraction", "vertical_limit_fraction"):
            value = _finite_number(profile[field], f"performance_profile.{field}")
            if not 0.0 <= value <= 0.5:
                raise ProtocolError(f"performance_profile.{field} must be in 0..0.5")
        for axis in ("roll_pid", "pitch_pid", "yaw_pid"):
            pid = profile[axis]
            if not isinstance(pid, dict) or set(pid) != {"kp", "ki", "kd"}:
                raise ProtocolError(f"performance_profile.{axis} must contain kp/ki/kd")
            for field in ("kp", "ki", "kd"):
                value = _finite_number(pid[field], f"performance_profile.{axis}.{field}")
                if value < 0.0:
                    raise ProtocolError(f"performance_profile.{axis}.{field} must be non-negative")
        if profile_fields in (
            _PERFORMANCE_PROFILE_V1_FIELDS,
            _PERFORMANCE_PROFILE_FIELDS,
        ):
            hover = _finite_number(
                profile["hover_force_per_thruster_n"],
                "performance_profile.hover_force_per_thruster_n",
            )
            if not 0.0 <= hover <= 10.0:
                raise ProtocolError(
                    "performance_profile.hover_force_per_thruster_n must be in 0..10 N"
                )
            if hover > 55.0 * float(profile["vertical_limit_fraction"]) + 1e-9:
                raise ProtocolError(
                    "performance profile hover force exceeds the vertical thrust limit"
                )
            trim_fields = (
                "vertical_ascent_front_rear_trim_percent",
                "vertical_descent_front_rear_trim_percent",
                "forward_front_rear_trim_percent",
                "backward_front_rear_trim_percent",
            )
            if profile_fields == _PERFORMANCE_PROFILE_FIELDS:
                trim_fields += (
                    "left_lateral_surge_trim_percent",
                    "right_lateral_surge_trim_percent",
                )
            for field in trim_fields:
                value = _finite_number(profile[field], f"performance_profile.{field}")
                if not -100.0 <= value <= 100.0:
                    raise ProtocolError(
                        f"performance_profile.{field} must be in -100..100 percent"
                    )
            for field in (
                "roll_max_moment_nm", "pitch_max_moment_nm", "yaw_max_moment_nm"
            ):
                value = _finite_number(profile[field], f"performance_profile.{field}")
                if not 0.1 <= value <= 20.0:
                    raise ProtocolError(
                        f"performance_profile.{field} must be in 0.1..20 Nm"
                    )

    if "pid_set" in payload and "pid_save" in payload:
        raise ProtocolError("pid_set and pid_save cannot be sent together")
    for field_name in ("pid_set", "pid_save"):
        if field_name not in payload:
            continue
        pid = payload[field_name]
        if not isinstance(pid, dict) or set(pid) != {"type", "kp", "ki", "kd"}:
            raise ProtocolError(f"{field_name} must contain exactly type/kp/ki/kd")
        if isinstance(pid["type"], bool) or not isinstance(pid["type"], int):
            raise ProtocolError(f"{field_name}.type must be an integer")
        if pid["type"] not in (1, 2, 3, 4, 5, 6):
            raise ProtocolError(f"{field_name}.type must be in 1..6")
        for field in ("kp", "ki", "kd"):
            _finite_number(pid[field], f"{field_name}.{field}", absolute_limit=100000.0)

    if "pid_get" in payload:
        value = payload["pid_get"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 6:
            raise ProtocolError("pid_get must be an integer in 0..6")

    return dict(payload)


def encode_control_packet(
    payload: Mapping[str, Any],
    shared_key: str,
    session: str,
    seq: int,
    *,
    protocol_version: int = PROTOCOL_VERSION,
    max_packet_bytes: int = DEFAULT_MAX_PACKET_BYTES,
) -> bytes:
    payload_copy = validate_control_payload(dict(payload))
    if not isinstance(session, str) or not _SESSION_RE.fullmatch(session):
        raise ProtocolError("session must be 32 lowercase hexadecimal characters")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0 or seq > 0x7FFFFFFFFFFFFFFF:
        raise ProtocolError("seq must be an integer in 0..2^63-1")
    if isinstance(protocol_version, bool) or not isinstance(protocol_version, int):
        raise ProtocolError("protocol_version must be an integer")

    body = {
        "protocol_version": protocol_version,
        "session": session,
        "seq": seq,
        "payload": payload_copy,
    }
    digest = hmac.new(_key_bytes(shared_key), _canonical_json(body), hashlib.sha256).hexdigest()
    packet = _canonical_json({**body, "hmac": digest})
    if len(packet) > int(max_packet_bytes):
        raise ProtocolError(f"packet exceeds {int(max_packet_bytes)} bytes")
    return packet


def decode_control_packet(
    data: bytes,
    shared_key: str,
    *,
    protocol_version: int = PROTOCOL_VERSION,
    max_packet_bytes: int = DEFAULT_MAX_PACKET_BYTES,
) -> VerifiedControlPacket:
    if not isinstance(data, bytes):
        raise ProtocolError("packet must be bytes")
    if not data or len(data) > int(max_packet_bytes):
        raise ProtocolError(f"packet size must be 1..{int(max_packet_bytes)} bytes")

    packet = _strict_json_loads(data)
    if not isinstance(packet, dict) or set(packet) != _ENVELOPE_FIELDS:
        raise ProtocolError("envelope fields must be exactly protocol_version/session/seq/payload/hmac")
    if packet["protocol_version"] != protocol_version:
        raise ProtocolError(
            f"unsupported protocol version {packet['protocol_version']!r}; expected {protocol_version}"
        )
    session = packet["session"]
    seq = packet["seq"]
    digest = packet["hmac"]
    if not isinstance(session, str) or not _SESSION_RE.fullmatch(session):
        raise ProtocolError("invalid session")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0 or seq > 0x7FFFFFFFFFFFFFFF:
        raise ProtocolError("invalid sequence")
    if not isinstance(digest, str) or not _HMAC_RE.fullmatch(digest):
        raise ProtocolError("invalid HMAC encoding")

    body = {
        "protocol_version": packet["protocol_version"],
        "session": session,
        "seq": seq,
        "payload": packet["payload"],
    }
    expected = hmac.new(_key_bytes(shared_key), _canonical_json(body), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(digest, expected):
        raise ProtocolError("HMAC authentication failed")

    payload = validate_control_payload(packet["payload"])
    return VerifiedControlPacket(session=session, seq=seq, payload=payload)


def encode_telemetry_packet(
    payload: Mapping[str, Any],
    shared_key: str,
    session: str,
    seq: int,
    *,
    protocol_version: int = PROTOCOL_VERSION,
    max_packet_bytes: int = 16384,
) -> bytes:
    if not isinstance(payload, Mapping):
        raise ProtocolError("telemetry payload must be an object")
    if not isinstance(session, str) or not _SESSION_RE.fullmatch(session):
        raise ProtocolError("telemetry session must be 32 lowercase hexadecimal characters")
    if isinstance(seq, bool) or not isinstance(seq, int) or not 0 <= seq <= 0x7FFFFFFFFFFFFFFF:
        raise ProtocolError("telemetry seq must be an integer in 0..2^63-1")
    body = {
        "protocol_version": int(protocol_version),
        "kind": "telemetry",
        "session": session,
        "seq": seq,
        "payload": dict(payload),
    }
    digest = hmac.new(_key_bytes(shared_key), _canonical_json(body), hashlib.sha256).hexdigest()
    packet = _canonical_json({**body, "hmac": digest})
    if len(packet) > int(max_packet_bytes):
        raise ProtocolError(f"telemetry packet exceeds {int(max_packet_bytes)} bytes")
    if len(packet) > TELEMETRY_WIRE_COMPRESSION_THRESHOLD:
        compressed = _TELEMETRY_COMPRESSED_MAGIC + zlib.compress(packet, level=6)
        if len(compressed) < len(packet):
            return compressed
    return packet


def _decode_telemetry_wire_packet(data: bytes, max_packet_bytes: int) -> bytes:
    """Expand one compressed telemetry datagram with a strict output limit."""
    if not data.startswith(_TELEMETRY_COMPRESSED_MAGIC):
        return data
    compressed = data[len(_TELEMETRY_COMPRESSED_MAGIC):]
    if not compressed:
        raise ProtocolError("compressed telemetry packet is empty")
    try:
        decoder = zlib.decompressobj()
        packet = decoder.decompress(compressed, int(max_packet_bytes) + 1)
        if decoder.unconsumed_tail or len(packet) > int(max_packet_bytes):
            raise ProtocolError(
                f"expanded telemetry packet exceeds {int(max_packet_bytes)} bytes"
            )
        packet += decoder.flush()
    except zlib.error as exc:
        raise ProtocolError(f"invalid compressed telemetry packet: {exc}") from exc
    if (
        not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
        or len(packet) > int(max_packet_bytes)
    ):
        raise ProtocolError("invalid or oversized compressed telemetry packet")
    return packet


def decode_telemetry_packet(
    data: bytes,
    shared_key: str,
    *,
    protocol_version: int = PROTOCOL_VERSION,
    max_packet_bytes: int = 16384,
) -> VerifiedTelemetryPacket:
    if not isinstance(data, bytes) or not data or len(data) > int(max_packet_bytes):
        raise ProtocolError(f"telemetry packet size must be 1..{int(max_packet_bytes)} bytes")
    data = _decode_telemetry_wire_packet(data, int(max_packet_bytes))
    packet = _strict_json_loads(data)
    if not isinstance(packet, dict) or set(packet) != _TELEMETRY_ENVELOPE_FIELDS:
        raise ProtocolError("invalid telemetry envelope fields")
    if packet["protocol_version"] != protocol_version or packet["kind"] != "telemetry":
        raise ProtocolError("unsupported telemetry protocol or kind")
    session = packet["session"]
    seq = packet["seq"]
    digest = packet["hmac"]
    if not isinstance(session, str) or not _SESSION_RE.fullmatch(session):
        raise ProtocolError("invalid telemetry session")
    if isinstance(seq, bool) or not isinstance(seq, int) or not 0 <= seq <= 0x7FFFFFFFFFFFFFFF:
        raise ProtocolError("invalid telemetry sequence")
    if not isinstance(digest, str) or not _HMAC_RE.fullmatch(digest):
        raise ProtocolError("invalid telemetry HMAC encoding")
    if not isinstance(packet["payload"], dict):
        raise ProtocolError("telemetry payload must be an object")
    body = {
        "protocol_version": packet["protocol_version"],
        "kind": packet["kind"],
        "session": session,
        "seq": seq,
        "payload": packet["payload"],
    }
    expected = hmac.new(_key_bytes(shared_key), _canonical_json(body), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(digest, expected):
        raise ProtocolError("telemetry HMAC authentication failed")
    return VerifiedTelemetryPacket(session=session, seq=seq, payload=dict(packet["payload"]))
