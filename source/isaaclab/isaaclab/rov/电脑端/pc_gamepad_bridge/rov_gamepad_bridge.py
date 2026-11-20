#!/usr/bin/env python3
"""Read a Flydigi APEX 5 directly and optionally send ArduSub control.

Monitoring is the default.  MAVLink output requires --send, and forwarding the
A/Arm button additionally requires --allow-arm.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from dataclasses import replace

import hid

from flydigi_apex5 import (
    PRODUCT_ID,
    VENDOR_ID,
    VENDOR_USAGE_PAGE,
    Apex5State,
    build_manual_control,
)
from control_profile import (
    LocalActionController,
    button_mask,
    load_profile,
    resolve_button_conflicts,
)
from flight_mode import FlightModeController
from runtime_support import ResumeGate, SingleInstance
from motion_input import MotionSlewLimiter, read_latest_input
from servo_control import ServoController


ACQUIRE_INTERVAL_SECONDS = 20.0
INPUT_TIMEOUT_SECONDS = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", help="send MAVLink to the Pi")
    parser.add_argument(
        "--allow-arm",
        action="store_true",
        help="allow the A button through; ignored without --send",
    )
    parser.add_argument("--pi-ip", default="192.168.1.3")
    parser.add_argument("--port", type=int, default=14550)
    parser.add_argument("--telemetry-port", type=int, default=14551)
    parser.add_argument("--target-system", type=int, default=1)
    parser.add_argument("--profile", default=None, help="controller profile JSON")
    parser.add_argument("--rate", type=float, default=25.0)
    parser.add_argument("--stick-deadzone", type=float, default=None)
    parser.add_argument("--trigger-deadzone", type=float, default=None)
    parser.add_argument(
        "--vertical-scale",
        type=float,
        default=None,
        help="vertical authority before the configured motion gear",
    )
    return parser.parse_args()


def find_vendor_interface() -> bytes:
    matches = [
        device
        for device in hid.enumerate(VENDOR_ID, PRODUCT_ID)
        if device.get("usage_page") == VENDOR_USAGE_PAGE
    ]
    if not matches:
        raise RuntimeError("APEX 5 vendor HID interface 37D7:2501/FFA0 not found")
    return matches[0]["path"]


def acquire_command() -> list[int]:
    command = [0x03, 0x5A, 0xA5, 0x1C, 23, 1] + list(b"ROVBridge")
    return (command + [0] * 32)[:32]


def open_controller() -> hid.device:
    controller = hid.device()
    try:
        controller.open_path(find_vendor_interface())
        controller.set_nonblocking(1)
        if controller.write(acquire_command()) < 0:
            raise RuntimeError("请在飞智空间站启用允许第三方应用接管映射")
    except Exception:
        controller.close()
        raise
    return controller


class UdpMavlinkOutput:
    """MAVLink UDP sender that leaves the Windows source port ephemeral.

    pymavlink's udpout transport binds the local socket to the destination port
    on Windows.  QGC already owns local UDP 14550, so that behaviour prevents
    the controller bridge from starting.  The Pi bridge accepts packets by PC
    address and does not require a fixed source port.
    """

    def __init__(self, host: str, port: int, source_system=255, source_component=190):
        from pymavlink import mavutil

        mavutil.set_dialect("ardupilotmega")
        self.destination = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mav = mavutil.mavlink.MAVLink(
            self,
            srcSystem=source_system,
            srcComponent=source_component,
        )

    def write(self, payload: bytes) -> int:
        return self.socket.sendto(payload, self.destination)

    def close(self) -> None:
        self.socket.close()


def open_mavlink(host: str, port: int):
    os.environ.setdefault("MAVLINK20", "1")
    return UdpMavlinkOutput(host, port)


def send_command(link, target_system: int, command) -> None:
    link.mav.manual_control_send(
        target_system,
        command.x,
        command.y,
        command.z,
        command.r,
        command.buttons,
        command.buttons2,
        command.enabled_extensions,
        command.s,
        command.t,
        command.aux1,
        command.aux2,
        command.aux3,
        command.aux4,
        command.aux5,
        command.aux6,
    )


def open_telemetry(port: int):
    os.environ.setdefault("MAVLINK20", "1")
    from pymavlink import mavutil
    return mavutil.mavlink_connection(f"udpin:0.0.0.0:{port}", dialect="ardupilotmega")


def poll_telemetry(receiver, modes: FlightModeController, now: float, *observers) -> None:
    if receiver is not None:
        for _ in range(100):
            message = receiver.recv_match(blocking=False)
            if message is None:
                break
            modes.observe(message, now)
            for observer in observers:
                observer.observe(message, now)
    modes.tick(now)


def request_mode(link, modes: FlightModeController, action: str, now: float) -> None:
    if link is None:
        modes.notice = "监视预览：未发送模式请求"
        return
    target = modes.request(action, now)
    if target is not None:
        # MAV_CMD_DO_SET_MODE, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED.
        link.mav.command_long_send(modes.target_system, 1, 176, 0,
                                   1, target, 0, 0, 0, 0, 0)


def request_power_telemetry(link, target_system: int) -> None:
    """Request current/voltage messages at 5 Hz on this MAVLink route."""
    from pymavlink import mavutil

    period_us = 200_000
    for message_id in (mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS,
                       mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS):
        link.mav.command_long_send(
            target_system,
            mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            message_id,
            period_us,
            0.0, 0.0, 0.0, 0.0, 0.0,
        )


def send_gimbal_angle(link, target_system: int, angle_deg: float) -> None:
    """Command the configured primary mount to an absolute pitch angle."""
    from pymavlink import mavutil

    link.mav.command_long_send(
        target_system,
        mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1,
        mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
        0,
        float(angle_deg),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        float(mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING),
    )


def send_servo_pwm(link, target_system: int, channel: int, pwm: int) -> None:
    """Send a raw PWM target to a disabled/manual AUX output."""
    from pymavlink import mavutil

    link.mav.command_long_send(
        target_system,
        mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        float(channel),
        float(pwm),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def run_bridge() -> int:
    args = parse_args()
    profile = load_profile(args.profile) if args.profile else load_profile()
    actions = LocalActionController(profile)
    motion_limiter = MotionSlewLimiter.from_profile(profile)
    servos = ServoController.from_profile(profile)
    forwarded_button_mask = button_mask(profile["button_forward_indices"])
    axis_settings = profile["axis"]
    stick_deadzone = (
        args.stick_deadzone
        if args.stick_deadzone is not None
        else float(axis_settings["stick_deadzone"])
    )
    trigger_deadzone = (
        args.trigger_deadzone
        if args.trigger_deadzone is not None
        else float(axis_settings["trigger_deadzone"])
    )
    vertical_scale = (
        args.vertical_scale
        if args.vertical_scale is not None
        else float(axis_settings["vertical_scale"])
    )
    axis_snap_inner_deg = float(axis_settings["axis_snap_inner_deg"])
    axis_snap_outer_deg = float(axis_settings["axis_snap_outer_deg"])
    if not 0.0 < args.rate <= 100.0:
        raise SystemExit("--rate must be in (0, 100]")
    if not 0.0 < vertical_scale <= 1.0:
        raise SystemExit("--vertical-scale must be in (0, 1]")
    if args.allow_arm and not args.send:
        print("NOTE: --allow-arm has no effect without --send")

    controller = open_controller()
    link = open_mavlink(args.pi_ip, args.port) if args.send else None
    receiver = open_telemetry(args.telemetry_port) if args.send else None
    modes = FlightModeController(args.target_system)
    resume = ResumeGate()
    default_requested = default_confirmed = False
    mode = "LIVE SEND" if args.send else "MONITOR ONLY"
    print(f"APEX 5 ready: {mode}; A/Arm={'enabled' if args.allow_arm else 'blocked'}")
    print("QGC joystick Enable must remain unchecked when LIVE SEND is used.")

    state = Apex5State.neutral()
    last_input = float("-inf")
    next_acquire = 0.0
    next_send = 0.0
    next_status = 0.0
    power_stream_requested = False
    send_period = 1.0 / args.rate

    try:
        while True:
            now = time.monotonic()
            if now >= next_acquire:
                controller.write(acquire_command())
                next_acquire = now + ACQUIRE_INTERVAL_SECONDS

            try:
                batch = read_latest_input(controller)
            except OSError as exc:
                raise RuntimeError(f"APEX 5 HID read failed: {exc}") from exc
            if batch.backlogged:
                state = Apex5State.neutral()
                last_input = float('-inf')
            elif batch.state is not None:
                state = batch.state
                last_input = now

            stale = now - last_input > INPUT_TIMEOUT_SECONDS
            effective_state = Apex5State.neutral() if stale else state
            poll_telemetry(receiver, modes, now, servos)
            if link is not None and modes.fresh(now) and not power_stream_requested:
                request_power_telemetry(link, args.target_system)
                power_stream_requested = True
            ready = resume.update(effective_state, not stale and modes.fresh(now), now)
            if link is not None and not default_confirmed:
                if ready and not default_requested and not modes.armed:
                    request_mode(link, modes, 'reset_manual', now)
                    default_requested = True
                if default_requested and modes.fresh(now) and modes.mode == 19 and modes.pending is None:
                    default_confirmed = True
                ready = ready and default_confirmed

            action_update = actions.update(effective_state.buttons)
            if action_update.gimbal_angle_deg is not None:
                print(
                    f"gimbal target: {action_update.gimbal_angle_deg:+.0f} deg",
                    flush=True,
                )
                if link is not None and ready:
                    send_gimbal_angle(
                        link, args.target_system, action_update.gimbal_angle_deg
                    )
            if action_update.motion_gain is not None:
                print(f"stick gain: {action_update.motion_gain:.0%}", flush=True)
            if action_update.vertical_gain is not None:
                print(f"vertical gain: {action_update.vertical_gain:.0%}", flush=True)
            if action_update.mode_action is not None:
                if ready or link is None:
                    request_mode(link, modes, action_update.mode_action, now)
                else:
                    modes.notice = "等待链路正常且全部按键/摇杆/扳机归中0.5秒"
                print(modes.notice, flush=True)

            safe_buttons = resolve_button_conflicts(effective_state.buttons)
            if batch.disarm_requested:
                safe_buttons = 1 << 1
            servo_commands = servos.update(
                safe_buttons & ~button_mask((15, 16)) if effective_state.buttons &
                (1 << int(profile["servos"]["angle"].get("reset_button", 14)))
                else safe_buttons,
                now,
                enabled=link is not None and ready and not stale,
            )
            for servo_command in servo_commands:
                send_servo_pwm(
                    link,
                    args.target_system,
                    servo_command.channel,
                    servo_command.pwm,
                )
            if action_update.angle_servo_reset_pwm is not None and link is not None and ready and not stale:
                preset = servos.command_preset("angle", action_update.angle_servo_reset_pwm, now)
                send_servo_pwm(link, args.target_system, preset.channel, preset.pwm)
                print(f"angle servo reset: {preset.pwm} us", flush=True)
            control_state = replace(effective_state, buttons=safe_buttons)
            if safe_buttons & (1 << 1):
                control_state = replace(Apex5State.neutral(), buttons=1 << 1)
            command = build_manual_control(
                control_state,
                stick_deadzone=stick_deadzone,
                trigger_deadzone=trigger_deadzone,
                vertical_scale=vertical_scale,
                motion_scale=actions.motion.gain,
                vertical_motion_scale=actions.vertical.gain,
                axis_snap_inner_deg=axis_snap_inner_deg,
                axis_snap_outer_deg=axis_snap_outer_deg,
                yaw_response_exponent=float(axis_settings.get("yaw_response_exponent", 1.0)),
                allow_arm=args.allow_arm and args.send and ready and not modes.armed
                    and command_neutral_for_arm(effective_state),
                button_mask=forwarded_button_mask,
            )
            if link is not None:
                command = motion_limiter.update(command, now,
                    active=ready and not stale and modes.pending is None,
                    motion_gain=actions.motion.gain, vertical_gain=actions.vertical.gain,
                    depth_hold=modes.fresh(now) and modes.mode in (2, 22))
            else:
                motion_limiter.reset()

            can_send = (not stale and ready) or bool(safe_buttons & (1 << 1))
            if link is not None and can_send and (now >= next_send or batch.disarm_requested):
                send_command(link, args.target_system, command)
                next_send = now + send_period

            if now >= next_status:
                extras = ",".join(sorted(effective_state.extra_buttons)) or "-"
                print(
                    f"x={command.x:+5d} y={command.y:+5d} z={command.z:4d} "
                    f"yaw={command.r:+5d} pitch={command.s:+4d} "
                    f"stick_gain={actions.motion.gain:.0%} vertical_gain={actions.vertical.gain:.0%} "
                    f"extra={extras} stale={stale} "
                    f"{modes.status(now)} | {modes.notice} | "
                    f"{servos.status_text().replace(chr(10), ' | ')}",
                    flush=True,
                )
                next_status = now + 0.25

            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        if link is not None:
            neutral = build_manual_control(Apex5State.neutral())
            for _ in range(5):
                send_command(link, args.target_system, neutral)
                time.sleep(0.02)
            link.close()
        if receiver is not None:
            receiver.close()
        controller.close()
    return 0


def command_neutral_for_arm(state):
    from runtime_support import neutral_input
    return neutral_input(replace(state, buttons=state.buttons & ~1))


def main() -> int:
    guard = SingleInstance()
    try:
        return run_bridge()
    finally:
        guard.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
