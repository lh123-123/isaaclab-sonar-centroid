"""PC pilot input conditioning; no PID, motor mapping or hardware ownership.

Drain nonblocking HID to its newest state, and limit only translation/vertical
command changes. Yaw/pitch neutral reaches the native FC target generator
immediately; depth-hold vertical release also bypasses the tail. Loss of control
and B/Disarm never leave a smoothing tail.
"""
from dataclasses import dataclass, replace
import math

from flydigi_apex5 import Apex5State, parse_input_report


@dataclass(frozen=True)
class HidBatch:
    state: Apex5State | None
    backlogged: bool = False
    disarm_requested: bool = False


def read_latest_input(controller, max_reports=64):
    """Bounded drain; an undrained FIFO is NOT evidence of fresh pilot input.

    Do not replay historical Arm/mode/servo edges. B/Disarm is the sole edge
    latched even in overflow, including a quick press-and-release.
    This cannot detect a driver's hidden queue or assign a device timestamp.
    """
    if type(max_reports) is not int or not 1 <= max_reports <= 256:
        raise ValueError('HID drain bound must be an integer in 1..256')
    latest = None
    disarm = False
    for _ in range(max_reports):
        report = controller.read(64)
        if not report:
            if latest is not None and disarm:
                latest = replace(Apex5State.neutral(), buttons=1 << 1)
            return HidBatch(latest, disarm_requested=disarm)
        raw = bytes(report)
        if len(raw) >= 32 and raw[1:4] == bytes((0x5A, 0xA5, 0xEF)):
            latest = parse_input_report(raw)
            disarm = disarm or bool(latest.buttons & (1 << 1))
    return HidBatch(None, backlogged=True, disarm_requested=disarm)


class MotionSlewLimiter:
    """Time-based normalized-command slew, not physical speed or braking.

    Defaults: 0->full request in ~0.25s, full->neutral in ~0.125s. No active
    reverse-thrust braking is inferred without velocity/position feedback.
    """
    def __init__(self, enabled=True, rise_per_s=4.0, fall_per_s=8.0):
        if type(enabled) is not bool:
            raise ValueError('motion smoothing enabled must be boolean')
        for value in (rise_per_s, fall_per_s):
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0.5 <= value <= 20):
                raise ValueError('motion slew rates must be finite in 0.5..20 /s')
        self.enabled, self.rise, self.fall = enabled, float(rise_per_s), float(fall_per_s)
        self.reset()

    @classmethod
    def from_profile(cls, profile):
        settings = profile.get('motion_smoothing', {})
        if not isinstance(settings, dict):
            raise ValueError('motion_smoothing must be a configuration object')
        return cls(settings.get('enabled', True), settings.get('rise_per_s', 4.0),
                   settings.get('fall_per_s', 8.0))

    def reset(self):
        self.xy = (0.0, 0.0)
        self.vertical = 0.0
        self.last_time = None

    @staticmethod
    def _move(current, desired, dt, rise, fall):
        old_size, new_size = math.hypot(*current), math.hypot(*desired)
        reversing = sum(a*b for a, b in zip(current, desired)) < -1e-9
        goal = (0.0,) * len(current) if reversing else desired
        rate = fall if reversing or new_size < old_size else rise
        delta = tuple(b-a for a, b in zip(current, goal))
        distance = math.hypot(*delta)
        if distance == 0:
            return goal
        fraction = min(1.0, rate * dt / distance)
        return tuple(a + d*fraction for a, d in zip(current, delta))

    @staticmethod
    def _cap(vector, maximum):
        size = math.hypot(*vector)
        scale = min(1.0, maximum / size) if size else 1.0
        return tuple(value * scale for value in vector)

    @staticmethod
    def _neutral(command):
        return replace(command, x=0, y=0, z=500, r=0, s=0, t=0,
                       aux1=0, aux2=0, aux3=0, aux4=0, aux5=0, aux6=0)

    def update(self, command, now, *, active, motion_gain=1.0, vertical_gain=1.0, depth_hold=False):
        for gain in (motion_gain, vertical_gain):
            if not math.isfinite(gain) or not 0 < gain <= 1:
                raise ValueError('motion gains must be finite in (0,1]')
        if not active or command.buttons & (1 << 1) or not math.isfinite(now):
            self.reset()
            return self._neutral(command)
        if not self.enabled:
            self.reset()
            return command
        dt = 0.0 if self.last_time is None else now - self.last_time
        if not 0 <= dt <= 0.1:
            self.reset()  # A paused GUI must not replay an old smoothing tail.
            dt = 0.0
        self.last_time = now
        desired_xy = self._cap((command.x / 1000, command.y / 1000), motion_gain)
        desired_vertical = max(-vertical_gain, min(vertical_gain, (command.z - 500) / 500))
        self.xy = self._cap(self._move(self.xy, desired_xy, dt, self.rise, self.fall), motion_gain)
        if depth_hold and command.z == 500:
            # Let the flight controller see release now, not after a PC ramp.
            # Its native vertical velocity/acceleration loop owns inertia braking.
            self.vertical = 0.0
        else:
            self.vertical = self._move((self.vertical,), (desired_vertical,), dt,
                                       self.rise, self.fall)[0]
        # Selecting a lower gear reduces authority immediately, without an old
        # high-gear tail. Rising gear remains slew-limited to its full endpoint.
        self.vertical = max(-vertical_gain, min(vertical_gain, self.vertical))
        return replace(command, x=round(self.xy[0] * 1000), y=round(self.xy[1] * 1000),
                       z=round(500 + self.vertical * 500))
