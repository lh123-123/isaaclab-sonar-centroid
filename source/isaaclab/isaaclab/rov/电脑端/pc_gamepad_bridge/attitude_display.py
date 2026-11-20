"""Native FC Euler display, without offsets, integration or frame rotation.

The caller converts MAVLink ATTITUDE radians to degrees. Only displayed heading
is wrapped to 0..360. FC sensor calibration is retained and never changed here.
"""
import math


class AttitudeDisplay:
    def __init__(self):
        self.absolute = self.relative = self.continuous = self.reference = None
        self.last_update = float('-inf')
        self.boot_ms = None
        # Legacy reference/relative/continuous fields remain None for old clients.
        self.notice = '原生姿态：直接读取飞控，不置零、不累加、不换算安装角。'

    def fresh(self, now, heartbeat_fresh=True):
        return heartbeat_fresh and self.absolute is not None and 0 <= now - self.last_update <= 1

    def observe(self, angles, now, *, disarmed, boot_ms=None):
        if len(angles) != 3 or not all(math.isfinite(v) for v in (*angles, now)):
            return False
        if boot_ms is not None:
            if type(boot_ms) is not int or not 0 <= boot_ms <= 0xffffffff:
                return False
            if self.boot_ms is not None:
                delta = (boot_ms - self.boot_ms) & 0xffffffff
                if delta == 0:
                    return False
                if delta > 0x7fffffff:
                    if self.boot_ms - boot_ms <= 1000:
                        return False  # small out-of-order packet, not a reboot
                    self.notice = '飞控重启：继续显示原生姿态，不重新置零。'
            self.boot_ms = boot_ms
        self.absolute = tuple(angles)
        self.last_update = now
        return True

    def zero(self, now, *, disarmed):
        """Compatibility no-op: old callbacks cannot establish a new datum."""
        self.notice = '原生姿态模式不执行置零，角度以飞控回传为准。'
        return False

    def angle_text(self, now, heartbeat_fresh):
        if not self.fresh(now, heartbeat_fresh):
            return '姿态未知（回传超时）'
        r, p, y = self.absolute
        return f'横滚 {r:+.1f}°    俯仰 {p:+.1f}°\n航向 {y % 360:.1f}°'

    def raw_text(self, now, heartbeat_fresh):
        if not self.fresh(now, heartbeat_fresh):
            return '飞控原生姿态：未知'
        r, p, y = self.absolute
        return f'飞控原生：横滚 {r:+.1f}° / 俯仰 {p:+.1f}°\n航向 {y % 360:.1f}°（与QGC一致）'
