"""Capture native FC roll/pitch as persistent targets, never calibrate or zero."""
from collections import deque
import math

PARAMS = ('ROV_LVL_R', 'ROV_LVL_P')
COMMAND = 31010  # MAV_CMD_USER_1 in the pinned ArduPilot MAVLink dialect
MAGIC = 61007


class LevelTarget:
    def __init__(self, target_system=1):
        self.target_system = target_system
        self.actual = None
        self.phase = None
        self.link = None
        self.values = {}
        self.samples = deque(maxlen=100)
        self.deadline = 0
        self.expected = None
        self.notice = '调平目标：等待读回飞控'

    @property
    def busy(self):
        return self.phase is not None

    def _request(self):
        self.values = {}
        for name in PARAMS:
            self.link.mav.param_request_read_send(self.target_system, 1, name.encode('ascii'), -1)

    def read(self, link, now):
        if self.busy:
            raise ValueError('调平目标正在处理')
        self.link, self.phase, self.deadline = link, 'read', now + 4
        self._request()

    def stable(self, now):
        recent = [s for s in self.samples if 0 <= now - s[0] <= 1.5]
        return (len(recent) >= 5 and recent[-1][0] - recent[0][0] >= .8
                and now - recent[-1][0] <= .4
                and all(math.hypot(*s[1]) <= 10 and math.sqrt(sum(v*v for v in s[2])) <= .05 for s in recent)
                and all(max(s[1][a] for s in recent) - min(s[1][a] for s in recent) <= .3 for a in (0, 1)))

    def capture(self, link, modes, now, *, stopped):
        if self.busy:
            raise ValueError('调平目标正在处理，请稍等')
        if self.actual is None:
            raise ValueError('未读到 ROV_LVL_R/P；需要支持调平目标的新固件')
        if (not stopped or not modes.fresh(now) or modes.armed or modes.mode != 19 or modes.pending is not None):
            raise ValueError('先上锁、切手动、停止手柄控制和电机测试')
        if not self.stable(now):
            raise ValueError('请将车身放平并静止约2秒；原生横滚/俯仰合成偏差须≤10°')
        self.expected = self.samples[-1][1]
        self.link, self.phase, self.deadline = link, 'capture', now + 4
        self.notice = '飞控正在保存当前调平目标；保持静止'
        # The FC captures its OWN current AHRS angles. No angles are sent by PC.
        link.mav.command_int_send(self.target_system, 1, 2, COMMAND, 0, 0,
                                  MAGIC, 1, 0, 0, 0, 0, 0)

    def observe(self, msg, now):
        if msg.get_srcSystem() != self.target_system or msg.get_srcComponent() != 1:
            return
        kind = msg.get_type()
        if kind == 'ATTITUDE':
            angles = (float(msg.roll), float(msg.pitch))
            rates = tuple(float(getattr(msg, a)) for a in ('rollspeed', 'pitchspeed', 'yawspeed'))
            if all(math.isfinite(v) for v in angles + rates):
                self.samples.append((now, tuple(math.degrees(v) for v in angles), rates))
        if not self.busy:
            return
        if kind == 'HEARTBEAT' and self.phase != 'read' and (msg.base_mode & 128 or msg.custom_mode != 19):
            self.fail('保存期间状态改变，调平目标未确认')
        elif kind == 'COMMAND_ACK' and msg.command == COMMAND and self.phase == 'capture':
            if msg.result != 0:
                self.fail(f'飞控未接受调平目标（{msg.result}）')
                return
            self.phase, self.deadline = 'verify', now + 4
            self._request()
        elif kind == 'PARAM_VALUE' and self.phase in ('read', 'verify'):
            name = msg.param_id.decode('ascii') if isinstance(msg.param_id, bytes) else msg.param_id
            name = name.rstrip('\x00')
            if name not in PARAMS:
                return
            value = float(msg.param_value)
            if msg.param_type != 9 or not math.isfinite(value) or abs(value) > 10:
                self.fail('飞控调平目标数据异常')
                return
            self.values[name] = value
            if len(self.values) == 2:
                pair = tuple(self.values[p] for p in PARAMS)
                if math.hypot(*pair) > 10.001:
                    self.fail('飞控调平目标超过允许范围')
                    return
                if self.phase == 'verify' and max(abs(a-b) for a, b in zip(pair, self.expected)) > .5:
                    self.fail('读回目标与静止采样不符，请保持上锁检查')
                    return
                self.actual, self.phase = pair, None
                self.notice = f'飞控调平目标：横滚 {pair[0]:+.2f}° / 俯仰 {pair[1]:+.2f}°（已读回）'

    def tick(self, modes, now):
        if self.busy and self.phase != 'read' and (not modes.fresh(now) or modes.armed or modes.mode != 19):
            self.fail('连接或上锁手动状态改变，调平目标未确认')
        elif self.busy and now >= self.deadline:
            self.fail('调平目标读取/保存超时；不自动重试')

    def fail(self, text):
        self.phase = None
        self.actual = None
        self.notice = text
