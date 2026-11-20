"""Explicit native ArduPilot level calibration, not a display-only offset.

Never runs at startup, never alters yaw/board rotation, never arms or commands
motion. Requires stopped control, fresh disarmed Manual and an operator-confirmed
horizontal, stationary body. Completion requires ACK, trim readback and actual
fresh near-level ATTITUDE; no synthetic angles are published.
"""
from collections import deque
import math


class LevelZero:
    PARAMS = ('AHRS_ORIENTATION', 'AHRS_TRIM_X', 'AHRS_TRIM_Y', 'AHRS_TRIM_Z')

    def __init__(self, target_system=1):
        self.target_system = target_system
        self.link = None
        self.phase = None
        self.samples = deque(maxlen=80)
        self.values = {}
        self.before = {}
        self.deadline = 0
        self.next_heartbeat = 0
        self.notice = '水平放置后可置零；零点保存在飞控，不随启动重置。'
        self.report = None
        self.ack_time = float('-inf')

    @property
    def busy(self):
        return self.phase is not None

    def stable(self, now, near_level=False):
        recent = [s for s in self.samples if 0 <= now - s[0] <= 1.5]
        limit = 1.5 if near_level else 9.0
        return (len(recent) >= 5 and recent[-1][0] - recent[0][0] >= .8
                and now - recent[-1][0] <= .4
                and all(max(abs(v) for v in s[2]) <= .05 for s in recent)
                and all(max(abs(v) for v in s[1][:2]) <= limit for s in recent)
                and all(max(s[1][axis] for s in recent) - min(s[1][axis] for s in recent) <= .6
                        for axis in (0, 1)))

    def start(self, link, modes, now, *, stopped, horizontal_confirmed):
        if self.busy:
            raise ValueError('水平置零正在进行，请等待结果')
        if not stopped or not horizontal_confirmed or not modes.fresh(now) or modes.armed or modes.mode != 19 or modes.pending is not None:
            raise ValueError('先上锁、切手动、停止控制与电机测试，并将车身放水平')
        if not self.stable(now):
            raise ValueError('姿态尚未静止或偏差超过9°；保持水平静止后重试，大偏差需安装校正')
        self.link, self.phase = link, 'before'
        self.mode_state = modes
        self.values, self.before, self.report = {}, {}, None
        self.deadline = now + 5
        self.notice = '正在读取飞控原水平参考…'
        self._request_params()

    def _request_params(self):
        self.values = {}
        for name in self.PARAMS:
            self.link.mav.param_request_read_send(self.target_system, 1, name.encode('ascii'), -1)

    def fail(self, text):
        self.phase = None
        self.notice = text + '；不自动重试，请保持上锁并检查QGC消息。'

    def observe(self, msg, now):
        if msg.get_srcSystem() != self.target_system or msg.get_srcComponent() != 1:
            return
        kind = msg.get_type()
        if kind == 'ATTITUDE':
            angles = tuple(getattr(msg, a, float('nan')) for a in ('roll', 'pitch', 'yaw'))
            rates = tuple(getattr(msg, a, float('nan')) for a in ('rollspeed', 'pitchspeed', 'yawspeed'))
            if all(math.isfinite(v) for v in angles + rates):
                self.samples.append((now, tuple(math.degrees(v) for v in angles), rates))
        if not self.busy:
            return
        if kind == 'HEARTBEAT' and (msg.type != 12 or msg.autopilot != 3 or msg.base_mode & 128 or msg.custom_mode != 19):
            self.fail('置零中车辆不再是上锁手动状态，结果未确认')
        elif kind == 'COMMAND_ACK' and msg.command == 241 and self.phase == 'calibrate':
            if msg.result == 0:
                self.ack_time = now
                self.phase, self.deadline = 'after', now + 8
                self.notice = '飞控接受水平校准，等待独立读回与真实姿态…'
                self.samples.clear()
                self._request_params()
            elif msg.result != 5:
                self.fail(f'飞控拒绝水平校准（结果{msg.result}）')
        elif kind == 'PARAM_VALUE' and self.phase in ('before', 'after'):
            name = msg.param_id.decode('ascii') if isinstance(msg.param_id, bytes) else msg.param_id
            name = name.rstrip('\x00')
            if name not in self.PARAMS or not math.isfinite(msg.param_value):
                return
            if name.startswith('AHRS_TRIM') and msg.param_type != 9:
                self.fail('水平参考参数类型异常')
                return
            if name == 'AHRS_ORIENTATION' and msg.param_type != 2:
                self.fail('安装方向参数类型异常')
                return
            self.values[name] = float(msg.param_value)
            if set(self.values) == set(self.PARAMS):
                if self.phase == 'before':
                    if (not self.stable(now) or not self.mode_state.fresh(now) or self.mode_state.armed
                            or self.mode_state.mode != 19 or self.mode_state.pending is not None):
                        self.fail('读取参数期间姿态不再静止，未发送校准')
                        return
                    if self.values['AHRS_ORIENTATION'] != 0 or abs(self.values['AHRS_TRIM_Z']) > 1e-6:
                        self.fail('已有安装旋转或航向修正，不能直接采用当前简易置零方案')
                        return
                    self.before = dict(self.values)
                    self.phase, self.deadline = 'calibrate', now + 35
                    self.notice = '正在校准真实横滚/俯仰零点，请勿移动车身…'
                    self.link.mav.command_long_send(self.target_system, 1, 241, 0, 0, 0, 0, 0, 2, 0, 0)
                else:
                    if (self.values['AHRS_ORIENTATION'] != self.before['AHRS_ORIENTATION']
                            or self.values['AHRS_TRIM_Z'] != self.before['AHRS_TRIM_Z']
                            or max(abs(self.values[n]) for n in ('AHRS_TRIM_X', 'AHRS_TRIM_Y')) > math.radians(10)):
                        self.fail('水平参考读回异常，结果未确认')
                        return
                    self.phase = 'settle'

    def tick(self, modes, now):
        if not self.busy:
            return
        if now >= self.next_heartbeat:
            self.link.mav.heartbeat_send(6, 8, 0, 0, 0)
            self.next_heartbeat = now + 1
        if modes.armed:
            self.fail('置零中收到解锁状态')
        elif now >= self.deadline:
            self.fail('水平置零结果未确认（超时或实际角度未回零）')
        elif self.phase == 'settle' and modes.fresh(now) and self.stable(now, near_level=True):
            self.report = {'native_level_calibration': True, 'before': self.before, 'after': dict(self.values),
                           'actual_attitude_deg': self.samples[-1][1], 'yaw_reference_reset': False,
                           'board_rotation_changed': False, 'hardware_balance_tested': False}
            self.phase = None
            self.notice = '横滚/俯仰零点已由飞控确认并保存；航向仍用原生角度。'
