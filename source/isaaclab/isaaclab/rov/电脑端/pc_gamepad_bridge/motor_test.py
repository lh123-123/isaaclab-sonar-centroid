"""Explicit start/stop ArduSub motor-test session; never writes servo functions.

DO_MOTOR_TEST uses zero-based logical motors, PWM throttle type and BOARD order.
ArduSub ignores param4: its real watchdog is 500 ms without a test packet.
GUI toggles start/stop; focus loss stops output. No fixed test-duration limit.
"""
from __future__ import annotations

import logging

PHYSICAL_TO_MOTOR = (2, 1, 4, 3, 6, 5, 8, 7)
EXPECTED_PARAMS = {"FRAME_CONFIG": 2, **{
    f"SERVO{i}_FUNCTION": 32 + motor
    for i, motor in enumerate(PHYSICAL_TO_MOTOR, 1)}}
POSITIONS = ("左前斜推", "右前斜推", "左后斜推", "右后斜推",
             "左前垂推", "右前垂推", "左后垂推", "右后垂推")
DIRECTION_PARAMS = tuple(f"MOT_{i}_DIRECTION" for i in range(1, 9))
HORIZONTAL_DIRECTION_PARAM = "MOT_H_REV"
HORIZONTAL_ACTIONS = frozenset(("左移", "右移", "左转", "右转"))
# ArduSub dbe79216, SUB_FRAME_VECTORED_6DOF logical motor factors.
# DO_MOTOR_TEST PWM bypasses these factors and MOT_n_DIRECTION: apply both here.
MOTION_FACTORS = {
    "前进": (-1, -1, 1, 1, 0, 0, 0, 0),
    "后退": (1, 1, -1, -1, 0, 0, 0, 0),
    "上升": (0, 0, 0, 0, -1, -1, -1, -1),
    "下降": (0, 0, 0, 0, 1, 1, 1, 1),
    "左移": (-1, 1, -1, 1, 0, 0, 0, 0),
    "右移": (1, -1, 1, -1, 0, 0, 0, 0),
    "左转": (-1, 1, 1, -1, 0, 0, 0, 0),
    "右转": (1, -1, -1, 1, 0, 0, 0, 0),
    "抬头": (0, 0, 0, 0, -1, -1, 1, 1),
    "低头": (0, 0, 0, 0, 1, 1, -1, -1),
    "左倾": (0, 0, 0, 0, -1, 1, -1, 1),
    "右倾": (0, 0, 0, 0, 1, -1, 1, -1),
}
MOTION_GROUPS = {
    "前后 / 升潜": ("前进", "后退", "上升", "下降"),
    "侧移 / 转向": ("左移", "右移", "左转", "右转"),
    "俯仰 / 横滚": ("抬头", "低头", "左倾", "右倾"),
}


def motion_pwm_plan(action, low_pwm, high_pwm, params):
    """Physical MAIN outputs, using read-back directions, not propeller guesses."""
    if action not in MOTION_FACTORS:
        raise ValueError("请选择有效的机身运动方向")
    if (isinstance(low_pwm, bool) or isinstance(high_pwm, bool)
            or not isinstance(low_pwm, int) or not isinstance(high_pwm, int)
            or not 1000 <= low_pwm <= 1500 <= high_pwm <= 2000):
        raise ValueError("低侧PWM须为1000～1500整数，高侧须为1500～2000整数")
    if not all(params.get(name) in (-1, 1) for name in DIRECTION_PARAMS):
        raise ValueError("尚未读齐8路飞控正反转参数，不能猜测运动方向")
    horizontal_sign = 1
    if action in HORIZONTAL_ACTIONS:
        reverse = params.get(HORIZONTAL_DIRECTION_PARAM)
        if reverse not in (0, 1):
            raise ValueError("尚未读到MOT_H_REV，侧移/转向测试需方向修正版固件及有效回传")
        horizontal_sign = -1 if reverse == 1 else 1
    result = {}
    for physical, logical in enumerate(PHYSICAL_TO_MOTOR, 1):
        sign = horizontal_sign * MOTION_FACTORS[action][logical - 1] * params[f"MOT_{logical}_DIRECTION"]
        result[physical] = high_pwm if sign > 0 else low_pwm if sign < 0 else 1500
    return result


class MotorTestSession:
    def __init__(self, modes, link):
        self.modes, self.link = modes, link
        self.phase = "monitor"
        self.params = {}
        self.direction_times = {}
        self.motion_action = None
        self.motion_plan = {}
        self.pwm = {}
        self.output_time = float("-inf")
        self.notice = "读取机架与输出映射；不会自动解锁"
        self.selected = 1
        self.direction = 0
        self.single_pwm = None
        self.level = 10
        self.deadline = 0.0
        self.cooldown = 0.0
        self.next_output = 0.0
        self.next_manual = 0.0
        self.next_params = 0.0
        self.ack_time = None
        self.test_started = None
        self.last_text = ""

    @property
    def mapping_ok(self):
        return all(self.params.get(k) == v for k, v in EXPECTED_PARAMS.items())

    def directions_fresh(self, now):
        return all(self.params.get(name) in (-1, 1)
                   and 0 <= now - self.direction_times.get(name, float("-inf")) <= 10
                   for name in DIRECTION_PARAMS)

    def horizontal_direction_fresh(self, action, now):
        return action not in HORIZONTAL_ACTIONS or (
            self.params.get(HORIZONTAL_DIRECTION_PARAM) in (0, 1) and
            0 <= now - self.direction_times.get(HORIZONTAL_DIRECTION_PARAM, float("-inf")) <= 10)

    def command(self, command, *params):
        self.link.mav.command_long_send(self.modes.target_system, 1, command, 0,
                                        *(tuple(params) + (0,) * (7 - len(params))))

    def observe(self, message, now):
        if message.get_srcSystem() != self.modes.target_system or message.get_srcComponent() != 1:
            return
        kind = message.get_type()
        if kind == "PARAM_VALUE":
            name = message.param_id
            if isinstance(name, bytes):
                name = name.decode("ascii", errors="replace")
            name = name.rstrip("\x00")
            if name in EXPECTED_PARAMS or name in DIRECTION_PARAMS or name == HORIZONTAL_DIRECTION_PARAM:
                previous = self.params.get(name)
                self.params[name] = float(message.param_value)
                if name in DIRECTION_PARAMS:
                    self.direction_times[name] = now
                    if self.motion_action and previous != self.params[name]:
                        self.stop(now, "正反转参数发生变化，已停止并请求上锁")
                elif name == HORIZONTAL_DIRECTION_PARAM:
                    self.direction_times[name] = now
                    if self.motion_action in HORIZONTAL_ACTIONS and previous != self.params[name]:
                        self.stop(now, "水平混控方向发生变化，已停止并请求上锁")
                if self.phase in ("mode", "arm", "ready") and not self.mapping_ok:
                    self.stop(now, "映射发生变化，已请求上锁")
        elif kind == "SERVO_OUTPUT_RAW" and getattr(message, "port", 0) == 0:
            self.pwm = {i: getattr(message, f"servo{i}_raw") for i in range(1, 9)}
            self.output_time = now
        elif kind == "STATUSTEXT":
            self.last_text = str(message.text).rstrip("\x00")
        elif kind == "COMMAND_ACK" and message.command in (176, 400, 209):
            # Ignore acknowledgements explicitly directed at another client.
            if getattr(message, "target_system", 0) not in (0, 255):
                return
            if getattr(message, "target_component", 0) not in (0, 190):
                return
            if self.phase in ("mode", "arm", "ready") and message.result not in (0, 5):
                self.stop(now, f"飞控拒绝命令 {message.command}，结果 {message.result}")
            elif message.command == 209 and message.result == 0 and self.phase == "ready":
                self.ack_time = now

    def prepare(self, now):
        if (self.phase != "monitor" or now < self.cooldown or not self.modes.fresh(now)
                or self.modes.armed or not self.mapping_ok):
            self.notice = "需新鲜回传、已上锁、映射核对通过，且冷却结束"
            return False
        self.phase, self.deadline = "mode", now + 3
        self.notice = "请求手动模式，等待飞控确认；尚未解锁"
        self.command(176, 1, 19)
        return True

    def hold(self, direction, level, now):
        if (self.phase != "ready" or not self.modes.fresh(now) or not self.modes.armed
                or self.modes.mode != 19 or not self.mapping_ok):
            return False
        if direction not in (-1, 1) or level not in (5, 10, 15) or self.direction or self.motion_action:
            return False
        self.direction, self.level = direction, level
        self.next_output = 0
        self.single_pwm = None
        self.notice = "持续测试中：再次点击测试按钮停止；失焦停止"
        logging.info("Motor test physical=%d logical=%d direction=%d level=%d",
                     self.selected, PHYSICAL_TO_MOTOR[self.selected - 1], direction, level)
        return True

    def start_single_pwm(self, pwm, now):
        if isinstance(pwm, bool) or not isinstance(pwm, int) or not 1000 <= pwm <= 2000:
            self.notice = "PWM须为1000～2000整数"
            return False
        if not self.hold(1 if pwm >= 1500 else -1, 10, now):
            return False
        self.single_pwm = pwm
        logging.info("Single PWM test physical=%d requested=%d output=%d", self.selected, pwm, pwm)
        return True

    def hold_motion(self, action, low_pwm, high_pwm, now):
        if (self.phase != "ready" or not self.modes.fresh(now) or not self.modes.armed
                or self.modes.mode != 19 or not self.mapping_ok
                or self.direction or self.motion_action):
            return False
        if not self.directions_fresh(now):
            self.notice = "方向参数未读齐或回传过期；等待读取，不能启动整组测试"
            return False
        if not self.horizontal_direction_fresh(action, now):
            self.notice = "需方向修正版固件并读回MOT_H_REV，暂不能启动侧移/转向测试"
            return False
        try:
            plan = motion_pwm_plan(action, low_pwm, high_pwm, self.params)
        except ValueError as exc:
            self.notice = str(exc)
            return False
        self.motion_action, self.motion_plan = action, plan
        self.next_output = 0
        self.notice = f"{action}启转测试：再次点击停止；失焦停止。不含姿态/定深闭环"
        logging.info("Motion PWM test action=%s physical_outputs=%s", action, plan)
        return True

    def _outputs(self):
        # Reset every motor, not only the selected one. Absolute PWM avoids
        # uncertainty in RC3 calibration. 1500 is this vehicle's verified neutral.
        for physical, logical in enumerate(PHYSICAL_TO_MOTOR, 1):
            pwm = 1500
            if self.motion_action:
                pwm = self.motion_plan[physical]
            elif physical == self.selected and self.direction:
                pwm = self.single_pwm if self.single_pwm is not None else 1500 + self.direction * self.level * 5
            self.command(209, logical - 1, 1, pwm, 0, 0, 2)

    def release(self, now):
        if self.direction or self.motion_action:
            self.direction = 0
            self.single_pwm = None
            self.motion_action, self.motion_plan = None, {}
            if self.phase == "ready":
                self._outputs()
                self.next_output = now + 0.1
            self.notice = "已发送8路1500中位停止；可再次点击启动"

    def stop(self, now, reason="已停止测试，等待飞控确认上锁"):
        active = self.phase in ("mode", "arm", "ready", "stopping")
        self.direction = 0
        self.single_pwm = None
        self.motion_action, self.motion_plan = None, {}
        try:
            if self.phase == "ready":
                self._outputs()
        finally:
            self.phase = "stopping" if active else "monitor"
            self.notice = reason
            if active:
                self.command(400, 0)  # ordinary disarm, never force-arm
                self.deadline = now + 3
                self.cooldown = now + 11

    def tick(self, now):
        if self.phase in ("mode", "arm", "ready"):
            if not self.modes.fresh(now) or not self.mapping_ok:
                self.stop(now, "链路/映射异常，已停止输出并请求上锁")
                return
            # Pilot-input failsafe remains enabled; neutral manual input only.
            if now >= self.next_manual:
                self.link.mav.manual_control_send(self.modes.target_system, 0, 0, 500, 0, 0)
                self.next_manual = now + 0.04
            if self.phase == "mode" and self.modes.mode == 19:
                self.phase, self.deadline = "arm", now + 3
                self.notice = "请求解锁，等待飞控心跳确认"
                self.command(400, 1)
            elif self.phase == "arm" and self.modes.armed:
                self.phase, self.test_started, self.ack_time = "ready", now, None
                self.notice = "测试就绪；点击启动，再次点击停止"
            if self.phase == "ready":
                if self.motion_action and not self.directions_fresh(now):
                    self.stop(now, "方向参数回传过期，已停止并请求上锁")
                    return
                if self.motion_action and not self.horizontal_direction_fresh(self.motion_action, now):
                    self.stop(now, "水平混控方向回传过期，已停止并请求上锁")
                    return
                if not self.modes.armed or self.modes.mode != 19:
                    self.stop(now, "飞控上锁或模式改变，测试已停止")
                    return
                if now >= self.next_output:
                    self._outputs()
                    self.next_output = now + 0.1
                if self.ack_time is None and now - self.test_started > 1.0:
                    self.stop(now, "电机测试未收到成功ACK，已请求上锁；查看QGC消息")
                elif self.ack_time is not None and now - self.ack_time > 1.0:
                    self.stop(now, "电机测试ACK中断，已请求上锁")
            elif now >= self.deadline:
                self.stop(now, "模式/解锁请求超时，已请求上锁")
        elif self.phase == "stopping":
            if self.modes.fresh(now) and not self.modes.armed:
                self.phase = "monitor"
                self.notice = "飞控已确认上锁"
            elif now >= self.deadline:
                self.notice = "上锁未确认！请在QGC确认或切断推进器供电"
        if self.phase in ("monitor", "ready") and now >= self.next_params:
            names = ((*EXPECTED_PARAMS, *DIRECTION_PARAMS, HORIZONTAL_DIRECTION_PARAM)
                     if self.phase == "monitor" else (*DIRECTION_PARAMS, HORIZONTAL_DIRECTION_PARAM))
            for name in names:
                self.link.mav.param_request_read_send(self.modes.target_system, 1, name.encode(), -1)
            self.next_params = now + 2
