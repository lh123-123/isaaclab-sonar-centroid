"""Read-only thrust setup audit. Never sends PARAM_SET or actuator commands."""
import math
from motor_test import PHYSICAL_TO_MOTOR, POSITIONS

TUNING_GROUPS = {
    '横滚角速度PID': ('ATC_RAT_RLL_P', 'ATC_RAT_RLL_I', 'ATC_RAT_RLL_D'),
    '俯仰角速度PID': ('ATC_RAT_PIT_P', 'ATC_RAT_PIT_I', 'ATC_RAT_PIT_D'),
    '偏航角速度PID': ('ATC_RAT_YAW_P', 'ATC_RAT_YAW_I', 'ATC_RAT_YAW_D'),
    # ArduSub 4.7 baseline uses Down-axis names, not old PSC_ACCZ aliases.
    '定深加速度PID': ('PSC_D_ACC_P', 'PSC_D_ACC_I', 'PSC_D_ACC_D'),
}
SAFETY_NAMES = ('ARMING_SKIPCHK', 'FS_EKF_ACTION', 'FS_GCS_ENABLE', 'FS_PILOT_INPUT',
                'FS_PILOT_TIMEOUT', 'FS_LEAK_ENABLE', 'LOG_BACKEND_TYPE')


class ThrustMonitor:
    def __init__(self, expected, target_system=1):
        self.target_system = target_system
        names = ("FRAME_CONFIG", "MOT_PWM_MIN", "MOT_PWM_MAX", "JS_GAIN_DEFAULT",
                 "JS_GAIN_MAX", "JS_GAIN_STEPS", "JS_THR_GAIN")
        self.expected = {name: (2 if name == "FRAME_CONFIG" else expected[name])
                         for name in names}
        for i in range(1, 9):
            for name in (f"MOT_{i}_DIRECTION", f"SERVO{i}_FUNCTION"):
                self.expected[name] = expected[name]
        self.params = {}
        self.read_order = dict.fromkeys([*self.expected, *SAFETY_NAMES,
                                        *(name for group in TUNING_GROUPS.values() for name in group)])
        self.pwm = None
        self.rc = None
        self.pwm_time = self.rc_time = float("-inf")
        self.next_request = 0.0

    def observe(self, message, now):
        if message.get_srcSystem() != self.target_system or message.get_srcComponent() != 1:
            return
        kind = message.get_type()
        if kind == "PARAM_VALUE":
            name = message.param_id
            if isinstance(name, bytes):
                name = name.decode("ascii", errors="replace")
            name = name.rstrip("\x00")
            value = float(message.param_value)
            if name in self.read_order and math.isfinite(value):
                self.params[name] = (value, now)
        elif kind == "SERVO_OUTPUT_RAW" and getattr(message, "port", 0) == 0:
            self.pwm = tuple(getattr(message, f"servo{i}_raw") for i in range(1, 9))
            self.pwm_time = now
        elif kind == "RC_CHANNELS":
            self.rc = tuple(getattr(message, f"chan{i}_raw") for i in range(1, 7))
            self.rc_time = now

    def value(self, name, now):
        value, received = self.params.get(name, (None, float("-inf")))
        return value if now - received <= 30 else None

    def request_one(self, link, modes, now):
        # Bounded named reads only. No full-list downloads; pause while armed.
        if link is None or not modes.fresh(now) or modes.armed or now < self.next_request:
            return
        missing = next((name for name in self.read_order if self.value(name, now) is None), None)
        if missing is not None:
            # Rotate unanswered names so a missing parameter cannot starve the rest.
            self.read_order.pop(missing)
            self.read_order[missing] = None
            link.mav.param_request_read_send(self.target_system, 1, missing.encode("ascii"), -1)
            self.next_request = now + 0.2

    def status(self, modes, now):
        if not modes.fresh(now):
            return "飞控离线：参数、PWM和RC输入均不可作为当前状态。"
        lines = [modes.status(now), modes.pilot_gain_status(now), modes.health_status(now)]
        lines.append('安全设置（只读，不代表已完成失控测试）：')
        safety = {name: self.value(name, now) for name in SAFETY_NAMES}
        lines.append(' / '.join(f'{name}={value:g}' if value is not None else f'{name}=未知'
                               for name, value in safety.items()))
        if safety['ARMING_SKIPCHK'] is not None and safety['ARMING_SKIPCHK'] > 0:
            lines.append('警告：存在被跳过的解锁检查；需核对具体项目，不能视为传感器已合格。')
        if safety['FS_EKF_ACTION'] == 0:
            lines.append('警告：EKF失效动作关闭；姿态/定深估计异常的处置尚需核验。')
        if safety['FS_GCS_ENABLE'] == 0 or safety['FS_PILOT_INPUT'] == 0:
            lines.append('警告：GCS或手动输入失控动作关闭；不要依赖电脑补救断链。')
        if safety['LOG_BACKEND_TYPE'] == 0:
            lines.append('警告：飞控日志后端关闭；无法据此完成机载日志PID分析。')
        lines.append('飞控PID实测参数（可在QGC Parameters修改，本面板不写入）：')
        for label, names in TUNING_GROUPS.items():
            values = [self.value(name, now) for name in names]
            lines.append(label + ' P/I/D：' + ' / '.join(f'{v:.5g}' if v is not None else '未知' for v in values))
        mismatches = []
        count = 0
        for name, desired in self.expected.items():
            actual = self.value(name, now)
            if actual is not None:
                count += 1
                if actual != desired:
                    mismatches.append(f"{name}: 实际{actual:g} / 目标{desired:g}")
        lines.append(f"参数核验 {count}/{len(self.expected)}（目标来自本地比赛配置）")
        if mismatches:
            lines.extend(mismatches)
        elif count == len(self.expected):
            lines.append("本次参数回传全部吻合；仍需验证实际水流和姿态纠偏。")
        else:
            lines.append("未读齐，不宣称参数已经生效；上锁后自动补读。")
        if modes.pilot_gain is not None and now - modes.last_pilot_gain <= 3 and modes.pilot_gain != 1:
            lines.append("注意：运行增益不是100%，前后/平移/偏航会再次缩小。")
        rc = self.rc if now - self.rc_time <= 2 else None
        lines.append("RC输入1–6：" + (" / ".join(map(str, rc)) if rc else "无新鲜回传"))
        pwm = self.pwm if now - self.pwm_time <= 2 else None
        lines.append("推进器实际输出PWM（不是实测推力）：")
        for physical, logical in enumerate(PHYSICAL_TO_MOTOR, 1):
            direction = self.value(f"MOT_{logical}_DIRECTION", now)
            output = f"{pwm[physical - 1]}µs" if pwm else "未知"
            lines.append(f"{physical} {POSITIONS[physical-1]} · Motor{logical} · "
                         f"方向{direction:g} · {output}" if direction is not None else
                         f"{physical} {POSITIONS[physical-1]} · Motor{logical} · 方向未知 · {output}")
        return "\n".join(lines)
