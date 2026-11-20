"""ArduSub mode requests confirmed by vehicle heartbeat, never by UI toggles."""

from __future__ import annotations
import math
from attitude_display import AttitudeDisplay


DEPTH_ONLY_MODE = 22  # Private mode in our pinned ROV firmware; capability checked below.
MODE_HOLDS = {19: (False, False), 0: (False, True), 2: (True, True),
              DEPTH_ONLY_MODE: (True, False)}  # depth, attitude
MODE_NAMES = {0: "仅姿态稳定", 2: "姿态稳定+定深", 19: "手动（两项关闭）",
              DEPTH_ONLY_MODE: "仅定深（姿态手动）"}


def health_fault_description(mask: int) -> str:
    """Decode MAV_SYS_STATUS_SENSOR bits, not ArduPilot InternalError bits."""
    names = ((0x00000001, '陀螺仪'), (0x00000002, '加速度计'),
             (0x00000004, '罗盘'), (0x00000008, '气压/深度传感器'))
    affected = [name for bit, name in names if mask & bit]
    return '、'.join(affected) if affected else '其他设备'


class FlightModeController:
    def __init__(self, target_system: int = 1):
        self.target_system = target_system
        self.mode: int | None = None
        self.armed = False
        self.last_heartbeat = float("-inf")
        self.pending: int | None = None
        self.deadline = 0.0
        self.notice = "等待飞控回传"
        self.pilot_gain = None
        self.last_pilot_gain = float("-inf")
        self.attitude = None
        self.last_attitude = float("-inf")
        self.attitude_time_boot_ms = None
        self.raw_yaw_rad = self.yaw_rate_deg_s = None
        self.attitude_display = AttitudeDisplay()
        self.rov_values: dict[str, tuple[float, float]] = {}
        self.system_health = None
        self.battery = {}

    def fresh(self, now: float) -> bool:
        return now - self.last_heartbeat <= 3.0

    def observe(self, message, now: float) -> None:
        if message.get_srcSystem() != self.target_system or message.get_srcComponent() != 1:
            return
        if message.get_type() == "HEARTBEAT":
            # ArduPilot submarine only; ignore GCS and other vehicle heartbeats.
            if message.type != 12 or message.autopilot != 3:
                return
            self.mode = int(message.custom_mode)
            self.armed = bool(message.base_mode & 128)
            self.last_heartbeat = now
            if self.pending == self.mode:
                self.pending = None
                self.notice = "模式已由飞控确认"
        elif message.get_type() == "COMMAND_ACK" and message.command == 176:
            if self.pending is not None and message.result not in (0, 5):
                self.pending = None
                self.notice = f"飞控拒绝模式切换（结果 {message.result}），查看QGC消息"
        elif message.get_type() == "ATTITUDE":
            values = (message.roll, message.pitch, message.yaw)
            if all(math.isfinite(value) for value in values):
                angles = tuple(math.degrees(value) for value in values)
                if self.attitude_display.observe(angles, now,
                    disarmed=self.fresh(now) and not self.armed,
                    boot_ms=getattr(message, 'time_boot_ms', None)):
                    self.attitude = angles
                    self.last_attitude = now
                    self.attitude_time_boot_ms = getattr(message, 'time_boot_ms', None)
                    self.raw_yaw_rad = message.yaw
                    rate = getattr(message, 'yawspeed', None)
                    self.yaw_rate_deg_s = math.degrees(rate) if isinstance(rate, (int, float)) and math.isfinite(rate) else None
        elif message.get_type() == "SYS_STATUS":
            masks = (message.onboard_control_sensors_present, message.onboard_control_sensors_enabled,
                     message.onboard_control_sensors_health)
            if all(type(value) is int and 0 <= value <= 0xffffffff for value in masks):
                failed = masks[0] & masks[1] & ~masks[2]
                load = message.load / 10.0 if type(message.load) is int and 0 <= message.load <= 1000 else None
                self.system_health = (failed, load, now)
            self._battery_sample('system', getattr(message, 'current_battery', -1),
                                 getattr(message, 'voltage_battery', 65535), now)
        elif message.get_type() == "BATTERY_STATUS" and getattr(message, 'id', -1) == 0:
            cells = getattr(message, 'voltages', [])
            voltage = sum(v for v in cells if type(v) is int and 0 < v < 65535)
            self._battery_sample('primary', getattr(message, 'current_battery', -1),
                                 voltage if voltage else 65535, now)
        elif message.get_type() == "NAMED_VALUE_FLOAT":
            name = message.name
            if isinstance(name, bytes):
                name = name.decode("ascii", errors="replace")
            name = name.rstrip("\x00")
            value = float(message.value)
            if not math.isfinite(value):
                return
            if name == "PilotGain":
                self.pilot_gain = value
                self.last_pilot_gain = now
            elif name in ("ROV_EN", "ROV_ACT", "ROV_PTGT", "ROV_YTGT", "ROV_DONLY"):
                if name in ("ROV_EN", "ROV_ACT", "ROV_DONLY") and value not in (0.0, 1.0):
                    return
                if name == "ROV_PTGT" and not -90 <= value <= 90:
                    return
                if name == "ROV_YTGT" and not -180 <= value <= 180:
                    return
                self.rov_values[name] = (value, now)

    def _battery_sample(self, source, current, voltage, now):
        amps = current / 100.0 if type(current) is int and -32768 <= current <= 32767 and current != -1 else None
        volts = voltage / 1000.0 if type(voltage) is int and 0 < voltage < 65535 else None
        self.battery[source] = (amps, volts, now)

    def power_sample(self, now):
        """Return the freshest recent sample which actually contains current.

        BATTERY_STATUS and SYS_STATUS are not guaranteed to arrive at the same
        rate.  Fixed source priority can therefore keep displaying an older
        current while a newer valid sample is already available.
        """
        fresh = [sample for sample in self.battery.values()
                 if 0 <= now - sample[2] <= 3]
        measured = [sample for sample in fresh if sample[0] is not None]
        candidates = measured or fresh
        return max(candidates, key=lambda sample: sample[2], default=None)

    def power_status(self, now):
        if not self.fresh(now):
            return '飞控电流/电压：未知（回传超时）'
        sample = self.power_sample(now)
        if sample is None:
            return '飞控电流/电压：未提供'
        amps, volts, _ = sample
        return ('电流 ' + (f'{amps:.2f} A' if amps is not None else '未提供（检查电流计）') +
                ' · 电压 ' + (f'{volts:.2f} V' if volts is not None else '未提供'))

    def hold_status(self, now):
        holds = MODE_HOLDS.get(self.mode) if self.fresh(now) else None
        if holds is None:
            return '姿态稳定：未知    定深：未知'
        depth, attitude = holds
        return f"姿态稳定：{'开' if attitude else '关'}    定深：{'开' if depth else '关'}"

    def depth_only_supported(self, now):
        cap = self.rov_values.get('ROV_DONLY')
        return self.fresh(now) and cap is not None and cap[0] == 1 and 0 <= now - cap[1] <= 3

    def attitude_status(self, now: float) -> str:
        if not self.fresh(now) or self.attitude is None or now - self.last_attitude > 1.0:
            return "实测姿态：未知（无新鲜回传）"
        roll, pitch, yaw = self.attitude
        return f"实测：横滚 {roll:+.1f}° · 俯仰 {pitch:+.1f}° · 航向 {yaw % 360:.1f}°"

    def pilot_control_status(self, now: float) -> str:
        if not self.fresh(now):
            return "回平控制：未确认（飞控离线）"
        if self.mode not in (0, 2):
            return "当前模式不提供此回平/定向功能"
        enabled = self.rov_values.get("ROV_EN")
        active = self.rov_values.get("ROV_ACT")
        if enabled is None or now - enabled[1] > 3.0:
            return "回平增强：未确认（标准固件不回传此状态）"
        if enabled[0] == 0:
            return "飞控标准回平 · 平顺增强关闭"
        if active is None or now - active[1] > 3.0:
            return "平顺增强已配置 · 当前执行状态未知"
        if active[0] == 0:
            return "平顺增强已配置 · 当前未执行"
        if not self.armed:
            return "平顺增强：等待上锁状态与执行状态一致"
        pitch = self.rov_values.get("ROV_PTGT")
        yaw = self.rov_values.get("ROV_YTGT")
        if all(item is not None and now - item[1] <= 1.0 for item in (pitch, yaw)):
            return f"平顺回平执行中 · 目标俯仰 {pitch[0]:+.1f}° / 航向 {yaw[0] % 360:.1f}°"
        return "平顺回平执行中 · 目标回传超时"

    def pilot_gain_status(self, now):
        if self.pilot_gain is None or now - self.last_pilot_gain > 3.0:
            return "飞控增益：未知（未收到新鲜PilotGain）"
        return f"飞控增益：{self.pilot_gain:.0%}（实测回传）"

    def health_status(self, now):
        if not self.fresh(now) or self.system_health is None or now - self.system_health[2] > 3:
            return '飞控健康/主循环负载：未知（无新鲜SYS_STATUS）'
        failed, load, _received = self.system_health
        result = (f'警告：飞控报告已启用设备异常位 0x{failed:08X}（{health_fault_description(failed)}），查看QGC消息/传感器页'
                  if failed else '已启用设备未报告异常（不代表校准或验收完成）')
        result += ' · 主循环负载' + (f'{load:.1f}%' if load is not None else '未知')
        if load is not None and load >= 90:
            result += '（负载偏高）'
        return result

    def tick(self, now: float) -> None:
        if self.pending is not None and now >= self.deadline:
            self.pending = None
            self.notice = "模式切换未确认；查看QGC消息，不自动重试"

    def request(self, action: str, now: float) -> int | None:
        self.tick(now)
        if not self.fresh(now):
            self.notice = "未发送：飞控回传缺失或超时"
            return None
        if self.pending is not None:
            self.notice = "等待上一条模式请求确认"
            return None
        if action == 'reset_manual':
            if self.armed:
                self.notice = '默认关闭未执行：飞控已解锁'
                return None
            target = 19
        elif action in ('attitude_toggle', 'depth_toggle') and self.mode in MODE_HOLDS:
            depth, attitude = MODE_HOLDS[self.mode]
            depth, attitude = (not depth, attitude) if action == 'depth_toggle' else (depth, not attitude)
            target = next(mode for mode, holds in MODE_HOLDS.items() if holds == (depth, attitude))
        else:
            self.notice = "当前模式不在手柄四种组合内，请先在QGC选择Manual"
            return None
        if target == DEPTH_ONLY_MODE and not self.depth_only_supported(now):
            self.notice = '仅定深未发送：当前固件未确认支持，需要新的独立控制固件'
            return None
        if target == self.mode and self.pending is None:
            self.notice = "当前已是手动，两项关闭"
            return None
        self.pending = target
        self.deadline = now + 3.0
        self.notice = f"请求{MODE_NAMES[target]}，等待飞控确认"
        return target

    def status(self, now: float) -> str:
        self.tick(now)
        if not self.fresh(now):
            return "飞控模式未知（无新鲜回传）"
        return f"{MODE_NAMES.get(self.mode, f'模式 {self.mode}')} / {'已解锁' if self.armed else '已上锁'}"
