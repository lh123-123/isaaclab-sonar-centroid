"""Read-only loopback snapshot for an external QGC Fly instrument.

No additional MAVLink receiver, no actuator/mode writes, no Tk work in HTTP.
The control thread publishes an immutable JSON copy at 5 Hz; it expires if
that thread stalls even while the HTTP thread is still running.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from flight_mode import MODE_HOLDS, MODE_NAMES, health_fault_description

STATUS_PORT = 18765


class CompetitionTelemetry:
    def __init__(self, target_system=1):
        self.target_system = target_system
        self.depth = self.descent = None
        self.depth_time = float('-inf')
        self.pwm = {}
        self.pwm_time = float('-inf')
        self.alert = None
        self.alert_severity = None
        self.alert_time = float('-inf')
        self.leak = None
        self.leak_time = float('-inf')

    def observe(self, message, now):
        if message.get_srcSystem() != self.target_system or message.get_srcComponent() != 1:
            return
        kind = message.get_type()
        if kind == 'GLOBAL_POSITION_INT':
            altitude, vz = getattr(message, 'relative_alt', None), getattr(message, 'vz', None)
            if type(altitude) is int and -10_000_000 <= altitude <= 10_000_000:
                # ArduSub get_alt_rel(): upward positive from home/water surface.
                self.depth = -altitude / 1000.0
                self.descent = vz / 100.0 if type(vz) is int and -32768 < vz < 32767 else None
                self.depth_time = now
        elif kind == 'SERVO_OUTPUT_RAW' and getattr(message, 'port', 0) == 0:
            self.pwm = {channel: value for channel in (9, 10, 11)
                        if type(value := getattr(message, f'servo{channel}_raw', None)) is int
                        and 500 <= value <= 2500}
            self.pwm_time = now
        elif kind == 'SYS_STATUS':
            present = getattr(message, 'onboard_control_sensors_present_extended', 0)
            enabled = getattr(message, 'onboard_control_sensors_enabled_extended', 0)
            health = getattr(message, 'onboard_control_sensors_health_extended', 0)
            self.leak = bool(health & 2) if present & enabled & 2 else None
            self.leak_time = now
        elif kind == 'STATUSTEXT':
            severity = getattr(message, 'severity', 7)
            if type(severity) is not int or not 0 <= severity <= 4:
                return
            value = message.text
            if isinstance(value, bytes):
                value = value.decode('utf-8', errors='replace')
            value = ''.join(c for c in str(value).split('\x00')[0] if c.isprintable())[:180]
            if value and (self.alert is None or self.alert_severity is None
                          or not 0 <= now - self.alert_time <= 10
                          or severity <= self.alert_severity):
                self.alert, self.alert_time = value, now
                self.alert_severity = severity


def make_snapshot(modes, telemetry, actions, servos, now, *, hid_fresh,
                  sending, ready, allow_arm, motor_testing=False, resources=None):
    online = modes.fresh(now)
    holds = MODE_HOLDS.get(modes.mode) if online else None
    if modes.mode == 22 and not modes.depth_only_supported(now):
        holds = None  # Mode 22 is private: do not interpret an unrelated firmware.
    attitude = modes.attitude if online and 0 <= now - modes.last_attitude <= 1 else None
    power = modes.power_sample(now) if online else None
    pwm = telemetry.pwm if online and 0 <= now - telemetry.pwm_time <= 3 else {}
    warnings = []
    if not online:
        warnings.append('飞控回传超时：勿按旧数值操纵')
    if not hid_fresh:
        warnings.append('手柄未连接或输入超时')
    if motor_testing:
        warnings.append('推进器测试面板已打开，请勿进行比赛操纵')
    if online and telemetry.leak is False and 0 <= now - telemetry.leak_time <= 3:
        warnings.insert(0, '飞控报告漏水传感器异常')
    if online and telemetry.alert and 0 <= now - telemetry.alert_time <= 10:
        warnings.insert(0, telemetry.alert)
    if online and modes.system_health is not None and 0 <= now - modes.system_health[2] <= 3:
        failed, load, _received = modes.system_health
        if failed:
            warnings.append(f'飞控设备异常 0x{failed:08X}（{health_fault_description(failed)}）：查看QGC消息/传感器页')
        if load is not None and load >= 90:
            warnings.append(f'飞控主循环负载偏高 {load:.0f}%')
    if resources and resources.get('temperature_c') is not None and resources['temperature_c'] >= 80:
        warnings.append('树莓派温度过高，检查散热')
    if modes.pending is not None:
        warnings.append('模式切换等待飞控确认')
    elif modes.notice and any(s in modes.notice for s in ('未发送', '拒绝', '未确认', '未执行')):
        warnings.append(modes.notice)
    return {
        'schema': 1, 'target_system': modes.target_system, 'fc_online': online,
        'armed': modes.armed if online else None,
        'mode': modes.mode if online else None,
        'mode_text': MODE_NAMES.get(modes.mode, f'模式 {modes.mode}') if online else '未知',
        'depth_hold': holds[0] if holds else None,
        'attitude_hold': holds[1] if holds else None,
        'pending_mode': modes.pending,
        'depth_m': telemetry.depth if online and 0 <= now - telemetry.depth_time <= 3 else None,
        'descent_mps': telemetry.descent if online and 0 <= now - telemetry.depth_time <= 3 else None,
        'roll_deg': attitude[0] if attitude else None,
        'pitch_deg': attitude[1] if attitude else None,
        'heading_deg': attitude[2] % 360 if attitude else None,
        # Native measured yaw only: never a target, relative datum or rate integration.
        'heading_source': 'ATTITUDE.yaw' if attitude else None,
        'heading_raw_yaw_rad': modes.raw_yaw_rad if attitude else None,
        'yaw_rate_deg_s': modes.yaw_rate_deg_s if attitude else None,
        'attitude_time_boot_ms': modes.attitude_time_boot_ms if attitude else None,
        'current_a': power[0] if power else None, 'voltage_v': power[1] if power else None,
        'hid_fresh': bool(hid_fresh), 'sending': bool(sending),
        'control_ready': bool(ready), 'allow_arm': bool(allow_arm),
        'motion_gain': actions.motion.gain, 'vertical_gain': actions.vertical.gain,
        'gimbal_target_deg': actions.gimbal.angle_deg,
        'gripper_next': servos.next_gripper_action_cn if online else None,
        'gripper_target_pwm': servos.channel_pwm(servos.axes[servos.gripper_name].channel) if online else None,
        'gripper_output_pwm': pwm.get(servos.axes[servos.gripper_name].channel),
        'warning': '；'.join(dict.fromkeys(warnings)),
        'resources': resources or {},
    }


class StatusServer:
    def __init__(self, port=STATUS_PORT):
        self.lock = threading.Lock()
        self.payload = None
        self.updated = float('-inf')
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(.5)

            def do_GET(self):
                # DNS rebinding protection: not reachable over a LAN address.
                if self.headers.get('Host') != f'127.0.0.1:{owner.port}':
                    self.send_error(403)
                    return
                if self.path.split('?', 1)[0] != '/status':
                    self.send_error(404)
                    return
                body = owner.response()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        self.server = HTTPServer(('127.0.0.1', port), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .1},
                         daemon=True, name='qgc-readonly-status').start()

    def publish(self, snapshot, now):
        payload = json.dumps(snapshot, ensure_ascii=False, allow_nan=False).encode('utf-8')
        with self.lock:
            self.payload, self.updated = payload, now

    def response(self):
        with self.lock:
            payload, updated = self.payload, self.updated
        valid = payload is not None and 0 <= time.monotonic() - updated <= 1
        if valid:
            return payload
        return b'{"schema":1,"bridge_stale":true}'

    def close(self):
        # No blocking server.shutdown() on the flight-control/Tk thread.
        def shutdown():
            self.server.shutdown()
            self.server.server_close()
        threading.Thread(target=shutdown, daemon=True).start()
