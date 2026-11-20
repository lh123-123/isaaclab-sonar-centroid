#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上位机 UDP 客户端 - 测试/调试工具
发送控制指令到下位机 (端口 5000)，接收遥测数据 (端口 5001)
"""

import socket
import time
import threading
import sys
from config_loader import load_config
from secure_protocol import (
    TelemetryReplayGuard,
    decode_telemetry_packet,
    encode_control_packet,
    new_session_id,
)


APP_CONFIG = load_config()


class PCClient:
    """上位机通信客户端"""

    def __init__(self,
                 rov_ip=None,
                 cmd_port=None,
                 joy_port=None,
                 listen_port=None):
        self.rov_ip = rov_ip or APP_CONFIG.get("network", "pi_ip")
        self.cmd_port = cmd_port or APP_CONFIG.getint("network", "command_port")
        self.joy_port = joy_port or APP_CONFIG.getint("network", "joystick_port")
        self.listen_port = listen_port or APP_CONFIG.getint("network", "telemetry_port")
        self.shared_key = APP_CONFIG.get("safety", "shared_key")
        self.protocol_version = APP_CONFIG.getint("safety", "protocol_version")
        self.max_packet_bytes = APP_CONFIG.getint("safety", "max_command_packet_bytes")
        self.session = new_session_id()
        self._seq = 0
        self._tx_lock = threading.Lock()

        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.telem_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.telem_sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
            )
        else:
            self.telem_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.telem_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.telem_sock.bind(("0.0.0.0", self.listen_port))
        self.telem_sock.settimeout(0.5)

        self.running = False
        self.latest_telemetry = None
        self.telemetry_lock = threading.Lock()
        self._telemetry_guard = TelemetryReplayGuard()

    def _send_payload(self, payload):
        with self._tx_lock:
            packet = encode_control_packet(
                payload, self.shared_key, self.session, self._seq,
                protocol_version=self.protocol_version,
                max_packet_bytes=self.max_packet_bytes,
            )
            self._seq += 1
            self.cmd_sock.sendto(packet, (self.rov_ip, self.cmd_port))

    def send_command(self, enable=None, mode=None, target=None,
                     save_data=None, pid_set=None, pid_save=None, arm=None,
                     emergency_stop=None, reset_emergency_stop=None,
                     servo_position=None, servo_min_position=None,
                     servo_immediate=None, performance_mode=None,
                     performance_profile=None):
        """发送 JSON 控制指令到端口 5000"""
        msg = {}
        if enable is not None: msg["enable"] = bool(enable)
        if mode is not None:
            if not isinstance(mode, dict):
                raise ValueError("mode must be an object")
            if bool(mode.get("heading_hold", False)):
                raise ValueError(
                    "heading_hold is temporarily disabled and must be false"
                )
            msg["mode"] = {**mode, "heading_hold": False}
        if target is not None:
            if not isinstance(target, dict):
                raise ValueError("target must be an object")
            if "heading_deg" in target:
                raise ValueError("heading target is temporarily disabled")
            msg["target"] = dict(target)
        if save_data is not None: msg["save_data"] = bool(save_data)
        if pid_set:
            msg["pid_set"] = pid_set
        if pid_save:
            msg["pid_save"] = pid_save
        if arm is not None: msg["arm"] = bool(arm)
        if emergency_stop is not None: msg["emergency_stop"] = bool(emergency_stop)
        if reset_emergency_stop is not None:
            msg["reset_emergency_stop"] = bool(reset_emergency_stop)
        if servo_position is not None:
            msg["servo_position"] = float(servo_position)
        if servo_min_position is not None:
            msg["servo_min_position"] = float(servo_min_position)
        if servo_immediate is not None:
            msg["servo_immediate"] = bool(servo_immediate)
        if performance_mode is not None:
            msg["performance_mode"] = str(performance_mode)
        if performance_profile is not None:
            msg["performance_profile"] = dict(performance_profile)
        if msg:
            self._send_payload(msg)

    def send_joystick(self, tau_x=0.0, tau_y=0.0, tau_z=0.0, M_z=0.0, M_y=0.0):
        """通过统一认证端口发送手柄力和链路心跳。"""
        self._send_payload({"joystick": {
            "tau_x": float(tau_x),
            "tau_y": float(tau_y),
            "tau_z": float(tau_z),
            "M_x": 0.0,
            "M_y": float(M_y),
            "M_z": float(M_z),
        }})

    def start_telemetry_listener(self):
        """启动遥测接收线程"""
        self.running = True

        def _listen():
            while self.running:
                try:
                    data, addr = self.telem_sock.recvfrom(16385)
                    if addr[0] != self.rov_ip:
                        continue
                    verified = decode_telemetry_packet(
                        data,
                        self.shared_key,
                        protocol_version=self.protocol_version,
                    )
                    telemetry = self._telemetry_guard.accept(verified)
                    with self.telemetry_lock:
                        self.latest_telemetry = telemetry
                except socket.timeout:
                    pass
                except Exception as e:
                    print(f"[遥测接收异常] {e}")

        thread = threading.Thread(target=_listen, daemon=True)
        thread.start()

    def get_telemetry(self):
        """获取最新遥测数据"""
        with self.telemetry_lock:
            return self.latest_telemetry

    def stop(self):
        self.running = False
        for _ in range(3):
            try:
                self.send_command(enable=False, arm=False)
                self.send_joystick()
            except OSError:
                break
            time.sleep(0.03)
        self.cmd_sock.close()
        self.telem_sock.close()

    def print_status(self):
        """打印最新状态"""
        tele = self.get_telemetry()
        if tele:
            imu = tele.get("imu", {})
            cmd = tele.get("cmd_enabled", False)
            joy = tele.get("joy_active", False)
            print(
                f"\r[状态] "
                f"R={imu.get('roll',0):6.1f} "
                f"P={imu.get('pitch',0):6.1f} "
                f"H={imu.get('heading',0):6.1f} | "
                f"D={tele.get('depth',0):6.3f}m | "
                f"Cmd={'ON' if cmd else 'OFF'} "
                f"Joy={'ON' if joy else 'OFF'}",
                f" ESTOP={'ON' if tele.get('emergency_stop') else 'OFF'}",
                end=""
            )


# ================================================================
#  命令行接口
# ================================================================
if __name__ == "__main__":
    ROV_IP = sys.argv[1] if len(sys.argv) > 1 else APP_CONFIG.get("network", "pi_ip")

    client = PCClient(rov_ip=ROV_IP)
    client.start_telemetry_listener()

    print("=" * 60)
    print(f"  BlueROV 上位机测试客户端")
    print(f"  目标 ROV: {ROV_IP}")
    print(f"  认证统一控制端口: {ROV_IP}:5000")
    print(f"  本地遥测: 0.0.0.0:5001")
    print("=" * 60)
    print()
    print("--- 模式指令 (端口5000) ---")
    print("  e        使能控制（默认定深）")
    print("  d        关闭控制 (停机)")
    print("  m <1/0>  切换模式 0=不开闭环 1=定深")
    print("  z <米>    设置目标深度")
    print("  s        切换数据保存")
    print("--- 手柄指令 (认证端口5000) ---")
    print("  w/a/s/d  前/左/后/右  (±10N)")
    print("  q/e      上浮/下潜 (±10N)")
    print("  j/l      左转/右转 (±5Nm)")
    print("  x        手柄回零")
    print("  Ctrl+C   退出")
    print()

    enabled = False
    mode = {"heading_hold": False, "depth_hold": True, "speed_hold": False}
    target = {"depth_m": 0.5, "speed_mps": 0.0}
    save_data = False

    # 手柄状态
    joy_tau_x, joy_tau_y, joy_tau_z, joy_Mz = 0.0, 0.0, 0.0, 0.0

    def send_joy():
        client.send_joystick(joy_tau_x, joy_tau_y, joy_tau_z, joy_Mz)

    try:
        while True:
            client.print_status()
            # 10 Hz authenticated heartbeat.  If this loop stops, the Pi locks
            # within the configured 500 ms and requires another manual arm.
            send_joy()

            import select
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                cmd = sys.stdin.readline().strip().split()
                if not cmd:
                    continue

                c = cmd[0]

                if c == "e":
                    enabled = True
                    client.send_command(enable=True, mode=mode, target=target, save_data=save_data)
                    print("\n[控制] 已使能")
                elif c == "d":
                    enabled = False
                    client.send_command(enable=False)
                    print("\n[控制] 已关闭")
                elif c == "m" and len(cmd) > 1:
                    m = int(cmd[1])
                    modes = {
                        0: {"heading_hold": False, "depth_hold": False, "speed_hold": False},
                        1: {"heading_hold": False, "depth_hold": True, "speed_hold": False},
                    }
                    mode = modes.get(m, modes[0])
                    if enabled:
                        client.send_command(enable=True, mode=mode, target=target, save_data=save_data)
                    print(f"\n[模式] -> {mode}")
                elif c == "z" and len(cmd) > 1:
                    target["depth_m"] = float(cmd[1])
                    if enabled:
                        client.send_command(enable=True, mode=mode, target=target, save_data=save_data)
                    print(f"\n[目标] 深度={target['depth_m']}m")
                elif c == "s":
                    save_data = not save_data
                    if enabled:
                        client.send_command(enable=True, mode=mode, target=target, save_data=save_data)
                    print(f"\n[保存] = {save_data}")

                # ---- 手柄指令 ----
                elif c == "w":  joy_tau_x = 10.0; send_joy(); print("\n[手柄] 前进")
                elif c == "s":  joy_tau_x = -10.0; send_joy(); print("\n[手柄] 后退")
                elif c == "a":  joy_tau_y = -10.0; send_joy(); print("\n[手柄] 左移")
                elif c == "d":  joy_tau_y = 10.0; send_joy(); print("\n[手柄] 右移")
                elif c == "q":  joy_tau_z = -10.0; send_joy(); print("\n[手柄] 上浮")
                elif c == "e":  joy_tau_z = 10.0; send_joy(); print("\n[手柄] 下潜")
                elif c == "j":  joy_Mz = 5.0; send_joy(); print("\n[手柄] 左转")
                elif c == "l":  joy_Mz = -5.0; send_joy(); print("\n[手柄] 右转")
                elif c == "x":
                    joy_tau_x = joy_tau_y = joy_tau_z = joy_Mz = 0.0
                    send_joy()
                    print("\n[手柄] 回零")

    except KeyboardInterrupt:
        print("\n退出")
    finally:
        client.send_command(enable=False)
        # 手柄回零
        client.send_joystick(0, 0, 0, 0)
        client.stop()
