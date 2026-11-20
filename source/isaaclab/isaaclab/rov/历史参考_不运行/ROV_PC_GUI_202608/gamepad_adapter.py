#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物理手柄适配器
读取 Xbox/PS/通用游戏手柄，转换为 UDP 手柄数据发给 ROV

依赖: pip install pygame
用法: python gamepad_adapter.py 192.168.1.3

按键映射 (Xbox 手柄):
  左摇杆上下  → 前进/后退  (tau_x)
  左摇杆左右  → 左移/右移  (tau_y)
  右摇杆上下  → 上浮/下潜  (tau_z)  或 RT/LT
  右摇杆左右  → 左转/右转  (M_z)
  A键        → 解锁/锁定推进器
  B键        → 急停(全零)
  RB键       → 夹爪打开/合拢切换（瞬时 PWM）
  R3按住     → 夹爪缓慢打开到180°
  L3按住     → 夹爪缓慢合拢到RB关闭角度
  Options键  → 退出

按键映射 (PS 手柄):
  左摇杆上下  → 前进/后退
  左摇杆左右  → 左移/右移
  右摇杆上下  → 上浮/下潜
  右摇杆左右  → 左转/右转
  ✕键        → 解锁/锁定
  ○键        → 急停
  R1键       → 夹爪打开/合拢切换（瞬时 PWM）
  R3按住     → 夹爪缓慢打开到180°
  L3按住     → 夹爪缓慢合拢到RB关闭角度
"""

import socket
import time
import sys

from config_loader import load_config
from input_shaping import remap_deadzone
from secure_protocol import encode_control_packet, new_session_id

try:
    import pygame
except ImportError:
    print("请先安装 pygame: pip install pygame")
    sys.exit(1)


class GamepadAdapter:
    # SDL/pygame-ce DualSense layout used by the main GUI.
    ARM_BUTTON = 0
    ESTOP_BUTTON = 1
    EXIT_BUTTON = 6       # Options/Start
    GRIPPER_CLOSE_JOG_BUTTON = 7  # L3
    GRIPPER_OPEN_JOG_BUTTON = 8   # R3
    JOG_INTERVAL_SECONDS = 0.2    # 5 Hz, same as the main GUI

    def __init__(self, rov_ip=None, cmd_port=None, joy_port=None):
        config = load_config()
        rov_ip = rov_ip or config.get("network", "pi_ip")
        self.rov_ip = rov_ip
        self.cmd_port = cmd_port or config.getint("network", "command_port")
        self.shared_key = config.get("safety", "shared_key")
        self.protocol_version = config.getint("safety", "protocol_version")
        self.max_packet_bytes = config.getint("safety", "max_command_packet_bytes")
        self.session = new_session_id()
        self.seq = 0

        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.armed = False
        self.running = True
        self.gripper_toggle_button = config.getint("servo", "toggle_button")
        self.gripper_open_position = config.getfloat("servo", "open_position")
        self.gripper_closed_position = config.getfloat("servo", "closed_position")
        self.gripper_jog_open_position = config.getfloat(
            "servo", "jog_open_position", fallback=1.0
        )
        self.gripper_jog_step = config.getfloat(
            "servo", "jog_step_per_refresh", fallback=0.05
        )
        self.servo_angle_min = config.getfloat("servo", "angle_min_deg")
        self.servo_angle_max = config.getfloat("servo", "angle_max_deg")
        self.gripper_position = self.gripper_closed_position
        # Startup/failsafe is already closed; the first deliberate press opens.
        self.gripper_close_on_next_press = False

        # 力缩放
        self.speed_scale = 20.0   # 平移力最大值 N
        self.yaw_scale = 15.0     # 偏航力矩最大值 Nm
        self.deadband = 0.15      # 摇杆/扳机差值死区，死区外连续重映射

        # 初始化 pygame 手柄
        pygame.init()
        pygame.joystick.init()

        self.joy_count = pygame.joystick.get_count()
        if self.joy_count == 0:
            print("未检测到手柄! 请插入 USB 手柄后重试")
            print("已检测到的输入设备:")
            for i in range(pygame.joystick.get_count()):
                print(f"  [{i}] {pygame.joystick.Joystick(i).get_name()}")
            sys.exit(1)

        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        print(f"✓ 已连接: {self.js.get_name()}")
        print(f"  轴数: {self.js.get_numaxes()}, 按键数: {self.js.get_numbuttons()}")
        print(f"  目标 ROV: {rov_ip}")
        print()

    def _send(self, payload):
        packet = encode_control_packet(
            payload, self.shared_key, self.session, self.seq,
            protocol_version=self.protocol_version,
            max_packet_bytes=self.max_packet_bytes,
        )
        self.seq += 1
        self.cmd_sock.sendto(packet, (self.rov_ip, self.cmd_port))

    def _deadband(self, val):
        """摇杆死区过滤并连续重映射，避免越过死区时突跳。"""
        return remap_deadzone(val, self.deadband)

    def _button_pressed(self, index):
        """Read one optional SDL button without breaking generic controllers."""
        try:
            return 0 <= index < self.js.get_numbuttons() and bool(
                self.js.get_button(index)
            )
        except Exception:
            return False

    def gripper_jog_direction(self):
        """Return +1 for R3, -1 for L3, and zero if neither/both are held."""
        open_jog = self._button_pressed(self.GRIPPER_OPEN_JOG_BUTTON)
        close_jog = self._button_pressed(self.GRIPPER_CLOSE_JOG_BUTTON)
        return int(open_jog) - int(close_jog)

    def read_joystick(self):
        """读取手柄并返回 4DOF 力 (兼容 Xbox 和 PS 手柄)"""
        pygame.event.pump()
        num_axes = self.js.get_numaxes()

        # 左摇杆 (Xbox/PS 通用: 轴0=X, 轴1=Y)
        lx = self._deadband(self.js.get_axis(0))
        ly = self._deadband(self.js.get_axis(1))

        # 检测手柄类型: PS手柄 L2 在轴3, Xbox LT 在轴2
        # PS: 0=LX 1=LY 2=RX 3=L2 4=R2 5=RY
        # XB: 0=LX 1=LY 2=LT 3=RX 4=RY 5=RT
        if num_axes >= 6 and abs(self.js.get_axis(3)) < 0.5:
            # Xbox: 轴3=右摇杆X, 轴4=右摇杆Y, 轴2=LT, 轴5=RT
            rx = self._deadband(self.js.get_axis(3))
            ry = self._deadband(self.js.get_axis(4))
            lt = (1.0 - self.js.get_axis(2)) / 2.0
            rt = (1.0 - self.js.get_axis(5)) / 2.0
        else:
            # PS: 轴2=右摇杆X, 轴5=右摇杆Y, 轴3=L2, 轴4=R2
            rx = self._deadband(self.js.get_axis(2))
            ry = self._deadband(self.js.get_axis(3)) if num_axes < 6 else self._deadband(self.js.get_axis(5))
            lt = (1.0 - (self.js.get_axis(3) if num_axes >= 4 else 0)) / 2.0
            rt = (1.0 - (self.js.get_axis(4) if num_axes >= 5 else 0)) / 2.0

        tau_x = -ly * self.speed_scale       # 前推 = 前进 +tau_x
        tau_y = -lx * self.speed_scale       # 左推 = 左移 -tau_y
        M_z = rx * self.yaw_scale            # 右推 = 右转 +M_z

        if ry != 0.0:
            tau_z = -ry * self.speed_scale
        else:
            tau_z = self._deadband(rt - lt) * self.speed_scale

        return tau_x, tau_y, tau_z, M_z

    def send_arm(self):
        """发送解锁指令"""
        self.armed = not self.armed
        self._send({"enable": False, "arm": self.armed})
        status = "🔓 已解锁" if self.armed else "🔒 已锁定"
        print(f"\r[{status}]", end="", flush=True)

    def send_stop(self):
        """急停"""
        self.armed = False
        self._send({
            "emergency_stop": True,
            "joystick": {
                "tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0,
                "M_x": 0.0, "M_y": 0.0, "M_z": 0.0,
            },
        })
        print("\r[🛑 急停]", end="", flush=True)

    def send_gripper_toggle(self):
        """Send one alternating close/open command for a button rising edge."""
        closing = self.gripper_close_on_next_press
        position = (
            self.gripper_closed_position
            if closing
            else self.gripper_open_position
        )
        self._send({
            "servo_position": position,
            "servo_min_position": self.gripper_closed_position,
            "servo_immediate": True,
        })
        self.gripper_position = position
        self.gripper_close_on_next_press = not closing
        action = "合拢" if closing else "打开"
        angle = self.servo_angle_min + position * (
            self.servo_angle_max - self.servo_angle_min
        )
        print(f"\r[夹爪已{action}: {angle:.1f}°]", end="", flush=True)

    def send_gripper_jog(self, direction):
        """Send one bounded slow L3/R3 step using the GUI's normalized protocol."""
        if direction not in (-1, 1):
            raise ValueError("gripper jog direction must be -1 or 1")
        lower = self.gripper_closed_position
        upper = self.gripper_jog_open_position
        position = max(lower, min(upper, self.gripper_position))
        position = max(lower, min(upper, position + direction * self.gripper_jog_step))
        if abs(position - self.gripper_position) <= 1e-9:
            return False
        self._send({
            "servo_position": position,
            "servo_min_position": lower,
            "servo_immediate": False,
        })
        self.gripper_position = position
        # Fine movement does not change RB's remembered completed action.
        angle = self.servo_angle_min + position * (
            self.servo_angle_max - self.servo_angle_min
        )
        action = "R3平缓打开" if direction > 0 else "L3平缓合拢"
        print(f"\r[{action}: {angle:.1f}°]", end="", flush=True)
        return True

    def run(self):
        """主循环: 40Hz 读取+发送"""
        print("操作说明:")
        print("  左摇杆上下 = 前进/后退")
        print("  左摇杆左右 = 左移/右移")
        print("  右摇杆上下 = 上浮/下潜")
        print("  右摇杆左右 = 左转/右转")
        print("  A键(✕)     = 解锁/锁定推进器")
        print("  B键(○)     = 急停")
        print(
            f"  RB键(R1)   = 夹爪打开/合拢 (pygame button {self.gripper_toggle_button})"
        )
        print("  R3(button 8) = 夹爪缓慢打开")
        print("  L3(button 7) = 夹爪缓慢合拢")
        print("  Options(button 6) = 退出")
        print()

        interval = 0.025  # 40Hz
        last_button = [False] * self.js.get_numbuttons()
        last_jog_tick = 0.0

        try:
            while self.running:
                start = time.time()
                pygame.event.pump()

                # 读取手柄
                tau_x, tau_y, tau_z, M_z = self.read_joystick()

                # 发送手柄力
                self._send({"joystick": {
                    "tau_x": round(tau_x, 2),
                    "tau_y": round(tau_y, 2),
                    "tau_z": round(tau_z, 2),
                    "M_x": 0.0,
                    "M_y": 0.0,
                    "M_z": round(M_z, 2),
                }})

                # 检查按键
                now = time.monotonic()
                jog_direction = self.gripper_jog_direction()
                if jog_direction:
                    if now - last_jog_tick >= self.JOG_INTERVAL_SECONDS:
                        self.send_gripper_jog(jog_direction)
                        last_jog_tick = now
                else:
                    # The next press should react immediately.
                    last_jog_tick = now - self.JOG_INTERVAL_SECONDS

                for i in range(self.js.get_numbuttons()):
                    pressed = self.js.get_button(i)
                    if pressed and not last_button[i]:
                        if i == self.ARM_BUTTON:    # A键 / ✕键 → 解锁
                            self.send_arm()
                        elif i == self.ESTOP_BUTTON:  # B键 / ○键 → 急停
                            self.send_stop()
                        elif i == self.gripper_toggle_button:  # RB / R1 → 夹爪切换
                            self.send_gripper_toggle()
                        elif i == self.EXIT_BUTTON:  # Options/Start → 退出
                            self.running = False
                            print("\n退出")
                    last_button[i] = pressed

                elapsed = time.time() - start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n退出")
        finally:
            # 归零 + 锁定
            for _ in range(3):
                self._send({
                    "enable": False,
                    "arm": False,
                    "servo_position": self.gripper_closed_position,
                    "servo_min_position": self.gripper_closed_position,
                    "servo_immediate": True,
                    "joystick": {
                        "tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0,
                        "M_x": 0.0, "M_y": 0.0, "M_z": 0.0,
                    },
                })
                time.sleep(0.03)
            self.cmd_sock.close()
            pygame.quit()
            print("已安全退出")


if __name__ == "__main__":
    rov_ip = sys.argv[1] if len(sys.argv) > 1 else None

    adapter = GamepadAdapter(rov_ip=rov_ip)
    adapter.run()
