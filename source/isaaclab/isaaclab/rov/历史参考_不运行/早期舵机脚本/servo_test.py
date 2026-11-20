#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PCA9685 单舵机台架测试工具（不控制推进器）。

默认模式不会导入 SMBus、不会打开 I2C、不会输出 PWM。真实硬件操作必须
显式传入 --enable，并在启动后的第二次提示中输入 ENABLE。

在树莓派上运行示例：
    python3 servo_test.py --enable
    python3 servo_test.py --enable --channel 9 --pulse-min 1000 --pulse-max 2000

交互命令：
    angle 45       转到指定角度（在允许范围内）
    pulse 1500     转到指定脉宽（微秒）
    + / -          按 --step 指定角度增量移动
    center          回到中心脉宽
    channel 9       切换到另一个辅助通道（0--7 默认禁止）
    status / help / quit

接线和安全：先拆下推进器螺旋桨或断开推进器电源，只给待测舵机供电；保留
独立物理急停。此脚本不能替代硬件急停，也没有经过本机实物验证。
"""

from __future__ import annotations

import argparse
import math
import sys
import time


PWM_FREQUENCY_HZ = 50
PWM_PERIOD_US = 20_000
PCA9685_MODE1 = 0x00
PCA9685_PRESCALE = 0xFE
PCA9685_LED0_ON_L = 0x06


def parse_int(value: str) -> int:
    return int(value, 0)


def bounded(value: float, low: float, high: float, label: str) -> float:
    if not low <= value <= high:
        raise ValueError(f"{label} 必须在 {low} 到 {high} 之间（当前为 {value}）")
    return value


class PCA9685:
    """最小 PCA9685 驱动；仅在 --enable 确认后才创建该对象。"""

    def __init__(self, address: int, bus_number: int) -> None:
        try:
            import smbus  # Raspberry Pi OS 包：python3-smbus
        except ImportError as exc:
            raise RuntimeError("找不到 smbus；请在树莓派安装 python3-smbus") from exc
        self.bus = smbus.SMBus(bus_number)
        self.address = address
        self._write(PCA9685_MODE1, 0x00)

    def _write(self, register: int, value: int) -> None:
        self.bus.write_byte_data(self.address, register, value & 0xFF)

    def _read(self, register: int) -> int:
        return int(self.bus.read_byte_data(self.address, register))

    def set_frequency(self, frequency_hz: int) -> None:
        prescale = math.floor(25_000_000.0 / 4096.0 / frequency_hz - 1.0 + 0.5)
        old_mode = self._read(PCA9685_MODE1)
        self._write(PCA9685_MODE1, (old_mode & 0x7F) | 0x10)
        self._write(PCA9685_PRESCALE, prescale)
        self._write(PCA9685_MODE1, old_mode)
        time.sleep(0.005)
        self._write(PCA9685_MODE1, old_mode | 0x80)

    def set_pwm(self, channel: int, on_count: int, off_count: int) -> None:
        base = PCA9685_LED0_ON_L + 4 * channel
        self._write(base, on_count)
        self._write(base + 1, on_count >> 8)
        self._write(base + 2, off_count)
        self._write(base + 3, off_count >> 8)

    def set_pulse_us(self, channel: int, pulse_us: int) -> None:
        counts = round(pulse_us * 4096 / PWM_PERIOD_US)
        self.set_pwm(channel, 0, counts)

    def disable_channel(self, channel: int) -> None:
        # PCA9685 的 full-off 位，停止该通道 PWM。
        self.set_pwm(channel, 0, 4096)

    def close(self) -> None:
        close = getattr(self.bus, "close", None)
        if close:
            close()


class ServoConsole:
    def __init__(self, args: argparse.Namespace, pwm: PCA9685 | None) -> None:
        self.args = args
        self.pwm = pwm
        self.channel = args.channel
        self.angle = 0.0
        self.last_pulse = args.pulse_center

    def _validate_channel(self, channel: int) -> int:
        bounded(channel, 0, 15, "通道")
        if channel < 8 and not self.args.allow_thruster_channel:
            raise ValueError("CH0--7 是现有工程的推进器通道；请使用 CH8--15，或明确传入 --allow-thruster-channel")
        return channel

    def _write_pulse(self, pulse: int) -> None:
        bounded(pulse, self.args.pulse_min, self.args.pulse_max, "脉宽")
        self.last_pulse = pulse
        if self.pwm:
            self.pwm.set_pulse_us(self.channel, pulse)
            print(f"已输出：CH{self.channel} = {pulse} us")
        else:
            print(f"模拟：CH{self.channel} = {pulse} us（未初始化 I2C）")

    def set_angle(self, angle: float) -> None:
        bounded(angle, self.args.angle_min, self.args.angle_max, "角度")
        ratio = (angle - self.args.angle_min) / (self.args.angle_max - self.args.angle_min)
        pulse = round(self.args.pulse_min + ratio * (self.args.pulse_max - self.args.pulse_min))
        self.angle = angle
        self._write_pulse(pulse)

    def center(self) -> None:
        self.angle = 0.0
        self._write_pulse(self.args.pulse_center)

    def show_status(self) -> None:
        mode = "真实 PWM 已启用" if self.pwm else "模拟模式（无 PWM 输出）"
        print(f"{mode} | CH{self.channel} | {self.last_pulse} us | 角度 {self.angle:g}° | 步进 {self.args.step:g}°")

    def run(self) -> None:
        print("输入 help 查看命令；Ctrl-C 与 quit 都会执行回中并关闭该通道 PWM。")
        self.show_status()
        while True:
            try:
                parts = input("servo> ").strip().split()
            except EOFError:
                return
            if not parts:
                continue
            command = parts[0].lower()
            try:
                if command in {"quit", "exit", "q"}:
                    return
                if command in {"help", "h", "?"}:
                    print("angle <度> | pulse <微秒> | + | - | center | channel <0-15> | status | quit")
                elif command == "angle" and len(parts) == 2:
                    self.set_angle(float(parts[1]))
                elif command == "pulse" and len(parts) == 2:
                    self._write_pulse(int(parts[1], 0))
                elif command == "+" and len(parts) == 1:
                    self.set_angle(self.angle + self.args.step)
                elif command == "-" and len(parts) == 1:
                    self.set_angle(self.angle - self.args.step)
                elif command in {"center", "c"} and len(parts) == 1:
                    self.center()
                elif command in {"channel", "ch"} and len(parts) == 2:
                    new_channel = self._validate_channel(int(parts[1], 0))
                    if new_channel != self.channel and self.pwm:
                        self.center()
                        self.pwm.disable_channel(self.channel)
                    self.channel = new_channel
                    self.center()
                elif command in {"status", "s"} and len(parts) == 1:
                    self.show_status()
                else:
                    print("命令无效。输入 help 查看用法。")
            except (ValueError, OSError) as exc:
                print(f"未执行：{exc}")

    def shutdown(self) -> None:
        if not self.pwm:
            return
        try:
            self.pwm.set_pulse_us(self.channel, self.args.pulse_center)
            print(f"退出保护：CH{self.channel} 回中到 {self.args.pulse_center} us。")
            time.sleep(self.args.center_hold)
            self.pwm.disable_channel(self.channel)
            print(f"退出保护：CH{self.channel} PWM 已关闭。")
        except OSError as exc:
            print(f"退出保护写入失败：{exc}", file=sys.stderr)
        finally:
            self.pwm.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCA9685 单舵机安全台架测试")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--enable", action="store_true", help="允许在二次确认后初始化 I2C/PWM")
    output.add_argument("--dry-run", action="store_true", help="模拟命令，不访问 I2C 或输出 PWM")
    parser.add_argument("--channel", type=parse_int, default=8, help="初始 PCA9685 通道，默认安全辅助通道 CH8")
    parser.add_argument("--address", type=parse_int, default=0x40, help="PCA9685 I2C 地址，默认 0x40")
    parser.add_argument("--bus", type=int, default=1, help="I2C 总线号，默认 1")
    parser.add_argument("--pulse-min", type=int, default=1000)
    parser.add_argument("--pulse-center", type=int, default=1500)
    parser.add_argument("--pulse-max", type=int, default=2000)
    parser.add_argument("--angle-min", type=float, default=-90.0)
    parser.add_argument("--angle-max", type=float, default=90.0)
    parser.add_argument("--step", type=float, default=5.0, help="+/- 的角度步进")
    parser.add_argument("--center-hold", type=float, default=0.5, help="退出回中后保持秒数")
    parser.add_argument("--allow-thruster-channel", action="store_true", help="允许 CH0--7（危险：这些是推进器通道）")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    bounded(args.address, 0x03, 0x77, "I2C 地址")
    bounded(args.bus, 0, 255, "I2C 总线号")
    bounded(args.channel, 0, 15, "通道")
    bounded(args.pulse_min, 500, 2500, "最小脉宽")
    bounded(args.pulse_center, 500, 2500, "中心脉宽")
    bounded(args.pulse_max, 500, 2500, "最大脉宽")
    if not args.pulse_min < args.pulse_center < args.pulse_max:
        raise ValueError("必须满足 pulse-min < pulse-center < pulse-max")
    if not -180 <= args.angle_min < args.angle_max <= 180:
        raise ValueError("角度范围必须满足 -180 <= angle-min < angle-max <= 180")
    bounded(args.step, 0.1, 45, "步进")
    bounded(args.center_hold, 0.2, 10, "回中保持时间")
    if args.channel < 8 and not args.allow_thruster_channel:
        raise ValueError("默认禁止 CH0--7（推进器通道）；请选择 CH8--15 或显式使用 --allow-thruster-channel")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2

    if not args.enable:
        if args.dry_run:
            print("模拟模式：不会初始化 I2C 或输出 PWM。")
            console = ServoConsole(args, None)
            console.run()
            return 0
        print("安全锁定：未传入 --enable，因此不会初始化 I2C 或输出 PWM。")
        print("真实台架测试前，请拆桨/断开推进器、接好物理急停，然后执行：python3 servo_test.py --enable")
        return 0

    print("危险确认：本程序会控制 PCA9685 上的单个通道。确认推进器已拆桨或断电，物理急停可用。")
    if input("输入 ENABLE 继续（其他任意输入取消）：").strip() != "ENABLE":
        print("已取消；未初始化 I2C 或 PWM。")
        return 0

    pwm = None
    console = None
    try:
        pwm = PCA9685(args.address, args.bus)
        pwm.set_frequency(PWM_FREQUENCY_HZ)
        console = ServoConsole(args, pwm)
        console.center()
        console.run()
        return 0
    except (RuntimeError, OSError) as exc:
        print(f"硬件初始化/通信失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C，执行退出保护。")
        return 0
    finally:
        if console:
            console.shutdown()
        elif pwm:
            pwm.close()


if __name__ == "__main__":
    raise SystemExit(main())
