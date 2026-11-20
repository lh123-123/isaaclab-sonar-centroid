#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""树莓派端 PCA9685 CH8 舵机角度测试工具。

本程序只写入配置中的辅助舵机通道（默认 CH8），不会启动 ROV 主控，
也不会给 CH0--CH7 推进器发送 PWM。舵机本身没有角度反馈，因此程序显示
的是根据脉宽换算出的目标角度，而不是实际机械测量角度。

真实测试示例（使用 config.ini 中的 CH8 映射和软件限位）：
    sudo systemctl stop rov-usst.service
    python3 pi_runtime/servo_angle_test.py --enable

进入交互后直接输入角度数字即可，例如 ``90``；当前配置范围为 0..180°，
并映射到 1000..2000 us。输入 ``q`` 退出。
"""

from __future__ import annotations

import argparse
import configparser
import math
import os
from pathlib import Path
import sys
import time


PWM_FREQUENCY_HZ = 50
PWM_PERIOD_US = 20_000
# The supplied STM32 reference drives the servo with 1000..2000 us pulses.
REFERENCE_PULSE_MIN_US = 1000
REFERENCE_PULSE_CENTER_US = 1500
REFERENCE_PULSE_MAX_US = 2000
REFERENCE_ANGLE_MIN_DEG = 0.0
REFERENCE_ANGLE_MAX_DEG = 180.0
PCA9685_MODE1 = 0x00
PCA9685_PRESCALE = 0xFE
PCA9685_LED0_ON_L = 0x06


def _number(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"不是有效数字：{value}") from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("数值必须是有限数字")
    return result


def _find_config(explicit: str | None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get("ROV_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    script_dir = Path(__file__).resolve().parent
    candidates.extend(
        (
            script_dir.parent / "config.ini",
            Path.cwd() / "config.ini",
            Path("/etc/rov/config.ini"),
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _read_settings(config_path: Path | None) -> dict[str, float | int]:
    parser = configparser.ConfigParser()
    if config_path is not None:
        parser.read(config_path, encoding="utf-8")
    if not parser.has_section("servo"):
        raise ValueError("找不到 [servo] 配置段；请使用 --config 指定 config.ini")

    servo = parser["servo"]
    hardware = parser["hardware"] if parser.has_section("hardware") else {}
    settings: dict[str, float | int] = {
        "channel": servo.getint("channel", fallback=8),
        "address": int(str(hardware.get("pwm_i2c_address", "0x40")), 0),
        "bus": 1,
    }
    settings.update(
        pulse_min=servo.getint("pulse_min_us", fallback=REFERENCE_PULSE_MIN_US),
        pulse_center=servo.getint(
            "pulse_center_us", fallback=REFERENCE_PULSE_CENTER_US
        ),
        pulse_max=servo.getint("pulse_max_us", fallback=REFERENCE_PULSE_MAX_US),
        angle_min=servo.getfloat("angle_min_deg", fallback=REFERENCE_ANGLE_MIN_DEG),
        angle_max=servo.getfloat("angle_max_deg", fallback=REFERENCE_ANGLE_MAX_DEG),
    )
    if not 0 <= int(settings["channel"]) <= 15:
        raise ValueError("舵机通道必须在 0..15；本测试通常应为 CH8")
    if int(settings["channel"]) != 8:
        raise ValueError(
            f"配置中的舵机通道是 CH{int(settings['channel'])}，本程序固定要求 CH8"
        )
    if not int(settings["pulse_min"]) < int(settings["pulse_center"]) < int(settings["pulse_max"]):
        raise ValueError("舵机脉宽必须满足 pulse_min_us < pulse_center_us < pulse_max_us")
    if not 0.0 <= float(settings["angle_min"]) < float(settings["angle_max"]) <= 180.0:
        raise ValueError("舵机角度范围必须在 0°..180° 内")
    if not 0x03 <= int(settings["address"]) <= 0x77:
        raise ValueError("PCA9685 I2C 地址无效")
    return settings


class PCA9685:
    """只提供本测试所需的 PCA9685 寄存器操作。"""

    def __init__(self, address: int, bus_number: int) -> None:
        try:
            import smbus  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "找不到 smbus，请在树莓派执行：sudo apt install -y python3-smbus"
            ) from exc
        self.bus = smbus.SMBus(bus_number)
        self.address = address
        self._write(PCA9685_MODE1, 0x00)

    def _write(self, register: int, value: int) -> None:
        self.bus.write_byte_data(self.address, register, value & 0xFF)

    def _read(self, register: int) -> int:
        return int(self.bus.read_byte_data(self.address, register))

    def set_frequency(self, frequency_hz: int) -> None:
        prescale = int(math.floor(25_000_000.0 / 4096.0 / frequency_hz - 1.0 + 0.5))
        old_mode = self._read(PCA9685_MODE1)
        self._write(PCA9685_MODE1, (old_mode & 0x7F) | 0x10)
        self._write(PCA9685_PRESCALE, prescale)
        self._write(PCA9685_MODE1, old_mode)
        time.sleep(0.005)
        self._write(PCA9685_MODE1, old_mode | 0x80)

    def set_pulse_us(self, channel: int, pulse_us: int) -> None:
        counts = max(0, min(4095, round(pulse_us * 4096 / PWM_PERIOD_US)))
        register = PCA9685_LED0_ON_L + 4 * channel
        self._write(register, 0)
        self._write(register + 1, 0)
        self._write(register + 2, counts)
        self._write(register + 3, counts >> 8)

    def disable(self, channel: int) -> None:
        register = PCA9685_LED0_ON_L + 4 * channel
        self._write(register, 0)
        self._write(register + 1, 0)
        self._write(register + 2, 0)
        self._write(register + 3, 0x10)  # full-off bit

    def close(self) -> None:
        close = getattr(self.bus, "close", None)
        if callable(close):
            close()


class ServoTester:
    def __init__(self, settings: dict[str, float | int], pwm: PCA9685 | None) -> None:
        self.settings = settings
        self.pwm = pwm
        self.angle = 0.0

    def pulse_for_angle(self, angle: float) -> int:
        angle_min = float(self.settings["angle_min"])
        angle_max = float(self.settings["angle_max"])
        if not angle_min <= angle <= angle_max:
            raise ValueError(f"角度必须在 {angle_min:g}° 到 {angle_max:g}° 之间")
        pulse_min = int(self.settings["pulse_min"])
        pulse_center = int(self.settings["pulse_center"])
        pulse_max = int(self.settings["pulse_max"])
        if angle_min < 0.0 and angle < 0.0:
            ratio = angle / angle_min if angle_min < 0.0 else 0.0
            return round(pulse_center + ratio * (pulse_min - pulse_center))
        if angle_min < 0.0:
            ratio = angle / angle_max
            return round(pulse_center + ratio * (pulse_max - pulse_center))
        ratio = (angle - angle_min) / (angle_max - angle_min)
        return round(pulse_min + ratio * (pulse_max - pulse_min))

    def angle_for_pulse(self, pulse: float) -> float:
        """Convert a configured pulse back to the displayed target angle."""
        angle_min = float(self.settings["angle_min"])
        angle_max = float(self.settings["angle_max"])
        pulse_min = float(self.settings["pulse_min"])
        pulse_max = float(self.settings["pulse_max"])
        ratio = (float(pulse) - pulse_min) / (pulse_max - pulse_min)
        return angle_min + ratio * (angle_max - angle_min)

    def set_angle(self, angle: float) -> None:
        pulse = self.pulse_for_angle(angle)
        self.angle = angle
        channel = int(self.settings["channel"])
        if self.pwm is not None:
            self.pwm.set_pulse_us(channel, pulse)
            prefix = "已输出"
        else:
            prefix = "模拟"
        print(f"{prefix}：CH{channel} 目标角度 {angle:g}° -> {pulse} us")

    def run(self) -> None:
        channel = int(self.settings["channel"])
        angle_min = float(self.settings["angle_min"])
        angle_max = float(self.settings["angle_max"])
        initial_angle = 0.0 if angle_min <= 0.0 <= angle_max else (angle_min + angle_max) / 2.0
        print(f"舵机测试已就绪：CH{channel}，直接输入角度数字；输入 q 退出。")
        print(
            f"允许范围：{angle_min:g}° .. {angle_max:g}°；"
            f"当前为 {initial_angle:g}°"
        )
        self.set_angle(initial_angle)
        while True:
            try:
                raw = input("角度> ").strip()
            except EOFError:
                break
            if raw.lower() in {"q", "quit", "exit"}:
                break
            if not raw:
                continue
            try:
                self.set_angle(_number(raw))
            except (ValueError, argparse.ArgumentTypeError) as exc:
                print(f"未执行：{exc}")

    def close(self) -> None:
        if self.pwm is not None:
            channel = int(self.settings["channel"])
            center = int(self.settings["pulse_center"])
            try:
                self.pwm.set_pulse_us(channel, center)
                center_angle = self.angle_for_pulse(center)
                print(f"退出保护：CH{channel} 回到 {center_angle:g}°（{center} us）")
                time.sleep(0.3)
                self.pwm.disable(channel)
            finally:
                self.pwm.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCA9685 CH8 舵机角度测试")
    parser.add_argument("--enable", action="store_true", help="允许初始化 I2C 并输出 PWM")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不访问 I2C")
    parser.add_argument("--config", help="config.ini 路径；默认自动查找")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.enable and args.dry_run:
        print("--enable 与 --dry-run 不能同时使用", file=sys.stderr)
        return 2
    try:
        config_path = _find_config(args.config)
        settings = _read_settings(config_path)
    except (OSError, ValueError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    if not args.enable and not args.dry_run:
        print("安全锁定：默认不会打开 I2C。要真实测试请执行：")
        print("  python3 pi_runtime/servo_angle_test.py --enable")
        print("先确认推进器断电/拆桨，舵机使用独立电源并共地。")
        return 0

    pwm = None
    tester = None
    try:
        if args.enable:
            print("危险确认：本程序将控制 PCA9685 CH8。输入 ENABLE 继续，其他输入取消：")
            if input().strip() != "ENABLE":
                print("已取消；未初始化 I2C 或输出 PWM。")
                return 0
            pwm = PCA9685(int(settings["address"]), int(settings["bus"]))
            pwm.set_frequency(PWM_FREQUENCY_HZ)
        tester = ServoTester(settings, pwm)
        tester.run()
        return 0
    except (RuntimeError, OSError) as exc:
        print(f"硬件初始化/通信失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C，执行退出保护。")
        return 0
    finally:
        if tester is not None:
            tester.close()
        elif pwm is not None:
            pwm.close()


if __name__ == "__main__":
    raise SystemExit(main())
