#!/usr/bin/env python3
"""Chinese APEX 5 monitor/control panel for the ROV competition profile.

The window is monitor-only unless --send is supplied.  It replaces QGC's
numeric 0..14 button monitor with meaningful Chinese names and also handles
the independent M1-M4/LM/RM vendor buttons.
"""

from __future__ import annotations

import argparse
import logging
import time
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox

from control_profile import (
    DEFAULT_PROFILE,
    LocalActionController,
    button_mask,
    load_profile,
    resolve_button_conflicts,
    save_default_motion_gain,
    save_quick_targets,
)
from flight_mode import FlightModeController
from system_monitor import SystemMonitor
from level_zero import LevelZero
from attitude_tolerance import AttitudeTolerance
from level_target import LevelTarget
from qgc_fly_status import CompetitionTelemetry, StatusServer, make_snapshot
from camera_focus import CameraFocusPanel
from motor_test import MotorTestSession
from motor_test_panel import MotorTestPanel
from thrust_monitor import ThrustMonitor
from runtime_support import ResumeGate, SingleInstance, neutral_input
from motion_input import MotionSlewLimiter, read_latest_input
from servo_control import ServoController, save_servo_ranges
from flydigi_apex5 import Apex5State, build_manual_control
from rov_gamepad_bridge import (
    ACQUIRE_INTERVAL_SECONDS,
    INPUT_TIMEOUT_SECONDS,
    acquire_command,
    command_neutral_for_arm,
    open_controller,
    open_mavlink,
    open_telemetry,
    poll_telemetry,
    request_power_telemetry,
    request_mode,
    send_command,
    send_gimbal_angle,
    send_servo_pwm,
)


VISIBLE_BUTTONS = (0, 1, 2, 3, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--allow-arm", action="store_true")
    parser.add_argument("--pi-ip", default="192.168.1.3")
    parser.add_argument("--port", type=int, default=14550)
    parser.add_argument("--telemetry-port", type=int, default=14551)
    parser.add_argument("--target-system", type=int, default=1)
    parser.add_argument("--rate", type=float, default=25.0)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--compact", action="store_true", help="small always-on-top QGC status dock")
    parser.add_argument("--receive", action="store_true", help="receive telemetry even in monitor mode")
    return parser.parse_args()


class ControlPanel:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.profile_path = Path(args.profile) if args.profile else DEFAULT_PROFILE
        self.profile = load_profile(args.profile) if args.profile else load_profile()
        self.actions = LocalActionController(self.profile)
        self.motion_limiter = MotionSlewLimiter.from_profile(self.profile)
        self.servos = ServoController.from_profile(self.profile)
        self.forward_mask = button_mask(self.profile["button_forward_indices"])
        self.controller = None
        self.controller_notice = "等待手柄连接"
        self.next_connect = 0.0
        self.link = open_mavlink(args.pi_ip, args.port) if args.send else None
        self.receiver = open_telemetry(args.telemetry_port) if args.send or getattr(args, "receive", False) else None
        self.modes = FlightModeController(args.target_system)
        self.level_zero = LevelZero(args.target_system)
        self.level_zero_link = None
        self.attitude_tolerance = AttitudeTolerance(args.target_system)
        self.attitude_tolerance_link = None
        self.attitude_tolerance_next_read = 0.0
        self.level_target = LevelTarget(args.target_system)
        self.level_target_link = None
        self.level_target_next_read = 0.0
        self.fly_telemetry = CompetitionTelemetry(args.target_system)
        self.fly_status_server = None
        self.fly_status_next = 0.0
        if args.send or getattr(args, 'receive', False):
            try:
                self.fly_status_server = StatusServer()
            except OSError as exc:
                logging.warning('QGC Fly只读仪表服务无法启动：%s', exc)
        self.default_requested = False
        self.default_confirmed = not args.send
        self.system_monitor = SystemMonitor(args.pi_ip,
            enabled=bool(getattr(args, 'receive', False) or args.send))
        self.resume = ResumeGate()
        self.state = Apex5State.neutral()
        self.last_input = float("-inf")
        self.next_acquire = 0.0
        self.next_send = 0.0
        self.power_next_request = 0.0
        self.last_gimbal_text = f"{self.actions.gimbal.angle_deg:+.0f}°"
        self.closing = False
        self.failed = False
        self.motor_dialog = None
        self.camera_dialog = None
        self.thrust_window = None
        self.thrust_link = None
        self.thrust_next_refresh = 0.0
        self.thrust_monitor = ThrustMonitor(self.profile["pixhawk_button_parameters"], args.target_system)
        self._last_logged_state = None

        self.root.title("水下机器人比赛控制面板｜飞智八爪鱼5")
        icon = Path(__file__).with_name("rov.ico")
        if icon.is_file():
            self.root.iconbitmap(str(icon))
        self.root.geometry("1180x940")
        self.root.minsize(1100, 900)
        self.root.configure(bg="#111827")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.colors = {
            "bg": "#111827",
            "panel": "#1f2937",
            "card": "#374151",
            "active": "#16a34a",
            "warning": "#f59e0b",
            "danger": "#dc2626",
            "text": "#f9fafb",
            "muted": "#9ca3af",
            "accent": "#38bdf8",
        }
        self.button_cards: dict[int, tk.Label] = {}
        self.status_var = tk.StringVar()
        self.attitude_var = tk.StringVar(value="实测姿态：未知（等待回传）")
        self.attitude_raw_var = tk.StringVar(value="飞控绝对姿态：未知")
        self.attitude_notice_var = tk.StringVar()
        self.attitude_tolerance_input = tk.StringVar(value="0.5")
        self.attitude_tolerance_notice = tk.StringVar(value=self.attitude_tolerance.notice)
        self.level_target_notice = tk.StringVar(value=self.level_target.notice)
        self.pilot_status_var = tk.StringVar()
        self.hold_var = tk.StringVar()
        self.system_var = tk.StringVar()
        self.power_var = tk.StringVar()
        self.attitude_next_refresh = 0.0
        self.axes_var = tk.StringVar()
        self.gimbal_var = tk.StringVar(value=self.last_gimbal_text)
        self.gain_var = tk.StringVar(value=self.actions.gain_text())
        self.servo_var = tk.StringVar(value=self.servos.status_text())
        self.arm_var = tk.BooleanVar(value=args.allow_arm and args.send)
        self.transport_var = tk.StringVar(value="停止控制" if args.send else "开启手柄控制")
        self.button_guide_text = self._button_guide_text()
        self.button_guide_var = tk.StringVar(value=self.button_guide_text)
        self.root.report_callback_exception = self.callback_failed
        if getattr(args, "compact", False):
            self._build_compact_ui()
        else:
            self._build_ui()
        self.refresh_attitude(time.monotonic())
        self.root.after(1, self.poll)

    def _scroll_body(self, parent, *, width=None):
        """Scrollable status column, so added telemetry cannot hide controls."""
        frame = tk.Frame(parent, bg=self.colors["panel"])
        canvas = tk.Canvas(frame, bg=self.colors["panel"], highlightthickness=0,
                           width=width or 480)
        bar = tk.Scrollbar(frame, command=canvas.yview)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=bar.set)
        content = tk.Frame(canvas, bg=self.colors["panel"], padx=14, pady=12)
        item = canvas.create_window(0, 0, anchor="nw", window=content)
        content.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(item, width=e.width))

        def wheel(event):
            # Only this column; do not intercept settings or motor-test dialogs.
            widget = event.widget
            while widget is not None and widget is not frame:
                widget = getattr(widget, "master", None)
            if widget is frame and canvas.yview() != (0.0, 1.0):
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        self.root.bind("<MouseWheel>", wheel, add="+")
        self.status_canvas = canvas
        return frame, content

    def _build_attitude_card(self, parent):
        card = tk.Frame(parent, bg="#172c40", padx=12, pady=10,
                        highlightbackground="#31516a", highlightthickness=1)
        card.pack(fill="x", pady=(4, 12))
        self._label(card, "飞控原生姿态 · 与QGC一致", bg="#172c40",
                    font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        self._label(card, textvariable=self.attitude_var, bg="#172c40",
                    fg=self.colors["accent"], justify="left", wraplength=400,
                    font=("Microsoft YaHei UI", 17, "bold"), pady=8).pack(anchor="w")
        self._label(card, textvariable=self.attitude_raw_var, bg="#172c40",
                    fg=self.colors["muted"], justify="left", wraplength=400,
                    font=("Microsoft YaHei UI", 9)).pack(anchor="w")
        self._label(card, textvariable=self.attitude_notice_var, bg="#172c40",
                    fg=self.colors['muted'], wraplength=400, justify='left',
                    font=('Microsoft YaHei UI', 9)).pack(anchor='w')
        tk.Button(card, text="设当前姿态为调平目标", command=self.capture_level_target,
                  bg="#28485f", fg="#edf5ff", relief="flat", padx=12).pack(anchor="w", pady=5)
        self._label(card, textvariable=self.level_target_notice, bg="#172c40",
                    fg=self.colors['muted'], wraplength=400, justify='left',
                    font=('Microsoft YaHei UI', 9)).pack(anchor='w')
        tolerance_row = tk.Frame(card, bg="#172c40")
        tolerance_row.pack(fill="x", pady=(8, 2))
        self._label(tolerance_row, "回平允许误差 ±", bg="#172c40",
                    font=("Microsoft YaHei UI", 10)).pack(side="left")
        self.tolerance_entry = tk.Entry(tolerance_row, textvariable=self.attitude_tolerance_input,
                                        width=6, justify="center", font=("Microsoft YaHei UI", 11))
        self.tolerance_entry.pack(side="left", padx=(3, 1))
        self._label(tolerance_row, "°", bg="#172c40").pack(side="left")
        tk.Button(tolerance_row, text="应用", command=self.apply_attitude_tolerance,
                  bg="#28485f", fg="#edf5ff", relief="flat", padx=12).pack(side="right")
        self._label(card, textvariable=self.attitude_tolerance_notice, bg="#172c40",
                    fg=self.colors["muted"], wraplength=400, justify="left",
                    font=("Microsoft YaHei UI", 9)).pack(anchor="w")
        self._label(card, textvariable=self.hold_var, bg="#172c40", fg=self.colors['accent'],
                    font=("Microsoft YaHei UI", 12, "bold")).pack(anchor='w', pady=5)
        self._label(card, textvariable=self.pilot_status_var, bg="#172c40",
                    justify="left", wraplength=400,
                    font=("Microsoft YaHei UI", 9), pady=5).pack(anchor="w")

    def _button_guide_text(self):
        buttons = self.profile["buttons"]
        groups = (
            ("安全与模式", (0, 1, 3, 2)),
            ("档位与夹爪", (4, 6, 9, 10)),
            ("十字快捷", (11, 12, 13, 14)),
            ("末端慢调", (15, 16, 17, 18, 19, 20)),
        )
        rows = ["运动轴", "左摇杆  前后/左右平移    右摇杆  偏航/俯仰",
                "RT  上升（线性）          LT  下潜（线性）"]
        for title, indices in groups:
            rows.extend(("", title))
            rows.extend(f'{buttons[str(index)]["name"]:<10} {buttons[str(index)]["action"]}'
                        for index in indices)
        return "\n".join(rows)

    def _build_button_guide(self, parent):
        card = tk.Frame(parent, bg=self.colors["card"], padx=12, pady=10)
        card.pack(fill="x", pady=(5, 12))
        self._label(card, "手柄按键对应功能", bg=self.colors["card"],
                    font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        self._label(card, "Y：姿态开/关　X：定深开/关",
                    bg=self.colors["card"], fg=self.colors["accent"],
                    font=("Microsoft YaHei UI", 10, "bold"), pady=5).pack(anchor="w")
        self._label(card, textvariable=self.button_guide_var, bg=self.colors["card"],
                    fg=self.colors["text"], justify="left", anchor="w",
                    font=("Microsoft YaHei UI", 9)).pack(fill="x")
        self._label(card, "模式切换以状态栏的飞控回传为准；显示‘请求中’不等于已经切换。",
                    bg=self.colors["card"], fg=self.colors["muted"],
                    justify="left", wraplength=450,
                    font=("Microsoft YaHei UI", 9), pady=6).pack(anchor="w")

    def refresh_attitude(self, now):
        display = self.modes.attitude_display
        fresh = self.modes.fresh(now)
        self.attitude_var.set(display.angle_text(now, fresh))
        self.attitude_raw_var.set('原生角度，不置零换算。\nY稳姿松杆回保存的调平目标；航向松杆锁定。')
        self.attitude_notice_var.set(display.notice)
        self.pilot_status_var.set(self.modes.pilot_control_status(now))
        self.hold_var.set(self.modes.hold_status(now))
        self.system_var.set(self.system_monitor.text(now))
        self.power_var.set(self.modes.power_status(now))

    def _build_system_card(self, parent):
        box = tk.Frame(parent, bg='#172c40', padx=12, pady=10)
        box.pack(fill='x', pady=(0, 12))
        self._label(box, '实时资源与供电', bg='#172c40',
                    font=('Microsoft YaHei UI', 12, 'bold')).pack(anchor='w')
        for var in (self.system_var, self.power_var):
            self._label(box, textvariable=var, bg='#172c40', justify='left',
                        wraplength=440, font=('Microsoft YaHei UI', 10), pady=5).pack(anchor='w')

    def zero_attitude_display(self):
        """Read-only UI reference; never sends a calibration or control command."""
        now = time.monotonic()
        self.modes.attitude_display.zero(now, disarmed=self.modes.fresh(now) and not self.modes.armed)
        self.refresh_attitude(now)

    def zero_attitude_control(self):
        """Legacy callback: native-angle mode must not change FC calibration."""
        self.zero_attitude_display()

    def apply_attitude_tolerance(self):
        now = time.monotonic()
        link = None
        try:
            if self.receiver is None:
                self.receiver = open_telemetry(self.args.telemetry_port)
            link = open_mavlink(self.args.pi_ip, self.args.port)
            self.attitude_tolerance.apply(
                link, self.modes, now, self.attitude_tolerance_input.get(),
                stopped=self.link is None and self.motor_dialog is None and not self.level_zero.busy and not self.level_target.busy,
            )
            self.attitude_tolerance_link = link
        except (ValueError, OSError) as exc:
            if link is not None:
                link.close()
            self.attitude_tolerance_notice.set(str(exc))

    def capture_level_target(self):
        now = time.monotonic()
        link = None
        try:
            if self.receiver is None:
                self.receiver = open_telemetry(self.args.telemetry_port)
            link = open_mavlink(self.args.pi_ip, self.args.port)
            self.level_target.capture(link, self.modes, now,
                stopped=self.link is None and self.motor_dialog is None
                and not self.level_zero.busy and not self.attitude_tolerance.busy)
            self.level_target_link = link
        except (ValueError, OSError) as exc:
            if link is not None:
                link.close()
            self.level_target.notice = str(exc)
        self.level_target_notice.set(self.level_target.notice)

    def _build_compact_ui(self):
        self.root.title("ROV状态｜关闭此窗会停止手柄桥接")
        height = max(650, min(940, self.root.winfo_screenheight() - 160))
        self.root.geometry(f"540x{height}+20+80")
        self.root.minsize(520, 650)
        self.root.attributes("-topmost", True)
        frame, box = self._scroll_body(self.root)
        frame.pack(fill="both", expand=True)
        self._label(box, "QGC比赛状态", font=("Microsoft YaHei UI", 16, "bold"),
                    bg=self.colors["panel"]).pack(anchor="w")
        self._label(box, textvariable=self.status_var, wraplength=385, justify="left",
                    bg=self.colors["panel"], pady=10).pack(fill="x")
        tk.Button(box, textvariable=self.transport_var, command=self.toggle_control).pack(fill="x", pady=6)
        tk.Checkbutton(box, text="允许A键解锁（开启控制后可用）", variable=self.arm_var,
                       command=self.update_arm_permission, bg=self.colors["panel"],
                       fg=self.colors["text"], selectcolor=self.colors["bg"]).pack(anchor="w")
        self._build_attitude_card(box)
        self._build_system_card(box)
        self._build_button_guide(box)
        self._label(box, "独立操纵档位（非功率上限）", bg=self.colors["panel"]).pack(anchor="w")
        self._label(box, textvariable=self.gain_var, fg=self.colors["accent"],
                    font=("Microsoft YaHei UI", 18, "bold"), bg=self.colors["panel"]).pack(anchor="w")
        tk.Button(box, text="默认运动档位设置 · LB复位", command=self.open_gain_settings).pack(fill="x", pady=4)
        self._label(box, "云台目标角度（非实测）", bg=self.colors["panel"]).pack(anchor="w")
        self._label(box, textvariable=self.gimbal_var, fg=self.colors["accent"],
                    font=("Microsoft YaHei UI", 22, "bold"), bg=self.colors["panel"]).pack(anchor="w")
        self._label(box, "机械臂舵机目标", bg=self.colors["panel"]).pack(anchor="w", pady=(8, 0))
        self._label(box, textvariable=self.servo_var, fg=self.colors["accent"],
                    font=("Microsoft YaHei UI", 11, "bold"), bg=self.colors["panel"],
                    justify="left").pack(anchor="w")
        camera_tools = tk.Frame(box, bg=self.colors["panel"])
        camera_tools.pack(fill="x", pady=(6, 4))
        tk.Button(camera_tools, text="舵机范围与速度", command=self.open_servo_settings).pack(side="left", fill="x", expand=True)
        tk.Button(camera_tools, text="十字快捷目标", command=self.open_quick_target_settings).pack(side="left", fill="x", expand=True, padx=(5, 0))
        tk.Button(camera_tools, text="摄像头对焦", command=self.open_camera_settings).pack(side="left", fill="x", expand=True, padx=(5, 0))
        tk.Button(box, text="动力与转向回传（只读）", command=self.open_thrust_monitor).pack(fill="x", pady=(4, 4))
        tk.Button(box, text="推进器逐个测试  →", command=self.open_motor_test,
                  bg="#155b70", fg="#edf5ff", relief="flat", pady=10,
                  font=("Microsoft YaHei UI", 11, "bold")).pack(fill="x", pady=(6, 4))
        self._label(box, "启动只监视；QGC Joystick须取消Enable。\n开启控制后默认两项关闭（等Manual回传）。\nX定深开/关 · Y姿态开/关 · B上锁",
                    bg=self.colors["panel"], fg=self.colors["muted"], justify="left").pack(anchor="w", pady=8)

    def update_arm_permission(self):
        self.args.allow_arm = bool(self.link and self.arm_var.get())
        self.arm_var.set(self.args.allow_arm)

    def toggle_control(self):
        if self.level_target.busy:
            self.modes.notice = '请等待调平目标读回后开启控制'
            return
        if self.level_zero.busy:
            self.modes.notice = '水平置零尚未结束，请保持静止、等待确认'
            return
        if self.failed:
            return
        if self.motor_dialog is not None:
            self.modes.notice = "请先停止测试、确认上锁并关闭推进器测试面板"
            return
        now = time.monotonic()
        if self.link is not None:
            if self.modes.fresh(now) and self.modes.armed:
                self.modes.notice = "飞控仍已解锁，请先按B并确认上锁，再停止桥接"
                return
            self.link.close()
            self.link = None
            self.args.send = False
            self.args.allow_arm = False
            self.arm_var.set(False)
            self.transport_var.set("开启手柄控制")
            self.modes.notice = "已停止控制流；这不等于飞控已上锁"
            return
        if not self.modes.fresh(now) or now - self.last_input > INPUT_TIMEOUT_SECONDS:
            self.modes.notice = "不能开启：手柄或飞控回传未就绪"
            return
        if self.modes.armed or not neutral_input(self.state):
            self.modes.notice = "先上锁并松开所有按键、摇杆、扳机"
            return
        try:
            self.link = open_mavlink(self.args.pi_ip, self.args.port)
        except OSError as exc:
            self.modes.notice = f"开启失败：{exc}"
            return
        self.args.send = True
        self.args.allow_arm = False
        self.arm_var.set(False)
        self.resume = ResumeGate()
        self.default_requested = self.default_confirmed = False
        self.transport_var.set("停止控制")
        self.modes.notice = "等待归中0.5秒，尚未解锁"
        logging.info("Control stream enabled by operator; A remains blocked")

    def callback_failed(self, kind, value, traceback):
        logging.error("Control callback failed; transmission stopped", exc_info=(kind, value, traceback))
        self.failed = True
        if self.level_target.busy:
            self.level_target.fail('程序异常，调平目标未确认')
        if self.level_target_link is not None:
            self.level_target_link.close()
            self.level_target_link = None
        if self.level_zero.busy:
            self.level_zero.fail('程序异常，水平校准结果未确认')
        if self.level_zero_link is not None:
            self.level_zero_link.close()
            self.level_zero_link = None
        if self.attitude_tolerance.busy:
            self.attitude_tolerance.fail('程序异常，稳姿误差写入结果未确认')
        if self.attitude_tolerance_link is not None:
            self.attitude_tolerance_link.close()
            self.attitude_tolerance_link = None
        if self.motor_dialog is not None:
            try:
                self.motor_dialog.session.stop(time.monotonic(), "程序异常，已请求停止及上锁")
            except OSError:
                logging.exception("Could not send motor-test stop; firmware watchdog must stop output")
        if self.link is not None:
            self.link.close()
            self.link = None
        self.args.send = self.args.allow_arm = False
        self.arm_var.set(False)
        self.transport_var.set("程序异常，请查看日志后重新启动")
        self.status_var.set("控制循环异常，已停止发送。\n飞控状态未知；请查看QGC。\n" + str(value))
        if self.motor_dialog is not None:
            self.root.after(100, self.poll_failed_test)

    def poll_failed_test(self):
        """After a callback failure, only receive status/disarm confirmation."""
        if self.closing or self.motor_dialog is None:
            return
        now = time.monotonic()
        try:
            poll_telemetry(self.receiver, self.modes, now, self.motor_dialog.session)
            self.motor_dialog.session.tick(now)
            self.motor_dialog.refresh(now)
        except Exception:
            logging.exception("Safety-only telemetry loop failed")
            self.motor_dialog.feedback.set("程序异常，上锁无法确认！请用QGC上锁或切断推进器供电")
        self.root.after(100, self.poll_failed_test)

    def _label(self, parent, text="", **kwargs):
        return tk.Label(
            parent,
            text=text,
            bg=kwargs.pop("bg", self.colors["bg"]),
            fg=kwargs.pop("fg", self.colors["text"]),
            font=kwargs.pop("font", ("Microsoft YaHei UI", 11)),
            **kwargs,
        )

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg=self.colors["bg"])
        top.pack(fill="x", padx=22, pady=(18, 10))
        tk.Button(top, text="推进器逐个测试", command=self.open_motor_test,
                  bg="#155b70", fg="#edf5ff", relief="flat", padx=12, pady=8).pack(side="right", padx=12)
        tk.Button(top, text="摄像头", command=self.open_camera_settings,
                  bg="#263951", fg="#edf5ff", relief="flat", padx=12, pady=8).pack(side="right", padx=4)
        tk.Button(top, text="动力回传", command=self.open_thrust_monitor,
                  bg="#263951", fg="#edf5ff", relief="flat", padx=12, pady=8).pack(side="right")
        self._label(
            top,
            "水下机器人｜比赛手柄",
            font=("Microsoft YaHei UI", 20, "bold"),
        ).pack(side="left")
        mode = "实机发送" if self.args.send else "安全监视（不发控制）"
        mode_color = self.colors["danger"] if self.args.send else self.colors["active"]
        self._label(
            top,
            mode,
            bg=mode_color,
            font=("Microsoft YaHei UI", 11, "bold"),
            padx=14,
            pady=6,
        ).pack(side="right")

        body = tk.Frame(self.root, bg=self.colors["bg"])
        body.pack(fill="both", expand=True, padx=22, pady=8)
        left_frame, left = self._scroll_body(body, width=470)
        left_frame.pack(side="left", fill="y", padx=(0, 12))
        right = tk.Frame(body, bg=self.colors["panel"], padx=18, pady=16)
        right.pack(side="left", fill="both", expand=True)

        self._build_attitude_card(left)
        self._build_system_card(left)
        self._label(left, "运动轴", bg=self.colors["panel"], font=("Microsoft YaHei UI", 15, "bold")).pack(anchor="w")
        self._label(
            left,
            "左摇杆：前后 / 左右平移\nRT：上升（线性）   LT：下潜（线性）\n右摇杆：偏航 / 俯仰\n稳定模式：俯仰/横滚松杆回平，偏航松杆定向\n正方向辅助，明确斜推仍保留",
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            justify="left",
            pady=10,
        ).pack(anchor="w")
        self._label(
            left,
            textvariable=self.axes_var,
            bg=self.colors["card"],
            font=("Consolas", 11),
            justify="left",
            padx=12,
            pady=12,
        ).pack(fill="x", pady=(4, 16))

        self._label(left, "云台目标角度（非实测）", bg=self.colors["panel"], font=("Microsoft YaHei UI", 15, "bold")).pack(anchor="w")
        self._label(
            left,
            textvariable=self.gimbal_var,
            bg=self.colors["card"],
            fg=self.colors["accent"],
            font=("Microsoft YaHei UI", 30, "bold"),
            padx=12,
            pady=10,
        ).pack(fill="x", pady=(8, 4))
        preset_text = "  /  ".join(f"{v:+g}°" for v in self.actions.gimbal.presets_deg)
        self._label(left, preset_text, bg=self.colors["panel"], fg=self.colors["muted"]).pack(anchor="w")

        self._label(left, "操纵增益档位", bg=self.colors["panel"], font=("Microsoft YaHei UI", 15, "bold"), pady=12).pack(anchor="w")
        self._label(
            left,
            textvariable=self.gain_var,
            bg=self.colors["card"],
            fg=self.colors["accent"],
            font=("Microsoft YaHei UI", 18, "bold"),
            wraplength=340,
            padx=12,
            pady=8,
        ).pack(fill="x")
        self._label(
            left,
            "左菜单：摇杆档｜右菜单：升潜档｜LB：恢复默认",
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            pady=5,
        ).pack(anchor="w")
        tk.Button(left, text="默认运动档位设置 · LB复位", command=self.open_gain_settings).pack(fill="x", pady=4)

        self._label(left, "机械臂舵机目标", bg=self.colors["panel"],
                    font=("Microsoft YaHei UI", 15, "bold"), pady=10).pack(anchor="w")
        self._label(left, textvariable=self.servo_var, bg=self.colors["card"],
                    fg=self.colors["accent"], justify="left", padx=12, pady=8).pack(fill="x")
        tk.Button(left, text="舵机范围与速度设置", command=self.open_servo_settings).pack(fill="x", pady=(6, 0))
        tk.Button(left, text="十字快捷目标设置", command=self.open_quick_target_settings).pack(fill="x", pady=(6, 0))

        self._label(left, "连接状态", bg=self.colors["panel"], font=("Microsoft YaHei UI", 15, "bold"), pady=14).pack(anchor="w")
        self._label(
            left,
            textvariable=self.status_var,
            bg=self.colors["panel"],
            justify="left",
            wraplength=330,
            font=("Microsoft YaHei UI", 10),
        ).pack(anchor="w")

        self._label(right, "常用按键（未使用键已隐藏）", bg=self.colors["panel"], font=("Microsoft YaHei UI", 15, "bold")).pack(anchor="w")
        grid = tk.Frame(right, bg=self.colors["panel"])
        grid.pack(fill="both", expand=True, pady=(12, 0))
        for position, index in enumerate(VISIBLE_BUTTONS):
            meta = self.profile["buttons"][str(index)]
            text = f'{meta["name"]}\n{meta["action"]}'
            card = self._label(
                grid,
                text,
                bg=self.colors["card"],
                font=("Microsoft YaHei UI", 10, "bold"),
                width=18,
                height=3,
                relief="flat",
            )
            card.grid(row=position // 3, column=position % 3, sticky="nsew", padx=5, pady=5)
            self.button_cards[index] = card
        for column in range(3):
            grid.columnconfigure(column, weight=1)
        for row in range(6):
            grid.rowconfigure(row, weight=1)

        foot = self._label(
            self.root,
            "X定深开/关｜Y姿态开/关（相互保留）｜十字左云台快捷角度／右上下舵机复位｜档位非功率上限。",
            bg=self.colors["warning"],
            fg="#111827",
            font=("Microsoft YaHei UI", 10, "bold"),
            pady=7,
        )
        foot.pack(fill="x", side="bottom")

    def open_motor_test(self):
        if self.level_target.busy:
            self.modes.notice = '请等待调平目标读回后测试'
            return
        if self.level_zero.busy:
            self.modes.notice = '水平校准正在进行，不能启动推进器测试'
            return
        if self.failed:
            return
        if self.motor_dialog is not None:
            self.motor_dialog.window.lift()
            return
        if self.link is not None:
            messagebox.showwarning("先停止手柄控制", "先上锁并点击“停止控制”，再进入逐个推进器测试。", parent=self.root)
            return
        if self.receiver is None:
            try:
                self.receiver = open_telemetry(self.args.telemetry_port)
            except OSError as exc:
                messagebox.showerror("遥测端口不可用", str(exc), parent=self.root)
                return
        try:
            link = open_mavlink(self.args.pi_ip, self.args.port)
        except OSError as exc:
            messagebox.showerror("测试链路不可用", str(exc), parent=self.root)
            return
        try:
            self.motor_dialog = MotorTestPanel(
                self, MotorTestSession(self.modes, link),
                startup_offset_us=self.profile["thruster_startup"]["reference_offset_us"],
            )
        except Exception:
            link.close()
            raise

    def open_thrust_monitor(self):
        if self.thrust_window is not None:
            self.thrust_window.lift()
            return
        try:
            if self.receiver is None:
                self.receiver = open_telemetry(self.args.telemetry_port)
            self.thrust_link = open_mavlink(self.args.pi_ip, self.args.port)
        except OSError as exc:
            messagebox.showerror("回传链路不可用", str(exc), parent=self.root)
            return
        window = tk.Toplevel(self.root)
        self.thrust_window = window
        self.thrust_next_refresh = 0.0
        window.title("ROV · 动力与姿态回传（只读）")
        window.geometry("760x700")
        window.configure(bg=self.colors["bg"])
        self._label(window, "实际回传 · 不解锁 / 不写参数 / 不驱动电机",
                    font=("Microsoft YaHei UI", 15, "bold"), pady=14).pack(anchor="w", padx=20)
        self.thrust_text = tk.Text(window, bg=self.colors["panel"], fg=self.colors["text"],
                                   relief="flat", font=("Microsoft YaHei UI", 11),
                                   padx=15, pady=15, state="disabled", wrap="word")
        self.thrust_text.pack(fill="both", expand=True, padx=20)
        self._label(window, "上锁时限速读取指定参数；解锁后暂停读参，仅显示已有回传。\n"
                    "100%是完整操纵输入，不保证定深/稳定模式中每台电机同时满输出。",
                    fg=self.colors["muted"], justify="left", pady=12).pack(anchor="w", padx=20)
        window.protocol("WM_DELETE_WINDOW", self.close_thrust_monitor)

    def close_thrust_monitor(self):
        if self.thrust_link is not None:
            self.thrust_link.close()
            self.thrust_link = None
        if self.thrust_window is not None:
            self.thrust_window.destroy()
            self.thrust_window = None

    def open_camera_settings(self):
        if self.camera_dialog is not None:
            self.camera_dialog.window.lift()
            return
        self.camera_dialog = CameraFocusPanel(self.root, self.args.pi_ip,
                                              lambda: setattr(self, "camera_dialog", None))

    def open_quick_target_settings(self) -> None:
        if self.link is not None or self.motor_dialog is not None or (
                self.modes.fresh(time.monotonic()) and self.modes.armed):
            messagebox.showwarning("先停止控制", "请先上锁并停止手柄控制，再修改快捷目标。", parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title("十字键快捷目标")
        window.geometry("390x210")
        window.resizable(False, False)
        window.transient(self.root)
        window.grab_set()
        frame = tk.Frame(window, padx=18, pady=16)
        frame.pack(fill="both", expand=True)
        gimbal = tk.StringVar(value=f"{self.profile['gimbal'].get('quick_angle_deg', -20):g}")
        reset = tk.StringVar(value=str(self.profile['servos']['angle'].get('reset_pwm', 920)))
        tk.Label(frame, text="十字左 · 云台角度（-90～20°）").grid(row=0, column=0, sticky="w", pady=9)
        tk.Entry(frame, textvariable=gimbal, width=11).grid(row=0, column=1, pady=9)
        tk.Label(frame, text="十字右 · 上下舵机复位（µs）").grid(row=1, column=0, sticky="w", pady=9)
        tk.Entry(frame, textvariable=reset, width=11).grid(row=1, column=1, pady=9)
        tk.Label(frame, text="复位PWM必须在上下舵机当前最小/最大范围内。",
                 fg="#596579").grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 12))

        def save() -> None:
            try:
                angle = float(gimbal.get())
                pwm = int(reset.get())
                save_quick_targets(self.profile_path, angle, pwm)
                self.profile["gimbal"]["quick_angle_deg"] = angle
                self.profile["servos"]["angle"]["reset_pwm"] = pwm
                self.profile["buttons"]["13"]["action"] = f"云台快捷角度 {angle:+g}°"
                self.profile["buttons"]["14"]["action"] = f"夹爪上下复位 {pwm}µs"
                self.button_guide_text = self._button_guide_text()
                self.button_guide_var.set(self.button_guide_text)
                for index in (13, 14):
                    if index in self.button_cards:
                        meta = self.profile["buttons"][str(index)]
                        self.button_cards[index].configure(text=f'{meta["name"]}\n{meta["action"]}')
            except (ValueError, OSError, KeyError) as exc:
                messagebox.showerror("未保存", str(exc), parent=window)
                return
            window.destroy()
            messagebox.showinfo("保存成功", "两个快捷目标已保存，下次启动仍保留。", parent=self.root)

        tk.Button(frame, text="取消", command=window.destroy).grid(row=3, column=0, sticky="e", padx=6)
        tk.Button(frame, text="保存", command=save).grid(row=3, column=1, sticky="e", padx=6)

    def open_servo_settings(self) -> None:
        if self.link is not None:
            messagebox.showwarning("先停止控制", "请先点击“停止控制”，再修改舵机范围。")
            return
        window = tk.Toplevel(self.root)
        window.title("机械臂 · 舵机范围与速度")
        window.geometry("660x370")
        window.resizable(False, False)
        window.transient(self.root)
        window.grab_set()
        frame = tk.Frame(window, padx=18, pady=16)
        frame.pack(fill="both", expand=True)
        tk.Label(
            frame,
            text="QGC显示Disabled是正常状态：AUX1～3由电脑直接控制。\n"
                 "端点500～2500µs；慢速20～500µs/s，数值越大越快。\n"
                 "建议先用100；按住连续、松手保持。RB快速开合不受此速度影响。",
            justify="left",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))
        tk.Label(frame, text="舵机").grid(row=1, column=0, padx=8)
        tk.Label(frame, text="最小PWM").grid(row=1, column=1, padx=8)
        tk.Label(frame, text="最大PWM").grid(row=1, column=2, padx=8)
        tk.Label(frame, text="慢速（µs/s）").grid(row=1, column=3, padx=8)
        labels = {"gripper": "夹爪开合", "angle": "夹爪上下", "rotation": "末端旋转"}
        variables: dict[str, tuple[tk.StringVar, tk.StringVar]] = {}
        speed_variables = {}
        for row, name in enumerate(("gripper", "angle", "rotation"), start=2):
            axis = self.servos.axes[name]
            minimum = tk.StringVar(value=str(axis.minimum_pwm))
            maximum = tk.StringVar(value=str(axis.maximum_pwm))
            variables[name] = (minimum, maximum)
            speed = tk.StringVar(value=f"{self.servos.speed_for(name):g}")
            speed_variables[name] = speed
            tk.Label(frame, text=labels[name]).grid(row=row, column=0, sticky="w", padx=8, pady=7)
            tk.Spinbox(frame, from_=500, to=2500, increment=10, textvariable=minimum,
                       width=10).grid(row=row, column=1, padx=8, pady=7)
            tk.Spinbox(frame, from_=500, to=2500, increment=10, textvariable=maximum,
                       width=10).grid(row=row, column=2, padx=8, pady=7)
            tk.Spinbox(frame, from_=20, to=500, increment=10, textvariable=speed,
                       width=12).grid(row=row, column=3, padx=8, pady=7)

        def save() -> None:
            try:
                ranges = {
                    name: (int(pair[0].get()), int(pair[1].get()))
                    for name, pair in variables.items()
                }
                speeds = {name: float(value.get()) for name, value in speed_variables.items()}
                save_servo_ranges(self.profile_path, ranges, speeds)
                self.servos.configure_ranges(ranges)
                self.servos.configure_speeds(speeds)
                self.profile["servos"]["angle"]["reset_pwm"] = min(
                    max(int(self.profile["servos"]["angle"].get("reset_pwm", 920)), ranges["angle"][0]),
                    ranges["angle"][1],
                )
                reset_pwm = self.profile["servos"]["angle"]["reset_pwm"]
                self.profile["buttons"]["14"]["action"] = f"夹爪上下复位 {reset_pwm}µs"
                self.button_guide_var.set(self._button_guide_text())
                if 14 in self.button_cards:
                    meta = self.profile["buttons"]["14"]
                    self.button_cards[14].configure(text=f'{meta["name"]}\n{meta["action"]}')
            except (ValueError, OSError, KeyError) as exc:
                messagebox.showerror("未保存", str(exc), parent=window)
                return
            self.servo_var.set(self.servos.status_text())
            window.destroy()
            messagebox.showinfo("保存成功", "三路舵机范围和速度已保存，下次启动仍保留。", parent=self.root)

        buttons = tk.Frame(frame)
        buttons.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(16, 0))
        tk.Button(buttons, text="取消", command=window.destroy).pack(side="right", padx=5)
        tk.Button(buttons, text="保存", command=save).pack(side="right", padx=5)

    def open_gain_settings(self):
        if self.link is not None or self.motor_dialog is not None or (
                self.modes.fresh(time.monotonic()) and self.modes.armed):
            messagebox.showwarning("先停止控制", "请先上锁、停止手柄控制和电机测试，再设置默认档位。", parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title("运动档位 · 启动与LB默认值")
        window.geometry("500x235")
        window.resizable(False, False)
        window.transient(self.root)
        window.grab_set()
        frame = tk.Frame(window, padx=20, pady=20)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="左菜单独立循环摇杆档；右菜单独立循环升潜档。\n"
                 "两组在启动时、按LB时恢复到下面保存的默认值。\n"
                 "保存只改变默认值，当前档位不跳变。", justify="left").pack(anchor="w")
        values = sorted(set(self.actions.motion.levels) & set(self.actions.vertical.levels))
        choices = {f"{value:.0%}": value for value in values}
        default = self.actions.motion.levels[self.actions.motion.default_index]
        selected = tk.StringVar(value=f"{default:.0%}")
        row = tk.Frame(frame)
        row.pack(fill="x", pady=15)
        tk.Label(row, text="共同默认档位").pack(side="left", padx=(0, 20))
        tk.OptionMenu(row, selected, *choices).pack(side="left")

        def save():
            try:
                gain = choices[selected.get()]
                save_default_motion_gain(self.profile_path, gain)
                self.actions.set_default_gain(gain)
            except (ValueError, KeyError, OSError) as exc:
                messagebox.showerror("未保存", str(exc), parent=window)
                return
            window.destroy()
            self.modes.notice = f"默认档位已保存为{gain:.0%}；按LB恢复两组或下次启动生效"

        tk.Button(frame, text="保存默认档位", command=save).pack(side="right")
        tk.Button(frame, text="取消", command=window.destroy).pack(side="right", padx=10)

    def poll(self) -> None:
        if self.closing:
            return
        now = time.monotonic()
        batch = None
        try:
            if self.controller is None and now >= self.next_connect:
                self.next_connect = now + 2.0
                self.controller = open_controller()
                self.controller_notice = "已连接"
                self.next_acquire = now + ACQUIRE_INTERVAL_SECONDS
                logging.info("APEX5 connected")
            if self.controller is not None:
                if now >= self.next_acquire:
                    if self.controller.write(acquire_command()) < 0:
                        raise OSError("手柄接管失败")
                    self.next_acquire = now + ACQUIRE_INTERVAL_SECONDS
                batch = read_latest_input(self.controller)
        except (OSError, RuntimeError) as exc:
            if self.controller is not None:
                self.controller.close()
                self.controller = None
            self.last_input = float("-inf")
            self.controller_notice = "未连接（每2秒重试）"
            logging.warning("Controller unavailable: %s", exc)
        if batch is not None:
            if batch.backlogged:
                self.state = Apex5State.neutral()
                self.last_input = float('-inf')
                self.controller_notice = '输入积压：等待新鲜输入并归中'
            elif batch.state is not None:
                self.state = batch.state
                self.last_input = now
                self.controller_notice = '已连接'

        stale = now - self.last_input > INPUT_TIMEOUT_SECONDS
        effective = Apex5State.neutral() if stale else self.state
        observers = [self.servos, self.thrust_monitor, self.fly_telemetry,
                     self.level_zero, self.attitude_tolerance, self.level_target]
        if self.motor_dialog is not None:
            observers.append(self.motor_dialog.session)
        poll_telemetry(self.receiver, self.modes, now, *observers)
        # This requests only telemetry timing, never arming or motor output.
        # QGC may later change its stream schedule, so renew sparingly.
        if (self.receiver is not None and self.modes.fresh(now)
                and now >= self.power_next_request):
            self.power_next_request = now + 30.0
            temporary_link = None
            try:
                power_link = self.link
                if power_link is None:
                    temporary_link = open_mavlink(self.args.pi_ip, self.args.port)
                    power_link = temporary_link
                request_power_telemetry(power_link, self.args.target_system)
            except OSError as exc:
                self.power_next_request = now + 5.0
                logging.warning('Cannot request 5 Hz power telemetry: %s', exc)
            finally:
                if temporary_link is not None:
                    temporary_link.close()
        self.level_zero.tick(self.modes, now)
        if (self.receiver is not None and self.modes.fresh(now)
                and self.level_target.actual is None and not self.level_target.busy
                and now >= self.level_target_next_read):
            self.level_target_next_read = now + 15
            try:
                self.level_target_link = open_mavlink(self.args.pi_ip, self.args.port)
                self.level_target.read(self.level_target_link, now)
            except OSError as exc:
                self.level_target.fail(f'无法读取调平目标：{exc}')
        self.level_target.tick(self.modes, now)
        self.level_target_notice.set(self.level_target.notice)
        if self.level_target_link is not None and not self.level_target.busy:
            self.level_target_link.close()
            self.level_target_link = None
        if (self.receiver is not None and self.modes.fresh(now)
                and self.attitude_tolerance.actual is None
                and now >= self.attitude_tolerance_next_read
                and not self.attitude_tolerance.busy):
            self.attitude_tolerance_next_read = now + 15.0
            try:
                self.attitude_tolerance_link = open_mavlink(self.args.pi_ip, self.args.port)
                self.attitude_tolerance.read(self.attitude_tolerance_link, now)
            except OSError as exc:
                self.attitude_tolerance.fail(f'无法读取稳姿误差：{exc}')
                if self.attitude_tolerance_link is not None:
                    self.attitude_tolerance_link.close()
                    self.attitude_tolerance_link = None
        self.attitude_tolerance.tick(self.modes, now)
        self.attitude_tolerance_notice.set(self.attitude_tolerance.notice)
        if (self.attitude_tolerance.actual is not None
                and self.root.focus_get() is not self.tolerance_entry
                and not self.attitude_tolerance.busy):
            self.attitude_tolerance_input.set(f'{self.attitude_tolerance.actual:g}')
        if self.attitude_tolerance_link is not None and not self.attitude_tolerance.busy:
            self.attitude_tolerance_link.close()
            self.attitude_tolerance_link = None
        if self.level_zero_link is not None and not self.level_zero.busy:
            self.level_zero_link.close()
            self.level_zero_link = None
            logging.info('Native level zero result: %s; evidence=%s', self.level_zero.notice, self.level_zero.report)
        if now >= self.attitude_next_refresh:
            self.attitude_next_refresh = now + 0.1
            self.refresh_attitude(now)
        if self.thrust_window is not None:
            self.thrust_monitor.request_one(self.thrust_link, self.modes, now)
            if now >= self.thrust_next_refresh:
                self.thrust_next_refresh = now + 0.25
                self.thrust_text.configure(state="normal")
                self.thrust_text.delete("1.0", "end")
                self.thrust_text.insert("1.0", self.modes.attitude_status(now) + "\n" +
                                        self.modes.pilot_control_status(now) + "\n\n" +
                                        self.thrust_monitor.status(self.modes, now))
                self.thrust_text.configure(state="disabled")
        if self.motor_dialog is not None:
            self.motor_dialog.session.tick(now)
            self.motor_dialog.refresh(now)
        ready = self.resume.update(effective, not stale and self.modes.fresh(now), now)
        if self.link is not None and not self.default_confirmed:
            if ready and not self.default_requested and not self.modes.armed:
                request_mode(self.link, self.modes, 'reset_manual', now)
                self.default_requested = True
            if self.default_requested and self.modes.fresh(now) and self.modes.mode == 19 and self.modes.pending is None:
                self.default_confirmed = True
            ready = ready and self.default_confirmed

        action_update = self.actions.update(effective.buttons)
        if action_update.gimbal_angle_deg is not None:
            self.last_gimbal_text = f"{action_update.gimbal_angle_deg:+.0f}°"
            self.gimbal_var.set(self.last_gimbal_text)
            if self.link is not None and ready:
                send_gimbal_angle(
                    self.link, self.args.target_system, action_update.gimbal_angle_deg
                )
        if action_update.motion_gain is not None or action_update.vertical_gain is not None:
            self.gain_var.set(self.actions.gain_text())
        if action_update.mode_action is not None:
            if ready or self.link is None:
                request_mode(self.link, self.modes, action_update.mode_action, now)
            else:
                self.modes.notice = "等待链路正常且全部按键/摇杆/扳机归中0.5秒"

        safe_buttons = resolve_button_conflicts(effective.buttons)
        if batch is not None and batch.disarm_requested:
            safe_buttons = 1 << 1  # Disarm wins even while dropping an old FIFO.
        servo_commands = self.servos.update(
            safe_buttons & ~button_mask((15, 16)) if effective.buttons &
            (1 << int(self.profile["servos"]["angle"].get("reset_button", 14)))
            else safe_buttons,
            now,
            enabled=self.link is not None and ready and not stale,
        )
        for servo_command in servo_commands:
            send_servo_pwm(
                self.link,
                self.args.target_system,
                servo_command.channel,
                servo_command.pwm,
            )
        if action_update.angle_servo_reset_pwm is not None and self.link is not None and ready and not stale:
            preset = self.servos.command_preset("angle", action_update.angle_servo_reset_pwm, now)
            send_servo_pwm(self.link, self.args.target_system, preset.channel, preset.pwm)
        self.servo_var.set(self.servos.status_text())
        control_state = replace(effective, buttons=safe_buttons)
        if safe_buttons & (1 << 1):
            control_state = replace(Apex5State.neutral(), buttons=1 << 1)
        axis = self.profile["axis"]
        command = build_manual_control(
            control_state,
            stick_deadzone=float(axis["stick_deadzone"]),
            trigger_deadzone=float(axis["trigger_deadzone"]),
            vertical_scale=float(axis["vertical_scale"]),
            motion_scale=self.actions.motion.gain,
            vertical_motion_scale=self.actions.vertical.gain,
            axis_snap_inner_deg=float(axis["axis_snap_inner_deg"]),
            axis_snap_outer_deg=float(axis["axis_snap_outer_deg"]),
            yaw_response_exponent=float(axis.get("yaw_response_exponent", 1.0)),
            allow_arm=self.args.send and self.args.allow_arm and ready and not self.modes.armed
                and command_neutral_for_arm(effective),
            button_mask=self.forward_mask,
        )
        if self.link is not None:
            command = self.motion_limiter.update(command, now,
                active=ready and not stale and self.modes.pending is None,
                motion_gain=self.actions.motion.gain, vertical_gain=self.actions.vertical.gain,
                depth_hold=self.modes.fresh(now) and self.modes.mode in (2, 22))
        else:
            self.motion_limiter.reset()

        can_send = (not stale and ready) or bool(safe_buttons & (1 << 1))
        urgent_disarm = batch is not None and batch.disarm_requested
        if self.link is not None and can_send and (now >= self.next_send or urgent_disarm):
            send_command(self.link, self.args.target_system, command)
            self.next_send = now + 1.0 / self.args.rate

        if self.fly_status_server is not None and now >= self.fly_status_next:
            self.fly_status_next = now + .2
            self.fly_status_server.publish(make_snapshot(
                self.modes, self.fly_telemetry, self.actions, self.servos, now,
                hid_fresh=not stale, sending=self.link is not None and can_send,
                ready=ready, allow_arm=self.args.send and self.args.allow_arm,
                motor_testing=self.motor_dialog is not None,
                resources=self.system_monitor.snapshot(now)), now)

        self.axes_var.set(
            f"前后 x  {command.x:+5d}\n"
            f"平移 y  {command.y:+5d}\n"
            f"升潜 z  {command.z:5d}\n"
            f"偏航    {command.r:+5d}\n"
            f"俯仰    {command.s:+5d}"
        )
        arm = "允许" if self.args.send and self.args.allow_arm else "锁止"
        self.status_var.set(
            f"手柄：{self.controller_notice if self.controller is None else ('超时' if stale else '正常')}\n"
            f"控制流：{'发送中' if self.link and can_send else '未发送'}\n"
            f"A键解锁：{arm}\n"
            f"{self.modes.status(now)}\n{self.modes.pilot_gain_status(now)}\n{self.modes.notice}"
        )
        snapshot = (stale, ready, bool(self.link), self.args.allow_arm, self.modes.mode,
                    self.modes.armed, self.modes.pending, self.modes.notice,
                    self.actions.motion.gain, self.actions.vertical.gain, self.actions.gimbal.angle_deg)
        if snapshot != self._last_logged_state:
            logging.info("State stale=%s ready=%s sending=%s arm_allowed=%s mode=%s armed=%s pending=%s notice=%s stick_gain=%s vertical_gain=%s gimbal_target=%s", *snapshot)
            self._last_logged_state = snapshot
        for index, card in self.button_cards.items():
            pressed = bool(effective.buttons & (1 << index))
            card.configure(bg=self.colors["active"] if pressed else self.colors["card"])

        self.root.after(10, self.poll)

    def close(self) -> None:
        if self.closing:
            return
        if self.level_target.busy:
            self.modes.notice = '调平目标正在读回，请稍等'
            return
        if self.level_zero.busy:
            self.modes.notice = '水平校准正在进行，请等待结果后再退出'
            return
        if self.attitude_tolerance.busy:
            self.modes.notice = '稳姿误差参数正在读写，请等待结果后再退出'
            return
        if self.motor_dialog is not None:
            self.motor_dialog.close()
            if self.motor_dialog is not None:
                self.modes.notice = "请先在推进器测试面板确认上锁后关闭"
                return
        if self.link is not None and self.modes.fresh(time.monotonic()) and self.modes.armed:
            self.modes.notice = "飞控仍已解锁，请先按B确认上锁，再关闭桥接"
            return
        self.closing = True
        if self.camera_dialog is not None:
            self.camera_dialog.close()
        self.system_monitor.close()
        if self.fly_status_server is not None:
            self.fly_status_server.close()
        self.close_thrust_monitor()
        if self.attitude_tolerance_link is not None:
            self.attitude_tolerance_link.close()
            self.attitude_tolerance_link = None
        if self.link is not None:
            neutral = build_manual_control(Apex5State.neutral())
            for _ in range(5):
                send_command(self.link, self.args.target_system, neutral)
                time.sleep(0.02)
            self.link.close()
        if self.receiver is not None:
            self.receiver.close()
        if self.controller is not None:
            self.controller.close()
        self.root.destroy()


def run_panel(args) -> int:
    if not 0 < args.rate <= 100:
        raise ValueError("发送频率必须在0～100Hz之间")
    if args.allow_arm and not args.send:
        print("NOTE: --allow-arm has no effect without --send")
    root = tk.Tk()
    try:
        panel = ControlPanel(root, args)
    except (RuntimeError, OSError) as exc:
        root.withdraw()
        messagebox.showerror("启动失败", str(exc))
        root.destroy()
        return 1
    try:
        root.mainloop()
    except KeyboardInterrupt:
        panel.close()
    return 0


def main() -> int:
    guard = SingleInstance()
    try:
        return run_panel(parse_args())
    finally:
        guard.close()


if __name__ == "__main__":
    raise SystemExit(main())
