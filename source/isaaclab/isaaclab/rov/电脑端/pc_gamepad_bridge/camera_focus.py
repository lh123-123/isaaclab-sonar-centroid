"""Camera-only SSH control. No MAVLink, mode, servo or motor commands."""
import ipaddress
import json
import os
import queue
import shutil
import subprocess
import threading
import tkinter as tk


class CameraFocusClient:
    def __init__(self, host, source_ip="192.168.1.4"):
        self.host = str(ipaddress.IPv4Address(host))
        self.source_ip = str(ipaddress.IPv4Address(source_ip))

    def request(self, mode=None, water_value=256):
        ssh = shutil.which("ssh")
        if ssh is None:
            raise RuntimeError("Windows SSH未安装")
        remote = "python3 /home/liang/underwater_robot/rov_video_stream.py "
        if mode is None:
            remote += "--focus-status"
        else:
            if mode not in ("water", "auto"):
                raise ValueError("仅支持水下默认或自动对焦")
            if mode == "water" and (isinstance(water_value, bool) or
                    not isinstance(water_value, int) or not 0 <= water_value <= 65535):
                raise ValueError("焦距须为有效整数")
            remote += "--set-focus " + mode + " --save-focus"
            if mode == "water":
                remote += " --water-focus " + str(water_value)
        result = subprocess.run([ssh, "-b", self.source_ip, "-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
                                 "-o", "ServerAliveInterval=3", "-o", "ServerAliveCountMax=2",
                                 "liang@" + self.host, remote], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=15,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout or "SSH连接失败").strip()[-500:])
        report = json.loads(result.stdout)
        if not isinstance(report.get("autofocus"), bool) or not all(
                type(report.get(key)) is int for key in ("minimum", "maximum", "step", "water_focus_value")):
            raise RuntimeError("摄像头状态回传不完整，未确认生效")
        if "focus_readable" not in report:
            # Old Pi code has no validity bit. Never label its auto-mode
            # focus_absolute cache as a live lens position.
            report["focus_readable"] = not report["autofocus"]
            if report["autofocus"]:
                report["focus_value"] = None
        value = report.get("focus_value")
        if type(report["focus_readable"]) is not bool or (
                report["focus_readable"] and type(value) is not int) or (
                not report["focus_readable"] and value is not None):
            raise RuntimeError("摄像头焦点位置回传无效")
        if report.get("configured_profile") not in ("water", "auto") or not (
                0 <= report["minimum"] < report["maximum"] <= 65535 and report["step"] > 0 and
                (value is None or report["minimum"] <= value <= report["maximum"])):
            raise RuntimeError("摄像头回传范围异常，未确认生效")
        if mode is not None and (report.get("saved") is not True or report["configured_profile"] != mode or
                report["autofocus"] != (mode == "auto") or
                (mode == "water" and (value != water_value or report["water_focus_value"] != water_value))):
            raise RuntimeError("对焦或默认配置未确认保存")
        return report


class CameraFocusPanel:
    def __init__(self, parent, host, on_close):
        self.client = CameraFocusClient(host)
        self.on_close = on_close
        self.closed = False
        self.results = queue.Queue()
        self.window = tk.Toplevel(parent)
        self.window.title("ROV · 摄像头对焦")
        self.window.geometry("590x345")
        self.window.minsize(550, 320)
        self.window.configure(bg="#111827")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.mode = tk.StringVar(value="water")
        self.value = tk.StringVar(value="256")
        self.readback = tk.StringVar(value="正在读取摄像头状态……")
        self.status = tk.StringVar(value="正在读取摄像头实际状态……")
        self.action_busy = False
        self.request_version = 0
        self.poll_after = None
        self.actual_report = None
        box = tk.Frame(self.window, bg="#1f2937", padx=20, pady=16)
        box.pack(fill="both", expand=True, padx=14, pady=14)
        tk.Label(box, text="摄像头对焦", bg="#1f2937", fg="#f9fafb",
                 font=("Microsoft YaHei UI", 16, "bold")).pack(anchor="w")
        choices = tk.Frame(box, bg="#1f2937")
        choices.pack(fill="x", pady=12)
        self.mode_buttons = []
        for label, mode in (("水下固定焦距", "water"), ("自动对焦", "auto")):
            button = tk.Radiobutton(choices, text=label, value=mode, variable=self.mode,
                                    command=lambda: self.begin(True), bg="#1f2937", fg="#f9fafb",
                                    selectcolor="#374151")
            button.pack(side="left", padx=5)
            self.mode_buttons.append(button)
        values = tk.Frame(box, bg="#1f2937")
        values.pack(fill="x")
        tk.Label(values, text="固定焦距", bg="#1f2937", fg="#f9fafb").pack(side="left")
        self.input = tk.Spinbox(values, from_=0, to=1023, width=8, textvariable=self.value)
        self.input.pack(side="left", padx=12)
        tk.Label(box, textvariable=self.readback, bg="#1f2937", fg="#38bdf8").pack(anchor="w", pady=8)
        self.range = tk.StringVar(value="范围待实机确认；256为历史起点")
        tk.Label(box, textvariable=self.range, bg="#1f2937", fg="#9ca3af").pack(anchor="w", pady=6)
        tk.Label(box, textvariable=self.status, bg="#1f2937", fg="#38bdf8", wraplength=530,
                 justify="left").pack(anchor="w", pady=8)
        actions = tk.Frame(box, bg="#1f2937")
        actions.pack(fill="x", pady=8)
        self.apply = tk.Button(actions, text="应用并保存固定焦距", command=lambda: self.begin(True))
        self.apply.pack(side="left", padx=(0, 10))
        self.refresh = tk.Button(actions, text="读取实际状态", command=lambda: self.begin(False))
        self.refresh.pack(side="left")
        tk.Label(box, text="水下看 QGC 画面，逐步调整固定焦距并保存；自动模式不提供实时焦距读数。",
                 bg="#1f2937", fg="#9ca3af", wraplength=530, justify="left").pack(anchor="w", pady=8)
        self.begin(False)
        self.poll_after = self.window.after(100, self.poll)

    def sync_controls(self):
        auto = self.mode.get() == "auto"
        self.input.configure(state="disabled" if auto or self.action_busy else "normal")
        self.apply.configure(state="disabled" if auto or self.action_busy else "normal")
        self.refresh.configure(state="disabled" if self.action_busy else "normal")
        for button in self.mode_buttons:
            button.configure(state="disabled" if self.action_busy else "normal")

    def start_action(self, request, message):
        self.request_version += 1
        version = self.request_version
        self.action_busy = True
        self.sync_controls()
        self.status.set(message)
        def worker():
            try:
                self.results.put((version, "action", request(), None))
            except Exception as exc:
                self.results.put((version, "action", None, str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def begin(self, apply):
        if self.closed or self.action_busy:
            return
        mode = self.mode.get() if apply else None
        value = 256
        if mode == "water":
            try:
                value = int(self.value.get())
            except ValueError:
                self.status.set("请输入整数固定焦距")
                self.sync_controls()
                return
        self.start_action(lambda: self.client.request(mode, value),
                          "正在切换并读回确认……" if apply else "正在读取实际状态……")

    def show_report(self, report):
        self.actual_report = report
        self.mode.set("auto" if report["autofocus"] else "water")
        self.value.set(str(report["water_focus_value"]))
        self.input.configure(from_=report["minimum"], to=report["maximum"], increment=report["step"])
        self.range.set(f"实机范围 {report['minimum']}–{report['maximum']}，步长 {report['step']}")
        if report["autofocus"] and not report["focus_readable"]:
            self.readback.set("自动对焦运行中 · 镜头实时位置不可读取")
        else:
            self.readback.set(f"摄像头回报：{report['focus_value']} · {'自动对焦' if report['autofocus'] else '固定焦距'}")
        self.status.set(f"水下默认：{report['water_focus_value']}；开机模式："
                        f"{'自动对焦' if report['configured_profile']=='auto' else '固定焦距'}。")
        self.sync_controls()

    def poll(self):
        if self.closed:
            return
        if self.poll_after is not None:
            self.window.after_cancel(self.poll_after)
            self.poll_after = None
        while True:
            try:
                version, kind, report, error = self.results.get_nowait()
            except queue.Empty:
                break
            # A mode change invalidates any read started before that change.
            if version != self.request_version:
                continue
            self.action_busy = False
            if error:
                self.status.set("未确认生效：" + error)
                if kind == "action" and self.actual_report:
                    self.mode.set("auto" if self.actual_report["autofocus"] else "water")
                self.sync_controls()
            else:
                self.show_report(report)
        self.poll_after = self.window.after(100, self.poll)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.poll_after is not None:
            self.window.after_cancel(self.poll_after)
            self.poll_after = None
        self.window.destroy()
        self.on_close()
