"""Physical-layout motor bench dialog, sharing the dock's telemetry receiver."""
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from motor_test import (EXPECTED_PARAMS, PHYSICAL_TO_MOTOR, POSITIONS,
                        MOTION_FACTORS, MOTION_GROUPS, motion_pwm_plan)

BG, CARD, BORDER = "#0b1220", "#131f32", "#263951"
TEXT, MUTED, CYAN = "#edf5ff", "#93a9c4", "#50d4ec"


class MotorTestPanel:
    def __init__(self, owner, session, startup_offset_us=50):
        if isinstance(startup_offset_us, bool) or not isinstance(startup_offset_us, int) or not 0 <= startup_offset_us <= 100:
            raise ValueError("启转参考偏移须为0～100整数μs")
        self.startup_offset_us = startup_offset_us
        self.owner, self.session = owner, session
        self.window = tk.Toplevel(owner.root)
        self.window.title("ROV · 推进器PWM测试")
        icon = Path(__file__).with_name("rov.ico")
        if icon.is_file():
            self.window.iconbitmap(str(icon))
        self.ui_scale = max(1.0, float(self.window.tk.call("tk", "scaling")) / (4/3))
        self.window.geometry(f"{round(1080*self.ui_scale)}x{round(800*self.ui_scale)}")
        self.window.minsize(round(1000*self.ui_scale), round(760*self.ui_scale))
        self.window.configure(bg=BG)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda _e: self.stop())
        self.window.bind("<FocusOut>", self.focus_out, add="+")
        self.level = tk.IntVar(self.window, value=10)
        self.confirm = tk.BooleanVar(self.window, value=False)
        self.title = tk.StringVar(self.window)
        self.subtitle = tk.StringVar(self.window)
        self.status = tk.StringVar(self.window)
        self.feedback = tk.StringVar(self.window)
        self.badge = tk.StringVar(self.window)
        self.selected_text = tk.StringVar(self.window)
        self.pwm_text = tk.StringVar(self.window)
        self.text_var = tk.StringVar(self.window)
        self.targets_var = tk.StringVar(self.window)
        self.test_mode = "single"
        self.preview_action = "前进"
        self.single_values = (tk.StringVar(self.window, value="1450"), tk.StringVar(self.window, value="1550"))
        self.motion_values = {action: (tk.StringVar(self.window, value="1490"), tk.StringVar(self.window, value="1510"))
                              for action in MOTION_FACTORS}
        self.motion_buttons = {}
        self.motion_rows = {}
        self.motion_group = tk.StringVar(self.window, value=next(iter(MOTION_GROUPS)))
        self.nodes = {}
        self.held = False
        self.closed = False
        self.build()
        self.refresh(time.monotonic())

    def label(self, parent, text="", size=11, color=TEXT, bg=CARD, **kw):
        return tk.Label(parent, text=text, font=("Microsoft YaHei UI", size),
                        bg=bg, fg=color, **kw)

    def button(self, parent, text, command=None, color=BORDER, **kw):
        padx, pady = kw.pop("padx", 14), kw.pop("pady", 8)
        return tk.Button(parent, text=text, command=command, bg=color, fg=TEXT,
                         activebackground="#31516e", activeforeground=TEXT,
                         disabledforeground="#68819b", font=("Microsoft YaHei UI", 11, "bold"),
                         relief="flat", bd=0, cursor="hand2", padx=padx, pady=pady, **kw)

    def pwm_entry(self, parent, variable, width=10):
        entry = tk.Entry(parent, textvariable=variable, width=width, bg=BG, fg=TEXT,
                         insertbackground=CYAN, relief="flat", justify="center",
                         font=("Microsoft YaHei UI", 12), highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=CYAN)
        # Changing inputs never changes a running output silently.
        entry.bind("<KeyPress>", lambda _e: self.release())
        return entry

    def apply_level(self):
        self.release()
        delta = self.level.get() * 5
        for var, pwm in zip(self.single_values, (1500-delta, 1500+delta)):
            var.set(str(pwm))

    def apply_startup_reference(self):
        # This only fills raw test values. Never starts or modifies a live test.
        self.release()
        pairs = (self.single_values, *self.motion_values.values())
        for pair in pairs:
            for var, pwm in zip(pair, (1500-self.startup_offset_us, 1500+self.startup_offset_us)):
                var.set(str(pwm))
        self.session.notice = f"已填入±{self.startup_offset_us}μs参考；不是实测启转阈值，须另点启动；常规控制补偿需自定义飞控固件"
        self.refresh(time.monotonic())

    def switch_mode(self, mode):
        self.release()
        self.test_mode = mode
        self.single_box.pack_forget()
        self.motion_box.pack_forget()
        (self.single_box if mode == "single" else self.motion_box).pack(fill="both", expand=True)
        self.refresh(time.monotonic())

    def switch_motion_group(self, group):
        self.release()  # Changing a category never switches a running test.
        for row in self.motion_rows.values():
            row.pack_forget()
        for action in MOTION_GROUPS[group]:
            self.motion_rows[action].pack(fill="x", pady=2)
        self.preview_action = MOTION_GROUPS[group][0]
        self.refresh(time.monotonic())

    def build(self):
        header = tk.Frame(self.window, bg=BG)
        header.pack(fill="x", padx=26, pady=(16, 10))
        self.label(header, "推进器测试", size=23, bg=BG).pack(side="left")
        self.label(header, " /  点击启动 · 再点停止 · 实际编号", color=MUTED, bg=BG).pack(side="left", padx=12)
        self.badge_widget = self.label(header, textvariable=self.badge, color=CYAN,
                                       bg=CARD, padx=16, pady=9)
        self.badge_widget.pack(side="right")
        body = tk.Frame(self.window, bg=BG)
        # Reserve the footer first so smaller/DPI-scaled windows never hide it.
        body.columnconfigure(0, weight=5)
        body.columnconfigure(1, weight=4)
        body.rowconfigure(0, weight=1)
        left = tk.Frame(body, bg=CARD, padx=20, pady=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        self.label(left, "01  选择推进器", size=14).pack(anchor="w")
        self.label(left, "俯视图 · 船头朝上 · 点击编号", color=MUTED).pack(anchor="w", pady=(4, 8))
        self.canvas = tk.Canvas(left, bg=CARD, highlightthickness=0, width=470, height=385)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.draw())
        self.label(left, textvariable=self.selected_text, size=14, color=CYAN).pack(anchor="w", pady=(4, 8))
        self.label(left, textvariable=self.pwm_text, color=MUTED).pack(anchor="w")
        self.label(left, textvariable=self.targets_var, size=10, color=CYAN,
                   justify="left").pack(anchor="w", pady=(8, 0))
        self.label(left, "回传PWM仅表示输出，不证明电机已经旋转。", size=10, color=MUTED,
                   wraplength=420, justify="left").pack(anchor="w", pady=(10, 0))
        right = tk.Frame(body, bg=CARD, padx=20, pady=12)
        right.grid(row=0, column=1, sticky="nsew")
        self.label(right, "02  PWM启转测试", size=14).pack(anchor="w")
        self.label(right, textvariable=self.title, size=17, color=CYAN).pack(anchor="w", pady=(6, 0))
        tabs = tk.Frame(right, bg=CARD)
        tabs.pack(fill="x", pady=(8, 8))
        self.button(tabs, "逐个测试", lambda: self.switch_mode("single"), color="#155b70").pack(side="left", expand=True, fill="x", padx=3)
        self.button(tabs, "运动启转", lambda: self.switch_mode("motion"), color="#30466a").pack(side="left", expand=True, fill="x", padx=3)
        area = tk.Frame(right, bg=CARD, height=round(250*self.ui_scale))
        area.pack(fill="x")
        area.pack_propagate(False)
        self.single_box = tk.Frame(area, bg=CARD)
        self.motion_box = tk.Frame(area, bg=CARD)
        self.single_box.pack(fill="both", expand=True)
        self.label(self.single_box, "输入PWM（原值输出，整数，1000～2000）", size=10, color=MUTED).pack(anchor="w")
        values = tk.Frame(self.single_box, bg=CARD)
        values.pack(fill="x", pady=(6, 8))
        for var in self.single_values:
            self.pwm_entry(values, var).pack(side="left", expand=True, fill="x", padx=3)
        self.label(self.single_box, "快捷填值（相对1500中位）", size=10, color=MUTED).pack(anchor="w")
        levels = tk.Frame(self.single_box, bg=CARD)
        levels.pack(fill="x", pady=(6, 10))
        for value in (5, 10, 15):
            tk.Radiobutton(levels, text=f"{value}%", value=value, variable=self.level,
                           indicatoron=False, bg=BORDER, fg=TEXT, selectcolor="#17647b",
                           command=self.apply_level,
                           activebackground="#31516e", font=("Microsoft YaHei UI", 12),
                           relief="flat", bd=0, padx=18, pady=6).pack(side="left", expand=True, fill="x", padx=3)
        row = tk.Frame(self.single_box, bg=CARD)
        row.pack(fill="x")
        self.forward = self.button(row, "高侧PWM · 启动", color="#155b70")
        self.backward = self.button(row, "低侧PWM · 启动", color="#30466a")
        for widget, direction in ((self.forward, 1), (self.backward, -1)):
            widget.pack(side="left", expand=True, fill="both", padx=3)
            widget.configure(command=lambda d=direction: self.press(d))
        self.label(self.single_box, "低/高侧仅指PWM，不代表机身前进/后退。", size=10,
                   color=MUTED).pack(anchor="w", pady=(8, 0))
        selector = tk.OptionMenu(self.motion_box, self.motion_group,
                                 *MOTION_GROUPS, command=self.switch_motion_group)
        selector.configure(bg=BORDER, fg=TEXT, activebackground="#31516e",
                           activeforeground=TEXT, highlightthickness=0, bd=0,
                           font=("Microsoft YaHei UI", 10))
        selector['menu'].configure(bg=CARD, fg=TEXT)
        selector.pack(fill="x", pady=(0, 3))
        headings = tk.Frame(self.motion_box, bg=CARD)
        headings.pack(fill="x")
        for text in ("低侧 ≤1500", "高侧 ≥1500", "机身方向"):
            self.label(headings, text, size=10, color=MUTED, width=12).pack(side="left", expand=True)
        self.motion_row_box = tk.Frame(self.motion_box, bg=CARD)
        self.motion_row_box.pack(fill="x")
        for action, pair in self.motion_values.items():
            motion_row = tk.Frame(self.motion_row_box, bg=CARD)
            self.motion_rows[action] = motion_row
            if action in MOTION_GROUPS[self.motion_group.get()]:
                motion_row.pack(fill="x", pady=2)
            for var in pair:
                self.pwm_entry(motion_row, var, width=8).pack(side="left", expand=True, fill="x", padx=3)
            button = self.button(motion_row, f"{action} · 启动", lambda a=action: self.press_motion(a), padx=8, pady=4)
            button.pack(side="left", expand=True, fill="x", padx=3)
            self.motion_buttons[action] = button
        self.label(self.motion_box, "按机架与飞控方向分配PWM；其余4台保持1500。\n抬头/低头/倾斜为开环方向测试，不是姿态稳定。", size=10,
                   color=MUTED, justify="left", wraplength=390).pack(anchor="w", pady=(6, 0))
        self.button(right, f"填入启转参考 ±{self.startup_offset_us}μs（不启动）",
                    self.apply_startup_reference, pady=4).pack(fill="x", pady=(4, 0))
        self.label(right, "再点任一测试按钮停止 · 失焦 / 断链停止", size=10,
                   color=MUTED).pack(anchor="w", pady=(8, 8))
        tk.Checkbutton(right, text="已关闭QGC电机测试及Joystick Enable\n已固定机体、无异物，符合推进器测试条件",
                       variable=self.confirm, bg=CARD, fg=TEXT, selectcolor=BG,
                       activebackground=CARD, activeforeground=TEXT,
                       justify="left", font=("Microsoft YaHei UI", 10)).pack(anchor="w")
        self.prepare = self.button(right, "进入测试 · 请求解锁", self.enable, color="#17647b")
        self.prepare.pack(fill="x", pady=(10, 8))
        self.button(right, "■  停止测试并上锁   Esc", self.stop, color="#853a45").pack(fill="x")
        self.label(right, textvariable=self.status, size=10, color=MUTED,
                   wraplength=355, justify="left").pack(anchor="w", pady=(8, 0))
        footer = tk.Frame(self.window, bg=BG)
        footer.pack(fill="x", side="bottom", padx=26, pady=10)
        self.label(footer, textvariable=self.feedback, bg=BG, color=CYAN,
                   wraplength=960, justify="left").pack(anchor="w")
        self.label(footer, textvariable=self.text_var, bg=BG, color=MUTED,
                   wraplength=960, justify="left", size=10).pack(anchor="w", pady=(5, 0))
        body.pack(fill="both", expand=True, padx=26)

    def draw(self):
        c = self.canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 400), max(c.winfo_height(), 300)
        cx, cy = w / 2, h / 2
        c.create_polygon(cx-140, 35, cx+140, 35, cx+185, 80, cx+185, h-65,
                         cx+145, h-25, cx-145, h-25, cx-185, h-65, cx-185, 80,
                         fill="#182a40", outline="#385671", width=2)
        c.create_line(cx, 65, cx, h-55, fill=BORDER, dash=(5, 7))
        c.create_polygon(cx, 8, cx-10, 26, cx+10, 26, fill=CYAN, outline="")
        c.create_text(cx, 48, text="船头", fill=MUTED, font=("Microsoft YaHei UI", 10))
        coords = ((cx-130, 93), (cx+130, 93), (cx-130, h-85), (cx+130, h-85),
                  (cx-74, cy-33), (cx+74, cy-33), (cx-74, cy+63), (cx+74, cy+63))
        for i, (x, y) in enumerate(coords, 1):
            tag = f"motor{i}"
            selected = i == self.session.selected
            fill = "#16647c" if selected else "#233950"
            outline = CYAN if selected else "#44617e"
            c.create_oval(x-31, y-31, x+31, y+31, fill=fill, outline=outline,
                          width=3 if selected else 1, tags=tag)
            c.create_text(x, y, text=str(i), fill=TEXT, font=("Microsoft YaHei UI", 20, "bold"), tags=tag)
            c.create_text(x, y+43, text="斜推" if i <= 4 else "垂推", fill=MUTED,
                          font=("Microsoft YaHei UI", 9), tags=tag)
            c.tag_bind(tag, "<Button-1>", lambda _e, n=i: self.select(n))
        c.create_text(cx, h-8, text="左舷                               右舷", fill=MUTED,
                      font=("Microsoft YaHei UI", 9))

    def select(self, number):
        self.release()
        self.session.selected = number
        self.draw()
        self.refresh(time.monotonic())

    def enable(self):
        if not self.confirm.get():
            messagebox.showwarning("确认测试条件", "请先确认QGC控制已关闭及电机测试条件。", parent=self.window)
            return
        self.session.prepare(time.monotonic())

    def press(self, direction):
        if self.session.direction or self.session.motion_action:
            self.release()
            self.refresh(time.monotonic())
            return "break"
        try:
            pwm = int(self.single_values[1 if direction > 0 else 0].get())
            if (direction > 0 and pwm < 1500) or (direction < 0 and pwm > 1500):
                raise ValueError()
        except ValueError:
            self.session.notice = "低侧填1000～1500整数，高侧填1500～2000整数"
            self.refresh(time.monotonic())
            return "break"
        self.held = self.session.start_single_pwm(pwm, time.monotonic())
        self.refresh(time.monotonic())
        return "break"

    def press_motion(self, action):
        if self.session.direction or self.session.motion_action:
            self.release()
        else:
            self.preview_action = action
            try:
                low, high = (int(var.get()) for var in self.motion_values[action])
            except ValueError:
                self.session.notice = "请输入整数PWM；编辑不会启动电机"
                self.refresh(time.monotonic())
                return
            self.held = self.session.hold_motion(action, low, high, time.monotonic())
        self.refresh(time.monotonic())

    def release(self):
        self.held = False
        self.session.release(time.monotonic())

    def focus_out(self, _event):
        # Moving focus between this dialog's widgets is not window focus loss.
        self.window.after_idle(self.check_focus)

    def check_focus(self):
        if not self.closed:
            focus = self.window.focus_displayof()
            if focus is None or focus.winfo_toplevel() != self.window:
                self.release()

    def stop(self):
        self.held = False
        self.session.stop(time.monotonic())

    def refresh(self, now):
        if self.closed:
            return
        s = self.session
        linked = s.modes.fresh(now)
        self.badge.set("● 飞控在线" if linked else "○ 等待飞控")
        n = s.selected
        self.selected_text.set(f"{n:02d}  {POSITIONS[n-1]}")
        self.pwm_text.set(f"MAIN {n}  →  Motor {PHYSICAL_TO_MOTOR[n-1]}  ·  零基测试索引 {PHYSICAL_TO_MOTOR[n-1]-1}")
        titles = {"monitor": "安全监视", "mode": "切换模式中", "arm": "等待解锁", "ready": "测试就绪", "stopping": "等待上锁"}
        active = bool(s.direction or s.motion_action)
        self.title.set(f"● {s.motion_action or f'{n}号'} 测试中" if active else titles[s.phase])
        self.subtitle.set("本面板不修改电机映射与正反转参数。\n正反向是PWM增减，不代表机身运动方向。")
        pwm = s.pwm.get(n) if now - s.output_time < 1 else None
        output = f"MAIN {n} 回传：{pwm} µs" if pwm is not None else f"MAIN {n} 回传：暂无新鲜PWM"
        mismatches = [f"{k}={s.params[k]:g}" for k,v in EXPECTED_PARAMS.items() if k in s.params and s.params[k] != v]
        count = sum(name in s.params for name in EXPECTED_PARAMS)
        mapping = "映射已核对通过" if s.mapping_ok else ("映射不符：" + ", ".join(mismatches) if mismatches else f"映射读取 {count}/9")
        direction_status = "方向回传已齐" if s.directions_fresh(now) else "等待8路方向回传"
        self.status.set(f"{s.modes.status(now)}\n{mapping} · {direction_status}\n" +
                        (f"冷却剩余 {max(0,s.cooldown-now):.0f} 秒" if now < s.cooldown else "点击启停、不限时 · 断流保护保留"))
        target = {i: 1500 for i in range(1, 9)}
        if s.motion_action:
            target = s.motion_plan
        elif s.direction:
            target[n] = s.single_pwm if s.single_pwm is not None else 1500+s.direction*s.level*5
        rows = []
        for i in range(1, 9, 2):
            rows.append("    ".join(f"{j}号 目标 {target[j]} / 回传 {s.pwm.get(j, '—') if now-s.output_time < 1 else '—'}"
                                    for j in (i, i+1)))
        self.targets_var.set("PWM µs · 目标 / 回传（非转速）\n" + "\n".join(rows))
        self.feedback.set(s.notice)
        self.text_var.set("飞控消息：" + (s.last_text[:140] or "暂无"))
        ready = s.phase == "ready" and linked and s.modes.armed
        self.forward.configure(state="normal" if ready else "disabled", text="■ 停止测试" if active else "高侧PWM · 启动")
        self.backward.configure(state="normal" if ready else "disabled", text="■ 停止测试" if active else "低侧PWM · 启动")
        for action, button in self.motion_buttons.items():
            button.configure(state="normal" if ready and (active or s.directions_fresh(now)) else "disabled",
                             text="■ 停止" if active else f"{action} · 启动")
        eligible = linked and s.mapping_ok and not s.modes.armed and s.phase == "monitor" and now >= s.cooldown
        self.prepare.configure(state="normal" if eligible else "disabled")

    def close(self):
        self.stop()
        # Stay visible until heartbeat proves disarm. No ambiguous silent close.
        if self.session.phase == "stopping":
            self.session.notice = "关闭前需确认上锁；已发送停止与上锁请求"
            return
        self.closed = True
        self.session.link.close()
        self.owner.motor_dialog = None
        self.window.destroy()
