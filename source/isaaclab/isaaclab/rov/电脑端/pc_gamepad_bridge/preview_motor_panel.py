"""Offline visual preview only: no UDP, HID, serial, or control commands."""
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import Mock

from flight_mode import FlightModeController
from motor_test import MotorTestSession
from motor_test_panel import MotorTestPanel


def main():
    root = tk.Tk()
    root.withdraw()
    owner = SimpleNamespace(root=root, motor_dialog=None)
    session = MotorTestSession(FlightModeController(), Mock())
    panel = MotorTestPanel(owner, session)
    owner.motor_dialog = panel
    panel.window.title("ROV · 推进器逐个测试 · 离线预览（不连接硬件）")
    panel.window.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
