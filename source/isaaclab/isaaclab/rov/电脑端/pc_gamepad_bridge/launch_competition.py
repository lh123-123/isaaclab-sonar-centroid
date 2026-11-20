"""QGC plus status dock; only telemetry/parameter requests until control starts."""

import argparse
import importlib.util
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from runtime_support import SingleInstance

BASE = Path(__file__).resolve().parent


def load_settings(path=BASE / "launch_settings.json"):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if not Path(settings["qgc_path"]).is_file():
        raise ValueError("找不到QGroundControl，请修改launch_settings.json中的qgc_path")
    for name in ("control_port", "telemetry_port"):
        if not 1 <= int(settings[name]) <= 65535:
            raise ValueError(f"invalid {name}")
    if settings["control_port"] == settings["telemetry_port"]:
        raise ValueError("控制端口和本地遥测监听端口不能相同")
    if not 0 < float(settings["rate_hz"]) <= 100:
        raise ValueError("invalid rate_hz")
    for module in ("hid", "pymavlink", "tkinter"):
        if importlib.util.find_spec(module) is None:
            raise ValueError(f"缺少依赖 {module}，请安装requirements.txt")
    return settings


def panel_arguments(settings):
    # Deliberately ignore any unsupported auto-send/auto-arm fields in JSON.
    return SimpleNamespace(send=False, allow_arm=False, receive=True, compact=True,
                           profile=None, pi_ip=settings["pi_ip"],
                           port=int(settings["control_port"]),
                           telemetry_port=int(settings["telemetry_port"]),
                           target_system=int(settings["target_system"]),
                           rate=float(settings["rate_hz"]))


def qgc_running():
    if os.name != "nt":
        return False
    result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq QGroundControl.exe", "/FO", "CSV", "/NH"],
                            capture_output=True, timeout=5,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode != 0:
        raise RuntimeError("无法检查QGC进程；请先检查任务管理器")
    return b'"qgroundcontrol.exe"' in result.stdout.lower()


def run(settings):
    guard = SingleInstance()
    try:
        logs = BASE / "logs"
        logs.mkdir(exist_ok=True)
        handler = RotatingFileHandler(logs / "competition.log", maxBytes=2_000_000,
                                      backupCount=3, encoding="utf-8")
        logging.basicConfig(level=logging.INFO, handlers=[handler],
                            format="%(asctime)s %(levelname)s %(message)s", force=True)
        logging.info("Starting QGC dock without motion control; telemetry requests allowed")
        if not qgc_running():
            subprocess.Popen([settings["qgc_path"]], cwd=str(Path(settings["qgc_path"]).parent),
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        from rov_control_panel import run_panel
        return run_panel(panel_arguments(settings))
    finally:
        guard.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="offline preflight; no app, HID or UDP started")
    parser.add_argument("--settings", type=Path, default=BASE / "launch_settings.json")
    args = parser.parse_args()
    try:
        settings = load_settings(args.settings)
        if args.check:
            print(json.dumps({"ok": True, "qgc": settings["qgc_path"],
                              "startup": "monitor only", "hardware_verified": False}, ensure_ascii=False))
            return 0
        return run(settings)
    except Exception as exc:
        logging.exception("Competition startup failed")
        if args.check:
            print(str(exc), file=sys.stderr)
        else:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("比赛启动失败", str(exc))
            root.destroy()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
