"""Identify the isolated, disarmed FC using pinned official uploader; never erase.

Requires a completed read-only parameter backup. Reboots to the bootloader,
records board capacity/serial, then boots the unchanged application again.
"""
import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path

from pymavlink import mavutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite identity report")
    backup = json.loads(args.backup.read_text())
    if len(backup["parameters"]) != backup["parameter_count"]:
        raise RuntimeError("Incomplete backup")
    port = str(Path(args.port).resolve(strict=True))
    owner = subprocess.run(["fuser", port], capture_output=True, text=True, timeout=5)
    if owner.returncode != 1 or owner.stdout.strip():
        raise RuntimeError("Serial port is not free")
    link = mavutil.mavlink_connection(port, baud=115200, source_system=254, source_component=191)
    try:
        link.port.exclusive = True
        hb = link.wait_heartbeat(timeout=10)
        if hb is None or hb.get_srcSystem() != 1 or hb.get_srcComponent() != 1 or hb.type != 12:
            raise RuntimeError("Expected ArduSub heartbeat missing")
        if hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            raise RuntimeError("Vehicle armed; abort")
        link.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
                                   0, 3, 0, 0, 0, 0, 0, 0)
    finally:
        link.close()
    spec = importlib.util.spec_from_file_location("pinned_uploader", args.source / "Tools/scripts/uploader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    deadline = time.monotonic() + 30
    up = None
    while time.monotonic() < deadline:
        try:
            up = module.uploader(port, 115200, [115200], identify_only=True)
            up.identify()
            break
        except Exception as exc:
            if up is not None:
                up.close()
            up = None
            print(f"Waiting for bootloader: {exc}", flush=True)
            time.sleep(0.5)
    if up is None:
        raise RuntimeError("No bootloader response; no erase or upload attempted")
    try:
        up.dump_board_info()
        serial = b"".join(up._uploader__getSN(i) for i in range(0, 12, 4)).hex()
        expected = bytes(backup["autopilot_version"]["uid2"][:12]).hex()
        if serial != expected:
            raise RuntimeError(f"Device serial mismatch: {serial} != {expected}")
        report = {"board_id": up.board_type, "board_revision": up.board_rev,
                  "bootloader_revision": up.bl_rev, "flash_size": up.fw_maxsize,
                  "serial": serial, "bootloader_git": getattr(up, "git_hash_bl", None)}
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        try:
            up._uploader__reboot()
        finally:
            up.close()


if __name__ == "__main__":
    main()
