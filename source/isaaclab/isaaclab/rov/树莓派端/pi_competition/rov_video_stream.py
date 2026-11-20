#!/usr/bin/env python3
"""USB MJPEG to QGC RTP/H264. Default is dry-run; --run explicitly starts video."""

import argparse
import ipaddress
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from urllib.parse import urlsplit

LOG = logging.getLogger("rov-video")
BASE = Path(__file__).resolve().parent


def rotation_degrees(config):
    angle = config.get("rotation_degrees", 0)
    if type(angle) is not int or angle not in (0, 180):
        raise ValueError("rotation_degrees must be integer 0 or 180")
    return angle


def enhancement_mode(config):
    mode = config.get("enhancement_mode", "off")
    if mode not in ("off", "mild"):
        raise ValueError("enhancement_mode must be off or mild")
    return mode


def load_profile(path=BASE / "video_profile.json"):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["source"] not in ("v4l2", "legacy-mjpeg"):
        raise ValueError("source must be v4l2 or legacy-mjpeg")
    ipaddress.IPv4Address(config["pc_ip"])
    if not 1024 <= int(config["port"]) <= 65534:
        raise ValueError("invalid RTP port (also needs port+1 for RTCP)")
    for name in ("width", "height"):
        if not 16 <= int(config[name]) <= 4096 or int(config[name]) % 2:
            raise ValueError("video dimensions must be even and 16..4096")
    if not 1 <= int(config["fps"]) <= 60 or not 100 <= int(config["bitrate_kbps"]) <= 20000:
        raise ValueError("invalid fps/bitrate")
    if not 1 <= int(config["encoder_threads"]) <= 4:
        raise ValueError("encoder_threads must be 1..4")
    if config["focus_profile"] not in ("water", "auto"):
        raise ValueError("focus_profile must be water or auto")
    if not 0 <= int(config["water_focus_value"]) <= 65535:
        raise ValueError("water_focus_value must be 0..65535; check the camera's supported range")
    if not config["device"].startswith("/dev/"):
        raise ValueError("device must be an explicit /dev path")
    if config["source"] == "legacy-mjpeg":
        url = urlsplit(config["legacy_url"])
        if url.scheme != "http" or not url.hostname or url.username or url.password:
            raise ValueError("legacy source must be a camera HTTP URL without credentials")
    rotation_degrees(config)
    enhancement_mode(config)
    return config


def build_command(config, ffmpeg="ffmpeg"):
    fps = int(config["fps"])
    size = f'{config["width"]}x{config["height"]}'
    bitrate = int(config["bitrate_kbps"])
    command = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "info"]
    if config["source"] == "v4l2":
        command += ["-f", "v4l2", "-input_format", "mjpeg", "-video_size", size,
                    "-framerate", str(fps), "-i", config["device"]]
    else:
        # HTTP fallback consumes the old fan-out server; never opens the camera twice.
        command += ["-rw_timeout", "3000000", "-fflags", "nobuffer", "-analyzeduration", "0",
                    "-probesize", "65536", "-i", config["legacy_url"]]
    # Both flips together are a 180-degree rotation, not a one-axis mirror.
    # Do this before encoding; flight-controller/body axes are unaffected.
    filters = ["hflip", "vflip"] if rotation_degrees(config) == 180 else []
    filters += [f"scale={config['width']}:{config['height']}", f"fps={fps}"]
    if enhancement_mode(config) == "mild":
        # Gentle contrast/saturation adjustment. No geometric correction can
        # be inferred from an uncalibrated acrylic port; avoid fake deblurring.
        filters.append("eq=contrast=1.04:saturation=1.04")
    command += ["-an", "-vf", ",".join(filters),
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                "-threads", str(config["encoder_threads"]), "-pix_fmt", "yuv420p",
                "-bf", "0", "-g", str(fps), "-keyint_min", str(fps),
                "-x264-params", "repeat-headers=1:scenecut=0",
                "-b:v", f"{bitrate}k", "-maxrate", f"{bitrate}k",
                "-bufsize", f"{max(100, bitrate // 5)}k",
                "-flush_packets", "1", "-f", "rtp", "-payload_type", "96",
                f"rtp://{config['pc_ip']}:{config['port']}?pkt_size=1200"]
    return command


def camera_controls(config, runner=subprocess.run):
    result = runner(["v4l2-ctl", "-d", config["device"], "--list-ctrls"],
                    capture_output=True, text=True, timeout=3)
    if result.returncode:
        raise RuntimeError("Cannot query camera controls: " + result.stderr.strip())
    controls = {}
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(\w+)\s+0x[0-9a-fA-F]+\s+\([^)]+\)\s*:\s*(.*)", line)
        if match:
            controls[match[1]] = {key: int(value) for key, value in
                                 re.findall(r"\b(min|max|step|value)=(-?\d+)", match[2])}
            flags = re.search(r"\bflags=([\w,]+)", match[2])
            controls[match[1]]["flags"] = set(flags.group(1).split(",")) if flags else set()
    return controls


def focus_status(config, runner=subprocess.run):
    if config["source"] != "v4l2":
        raise RuntimeError("Focus control requires the direct V4L2 camera source")
    controls = camera_controls(config, runner)
    auto_name = next((name for name in ("focus_automatic_continuous", "focus_auto")
                      if name in controls), None)
    manual = controls.get("focus_absolute", {})
    if auto_name is None or "value" not in controls[auto_name] or not all(
            key in manual for key in ("min", "max", "step", "value")) or manual["step"] <= 0:
        raise RuntimeError("Camera does not expose readable auto/manual focus controls")
    autofocus = bool(controls[auto_name]["value"])
    # In automatic mode, many UVC drivers report the cached last manual
    # focus_absolute value. It is not the lens position. A driver must expose
    # a volatile, active control before we claim a live autofocus readback.
    focus_readable = "inactive" not in manual["flags"] and (
        not autofocus or "volatile" in manual["flags"])
    return {"configured_profile": config["focus_profile"], "water_focus_value": config["water_focus_value"], "auto_control": auto_name,
            "autofocus": autofocus, "focus_readable": focus_readable,
            "focus_value": manual["value"] if focus_readable else None,
            "minimum": manual["min"], "maximum": manual["max"], "step": manual["step"]}


def apply_focus(config, runner=subprocess.run, pause=time.sleep):
    if config["source"] != "v4l2":
        return
    if not shutil.which("v4l2-ctl"):
        raise RuntimeError("v4l2-ctl missing; configured focus cannot be verified")
    before = focus_status(config, runner)
    value = int(config["water_focus_value"])
    if config["focus_profile"] == "water" and (not before["minimum"] <= value <= before["maximum"] or
            (value - before["minimum"]) % before["step"]):
        raise ValueError("Water focus value is outside the camera's range/step")
    auto = before["auto_control"]
    if config["focus_profile"] == "auto":
        controls = (f"{auto}=1",)
    else:
        # Historical camera rejects combining auto/manual focus in one ioctl.
        controls = (f"{auto}=0", f"focus_absolute={value}")
    for control in controls:
        result = runner(["v4l2-ctl", "-d", config["device"], "-c", control],
                        capture_output=True, text=True, timeout=3)
        if result.returncode:
            raise RuntimeError(f"Camera rejected {control}: {result.stderr.strip()}")
        if control == f"{auto}=0":
            pause(0.2)
    after = focus_status(config, runner)
    if after["autofocus"] != (config["focus_profile"] == "auto") or (
            config["focus_profile"] == "water" and after["focus_value"] != value):
        raise RuntimeError("Camera focus independent readback does not match request")
    return after


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_encoder(command, config, focus, status_path, environment=None):
    # Keep one bounded status file, instead of an unbounded FFmpeg progress log.
    process = subprocess.Popen(command[:1] + ["-nostats", "-stats_period", "1", "-progress", "pipe:1"] + command[1:],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env=environment)
    progress = {}
    def stop(_signal, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    report = {"state": "starting", "pid": process.pid, "supervisor_pid": os.getpid(), "updated_at": time.time(),
              "width": config["width"], "height": config["height"], "target_fps": config["fps"],
              "pc_ip": config["pc_ip"], "port": config["port"], "focus_at_start": focus,
              "rotation_degrees": rotation_degrees(config)}
    atomic_json(status_path, report)
    try:
        for line in process.stdout:
            line = line.strip()
            key, _, value = line.partition("=")
            if key in ("frame", "fps", "out_time_us", "total_size", "bitrate", "speed", "progress"):
                progress[key] = value
                if key == "progress":
                    report.update(state="streaming" if value == "continue" else "ending",
                                  updated_at=time.time(), encoder_progress=progress.copy())
                    atomic_json(status_path, report)
            elif line:
                LOG.info("ffmpeg: %s", line)
        if process.wait() != 0:
            raise RuntimeError(f"FFmpeg exited with status {process.returncode}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        report.update(state="stopped", updated_at=time.time(), exit_code=process.returncode)
        atomic_json(status_path, report)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=BASE / "video_profile.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check", action="store_true", help="read camera formats/controls, never change them")
    mode.add_argument("--focus-status", action="store_true", help="read actual focus without changing it")
    mode.add_argument("--set-focus", choices=("water", "auto"), help="change focus without restarting video")
    parser.add_argument("--water-focus", type=int)
    parser.add_argument("--save-focus", action="store_true", help="persist only after successful focus readback")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_profile(args.config)
    if args.save_focus and args.set_focus is None:
        parser.error("--save-focus requires --set-focus")
    if args.water_focus is not None and args.set_focus != "water":
        parser.error("--water-focus requires --set-focus water")
    if args.focus_status:
        print(json.dumps(focus_status(config), ensure_ascii=False))
        return 0
    if args.set_focus:
        config["focus_profile"] = args.set_focus
        if args.water_focus is not None:
            config["water_focus_value"] = args.water_focus
        actual = apply_focus(config)
        if actual is None:
            raise RuntimeError("Focus requires direct V4L2 capture, not legacy HTTP")
        if args.save_focus:
            atomic_json(args.config, config)
        print(json.dumps(dict(actual, saved=args.save_focus), ensure_ascii=False))
        return 0
    command = build_command(config)
    if args.check:
        print("ffmpeg:", shutil.which("ffmpeg"))
        if config["source"] == "v4l2":
            for option in ("--list-formats-ext", "--list-ctrls"):
                subprocess.run(["v4l2-ctl", "-d", config["device"], option], check=True, timeout=8)
        else:
            print("Legacy HTTP source selected; no device access:", config["legacy_url"])
        return 0
    if not args.run:
        print(json.dumps({"dry_run": True, "command": command, "focus": config["focus_profile"],
                          "enhancement_mode": enhancement_mode(config),
                          "fps_is_target_not_measurement": True}, ensure_ascii=False, indent=2))
        return 0
    binary = shutil.which("ffmpeg")
    if not binary:
        raise RuntimeError("ffmpeg missing; install ffmpeg and v4l-utils on the Pi")
    # Fail explicitly when another capture process owns this device; never stop it here.
    if config["source"] == "v4l2" and shutil.which("fuser"):
        owners = subprocess.run(["fuser", config["device"]], capture_output=True, timeout=3)
        if owners.returncode == 0:
            raise RuntimeError("Camera device is already in use. Stop the old camera service or select legacy-mjpeg.")
    focus = apply_focus(config)
    environment = os.environ.copy()
    # Local camera traffic must not pass through a user's web proxy.
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        environment.pop(key, None)
    command[0] = binary
    LOG.info("Starting QGC RTP/H264: %sx%s @ %s target fps; focus preset=%s",
             config["width"], config["height"], config["fps"], config["focus_profile"])
    # Child environment excludes proxies but the parent's configuration stays intact.
    run_encoder(command, config, focus, args.config.with_name("video_status.json"), environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
