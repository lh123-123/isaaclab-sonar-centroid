#!/usr/bin/env python3
"""Fan out MJPEG and expose camera-only quality presets."""
import argparse
import json
import logging
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = logging.getLogger("usb_mjpeg")
BOUNDARY = b"rovframe"

# Keep these presets deliberately limited to UVC image controls.  Neither
# profile can arm the ROV or write to the motion-control service.
_COMMON_IMAGE_CONTROLS = (
    ("white_balance_temperature_auto", 1),
    ("exposure_auto", 3),
    ("exposure_auto_priority", 0),
    ("brightness", 5),
    ("contrast", 54),
    ("saturation", 64),
    ("hue", 0),
    ("gamma", 250),
    ("sharpness", 98),
    ("backlight_compensation", 1),
)
CAMERA_PROFILES = {
    # Measured in the pool on 2026-08-21.  Manual focus avoids underwater
    # hunting on suspended particles and preserves the calibrated value.
    "water": (
        ("focus_auto", 0),
        ("focus_absolute", 256),
    ) + _COMMON_IMAGE_CONTROLS,
    # Re-measured on shore at 1080p on 2026-08-22 with the gripper/lower centre
    # carrying 65% of the clarity score.  Coarse-to-fine scans converged on the
    # 218-224 plateau; the final 11-sample validation selected 222.  Manual
    # focus makes the result repeatable and avoids focus hunting.
    "shore": (
        ("focus_auto", 0),
        ("focus_absolute", 222),
    ) + _COMMON_IMAGE_CONTROLS,
    # Restore continuous UVC autofocus while retaining the proven common image
    # controls.  This is useful when target distance changes too much for one
    # calibrated manual-focus value.
    "autofocus": (
        ("focus_auto", 1),
    ) + _COMMON_IMAGE_CONTROLS,
}


def apply_camera_profile(device, profile, runner=subprocess.run, pause=time.sleep):
    """Apply one named UVC profile with one control per ioctl."""
    if profile not in CAMERA_PROFILES:
        raise ValueError("unknown camera profile")
    for control, value in CAMERA_PROFILES[profile]:
        runner(
            ["v4l2-ctl", "-d", device, "-c", "%s=%s" % (control, value)],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=3,
        )
        # This UVC driver rejects absolute focus immediately after switching
        # out of autofocus, so preserve the proven inter-control delay.
        if control == "focus_auto" and value == 0:
            pause(0.2)


class Frames:
    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0

    def put(self, frame):
        with self.condition:
            self.frame, self.sequence = frame, self.sequence + 1
            self.condition.notify_all()

    def next_after(self, sequence):
        with self.condition:
            self.condition.wait_for(lambda: self.sequence != sequence, timeout=10)
            return self.frame, self.sequence


def build_capture_command(args):
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-f", "v4l2",
        "-input_format", "mjpeg", "-video_size", args.video_size,
        "-framerate", str(args.framerate), "-i", args.device, "-an",
    ]
    if args.output_framerate and args.output_framerate != args.framerate:
        command.extend(["-vf", "fps=%s" % args.output_framerate])
    command.extend([
        # Some UVC cameras emit JPEG frames with a missing EOI marker.  Decode
        # and re-encode here so clients always receive complete JPEG images.
        "-c:v", "mjpeg", "-q:v", str(args.jpeg_quality),
        "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    ])
    return command


def capture(frames, args, stop):
    command = build_capture_command(args)
    while not stop.is_set():
        process = None
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, bufsize=0)
            buffer = bytearray()
            while not stop.is_set():
                chunk = process.stdout.read(8192)
                if not chunk:
                    break
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                    if start < 0:
                        if len(buffer) > 4 * 1024 * 1024:
                            buffer.clear()
                        break
                    if end < 0:
                        if start:
                            del buffer[:start]
                        break
                    frames.put(bytes(buffer[start:end + 2]))
                    del buffer[:end + 2]
        except FileNotFoundError:
            LOG.error("ffmpeg is not installed")
            stop.set()
        except Exception:
            LOG.exception("camera capture failed")
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
        if not stop.is_set():
            LOG.warning("ffmpeg exited; retrying in 1 second")
            stop.wait(1)


def handler_factory(
    frames,
    allowed_client,
    device="/dev/video0",
    profile_state=None,
    profile_lock=None,
    profile_applier=apply_camera_profile,
):
    profile_state = profile_state if profile_state is not None else {"current": "water"}
    profile_lock = profile_lock if profile_lock is not None else threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _client_allowed(self):
            if self.client_address[0] == allowed_client:
                return True
            self.send_error(403)
            return False

        def _send_json(self, status, payload):
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._client_allowed():
                return
            if self.path == "/camera/profile":
                with profile_lock:
                    current = profile_state.get("current", "water")
                self._send_json(200, {
                    "ok": True,
                    "profile": current,
                    "profiles": sorted(CAMERA_PROFILES),
                })
                return
            if self.path != "/stream.mjpg":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Cache-Control", "no-store")
            # FFmpeg's mpjpeg demuxer expects the boundary token immediately
            # after ``boundary=`` and every following delimiter on a fresh
            # CRLF-delimited line.
            self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=rovframe")
            self.end_headers()
            sequence = -1
            first_part = True
            try:
                while True:
                    frame, sequence = frames.next_after(sequence)
                    if frame is None:
                        continue
                    prefix = b"" if first_part else b"\r\n"
                    first_part = False
                    self.wfile.write(prefix + b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(("Content-Length: %d\r\n\r\n" % len(frame)).encode())
                    self.wfile.write(frame)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_POST(self):
            if not self._client_allowed():
                return
            if self.path != "/camera/profile":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"ok": False, "error": "invalid length"})
                return
            if length <= 0 or length > 1024:
                self._send_json(400, {"ok": False, "error": "invalid body"})
                return
            profile = None
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                profile = payload.get("profile") if isinstance(payload, dict) else None
                if profile not in CAMERA_PROFILES:
                    raise ValueError("unknown camera profile")
                with profile_lock:
                    profile_applier(device, profile)
                    profile_state["current"] = profile
            except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return
            except (OSError, subprocess.SubprocessError) as exc:
                LOG.warning("unable to apply camera profile %r: %s", profile, exc)
                self._send_json(503, {"ok": False, "error": "camera control unavailable"})
                return
            LOG.info("camera profile applied: %s", profile)
            self._send_json(200, {"ok": True, "profile": profile})

        def log_message(self, fmt, *values):
            LOG.info("%s - %s", self.client_address[0], fmt % values)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--bind-host", default="192.168.1.3")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--allow-client", default="192.168.1.4")
    parser.add_argument("--video-size", default="640x480")
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--output-framerate", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be 1..65535")
    if not 1 <= args.framerate <= 120:
        parser.error("--framerate must be 1..120")
    if not 0 <= args.output_framerate <= args.framerate:
        parser.error("--output-framerate must be 0..framerate")
    if not 1 <= args.jpeg_quality <= 31:
        parser.error("--jpeg-quality must be 1..31")
    stop, frames = threading.Event(), Frames()
    server = ThreadingHTTPServer(
        (args.bind_host, args.port),
        handler_factory(frames, args.allow_client, device=args.device),
    )
    server.daemon_threads = True
    thread = threading.Thread(target=capture, args=(frames, args, stop), daemon=True)
    thread.start()
    def shutdown(_signum, _frame):
        stop.set()
        # shutdown() must run outside serve_forever's own thread; calling it
        # directly from this signal handler deadlocks on Python 3.7.
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    LOG.warning("serving http://%s:%s/stream.mjpg", args.bind_host, args.port)
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()
        thread.join(timeout=3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
