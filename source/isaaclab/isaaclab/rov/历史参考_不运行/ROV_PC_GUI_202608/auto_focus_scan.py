#!/usr/bin/env python3
"""Measure the sharpest manual UVC focus value from the live MJPEG feed."""

import argparse
import json
import statistics
import subprocess
import time
from urllib.request import ProxyHandler, build_opener

import cv2
import numpy as np


def read_mjpeg_frame(opener, stream_url, timeout=5):
    """Read one complete JPEG from an MJPEG response, bypassing proxies."""
    with opener.open(stream_url, timeout=timeout) as response:
        buffer = bytearray()
        while len(buffer) < 4 * 1024 * 1024:
            chunk = response.read(65536)
            if not chunk:
                break
            buffer.extend(chunk)
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
            if start >= 0 and end >= 0:
                encoded = np.frombuffer(buffer[start:end + 2], dtype=np.uint8)
                frame = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    raise RuntimeError("camera returned an undecodable JPEG")
                return frame
    raise RuntimeError("camera stream did not return a complete JPEG")


def clarity_score(frame, roi="center"):
    """Return robust edge energy from the requested task-relevant image area."""
    height, width = frame.shape
    if roi == "far":
        # Ignore the ROV frame in the lower/central foreground when calibrating
        # for navigation targets farther from the camera.
        y0, y1 = height // 20, height * 11 // 20
        x0, x1 = width // 10, width - width // 10
        crops = ((frame[y0:y1, x0:x1], 1.0),)
    elif roi == "center":
        y0, y1 = height // 10, height - height // 10
        x0, x1 = width // 10, width - width // 10
        crops = ((frame[y0:y1, x0:x1], 1.0),)
    elif roi == "claw_center":
        # The gripper/vehicle frame occupies the lower central portion of the
        # live view.  Give it the highest weight while retaining enough of the
        # geometric centre to avoid focusing only on a single foreground edge.
        center = frame[
            height // 10:height * 9 // 10,
            width * 15 // 100:width * 85 // 100,
        ]
        claw = frame[
            height * 42 // 100:height * 98 // 100,
            width * 28 // 100:width * 72 // 100,
        ]
        crops = ((center, 0.35), (claw, 0.65))
    else:
        raise ValueError("unknown clarity ROI")

    score = 0.0
    for crop, weight in crops:
        # A slight blur suppresses JPEG blocking; Laplacian variance then
        # tracks optical focus instead of compression noise.
        crop = cv2.GaussianBlur(crop, (3, 3), 0)
        score += weight * float(cv2.Laplacian(crop, cv2.CV_64F).var())
    return score


class FocusController:
    def __init__(self, host, bind_address, device):
        self.host = host
        self.bind_address = bind_address
        self.device = device

    def _run(self, control):
        command = [
            "ssh", "-b", self.bind_address,
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            self.host,
            "v4l2-ctl -d %s -c %s" % (self.device, control),
        ]
        subprocess.run(command, check=True, timeout=8)

    def use_manual_focus(self):
        self._run("focus_auto=0")
        time.sleep(0.25)

    def set_focus(self, value):
        self._run("focus_absolute=%d" % value)


def measure_focus(controller, opener, stream_url, value, samples, settle, roi):
    controller.set_focus(value)
    time.sleep(settle)
    scores = [
        clarity_score(read_mjpeg_frame(opener, stream_url), roi=roi)
        for _ in range(samples)
    ]
    return statistics.median(scores), scores


def scan_phase(controller, opener, stream_url, candidates, samples, settle, results, roi):
    normalized = sorted(set(max(0, min(1023, item)) for item in candidates))
    for value in normalized:
        score, samples_raw = measure_focus(
            controller, opener, stream_url, value,
            samples=samples, settle=settle, roi=roi,
        )
        results[value] = {"score": score, "samples": samples_raw}
        print("focus=%4d  clarity=%10.3f" % (value, score), flush=True)
    return max(normalized, key=lambda item: results[item]["score"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default="http://192.168.1.3:8080/stream.mjpg")
    parser.add_argument("--ssh-host", default="pi@192.168.1.3")
    parser.add_argument("--bind-address", default="192.168.1.4")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--roi", choices=("claw_center", "center", "far"),
        default="claw_center",
        help="image region used for the clarity score",
    )
    parser.add_argument(
        "--values",
        help="comma-separated focus values for a validation-only scan",
    )
    parser.add_argument("--output", default="camera_focus_scan_result.json")
    args = parser.parse_args()

    opener = build_opener(ProxyHandler({}))
    controller = FocusController(args.ssh_host, args.bind_address, args.device)
    controller.use_manual_focus()
    results = {}

    if args.values:
        candidates = [int(value.strip()) for value in args.values.split(",")]
        print("validation scan", flush=True)
        best = scan_phase(
            controller, opener, args.stream, candidates, args.samples, 0.40,
            results, args.roi,
        )
    else:
        print("[1/3] coarse scan", flush=True)
        coarse = list(range(0, 1024, 64)) + [1023]
        best = scan_phase(
            controller, opener, args.stream, coarse, args.samples, 0.45,
            results, args.roi,
        )

        print("[2/3] medium scan around %d" % best, flush=True)
        medium = range(best - 64, best + 65, 16)
        best = scan_phase(
            controller, opener, args.stream, medium, args.samples, 0.35,
            results, args.roi,
        )

        print("[3/3] fine scan around %d" % best, flush=True)
        fine = range(best - 16, best + 17, 4)
        best = scan_phase(
            controller, opener, args.stream, fine, args.samples, 0.30,
            results, args.roi,
        )

    controller.set_focus(best)
    ranked = sorted(results.items(), key=lambda item: item[1]["score"], reverse=True)
    report = {
        "roi": args.roi,
        "best_focus": best,
        "best_score": results[best]["score"],
        "top_five": [
            {"focus": value, "score": detail["score"]}
            for value, detail in ranked[:5]
        ],
        "measurements": {str(value): detail for value, detail in sorted(results.items())},
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print("BEST_FOCUS=%d BEST_SCORE=%.3f" % (best, results[best]["score"]), flush=True)


if __name__ == "__main__":
    main()
