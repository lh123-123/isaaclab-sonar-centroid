#!/usr/bin/env python3
"""Capture and validate depth-sensor frames without starting ROV control."""

import argparse
import math
import statistics
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Read-only raw depth sensor capture.")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=19200)
    parser.add_argument("--address", type=int, default=1)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    if not 1 <= args.address <= 99:
        parser.error("--address must be 1..99")
    if not 1 <= args.samples <= 240:
        parser.error("--samples must be 1..240")
    if not 0.1 <= args.interval <= 10.0:
        parser.error("--interval must be 0.1..10 seconds")
    return args


def parse_frame(frame, address):
    """Return raw pressure and temperature, or a concise validation error."""
    if len(frame) != 22:
        return None, f"length={len(frame)}"
    if frame[:2] != b"sD":
        return None, f"prefix={frame[:2]!r}"
    try:
        received_address = int(frame[2:4].decode("ascii"))
        pressure_digits = frame[5:13].decode("ascii")
        temperature_digits = frame[14:19].decode("ascii")
        checksum_digits = frame[19:22].decode("ascii")
        if not pressure_digits.isdigit() or not temperature_digits.isdigit():
            return None, "non-numeric field"
        if not checksum_digits.isdigit():
            return None, "non-numeric checksum"
        if frame[4] not in (ord("0"), ord("1")) or frame[13] not in (ord("0"), ord("1")):
            return None, "invalid sign"
    except (UnicodeDecodeError, ValueError):
        return None, "non-ascii field"
    if received_address != address:
        return None, f"address={received_address}"
    expected_checksum = sum(byte - ord("0") for byte in frame[2:19])
    if int(checksum_digits) != expected_checksum:
        return None, f"checksum={checksum_digits}, expected={expected_checksum:03d}"

    pressure = float(pressure_digits) / 10000.0
    temperature = float(temperature_digits) / 1000.0
    if frame[4] == ord("1"):
        pressure = -pressure
    if frame[13] == ord("1"):
        temperature = -temperature
    return (pressure, temperature), None


def trend_per_second(times, values):
    if len(values) < 2:
        return math.nan
    # Raspberry Pi OS's bundled Python may not provide statistics.fmean.
    time_mean = sum(times) / len(times)
    value_mean = sum(values) / len(values)
    numerator = sum((t - time_mean) * (v - value_mean) for t, v in zip(times, values))
    denominator = sum((t - time_mean) ** 2 for t in times)
    return numerator / denominator if denominator else math.nan


def read_response(device, timeout=0.5, quiet_time=0.03):
    """Read every byte of one response, stopping only after serial quiet."""
    deadline = time.monotonic() + timeout
    last_data_at = None
    response = bytearray()
    while time.monotonic() < deadline:
        available = device.in_waiting
        if available:
            response.extend(device.read(available))
            last_data_at = time.monotonic()
            continue
        if response and last_data_at is not None and time.monotonic() - last_data_at >= quiet_time:
            break
        time.sleep(0.002)
    return bytes(response)


def main():
    args = parse_args()
    print(
        "READ-ONLY raw capture: "
        f"port={args.port}, baud={args.baud}, address={args.address}, "
        f"samples={args.samples}"
    )
    print("No IMU, controller, MPC, PCA9685, GPIO, or PWM is started.")

    try:
        import serial

        device = serial.Serial(args.port, args.baud, timeout=0.35)
    except Exception as exc:
        print(f"OPEN FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    values, temperatures, times = [], [], []
    invalid = 0
    try:
        for index in range(1, args.samples + 1):
            device.reset_input_buffer()
            device.write(f"sD{args.address:02d}\r\n".encode("ascii"))
            device.flush()
            frame = read_response(device)
            parsed, error = parse_frame(frame, args.address)
            elapsed = time.monotonic()
            if error:
                invalid += 1
                print(f"{index:03d}/{args.samples}: INVALID {error}; frame={frame.hex()}")
            else:
                pressure, temperature = parsed
                values.append(pressure)
                temperatures.append(temperature)
                times.append(elapsed)
                print(
                    f"{index:03d}/{args.samples}: raw={pressure:+.6f} "
                    f"temp={temperature:+.3f} C frame={frame.decode('ascii', errors='replace')}"
                )
            if index < args.samples:
                time.sleep(args.interval)
    finally:
        device.close()

    if not values:
        print(f"RESULT: FAIL; valid=0 invalid={invalid}")
        return 1

    print(
        "RESULT: "
        f"valid={len(values)}/{args.samples}; invalid={invalid}; "
        f"raw_median={statistics.median(values):+.6f}; "
        f"raw_stddev={statistics.pstdev(values):.6f}; "
        f"raw_span={max(values) - min(values):.6f}; "
        f"raw_trend={trend_per_second(times, values):+.6f}/s; "
        f"temp_span={max(temperatures) - min(temperatures):.3f} C"
    )
    return 0 if invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
