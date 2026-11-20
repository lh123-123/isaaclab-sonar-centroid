#!/usr/bin/env python3
"""Read only the depth sensor serial protocol; never start ROV control/PWM."""

import argparse
import math
import statistics
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone read-only diagnostic for the ROV depth sensor."
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=19200)
    parser.add_argument("--address", type=int, default=1)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    if not 1 <= args.address <= 99:
        parser.error("--address must be 1..99")
    if not 1 <= args.samples <= 120:
        parser.error("--samples must be 1..120")
    if not 0.1 <= args.interval <= 10.0:
        parser.error("--interval must be 0.1..10 seconds")
    return args


def main():
    args = parse_args()
    print(
        "READ-ONLY depth diagnostic: "
        f"port={args.port}, baud={args.baud}, address={args.address}, "
        f"samples={args.samples}"
    )
    print("No ROV control process, PCA9685, or PWM output is started.")

    try:
        from imu_depth_combined import DepthSensor

        sensor = DepthSensor(
            port=args.port,
            baudrate=args.baud,
            address=args.address,
            timeout=0.3,
        )
    except Exception as exc:
        print(f"OPEN FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    pressures, temperatures, depths = [], [], []
    try:
        for index in range(1, args.samples + 1):
            result = sensor.query()
            if result is None:
                print(f"{index:02d}/{args.samples}: TIMEOUT or invalid frame")
            else:
                pressure = sensor.get_pressure()
                temperature = sensor.get_temperature()
                depth = sensor.get_depth()
                if all(math.isfinite(value) for value in (pressure, temperature, depth)):
                    pressures.append(pressure)
                    temperatures.append(temperature)
                    depths.append(depth)
                    print(
                        f"{index:02d}/{args.samples}: pressure_raw={pressure:+.6f} "
                        f"temperature={temperature:+.3f} C depth={depth:+.3f} m"
                    )
                else:
                    print(f"{index:02d}/{args.samples}: NON-FINITE sensor value")
            if index < args.samples:
                time.sleep(args.interval)
    finally:
        sensor.close()

    if not depths:
        print("RESULT: FAIL — no valid response. Check power, RS-485/USB adapter, port, baud, and address.")
        return 1

    valid = len(depths)
    print(
        "RESULT: "
        f"{valid}/{args.samples} valid; "
        f"pressure median={statistics.median(pressures):+.6f}; "
        f"temperature median={statistics.median(temperatures):+.3f} C; "
        f"depth median={statistics.median(depths):+.3f} m; "
        f"depth span={max(depths) - min(depths):.3f} m"
    )
    return 0 if valid == args.samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
