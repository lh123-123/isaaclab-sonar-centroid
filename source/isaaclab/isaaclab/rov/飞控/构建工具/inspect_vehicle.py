"""Read-only serial inspection/full parameter backup before or after flashing.

The bridge must be stopped and actuator power disconnected by the operator.
No mode, arm, parameter-set, reboot or motor commands are sent.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

from pymavlink import mavutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()
    outputs = (args.output.with_suffix(".json"), args.output.with_suffix(".params"))
    if any(path.exists() for path in outputs):
        raise RuntimeError("Backup destination exists; refusing to overwrite")
    port = str(Path(args.port).resolve(strict=True))
    owner = subprocess.run(["fuser", port], capture_output=True, text=True, timeout=5)
    if owner.returncode == 0 or owner.stdout.strip():
        raise RuntimeError("Serial port is owned by another process: " + owner.stdout.strip())
    if owner.returncode != 1:
        raise RuntimeError("Unable to establish exclusive serial ownership")
    link = mavutil.mavlink_connection(port, baud=args.baud, source_system=254, source_component=191)
    try:
        link.port.exclusive = True
        hb = link.wait_heartbeat(timeout=10)
        if hb is None or hb.get_srcSystem() != 1 or hb.get_srcComponent() != 1:
            raise RuntimeError("No expected autopilot heartbeat")
        if hb.type != mavutil.mavlink.MAV_TYPE_SUBMARINE:
            raise RuntimeError("Connected vehicle is not ArduSub")
        if hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            raise RuntimeError("Vehicle is armed; inspection aborted")
        print("CONFIRMED disarmed ArduSub, sysid=1 compid=1", flush=True)
        link.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        link.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
                                   mavutil.mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION, 0, 0, 0, 0, 0, 0)
        version = None
        params = {}
        indices = {}
        expected_count = None
        deadline = time.monotonic() + 150
        next_gap = time.monotonic() + 15
        last_progress = 0
        next_heartbeat = time.monotonic() + 1
        link.mav.param_request_list_send(1, 1)
        while time.monotonic() < deadline:
            msg = link.recv_match(blocking=True, timeout=0.2)
            if msg is not None and msg.get_srcSystem() == 1 and msg.get_srcComponent() == 1:
                kind = msg.get_type()
                if kind == "HEARTBEAT" and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                    raise RuntimeError("Vehicle armed during inspection; aborted")
                if kind == "AUTOPILOT_VERSION":
                    version = msg.to_dict()
                if kind == "PARAM_VALUE":
                    count = int(msg.param_count)
                    index = int(msg.param_index)
                    if expected_count is not None and count != expected_count:
                        raise RuntimeError("Parameter count changed during backup")
                    if not 0 <= index < count:
                        continue
                    expected_count = count
                    name = msg.param_id.decode("ascii") if isinstance(msg.param_id, bytes) else msg.param_id
                    name = name.rstrip("\x00")
                    params[name] = {"value": float(msg.param_value), "type": int(msg.param_type), "index": index}
                    indices[index] = name
            now = time.monotonic()
            if now >= next_heartbeat:
                link.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                next_heartbeat = now + 1
            if now - last_progress > 5:
                print(f"Parameters: {len(indices)}/{expected_count or '?'}", flush=True)
                last_progress = now
            if expected_count and len(indices) == expected_count and version is not None:
                break
            if expected_count and now >= next_gap:
                missing = next((i for i in range(expected_count) if i not in indices), None)
                if missing is not None:
                    link.mav.param_request_read_send(1, 1, b"", missing)
                next_gap = now + 0.1
        if not expected_count or len(indices) != expected_count or len(params) != expected_count or version is None:
            raise RuntimeError("Incomplete backup or missing firmware identity; no files written")
        stamp = {"heartbeat": hb.to_dict(), "autopilot_version": version,
                 "parameter_count": expected_count, "parameters": params}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with outputs[0].open("x", encoding="utf-8") as stream:
            json.dump(stamp, stream, indent=2)
        with outputs[1].open("x", encoding="utf-8") as stream:
            stream.write("# Full read-back backup: sysid 1, compid 1; no parameter writes\n")
            for name, item in sorted(params.items(), key=lambda pair: pair[1]["index"]):
                stream.write(f"1\t1\t{name}\t{item['value']:.9g}\t{item['type']}\n")
        print(json.dumps({"version": version, "parameter_count": expected_count,
                          "outputs": list(map(str, outputs))}, indent=2), flush=True)
    finally:
        link.close()


if __name__ == "__main__":
    main()
