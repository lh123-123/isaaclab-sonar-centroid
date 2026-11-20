"""Collect stationary telemetry without calibration, parameter writes or motion."""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

from pymavlink import mavutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Evidence already exists')
    port = str(Path(args.port).resolve(strict=True))
    owner = subprocess.run(['fuser', port], capture_output=True, text=True, timeout=5)
    if owner.returncode != 1 or owner.stdout.strip():
        raise RuntimeError('Serial port occupied')
    link = mavutil.mavlink_connection(port, baud=115200, source_system=254, source_component=191)
    rows = []
    neutral = 0
    heartbeats = 0
    start = time.monotonic()
    last_hb = start
    next_request = start
    try:
        link.port.exclusive = True
        while time.monotonic() - start < 30:
            now = time.monotonic()
            if now >= next_request:
                link.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                       mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                for message_id in (1, 26, 30, 36, 116, 137):
                    link.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
                                              0, message_id, 0, 0, 0, 0, 0, 0)
                next_request = now + 1
            msg = link.recv_match(blocking=True, timeout=.1)
            if msg is not None and msg.get_srcSystem() == 1 and msg.get_srcComponent() == 1:
                kind = msg.get_type()
                if kind == 'HEARTBEAT':
                    if msg.type != 12 or msg.base_mode & 128 or msg.custom_mode != 19:
                        raise RuntimeError('Requires disarmed ArduSub Manual mode')
                    last_hb = now
                    heartbeats += 1
                elif kind == 'SERVO_OUTPUT_RAW' and msg.port == 0:
                    if [getattr(msg, f'servo{i}_raw') for i in range(1, 9)] != [1500]*8:
                        raise RuntimeError('Non-neutral motor outputs')
                    neutral += 1
                if kind in ('ATTITUDE', 'SCALED_IMU', 'SCALED_IMU2', 'SCALED_PRESSURE2', 'SYS_STATUS', 'STATUSTEXT'):
                    rows.append({'elapsed': now-start, **msg.to_dict()})
                    if kind == 'STATUSTEXT' and any(s in msg.text.lower() for s in ('deadlock', 'panic', 'watchdog', 'internal error')):
                        raise RuntimeError('Firmware fault: '+msg.text)
            if now-last_hb > 5:
                raise RuntimeError('Heartbeat interrupted')
        summary = {}
        fields = {'ATTITUDE': ('roll', 'pitch', 'yaw', 'rollspeed', 'pitchspeed', 'yawspeed'),
                  'SCALED_IMU': ('xgyro', 'ygyro', 'zgyro'),
                  'SCALED_IMU2': ('xgyro', 'ygyro', 'zgyro'),
                  'SCALED_PRESSURE2': ('press_abs', 'temperature')}
        for kind, names in fields.items():
            samples = [r for r in rows if r['mavpackettype'] == kind]
            if len(samples) < 15:
                raise RuntimeError('Insufficient samples: '+kind)
            summary[kind] = {'samples': len(samples)}
            for name in names:
                values = [float(r[name]) for r in samples]
                if not all(math.isfinite(v) for v in values):
                    raise RuntimeError('Nonfinite sensor reading')
                summary[kind][name] = {'min': min(values), 'max': max(values),
                                     'mean': sum(values)/len(values), 'first': values[0], 'last': values[-1]}
        if heartbeats < 20 or neutral < 20:
            raise RuntimeError('Incomplete safety evidence')
        report = {'read_only': True, 'duration': time.monotonic()-start,
                  'heartbeats': heartbeats, 'neutral_samples': neutral,
                  'units': {'ATTITUDE': 'rad and rad/s', 'SCALED_IMU': 'mrad/s',
                            'SCALED_IMU2': 'mrad/s', 'SCALED_PRESSURE2': 'hPa, centidegC'},
                  'summary': summary, 'rows': rows, 'real_water_acceptance': False}
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)
    finally:
        link.close()


if __name__ == '__main__':
    main()
