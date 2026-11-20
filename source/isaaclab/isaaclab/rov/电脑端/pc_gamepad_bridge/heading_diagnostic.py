"""Read-only native heading audit over the existing Pi telemetry mirror.

Run only with the PC control window closed. Does not own the FC serial port,
arm, change modes, calibrate, set parameters, or command actuators.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

from control_profile import LocalActionController, load_profile
from flight_mode import FlightModeController
from qgc_fly_status import CompetitionTelemetry, make_snapshot
from rov_gamepad_bridge import open_mavlink, open_telemetry
from servo_control import ServoController


PARAMS = ('INS_GYR_CAL', 'INS_GYROFFS_Z', 'INS_GYR2OFFS_Z', 'COMPASS_ENABLE',
          'COMPASS_USE', 'COMPASS_OFS_X', 'COMPASS_OFS_Y', 'COMPASS_OFS_Z',
          'EK3_SRC1_YAW', 'AHRS_EKF_TYPE', 'AHRS_ORIENTATION')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-parameters', action='store_true', help='also save a complete live parameter readback')
    args = parser.parse_args()
    if not 5 <= args.seconds <= 60:
        parser.error('seconds must be 5..60')
    modes, telemetry = FlightModeController(), CompetitionTelemetry()
    profile = load_profile()
    actions, servos = LocalActionController(profile), ServoController.from_profile(profile)
    receiver, sender = open_telemetry(14551), open_mavlink('192.168.1.3', 14550)
    sender.socket.bind(('192.168.1.4', 0))
    rows, params, health, ekf, mags = [], {}, [], [], []
    all_params, indices, expected_count = {}, {}, None
    list_requested = False
    started, next_query = time.monotonic(), 0
    try:
        while time.monotonic() - started < args.seconds:
            now = time.monotonic()
            if modes.fresh(now) and (modes.armed or modes.mode != 19):
                raise RuntimeError('Fresh disarmed Manual required; no writes sent')
            if modes.fresh(now) and now >= next_query:
                if args.all_parameters and not list_requested:
                    sender.mav.param_request_list_send(1, 1)
                    list_requested = True
                for name in PARAMS:
                    if name not in params:
                        sender.mav.param_request_read_send(1, 1, name.encode('ascii'), -1)
                next_query = now + 3
                if args.all_parameters and expected_count:
                    for index in [i for i in range(expected_count) if i not in indices][:12]:
                        sender.mav.param_request_read_send(1, 1, b'', index)
            msg = receiver.recv_match(blocking=True, timeout=.1)
            if msg is None or msg.get_srcSystem() != 1 or msg.get_srcComponent() != 1:
                continue
            now = time.monotonic()
            modes.observe(msg, now)
            telemetry.observe(msg, now)
            kind = msg.get_type()
            if kind == 'ATTITUDE' and modes.last_attitude == now:
                snapshot = make_snapshot(modes, telemetry, actions, servos, now,
                    hid_fresh=False, sending=False, ready=False, allow_arm=False)
                rows.append({'elapsed_s': now-started, 'time_boot_ms': msg.time_boot_ms,
                    'native_yaw_deg': math.degrees(msg.yaw) % 360,
                    'display_heading_deg': snapshot['heading_deg'],
                    'raw_yawspeed_deg_s': math.degrees(msg.yawspeed),
                    'roll_deg': math.degrees(msg.roll), 'pitch_deg': math.degrees(msg.pitch)})
            elif kind == 'VFR_HUD' and rows:
                rows[-1]['native_vfr_heading_deg'] = msg.heading
            elif kind == 'PARAM_VALUE':
                name = msg.param_id.decode('ascii') if isinstance(msg.param_id, bytes) else msg.param_id
                name = name.rstrip('\x00')
                if name in PARAMS:
                    params[name] = {'value': msg.param_value, 'type': msg.param_type}
                if args.all_parameters and math.isfinite(msg.param_value):
                    if expected_count is not None and expected_count != msg.param_count:
                        raise RuntimeError('Parameter count changed during readback')
                    expected_count = msg.param_count
                    if 0 <= msg.param_index < expected_count:
                        all_params[name] = {'value': msg.param_value, 'type': msg.param_type, 'index': msg.param_index}
                        indices[msg.param_index] = name
            elif kind == 'SYS_STATUS':
                health.append({'elapsed_s': now-started,
                    'present': msg.onboard_control_sensors_present,
                    'enabled': msg.onboard_control_sensors_enabled,
                    'health': msg.onboard_control_sensors_health})
            elif kind == 'EKF_STATUS_REPORT':
                ekf.append({'elapsed_s': now-started, **msg.to_dict()})
            elif kind in ('RAW_IMU', 'SCALED_IMU', 'SCALED_IMU2'):
                mags.append({'elapsed_s': now-started, 'kind': kind,
                    'xmag': msg.xmag, 'ymag': msg.ymag, 'zmag': msg.zmag})
        if not modes.fresh(time.monotonic()) or modes.armed or len(rows) < 20:
            raise RuntimeError('Insufficient fresh disarmed telemetry')
        if args.all_parameters and (not expected_count or len(indices) != expected_count or len(all_params) != expected_count):
            raise RuntimeError(f'Incomplete parameter readback: {len(indices)}/{expected_count}')
        deltas = [(b['native_yaw_deg'] - a['native_yaw_deg'] + 180) % 360 - 180
                  for a, b in zip(rows, rows[1:])]
        total = sum(deltas)
        duration = rows[-1]['elapsed_s'] - rows[0]['elapsed_s']
        errors = [abs(r['native_yaw_deg']-r['display_heading_deg']) for r in rows
                  if r['display_heading_deg'] is not None]
        report = {'read_only': True, 'armed': modes.armed, 'mode': modes.mode,
            'summary': {'samples': len(rows), 'duration_s': duration,
                'native_heading_change_deg': total, 'native_drift_deg_s': total/duration,
                'max_native_display_error_deg': max(errors),
                'median_raw_yawspeed_deg_s': statistics.median(r['raw_yawspeed_deg_s'] for r in rows)},
            'parameters': params, 'samples': rows, 'sensor_health': health,
            'ekf_status': ekf, 'mag_samples': mags}
        if args.all_parameters:
            report['complete_parameters'] = all_params
            report['parameter_count'] = expected_count
            args.output.with_suffix('.params').write_text(
                '# Full live readback after native gyro calibration and INS_GYR_CAL=1\n' +
                ''.join(f'1\t1\t{name}\t{value["value"]:.10g}\t{value["type"]}\n'
                        for name, value in sorted(all_params.items(), key=lambda item: item[1]['index'])), encoding='utf-8')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: report[k] for k in ('read_only', 'armed', 'mode', 'summary', 'parameters')}, indent=2))
    finally:
        sender.close()
        receiver.close()


if __name__ == '__main__':
    main()
