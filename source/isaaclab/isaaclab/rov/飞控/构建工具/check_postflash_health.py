"""Read-only, 30-second custom-firmware health/neutral-output check."""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

from pymavlink import mavutil
from activate_deadzone import CONTROL_PARAMETERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control-bundle', action='store_true')
    parser.add_argument('--before-activation', action='store_true')
    parser.add_argument('--expect-tilt-tolerance', type=float)
    parser.add_argument('--expect-pwm-min', type=int, default=1000)
    parser.add_argument('--expect-pwm-max', type=int, default=2000)
    parser.add_argument('--expect-horizontal-reverse', type=int, choices=(0, 1))
    parser.add_argument('--expect-current-max', type=float)
    parser.add_argument('--expect-current-tc', type=float)
    parser.add_argument('--expect-level-roll', type=float)
    parser.add_argument('--expect-level-pitch', type=float)
    parser.add_argument('--expect-angle-p', type=float)
    parser.add_argument('--expect-rp-max', type=float)
    parser.add_argument('--expect-depth-max', type=float)
    args = parser.parse_args()
    if not 1000 <= args.expect_pwm_min < 1500 < args.expect_pwm_max <= 2000:
        raise RuntimeError('Invalid expected motor PWM bounds')
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite health report')
    port = str(Path(args.port).resolve(strict=True))
    owner = subprocess.run(['fuser', port], capture_output=True, text=True, timeout=5)
    if owner.returncode != 1 or owner.stdout.strip():
        raise RuntimeError('Serial port not free')
    link = mavutil.mavlink_connection(port, baud=115200, source_system=254, source_component=191)
    report = {'heartbeats': 0, 'neutral_output_samples': 0, 'params': {}, 'messages': []}
    targets = {'MOT_DZ_PWM': 50.0, 'MOT_DZ_IN': 0.01, 'MOT_PWM_MIN': float(args.expect_pwm_min),
               'MOT_PWM_MAX': float(args.expect_pwm_max), 'FRAME_CONFIG': 2.0}
    if args.expect_horizontal_reverse is not None:
        targets['MOT_H_REV'] = float(args.expect_horizontal_reverse)
    for name, value in (('MOT_BAT_CURR_MAX', args.expect_current_max),
                        ('MOT_BAT_CURR_TC', args.expect_current_tc),
                        ('ROV_LVL_R', args.expect_level_roll), ('ROV_LVL_P', args.expect_level_pitch),
                        ('ATC_ANG_RLL_P', args.expect_angle_p), ('ATC_ANG_PIT_P', args.expect_angle_p),
                        ('MOT_RP_MAX', args.expect_rp_max),
                        ('MOT_DH_MAX', args.expect_depth_max)):
        if value is not None:
            targets[name] = float(value)
    if args.before_activation:
        targets['MOT_DZ_PWM'] = 0.0
    if args.control_bundle:
        targets.update({name: default if args.before_activation else active
                        for name, default, active, _kind in CONTROL_PARAMETERS})
    if args.expect_tilt_tolerance is not None:
        if not 0.0 <= args.expect_tilt_tolerance <= 5.0:
            raise RuntimeError('Invalid expected tilt tolerance')
        targets['ROV_T_ERR'] = args.expect_tilt_tolerance
    try:
        link.port.exclusive = True
        start = time.monotonic()
        last_hb = start
        next_request = start
        while time.monotonic() - start < 30:
            now = time.monotonic()
            if now >= next_request:
                link.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                link.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
                                           mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW, 0, 0, 0, 0, 0, 0)
                for name in targets:
                    link.mav.param_request_read_send(1, 1, name.encode('ascii'), -1)
                next_request = now + 2
            msg = link.recv_match(blocking=True, timeout=0.2)
            if msg is not None and msg.get_srcSystem() == 1 and msg.get_srcComponent() == 1:
                kind = msg.get_type()
                if kind == 'HEARTBEAT':
                    if msg.type != 12 or msg.base_mode & 128:
                        raise RuntimeError('Wrong vehicle type or armed heartbeat')
                    last_hb = now
                    report['heartbeats'] += 1
                elif kind == 'SERVO_OUTPUT_RAW' and msg.port == 0:
                    outputs = [getattr(msg, f'servo{i}_raw') for i in range(1, 9)]
                    if outputs != [1500] * 8:
                        raise RuntimeError('Non-neutral motor output while disarmed: ' + repr(outputs))
                    report['neutral_output_samples'] += 1
                elif kind == 'PARAM_VALUE':
                    name = msg.param_id.rstrip('\x00')
                    if name in targets:
                        value = float(msg.param_value)
                        if not math.isclose(value, targets[name], abs_tol=1e-7):
                            raise RuntimeError(f'Unexpected {name}={value}')
                        report['params'][name] = value
                elif kind == 'STATUSTEXT':
                    text = msg.text.rstrip('\x00')
                    entry = {'severity': msg.severity, 'text': text}
                    if entry not in report['messages']:
                        report['messages'].append(entry)
                    if any(word in text.lower() for word in ('deadlock', 'watchdog', 'internal error', 'panic')):
                        raise RuntimeError('Firmware health fault: ' + text)
            if now - last_hb > 5:
                raise RuntimeError('Autopilot heartbeat interruption')
        if report['heartbeats'] < 20 or not report['neutral_output_samples'] or set(report['params']) != set(targets):
            raise RuntimeError('Incomplete health/parameter/output evidence: ' + repr(report))
        report['duration_seconds'] = time.monotonic() - start
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        link.close()


if __name__ == '__main__':
    main()
