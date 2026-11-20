"""Explicit native gyro calibration, with disarmed/neutral/stationary gates.

Does not calibrate level/compass, reset yaw, arm, reboot, set parameters or
command motors. Readbacks and ACK are saved; physical quiet must be confirmed.
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flight_mode import FlightModeController
from rov_gamepad_bridge import open_mavlink, open_telemetry

PARAMS = tuple(f'INS_{stem}{axis}' for stem in ('GYROFFS_', 'GYR2OFFS_')
               for axis in 'XYZ') + ('INS_GYR_CAL', 'AHRS_ORIENTATION',
                                     'AHRS_TRIM_X', 'AHRS_TRIM_Y', 'AHRS_TRIM_Z')


def stationary(samples, now):
    recent = [s for s in samples if 0 <= now-s[0] <= 1.5]
    return (len(recent) >= 10 and recent[-1][0]-recent[0][0] >= 1
            and now-recent[-1][0] <= .2
            and all(max(abs(v) for v in s[2]) <= .035 for s in recent)
            and all(max(s[1][a] for s in recent)-min(s[1][a] for s in recent) <= .4
                    for a in (0, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stationary-power-off-confirmed', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.stationary_power_off_confirmed:
        parser.error('Physical stationary body, disconnected actuator power and stopped PC control required')
    receiver, sender = open_telemetry(14551), open_mavlink('192.168.1.3', 14550)
    sender.socket.bind(('192.168.1.4', 0))
    modes, samples = FlightModeController(), deque(maxlen=80)
    pwm_at, before, after, texts = float('-inf'), {}, {}, []
    phase, deadline, query_at = 'before', time.monotonic()+8, 0
    ack = None
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if modes.fresh(now) and (modes.armed or modes.mode != 19):
                raise RuntimeError('Not disarmed Manual; calibration aborted')
            values = before if phase == 'before' else after
            if phase in ('before', 'after') and modes.fresh(now) and now >= query_at:
                for name in PARAMS:
                    if name not in values:
                        sender.mav.param_request_read_send(1, 1, name.encode(), -1)
                query_at = now+2
            msg = receiver.recv_match(blocking=True, timeout=.05)
            if msg is not None and msg.get_srcSystem() == 1 and msg.get_srcComponent() == 1:
                now = time.monotonic()
                modes.observe(msg, now)
                kind = msg.get_type()
                if kind == 'HEARTBEAT' and (msg.base_mode & 128 or msg.custom_mode != 19):
                    raise RuntimeError('Armed/non-Manual heartbeat; calibration aborted')
                if kind == 'SERVO_OUTPUT_RAW' and msg.port == 0:
                    if any(getattr(msg, f'servo{n}_raw') != 1500 for n in range(1, 9)):
                        raise RuntimeError('Non-neutral motor output; calibration aborted')
                    pwm_at = now
                elif kind == 'ATTITUDE':
                    angles = (msg.roll, msg.pitch, msg.yaw)
                    rates = (msg.rollspeed, msg.pitchspeed, msg.yawspeed)
                    if all(math.isfinite(v) for v in angles+rates):
                        samples.append((now, tuple(math.degrees(v) for v in angles), rates))
                elif kind == 'PARAM_VALUE' and phase in ('before', 'after'):
                    name = msg.param_id.decode() if isinstance(msg.param_id, bytes) else msg.param_id
                    name = name.rstrip('\x00')
                    if name in PARAMS and math.isfinite(msg.param_value):
                        expected = 2 if name in ('INS_GYR_CAL', 'AHRS_ORIENTATION') else 9
                        if msg.param_type != expected:
                            raise RuntimeError('Unexpected parameter type: '+name)
                        values[name] = msg.param_value
                elif kind == 'STATUSTEXT':
                    texts.append(str(msg.text))
                elif kind == 'COMMAND_ACK' and msg.command == 241 and phase == 'calibrate':
                    if getattr(msg, 'target_system', 255) not in (0, 255) or getattr(msg, 'target_component', 190) not in (0, 190):
                        continue
                    if msg.result not in (0, 5):
                        raise RuntimeError('Native gyro calibration rejected: '+str(msg.result))
                    if msg.result == 0:
                        ack = msg.to_dict()
                        phase, deadline, query_at = 'after', now+10, 0
                        samples.clear()
            if phase == 'before' and len(before) == len(PARAMS):
                if modes.fresh(now) and not modes.armed and modes.mode == 19 and now-pwm_at <= .5 and stationary(samples, now):
                    # MAV_CMD_PREFLIGHT_CALIBRATION: param1=1 only, all others=0.
                    sender.mav.command_long_send(1, 1, 241, 0, 1, 0, 0, 0, 0, 0, 0)
                    phase, deadline = 'calibrate', now+35
                    print('Native gyro calibration sent; no level/yaw/compass command', flush=True)
            elif phase == 'after' and len(after) == len(PARAMS):
                protected = ('AHRS_ORIENTATION', 'AHRS_TRIM_X', 'AHRS_TRIM_Y', 'AHRS_TRIM_Z', 'INS_GYR_CAL')
                if any(before[n] != after[n] for n in protected):
                    raise RuntimeError('Unexpected change to level/orientation/startup setting')
                if modes.fresh(now) and now-pwm_at <= .5 and stationary(samples, now):
                    report = {'native_gyro_calibration': True, 'ack': ack,
                        'before': before, 'after': after, 'statustext': texts,
                        'armed': modes.armed, 'mode': modes.mode, 'yaw_reset_sent': False,
                        'level_calibration_sent': False, 'parameter_set_sent': False}
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
                    print(json.dumps(report, indent=2))
                    return
        raise RuntimeError('Calibration gate/ACK/readback timed out; not automatically retried')
    finally:
        sender.close()
        receiver.close()


if __name__ == '__main__':
    main()
