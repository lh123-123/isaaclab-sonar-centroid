"""Restore one stock parameter: INS_GYR_CAL=1, with independent readback.

No immediate calibration/reboot, no mode/arm/actuator commands. Requires a
fresh disarmed Manual heartbeat and neutral 8-motor outputs before the write.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flight_mode import FlightModeController
from rov_gamepad_bridge import open_mavlink, open_telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receiver, sender = open_telemetry(14551), open_mavlink('192.168.1.3', 14550)
    sender.socket.bind(('192.168.1.4', 0))
    modes, before, phase = FlightModeController(), None, 'before'
    pwm_at, next_query, verify_at = float('-inf'), 0, float('inf')
    deadline = time.monotonic()+12
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if modes.fresh(now) and (modes.armed or modes.mode != 19):
                raise RuntimeError('Not disarmed Manual')
            if modes.fresh(now) and now >= next_query and phase in ('before', 'verify'):
                sender.mav.param_request_read_send(1, 1, b'INS_GYR_CAL', -1)
                next_query = now+2
            msg = receiver.recv_match(blocking=True, timeout=.05)
            if msg is not None and msg.get_srcSystem() == 1 and msg.get_srcComponent() == 1:
                now = time.monotonic()
                modes.observe(msg, now)
                if msg.get_type() == 'HEARTBEAT' and (msg.base_mode & 128 or msg.custom_mode != 19):
                    raise RuntimeError('Not disarmed Manual')
                elif msg.get_type() == 'SERVO_OUTPUT_RAW' and msg.port == 0:
                    if any(getattr(msg, f'servo{n}_raw') != 1500 for n in range(1, 9)):
                        raise RuntimeError('Motor outputs not neutral')
                    pwm_at = now
                elif msg.get_type() == 'PARAM_VALUE':
                    name = msg.param_id.decode() if isinstance(msg.param_id, bytes) else msg.param_id
                    if name.rstrip('\x00') == 'INS_GYR_CAL':
                        if msg.param_type != 2 or msg.param_value not in (0, 1):
                            raise RuntimeError('Unexpected INS_GYR_CAL value/type')
                        if phase == 'before':
                            before = int(msg.param_value)
                        elif phase == 'verify':
                            if msg.param_value != 1 or not modes.fresh(now) or now-pwm_at > .5:
                                raise RuntimeError('Independent readback failed')
                            report = {'parameter': 'INS_GYR_CAL', 'before': before,
                                'after': int(msg.param_value), 'type': msg.param_type,
                                'independent_readback': True, 'armed': modes.armed,
                                'mode': modes.mode, 'immediate_reboot_sent': False}
                            args.output.parent.mkdir(parents=True, exist_ok=True)
                            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
                            print(json.dumps(report, indent=2))
                            return
            if phase == 'before' and before is not None and modes.fresh(now) and now-pwm_at <= .5:
                sender.mav.param_set_send(1, 1, b'INS_GYR_CAL', 1, 2)
                phase, verify_at = 'drain_echo', now+.5
            elif phase == 'drain_echo' and now >= verify_at:
                phase, next_query = 'verify', 0
        raise RuntimeError('Startup calibration setting not confirmed; not retried automatically')
    finally:
        sender.close()
        receiver.close()


if __name__ == '__main__':
    main()
