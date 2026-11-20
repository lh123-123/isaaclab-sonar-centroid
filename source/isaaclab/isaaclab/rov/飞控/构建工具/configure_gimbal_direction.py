"""Set only SERVO12_REVERSED=1, with fresh disarmed/neutral/readback gates.

Actuator power must be disconnected by the operator. No arm/mode/servo commands.
The calibrated limits and mount function are never changed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from pymavlink import mavutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Evidence path already exists')
    port = str(Path(args.port).resolve(strict=True))
    owner = subprocess.run(['fuser', port], capture_output=True, text=True, timeout=5)
    if owner.returncode != 1 or owner.stdout.strip():
        raise RuntimeError('Serial port not free')
    link = mavutil.mavlink_connection(port, baud=115200, source_system=254, source_component=191)
    next_hb = 0
    armed = None
    last_hb = 0
    neutral = False

    def receive():
        nonlocal next_hb, armed, last_hb, neutral
        now = time.monotonic()
        if now >= next_hb:
            link.mav.heartbeat_send(6, 8, 0, 0, 0)
            next_hb = now + 1
        msg = link.recv_match(blocking=True, timeout=.2)
        if msg is None or msg.get_srcSystem() != 1 or msg.get_srcComponent() != 1:
            return None
        if msg.get_type() == 'HEARTBEAT':
            if msg.type != 12 or msg.autopilot != 3 or msg.base_mode & 128:
                raise RuntimeError('Only disarmed ArduSub is allowed')
            armed, last_hb = False, time.monotonic()
        elif msg.get_type() == 'SERVO_OUTPUT_RAW' and msg.port == 0:
            if [getattr(msg, f'servo{i}_raw') for i in range(1, 9)] != [1500] * 8:
                raise RuntimeError('Thruster outputs are not neutral')
            neutral = True
        return msg

    def read(name):
        # Drain old traffic before a distinct read request. Never treat SET's
        # echoed PARAM_VALUE as independent confirmation of persisted settings.
        while link.recv_match(blocking=False) is not None:
            pass
        link.mav.param_request_read_send(1, 1, name.encode('ascii'), -1)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            msg = receive()
            if msg is not None and msg.get_type() == 'PARAM_VALUE':
                key = msg.param_id.decode('ascii') if isinstance(msg.param_id, bytes) else msg.param_id
                if key.rstrip('\x00') == name:
                    return float(msg.param_value), int(msg.param_type)
        raise RuntimeError('Independent read timed out: ' + name)

    try:
        link.port.exclusive = True
        deadline = time.monotonic() + 6
        link.mav.command_long_send(1, 1, 512, 0, 36, 0, 0, 0, 0, 0, 0)
        while time.monotonic() < deadline and (armed is not False or not neutral):
            receive()
        if armed is not False or not neutral:
            raise RuntimeError('Disarmed/neutral evidence missing')
        function, function_type = read('SERVO12_FUNCTION')
        before, kind = read('SERVO12_REVERSED')
        if function != 7 or function_type != 4 or kind != 2 or before not in (0, 1):
            raise RuntimeError('Expected mount_tilt channel and binary reverse setting missing')
        if time.monotonic() - last_hb > 3:
            raise RuntimeError('Heartbeat stale before parameter write')
        if before == 0:
            link.mav.param_set_send(1, 1, b'SERVO12_REVERSED', 1.0, kind)
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                receive()
        after, after_type = read('SERVO12_REVERSED')
        if after != 1 or after_type != kind or time.monotonic() - last_hb > 3:
            raise RuntimeError('Gimbal reverse independent readback failed; keep power disconnected')
        report = {'parameter': 'SERVO12_REVERSED', 'before': before, 'after': after,
                  'function': function, 'disarmed': True, 'written': before == 0,
                  'independent_readback': True, 'mechanical_direction_tested': False}
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report), flush=True)
    finally:
        link.close()


if __name__ == '__main__':
    main()
