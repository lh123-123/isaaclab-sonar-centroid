"""Confirm four real modes while disarmed/isolated; finish in Manual.

Never sends ARM, MANUAL_CONTROL, servo or motor-test commands. Mode requests
are state-changing: operator must disconnect all actuator power beforehand.
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
        raise RuntimeError('Refusing to overwrite verification evidence')
    port = str(Path(args.port).resolve(strict=True))
    owner = subprocess.run(['fuser', port], capture_output=True, text=True, timeout=5)
    if owner.returncode != 1 or owner.stdout.strip():
        raise RuntimeError('Serial port is not free')
    link = mavutil.mavlink_connection(port, baud=115200, source_system=254, source_component=191)
    records = []
    next_request = 0
    capability = False
    mode = None
    armed = None
    neutral_samples = 0
    def pump(deadline, target=None):
        nonlocal next_request, capability, mode, armed, neutral_samples
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_request:
                link.mav.heartbeat_send(6, 8, 0, 0, 0)
                link.mav.command_long_send(1, 1, 512, 0, 36, 0, 0, 0, 0, 0, 0)
                next_request = now + 1
            msg = link.recv_match(blocking=True, timeout=.2)
            if msg is None or msg.get_srcSystem() != 1 or msg.get_srcComponent() != 1:
                continue
            kind = msg.get_type()
            if kind == 'HEARTBEAT':
                if msg.type != 12 or msg.autopilot != 3 or msg.base_mode & 128:
                    raise RuntimeError('Expected disarmed ArduSub only; abort')
                mode, armed = int(msg.custom_mode), False
                if target is not None and mode == target:
                    return True
            elif kind == 'SERVO_OUTPUT_RAW' and msg.port == 0:
                if [getattr(msg, f'servo{i}_raw') for i in range(1, 9)] != [1500] * 8:
                    raise RuntimeError('Non-neutral thruster output during mode check')
                neutral_samples += 1
            elif kind == 'NAMED_VALUE_FLOAT':
                name = msg.name.decode('ascii') if isinstance(msg.name, bytes) else msg.name
                if name.rstrip('\x00') == 'ROV_DONLY' and msg.value == 1:
                    capability = True
            elif kind == 'COMMAND_ACK' and msg.command == 176 and msg.result not in (0, 5):
                raise RuntimeError(f'Mode rejected: result {msg.result}')
            elif kind == 'STATUSTEXT':
                if any(word in msg.text.lower() for word in ('deadlock', 'watchdog', 'internal error', 'panic')):
                    raise RuntimeError('Firmware fault: ' + msg.text)
        return False
    try:
        link.port.exclusive = True
        pump(time.monotonic() + 5)
        if armed is not False or not capability or not neutral_samples:
            raise RuntimeError('Fresh disarmed/capability/neutral evidence missing')
        # Each hold combination, including both directions of attitude toggle
        # while depth hold stays enabled, confirmed by actual HEARTBEAT.
        for target in (19, 0, 2, 22, 2, 0, 19):
            link.mav.command_long_send(1, 1, 176, 0, 1, target, 0, 0, 0, 0, 0)
            if not pump(time.monotonic() + 5, target):
                raise RuntimeError(f'Mode {target} heartbeat confirmation timeout')
            records.append({'mode': target, 'confirmed_disarmed': True})
            print('CONFIRMED disarmed mode', target, flush=True)
            pump(time.monotonic() + .7)
        report = {'modes': records, 'ROV_DONLY': 1, 'neutral_samples': neutral_samples,
                  'final_mode': mode, 'armed': armed, 'hardware_motion_tested': False}
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report), flush=True)
    finally:
        if armed is False and mode != 19:
            link.mav.command_long_send(1, 1, 176, 0, 1, 19, 0, 0, 0, 0, 0)
            if not pump(time.monotonic() + 5, 19):
                print('WARNING: Manual restore unconfirmed; do not reconnect actuator power', flush=True)
        link.close()


if __name__ == '__main__':
    main()
