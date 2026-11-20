"""Activate only the two verified custom parameters while isolated/disarmed.

Requires pre/post full readbacks to match for all pre-existing parameters.
No arm, mode, manual-control, or motor commands are sent.
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path



CONTROL_PARAMETERS = [
    ('ROV_T_RATE', 30.0, 30.0, 9), ('ROV_T_ACC', 90.0, 90.0, 9),
    ('ROV_Y_RATE', 45.0, 45.0, 9), ('ROV_Y_ACC', 120.0, 120.0, 9),
    ('ROV_Y_LEAD', 12.0, 12.0, 9), ('ROV_IN_TMO', 0.35, 0.35, 9),
    ('MOT_DZ_IN', 0.01, 0.01, 9), ('MOT_DZ_PWM', 0.0, 50.0, 4),
    ('MOT_ATT_PRIO', 0.0, 1.0, 2), ('ROV_EN', 0.0, 1.0, 2),
]


def same_parameters(before, after, expected_additions=None):
    for snapshot in (before, after):
        if snapshot['parameter_count'] != len(snapshot['parameters']):
            raise RuntimeError('Incomplete parameter snapshot')
    differences = {}
    runtime_counters = {}
    old_boot = before['parameters'].get('STAT_BOOTCNT', {}).get('value', 0)
    new_boot = after['parameters'].get('STAT_BOOTCNT', {}).get('value', 0)
    reboot_observed = (math.isfinite(old_boot) and math.isfinite(new_boot)
                       and 0 < new_boot - old_boot <= 10)
    for name, old in before['parameters'].items():
        new = after['parameters'].get(name)
        if new is None or old['type'] != new['type'] or old['value'] != new['value']:
            # These two read-only counters advance during normal boot/runtime.
            # Never waive configuration, flight-count or flight-time changes.
            if name in ('STAT_BOOTCNT', 'STAT_RUNTIME') and new is not None and old['type'] == new['type']:
                old_value, new_value = float(old['value']), float(new['value'])
                bound = 10 if name == 'STAT_BOOTCNT' else 86400
                if (math.isfinite(old_value) and math.isfinite(new_value)
                        and old_value.is_integer() and new_value.is_integer()
                        and 0 <= new_value - old_value <= bound):
                    runtime_counters[name] = {'before': old_value, 'after': new_value}
                    continue
            # Pinned AP_Baro.cpp marks these internal values read-only/volatile
            # and calibrates them on boot. Only allow a small shore-pressure
            # change across a proven reboot; never restore an obsolete zero.
            if (name in ('BARO1_GND_PRESS', 'BARO2_GND_PRESS', 'BARO3_GND_PRESS')
                    and reboot_observed and new is not None
                    and old['type'] == new['type'] == 9):
                old_value, new_value = float(old['value']), float(new['value'])
                if (math.isfinite(old_value) and math.isfinite(new_value)
                        and 80000 <= old_value <= 120000
                        and 80000 <= new_value <= 120000
                        and abs(new_value - old_value) <= 2000):
                    runtime_counters[name] = {'before': old_value, 'after': new_value,
                                              'reason': 'volatile boot ground-pressure calibration'}
                    continue
            differences[name] = {'before': old, 'after': new}
    if differences:
        raise RuntimeError('Pre-existing parameters changed: ' + json.dumps(differences))
    additions = set(after['parameters']) - set(before['parameters'])
    expected_additions = {'MOT_DZ_PWM', 'MOT_DZ_IN'} if expected_additions is None else set(expected_additions)
    if additions != expected_additions:
        raise RuntimeError('Unexpected firmware parameter additions: ' + repr(additions))
    if before['autopilot_version']['uid2'] != after['autopilot_version']['uid2']:
        raise RuntimeError('Vehicle identity changed')
    return runtime_counters


def main():
    from pymavlink import mavutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--control-bundle', action='store_true',
                        help='verify all 10 combined-firmware additions; enable tilt return and priority last')
    args = parser.parse_args()
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    parameters = CONTROL_PARAMETERS if args.control_bundle else [
        ('MOT_DZ_IN', 0.01, 0.01, 9), ('MOT_DZ_PWM', 0.0, 50.0, 4)]
    counters = same_parameters(before, after, {item[0] for item in parameters})
    print('CONFIGURATION VERIFIED; volatile boot/runtime changes: ' + json.dumps(counters), flush=True)
    for name, expected, _target, kind in parameters:
        item = after['parameters'][name]
        if item['type'] != kind or not math.isclose(item['value'], expected, abs_tol=1e-7):
            raise RuntimeError('Unexpected default for ' + name)
    port = str(Path(args.port).resolve(strict=True))
    owner = subprocess.run(['fuser', port], capture_output=True, text=True, timeout=5)
    if owner.returncode != 1 or owner.stdout.strip():
        raise RuntimeError('Serial port is not free')
    link = mavutil.mavlink_connection(port, baud=115200, source_system=254, source_component=191)
    try:
        link.port.exclusive = True
        hb = link.wait_heartbeat(timeout=10)
        if hb is None or hb.get_srcSystem() != 1 or hb.get_srcComponent() != 1 or hb.type != 12:
            raise RuntimeError('Expected ArduSub heartbeat missing')
        if hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            raise RuntimeError('Vehicle armed; no writes')

        def read(name):
            link.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                    mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            link.mav.param_request_read_send(1, 1, name.encode('ascii'), -1)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                msg = link.recv_match(blocking=True, timeout=0.2)
                if msg is None or msg.get_srcSystem() != 1 or msg.get_srcComponent() != 1:
                    continue
                if msg.get_type() == 'HEARTBEAT' and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                    raise RuntimeError('Vehicle armed during transaction; abort')
                if msg.get_type() == 'PARAM_VALUE' and msg.param_id.rstrip('\x00') == name:
                    return float(msg.param_value), int(msg.param_type)
            raise RuntimeError('No independent readback for ' + name)

        for name, _default, target, kind in parameters:
            previous, actual_type = read(name)
            if actual_type != kind:
                raise RuntimeError('Parameter type mismatch: ' + name)
            if not math.isclose(previous, target, abs_tol=1e-7):
                # Recheck fresh disarmed heartbeat immediately before mutation.
                hb = link.recv_match(type='HEARTBEAT', blocking=True, timeout=3)
                if hb is None or hb.get_srcSystem() != 1 or hb.get_srcComponent() != 1 or hb.base_mode & 128:
                    raise RuntimeError('No fresh disarmed heartbeat before write')
                link.mav.param_set_send(1, 1, name.encode('ascii'), target, kind)
                # Consume set acknowledgement, then independently request the value.
                time.sleep(0.3)
                for _ in range(1000):
                    if link.recv_match(blocking=False) is None:
                        break
                else:
                    raise RuntimeError('Unable to drain set acknowledgements')
            value, actual_type = read(name)
            if actual_type != kind or not math.isclose(value, target, abs_tol=1e-7):
                raise RuntimeError('Activation readback mismatch: ' + name)
            print(f'VERIFIED {name}={value:g}', flush=True)
    finally:
        link.close()


if __name__ == '__main__':
    main()
