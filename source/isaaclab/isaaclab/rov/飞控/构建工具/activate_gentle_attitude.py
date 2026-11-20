"""Activate gentle roll/pitch correction while disarmed and in Manual."""
import argparse
import math
from pathlib import Path
import subprocess
import time


PARAMS = (
    ('ATC_ANG_RLL_P', 2.0, 9),
    ('ATC_ANG_PIT_P', 2.0, 9),
    ('MOT_RP_MAX', 25.0, 9),
)


def main():
    from pymavlink import mavutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--angle-p', type=float, default=2.0)
    parser.add_argument('--rp-max-percent', type=float, default=25.0)
    parser.add_argument('--depth-max-percent', type=float, default=None,
                        help='Set MOT_DH_MAX only on the gentle depth successor firmware')
    args = parser.parse_args()
    if not math.isfinite(args.angle_p) or not 0.5 <= args.angle_p <= 6.0:
        raise RuntimeError('Angle P must be 0.5..6.0')
    if not math.isfinite(args.rp_max_percent) or not 0 <= args.rp_max_percent <= 100:
        raise RuntimeError('Roll/pitch cap must be 0..100 percent')
    if args.depth_max_percent is not None and (not math.isfinite(args.depth_max_percent)
                                               or not 0 <= args.depth_max_percent <= 100):
        raise RuntimeError('Depth hold cap must be 0..100 percent')
    targets = [
        ('ATC_ANG_RLL_P', args.angle_p, 9),
        ('ATC_ANG_PIT_P', args.angle_p, 9),
        ('MOT_RP_MAX', args.rp_max_percent, 9),
    ]
    if args.depth_max_percent is not None:
        targets.append(('MOT_DH_MAX', args.depth_max_percent, 9))
    targets.append(('MOT_ATT_PRIO', 1.0, 2))
    port = str(Path(args.port).resolve(strict=True))
    owner = subprocess.run(['fuser', port], capture_output=True, text=True, timeout=5)
    if owner.returncode != 1 or owner.stdout.strip():
        raise RuntimeError('Serial port is not free')
    link = mavutil.mavlink_connection(port, baud=115200, source_system=254, source_component=191)
    try:
        link.port.exclusive = True
        hb = link.wait_heartbeat(timeout=10)
        if hb is None or hb.type != mavutil.mavlink.MAV_TYPE_SUBMARINE or hb.base_mode & 128 or hb.custom_mode != 19:
            raise RuntimeError('Require fresh disarmed ArduSub in Manual; no writes')

        def read(name):
            link.mav.param_request_read_send(1, 1, name.encode('ascii'), -1)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                msg = link.recv_match(blocking=True, timeout=.2)
                if msg is None or msg.get_srcSystem() != 1 or msg.get_srcComponent() != 1:
                    continue
                if msg.get_type() == 'HEARTBEAT' and (msg.base_mode & 128 or msg.custom_mode != 19):
                    raise RuntimeError('Vehicle state changed; abort')
                if msg.get_type() == 'PARAM_VALUE':
                    incoming = msg.param_id.decode('ascii') if isinstance(msg.param_id, bytes) else msg.param_id
                    if incoming.rstrip('\x00') == name:
                        return float(msg.param_value), int(msg.param_type)
            raise RuntimeError('No readback: ' + name)

        for name, target, kind in targets:
            before, actual_kind = read(name)
            if actual_kind != kind or not math.isfinite(before):
                raise RuntimeError('Invalid parameter: ' + name)
            if not math.isclose(before, target, abs_tol=1e-7):
                fresh = link.recv_match(type='HEARTBEAT', blocking=True, timeout=3)
                if fresh is None or fresh.get_srcSystem() != 1 or fresh.get_srcComponent() != 1 or fresh.base_mode & 128 or fresh.custom_mode != 19:
                    raise RuntimeError('No fresh disarmed Manual heartbeat before write')
                link.mav.param_set_send(1, 1, name.encode('ascii'), target, kind)
                time.sleep(.3)
            after, actual_kind = read(name)
            if actual_kind != kind or not math.isclose(after, target, abs_tol=1e-7):
                raise RuntimeError('Independent readback mismatch: ' + name)
            print(f'VERIFIED {name}={after:g}', flush=True)
    finally:
        link.close()


if __name__ == '__main__':
    main()
