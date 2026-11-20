"""Read-only MAVLink query probe; no arm, parameter writes or reboot."""
import argparse
import collections
import json
import time

from pymavlink import mavutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--source-system', type=int, default=255)
    args = parser.parse_args()
    link = mavutil.mavlink_connection(args.port, baud=115200,
                                    source_system=args.source_system, source_component=191)
    link.port.exclusive = True
    counts = collections.Counter()
    try:
        hb = link.wait_heartbeat(timeout=10)
        if hb is None or hb.get_srcSystem() != 1 or hb.get_srcComponent() != 1 or hb.type != 12 or hb.base_mode & 128:
            raise RuntimeError('Fresh disarmed ArduSub heartbeat required')
        print('HEARTBEAT ' + json.dumps(hb.to_dict()), flush=True)
        print('WIRE_PROTOCOL ' + str(link.WIRE_PROTOCOL_VERSION), flush=True)
        deadline = time.monotonic() + 10
        next_query = 0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_query:
                link.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                link.mav.param_request_read_send(1, 1, b'MAV_OPTIONS', -1)
                for message_id in (148, 36, 1):
                    link.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
                                              0, message_id, 0, 0, 0, 0, 0, 0)
                link.port.flush()
                next_query = now + 2
            msg = link.recv_match(blocking=True, timeout=0.1)
            if msg is None:
                continue
            counts[msg.get_type()] += 1
            if msg.get_srcSystem() != 1 or msg.get_srcComponent() != 1:
                continue
            if msg.get_type() == 'HEARTBEAT' and msg.base_mode & 128:
                raise RuntimeError('Unexpected armed heartbeat')
            if counts[msg.get_type()] == 1 and msg.get_type() in ('AUTOPILOT_VERSION', 'SERVO_OUTPUT_RAW', 'PARAM_VALUE', 'COMMAND_ACK', 'STATUSTEXT', 'SYS_STATUS'):
                print(json.dumps(msg.to_dict()), flush=True)
        print('MESSAGE_COUNTS ' + json.dumps(counts), flush=True)
    finally:
        link.close()


if __name__ == '__main__':
    main()
