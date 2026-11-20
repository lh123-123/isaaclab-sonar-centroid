"""Flash verified task artifact via pinned official uploader, with safe gates.

Operator must disconnect actuator power and stop the bridge first. Only the
application is flashed; board/size guards and CRC verification stay enabled.
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

from pymavlink import mavutil


def usb_interface(port):
    """Remember the physical USB interface, not its mutable ttyACM number."""
    return (Path('/sys/class/tty') / Path(port).name / 'device').resolve(strict=True)


def rediscover_port(interface):
    candidates = [str(Path('/dev') / tty.name)
                  for tty in Path('/sys/class/tty').glob('ttyACM*')
                  if (tty / 'device').resolve() == interface
                  and (Path('/dev') / tty.name).exists()]
    if len(candidates) > 1:
        raise RuntimeError('Ambiguous USB interface; no erase')
    return candidates[0] if candidates else None


def is_pixhawk_bootloader(port):
    """Wait for the bootloader USB identity, not the departing app tty.

    The FMU v3 bootloader is visible for only a few seconds. Opening the
    application tty during its disconnect consumed that window on the first
    no-erase attempt, so identify only after USB VID:PID is 26ac:0011.
    """
    try:
        usb_device = (Path('/sys/class/tty') / Path(port).name / 'device').resolve(strict=True).parent
        return ((usb_device / 'idVendor').read_text().strip().lower() == '26ac'
                and (usb_device / 'idProduct').read_text().strip().lower() == '0011')
    except (FileNotFoundError, OSError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--firmware', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    identity = json.loads(args.identity.read_text())
    backup = json.loads(args.backup.read_text())
    if backup['parameter_count'] != len(backup['parameters']):
        raise RuntimeError('Incomplete pre-flash backup')
    if hashlib.sha256(args.firmware.read_bytes()).hexdigest().lower() != args.sha256.lower():
        raise RuntimeError('Firmware file hash mismatch')
    spec = importlib.util.spec_from_file_location('pinned_uploader', args.source / 'Tools/scripts/uploader.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    firmware = module.firmware(str(args.firmware))
    if firmware.property('board_id') != identity['board_id'] or identity['board_id'] != 9:
        raise RuntimeError('Artifact/board ID mismatch')
    if firmware.property('image_size') > identity['flash_size']:
        raise RuntimeError('Firmware exceeds identified flash capacity')
    port = str(Path(args.port).resolve(strict=True))
    interface = usb_interface(port)
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
            raise RuntimeError('Vehicle armed; flash aborted')
        version = None
        neutral = False
        deadline = time.monotonic() + 10
        next_request = 0.0
        while time.monotonic() < deadline:
            if time.monotonic() >= next_request:
                link.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                for msg_id in (mavutil.mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION,
                               mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW):
                    link.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
                                               msg_id, 0, 0, 0, 0, 0, 0)
                link.port.flush()
                next_request = time.monotonic() + 1.0
            msg = link.recv_match(blocking=True, timeout=0.2)
            if msg is None or msg.get_srcSystem() != 1 or msg.get_srcComponent() != 1:
                continue
            if msg.get_type() == 'HEARTBEAT' and msg.base_mode & 128:
                raise RuntimeError('Vehicle armed during preflight; abort')
            if msg.get_type() == 'AUTOPILOT_VERSION':
                version = msg.to_dict()
            if msg.get_type() == 'SERVO_OUTPUT_RAW' and msg.port == 0:
                outputs = [getattr(msg, f'servo{i}_raw') for i in range(1, 9)]
                if outputs != [1500] * 8:
                    raise RuntimeError('Motor outputs are not neutral: ' + repr(outputs))
                neutral = True
            if version is not None and neutral:
                break
        if version is None or version['uid2'] != backup['autopilot_version']['uid2'] or not neutral:
            raise RuntimeError(f'Fresh identity/neutral outputs not confirmed: version={version is not None}, neutral={neutral}')
        print('SAFE GATES PASSED: disarmed, same UID, 8 outputs=1500, artifact hash/board/size match', flush=True)
        link.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
                                   0, 3, 0, 0, 0, 0, 0, 0)
        link.port.flush()
        # Acknowledgement is diagnostic; USB may disappear before it arrives.
        # Do not spend seconds here and miss an older bootloader's short window.
        ack_deadline = time.monotonic() + 0.25
        while time.monotonic() < ack_deadline:
            try:
                msg = link.recv_match(blocking=True, timeout=0.02)
            except (OSError, module.serial.SerialException) as exc:
                print(f'Reboot USB transition: {exc}', flush=True)
                break
            if (msg is not None and msg.get_type() == 'COMMAND_ACK'
                    and msg.get_srcSystem() == 1 and msg.get_srcComponent() == 1
                    and msg.command == mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN):
                print(f'Reboot acknowledgement: result={msg.result}', flush=True)
                if msg.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    raise RuntimeError('Bootloader reboot rejected; no erase')
                break
    finally:
        link.close()
    up = None
    deadline = time.monotonic() + 45
    last_report = 0.0
    while time.monotonic() < deadline:
        try:
            candidate = rediscover_port(interface)
            if candidate is None or not is_pixhawk_bootloader(candidate):
                time.sleep(0.02)
                continue
            up = module.uploader(candidate, 115200, [115200])
            up.port.timeout = 0.25
            up.identify()
            up.port.timeout = 2.0
            print(f'Bootloader found on {candidate}', flush=True)
            break
        except Exception as exc:
            if up is not None:
                up.close()
            up = None
            if time.monotonic() - last_report >= 1.0:
                print(f'Waiting for bootloader: {exc}', flush=True)
                last_report = time.monotonic()
            time.sleep(0.05)
    if up is None:
        raise RuntimeError('No bootloader response; no erase attempted')
    try:
        serial = b''.join(up._uploader__getSN(i) for i in range(0, 12, 4)).hex()
        if serial != identity['serial'] or up.board_type != identity['board_id'] or up.fw_maxsize != identity['flash_size']:
            raise RuntimeError('Bootloader identity changed; no erase')
        up.upload(firmware, force=False)
        print('FLASH SUCCESS: official uploader completed CRC verification and reboot', flush=True)
    finally:
        up.close()


if __name__ == '__main__':
    main()
