"""Run the already guarded uploader independently of an SSH connection.

For this deployment only. Requires the operator's fresh actuator-power-off
confirmation. Never bypasses guarded_flash.py's UID, disarm, neutral, board,
capacity, exclusive-serial and official CRC gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('/home/liang/rov_firmware_build_20261004')
BUNDLE = ROOT / 'control_release_20261005'
COMMIT = 'dbe792162d06cab66c3475fd5556bf7a120f119e'
PATCH = '8b32a68c4d8b9da764df97b6d977a5daa096e4309841f46c48bae22dce143383'
PORT = '/dev/serial/by-id/usb-ArduPilot_Pixhawk1_3B0042000450325635393420-if00'


def check_bundle(bundle):
    if bundle.resolve(strict=True) != BUNDLE:
        raise RuntimeError('Unexpected deployment directory')
    built = json.loads((bundle / 'build_status.json').read_text())
    if (built['status'] != 'BUILT_NOT_FLASHED' or built['source_commit'] != COMMIT
            or built['patch_sha256'] != PATCH or built['board_id'] != 9):
        raise RuntimeError('Expected completed release build missing')
    firmware = bundle / 'source/build/Pixhawk1/bin/ardusub.apj'
    if Path(built['apj']) != firmware:
        raise RuntimeError('Unexpected firmware path')
    if hashlib.sha256(firmware.read_bytes()).hexdigest() != built['apj_sha256']:
        raise RuntimeError('Built artifact changed')
    for service in ('rov-mavlink.service', 'rov-video.service'):
        state = subprocess.check_output(['systemctl', 'show', '-p', 'ActiveState',
                                         '--value', service], text=True).strip()
        if state != 'inactive':
            raise RuntimeError('Service is not fully stopped: ' + service)
    temperature = int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000
    if temperature > 78:
        raise RuntimeError('Pi too hot to flash')
    return built, firmware, temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    built, firmware, temperature = check_bundle(bundle)
    result_path = bundle / 'flash_status.json'
    if result_path.exists():
        raise RuntimeError('Flash result already exists; never flash twice implicitly')
    if args.run:
        command = [str(ROOT / 'venv/bin/python'), str(bundle / 'guarded_flash.py'),
                   '--source', str(bundle / 'source'), '--port', PORT,
                   '--backup', str(bundle / 'vehicle_preflash.json'),
                   '--identity', str(ROOT / 'board_identity.json'),
                   '--firmware', str(firmware), '--sha256', built['apj_sha256']]
        code = subprocess.run(command).returncode
        result = {'status': 'FLASH_SUCCESS' if code == 0 else 'FLASH_FAILED',
                  'exit_code': code, 'apj_sha256': built['apj_sha256'],
                  'finished_unix': time.time(),
                  'water_acceptance': False}
        with result_path.open('x') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result), flush=True)
        raise SystemExit(code)
    launch_path = bundle / 'flash_launch.json'
    if launch_path.exists():
        raise RuntimeError('Uploader already launched; inspect it instead of duplicating')
    with (bundle / 'flash.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                  '--bundle', str(bundle), '--run'],
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    launch = {'pid': child.pid, 'initial_temperature_c': temperature,
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'launched_unix': time.time(), 'apj_sha256': built['apj_sha256']}
    with launch_path.open('x') as stream:
        json.dump(launch, stream, indent=2)
    print(json.dumps(launch), flush=True)


if __name__ == '__main__':
    main()
