"""Detached guarded upload for the gentle roll/pitch and depth release."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/liang/rov_firmware_build_20261004')
BUNDLE = ROOT / 'control_gentle_depth_20261008'
PREPARED = 'prepared_gentle_depth_20261008'
PORT = '/dev/serial/by-id/usb-ArduPilot_Pixhawk1_3B0042000450325635393420-if00'


def main():
    run = sys.argv[1:] == ['--run']
    if sys.argv[1:] not in ([], ['--run']):
        raise RuntimeError('Unexpected arguments')
    manifest = json.loads((BUNDLE / PREPARED / 'manifest.json').read_text())
    built = json.loads((BUNDLE / 'build_status.json').read_text())
    firmware = BUNDLE / 'source/build/Pixhawk1/bin/ardusub.apj'
    if (built['status'] != 'BUILT_NOT_FLASHED' or built['board_id'] != 9 or
            built['source_commit'] != 'dbe792162d06cab66c3475fd5556bf7a120f119e' or
            built['patch_sha256'] != manifest['patch_sha256'] or Path(built['apj']) != firmware or
            hashlib.sha256(firmware.read_bytes()).hexdigest() != built['apj_sha256']):
        raise RuntimeError('Build artifact/manifest mismatch')
    for name, expected in manifest['source_sha256'].items():
        if hashlib.sha256((BUNDLE / 'source' / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Source changed: ' + name)
    for service in ('rov-video.service', 'rov-mavlink.service'):
        state = subprocess.check_output(
            ['systemctl', 'show', '-p', 'ActiveState', '--value', service], text=True).strip()
        if state != 'inactive':
            raise RuntimeError('Service must be stopped: ' + service)
    temperature = int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000
    if temperature > 78:
        raise RuntimeError('Too hot to flash')
    result_path = BUNDLE / 'flash_status.json'
    if result_path.exists():
        raise RuntimeError('Flash already attempted; do not retry implicitly')
    if run:
        command = [str(ROOT / 'venv/bin/python'), str(BUNDLE / 'guarded_flash.py'),
                   '--source', str(BUNDLE / 'source'), '--port', PORT,
                   '--backup', str(BUNDLE / 'vehicle_preflash.json'),
                   '--identity', str(ROOT / 'board_identity.json'),
                   '--firmware', str(firmware), '--sha256', built['apj_sha256']]
        code = subprocess.run(command).returncode
        record = {'status': 'FLASH_SUCCESS' if code == 0 else 'FLASH_FAILED',
                  'exit_code': code, 'apj_sha256': built['apj_sha256'],
                  'finished_unix': time.time(), 'water_acceptance': False}
        with result_path.open('x') as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps(record), flush=True)
        raise SystemExit(code)
    launch_path = BUNDLE / 'flash_launch.json'
    if launch_path.exists():
        raise RuntimeError('Uploader already launched; inspect state first')
    with (BUNDLE / 'flash.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--run'],
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record = {'pid': child.pid, 'initial_temperature_c': temperature,
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'launched_unix': time.time(), 'apj_sha256': built['apj_sha256']}
    with launch_path.open('x') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
