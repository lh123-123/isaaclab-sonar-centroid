"""Explicit second one-shot upload after a confirmed no-erase bootloader miss."""
import json
import subprocess
from pathlib import Path

import launch_depth_bottom_guard_flash as guarded_launcher


ROOT = Path('/home/liang/rov_firmware_build_20261004')
PREVIOUS = ROOT / 'control_full_trigger_release_safe_20261013'
guarded_launcher.BUNDLE = ROOT / 'control_full_trigger_release_safe_retry_20261013'
guarded_launcher.PREPARED = 'prepared_full_trigger_release_safe_20261013'
guarded_launcher.__file__ = __file__


def main():
    previous = json.loads((PREVIOUS / 'flash_status.json').read_text())
    log = (PREVIOUS / 'flash.log').read_text(errors='replace')
    if (previous.get('status') != 'FLASH_FAILED' or previous.get('exit_code') != 1
            or 'No bootloader response; no erase attempted' not in log
            or 'Bootloader found' in log or 'Erase  :' in log):
        raise RuntimeError('Previous attempt is not proved no-erase; no retry')
    after = json.loads((PREVIOUS / 'after_failed_attempt.json').read_text())
    before = json.loads((guarded_launcher.BUNDLE / 'vehicle_preflash.json').read_text())
    if (before['parameter_count'] != after['parameter_count'] or
            before['autopilot_version']['uid2'] != after['autopilot_version']['uid2'] or
            len(after['parameters']) != after['parameter_count']):
        raise RuntimeError('Complete original-controller readback missing')
    state = subprocess.check_output(['vcgencmd', 'get_throttled'], text=True).strip()
    if int(state.split('=', 1)[1], 16) & 0xF:
        raise RuntimeError('Current power or thermal fault; retry aborted: ' + state)
    guarded_launcher.main()


if __name__ == '__main__':
    main()
