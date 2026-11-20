"""One-shot guarded flash of the isolated full-trigger-release candidate.

Never auto-run. The previous firmware and parameter backup must exist first.
"""
import subprocess
from pathlib import Path

import launch_depth_bottom_guard_flash as guarded_launcher


ROOT = Path('/home/liang/rov_firmware_build_20261004')
guarded_launcher.BUNDLE = ROOT / 'control_full_trigger_release_safe_20261013'
guarded_launcher.PREPARED = 'prepared_full_trigger_release_safe_20261013'
# The shared launcher respawns Path(__file__) for its guarded --run phase.
# Point that child at this wrapper, not the older release's entry point.
guarded_launcher.__file__ = __file__


def main():
    # Flashing while the Pi is currently undervolted or throttled risks
    # interrupting an erase/program cycle. Historic flags are diagnostic only.
    state = subprocess.check_output(['vcgencmd', 'get_throttled'], text=True).strip()
    if int(state.split('=', 1)[1], 16) & 0xF:
        raise RuntimeError('Current power or thermal fault; flash aborted: ' + state)
    guarded_launcher.main()


if __name__ == '__main__':
    main()
