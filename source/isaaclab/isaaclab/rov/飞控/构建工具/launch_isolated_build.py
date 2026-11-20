"""Start only this task's compiler + thermal supervisor, no vehicle I/O."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    root = Path('/home/liang/rov_firmware_build_20261004')
    approved = {'control_independent_2030': 'prepared_independent_20261004',
                'control_release_20261005': 'prepared_independent_20261005',
                'control_tolerance_20261005': 'prepared_tolerance_20261005',
                'control_horizontal_20261007': 'prepared_horizontal_20261007',
                'control_level_target_20261007': 'prepared_level_target_20261007'}
    approved['control_current_guard_20261008'] = 'prepared_current_guard_20261008'
    approved['control_current_guard_20261008_v2'] = 'prepared_current_guard_20261008_v2'
    approved['control_gentle_attitude_20261008'] = 'prepared_gentle_attitude_20261008'
    approved['control_gentle_depth_20261008'] = 'prepared_gentle_depth_20261008'
    approved['control_depth_bottom_guard_20261008'] = 'prepared_depth_bottom_guard_20261008'
    approved['control_full_trigger_release_20261013'] = 'prepared_full_trigger_release_20261013'
    approved['control_full_trigger_release_safe_20261013'] = 'prepared_full_trigger_release_safe_20261013'
    if bundle.parent != root or bundle.name not in approved:
        raise RuntimeError('Unexpected build target')
    prior_launches = sorted(bundle.glob('launched*.json'))
    if prior_launches:
        if not args.resume:
            raise RuntimeError('Build already launched; inspect state, never duplicate it')
        # Check retries too, not just the original launch. A second retry must
        # never start another compiler beside the latest supervised build.
        for prior_path in prior_launches:
            prior = json.loads(prior_path.read_text())
            for key in ('build_pid', 'thermal_pid'):
                try:
                    os.kill(prior[key], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError('Previous build or supervisor still alive: ' + str(prior_path))
    for service in ('rov-video.service', 'rov-mavlink.service'):
        if subprocess.run(['systemctl', 'is-active', '--quiet', service]).returncode == 0:
            raise RuntimeError('Stop video and bridge before isolated build')
    temp = int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000
    if temp > 78:
        raise RuntimeError(f'Pi still too hot to start: {temp:.1f} C')
    throttled = subprocess.check_output(['vcgencmd', 'get_throttled'], text=True).strip()
    if int(throttled.split('=', 1)[1], 16) & 0xF:
        raise RuntimeError('Current Pi undervoltage or throttling: ' + throttled)
    suffix = '_' + str(int(time.time())) if args.resume else ''
    command = ['python3', str(bundle / 'build_control_on_pi.py'), '--bundle', str(bundle),
               '--prepared-name', approved[bundle.name], '--jobs', str(args.jobs)]
    if args.resume:
        command.append('--resume')
    if (bundle / 'cache_migration.json').is_file():
        command.append('--verified-cache-path')
    with (bundle / ('launcher' + suffix + '.log')).open('x') as log:
        build = subprocess.Popen(command, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    with (bundle / ('thermal' + suffix + '.log')).open('x') as log:
        supervisor = subprocess.Popen(['python3', str(bundle / 'supervise_build_heat.py'), '--pid', str(build.pid),
            '--bundle', str(bundle)], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    result = {'build_pid': build.pid, 'thermal_pid': supervisor.pid, 'initial_temperature_c': temp}
    (bundle / ('launched' + suffix + '.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
