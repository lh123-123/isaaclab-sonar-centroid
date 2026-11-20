"""Make a verified deployment archive; does not contact hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepared-name', choices=('prepared_independent_20261004',
                                                  'prepared_independent_20261005',
                                                  'prepared_tolerance_20261005', 'prepared_horizontal_20261007', 'prepared_level_target_20261007',
                                                  'prepared_current_guard_20261008', 'prepared_current_guard_20261008_v2',
                                                  'prepared_gentle_attitude_20261008',
                                                  'prepared_gentle_depth_20261008',
                                                  'prepared_depth_bottom_guard_20261008'),
                        default='prepared_independent_20261005')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    prepared = root / args.prepared_name
    manifest = json.loads((prepared / 'manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        if hashlib.sha256((prepared / 'source' / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Source mismatch: ' + name)
    for name, digest in manifest['baseline_sha256'].items():
        if hashlib.sha256((root / 'baseline_dbe79216' / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Baseline mismatch: ' + name)
    if hashlib.sha256((prepared / 'ardusub_rov_control.patch').read_bytes()).hexdigest() != manifest['patch_sha256']:
        raise RuntimeError('Patch mismatch')
    with tarfile.open(args.output, 'x:gz') as archive:
        archive.add(prepared, arcname=prepared.name)
        archive.add(root / 'baseline_dbe79216', arcname='baseline_dbe79216')
        for script in ('build_control_on_pi.py', 'supervise_build_heat.py', 'inspect_vehicle.py',
                       'guarded_flash.py', 'check_postflash_health.py', 'activate_deadzone.py',
                       'activate_current_guard.py', 'activate_gentle_attitude.py',
                       'verify_independent_modes.py', 'configure_gimbal_direction.py',
                       'launch_isolated_build.py', 'launch_level_target_flash.py',
                       'launch_gentle_attitude_flash.py', 'launch_gentle_depth_flash.py',
                       'launch_depth_bottom_guard_flash.py',
                       'verify_parameter_preservation.py'):
            path = root / script
            if path.is_file():
                archive.add(path, arcname=script)
        # These cache-repair helpers are pinned to the older failed build only.
        if args.prepared_name == 'prepared_independent_20261004':
            for script in ('reuse_verified_build_cache.py', 'refresh_verified_prepared.py'):
                archive.add(root / script, arcname=script)
    print(args.output, hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
