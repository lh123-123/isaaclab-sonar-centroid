"""Preserve old task tree, retain compiler path and reuse Waf's dependency cache.

No serial/vehicle access. Waf recompiles changed inputs/dependencies normally.
All old/new source payloads and the old APJ are checked before any move.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path('/home/liang/rov_firmware_build_20261004')
BUNDLE = ROOT / 'control_independent_2030'
OLD = ROOT / 'control_1715/source'
NEW = BUNDLE / 'source'
SAVED = ROOT / 'control_1715/source_before_independent_20261004'
COMMIT = 'dbe792162d06cab66c3475fd5556bf7a120f119e'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for path in (ROOT, BUNDLE, OLD, NEW):
        if path.resolve(strict=True) != path or path.is_symlink():
            raise RuntimeError('Unexpected physical task path')
    if SAVED.exists() or (BUNDLE / 'cache_migration.json').exists():
        raise RuntimeError('Migration already started; inspect before continuing')
    for service in ('rov-mavlink.service', 'rov-video.service'):
        if subprocess.run(['systemctl', 'is-active', '--quiet', service]).returncode == 0:
            raise RuntimeError('Managed bridge/video must remain stopped')
    old_manifest = json.loads((ROOT / 'control_1715/prepared_control_20261004/manifest.json').read_text())
    new_manifest = json.loads((BUNDLE / 'prepared_independent_20261004/manifest.json').read_text())
    for tree, manifest in ((OLD, old_manifest), (NEW, new_manifest)):
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree).decode().strip() != COMMIT:
            raise RuntimeError('Pinned tree commit mismatch')
        for name, expected in manifest['source_sha256'].items():
            if digest(tree / name) != expected:
                raise RuntimeError('Source payload changed: ' + str(tree / name))
    for name, expected in new_manifest['baseline_sha256'].items():
        old_expected = old_manifest['source_sha256'].get(name, expected)
        if digest(OLD / name) != old_expected:
            raise RuntimeError('Old baseline file changed: ' + name)
    old_status = json.loads((ROOT / 'control_1715/build_status.json').read_text())
    old_apj = OLD / 'build/Pixhawk1/bin/ardusub.apj'
    if digest(old_apj) != old_status['apj_sha256']:
        raise RuntimeError('Previous verified APJ mismatch')
    # Only terminate the exact two processes recorded by this task's launch.
    launch = json.loads(sorted(BUNDLE.glob('launched*.json'), key=lambda p: p.stat().st_mtime)[-1].read_text())
    pids = [launch['build_pid'], launch['thermal_pid']]
    expected_args = [str(BUNDLE / 'build_control_on_pi.py').encode(), str(BUNDLE / 'supervise_build_heat.py').encode()]
    for pid, expected in zip(pids, expected_args):
        proc = Path('/proc') / str(pid)
        if proc.stat().st_uid != os.getuid() or expected not in (proc / 'cmdline').read_bytes().split(b'\0'):
            raise RuntimeError('Task process identity mismatch; nothing moved')
    group = os.getpgid(pids[0])
    if group != pids[0] or group <= 1 or group == os.getpgrp():
        raise RuntimeError('Unsafe compiler group')
    os.kill(pids[1], signal.SIGTERM)
    os.killpg(group, signal.SIGCONT)
    os.killpg(group, signal.SIGTERM)
    time.sleep(2)
    processes = subprocess.check_output(['ps', '-eo', 'pgid=,stat=']).decode().splitlines()
    if any(int(line.split()[0]) == group and not line.split()[1].startswith('Z') for line in processes if line.split()):
        raise RuntimeError('Compiler children still alive; nothing moved')
    # Preserve the complete old source AND build. Move only the checked task
    # directories; keep the absolute cache path so Waf UIDs remain valid.
    OLD.rename(SAVED)
    NEW.rename(OLD)
    (OLD / 'build').rename(BUNDLE / 'partial_full_build_preserved')
    shutil.copytree(SAVED / 'build', OLD / 'build', symlinks=True)
    for name, expected in new_manifest['source_sha256'].items():
        if digest(OLD / name) != expected:
            raise RuntimeError('New tree changed during migration')
    evidence = {'old_tree_preserved': str(SAVED), 'prepared_tree': str(OLD),
                'old_apj_sha256': old_status['apj_sha256'], 'new_patch_sha256': new_manifest['patch_sha256'],
                'compiler_dependency_checks_required': True, 'old_build_pids_stopped': pids}
    with (BUNDLE / 'cache_migration.json').open('x') as stream:
        json.dump(evidence, stream, indent=2)
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
