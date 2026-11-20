"""Apply a hash-checked compile correction to this inactive task build only."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/liang/rov_firmware_build_20261004')
BUNDLE = ROOT / 'control_independent_2030'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tree = ROOT / 'control_1715/source'
    current = BUNDLE / 'prepared_independent_20261004'
    incoming = BUNDLE / 'compile_fix/prepared_independent_20261004'
    saved = BUNDLE / 'prepared_before_private_fix'
    if saved.exists():
        raise RuntimeError('Correction already started')
    if json.loads((BUNDLE / 'build_status.json').read_text())['status'] != 'BUILD_FAILED':
        raise RuntimeError('Expected inactive failed build')
    for path in (tree, current, incoming):
        if path.resolve(strict=True) != path:
            raise RuntimeError('Unexpected target path')
    old = json.loads((current / 'manifest.json').read_text())
    new = json.loads((incoming / 'manifest.json').read_text())
    if old['source_commit'] != new['source_commit'] or old['baseline_sha256'] != new['baseline_sha256']:
        raise RuntimeError('Baseline changed')
    changes = [name for name in new['source_sha256'] if old['source_sha256'].get(name) != new['source_sha256'][name]]
    if changes != ['ArduSub/Sub.h'] or set(old['source_sha256']) != set(new['source_sha256']):
        raise RuntimeError('Correction exceeds reviewed private-access fix')
    for name, digest in old['source_sha256'].items():
        if sha(tree / name) != digest:
            raise RuntimeError('Current build source changed: ' + name)
    for name, digest in new['source_sha256'].items():
        if sha(incoming / 'source' / name) != digest:
            raise RuntimeError('Incoming source hash mismatch')
    if sha(incoming / 'ardusub_rov_control.patch') != new['patch_sha256']:
        raise RuntimeError('Incoming patch hash mismatch')
    for service in ('rov-mavlink.service', 'rov-video.service'):
        if subprocess.run(['systemctl', 'is-active', '--quiet', service]).returncode == 0:
            raise RuntimeError('Keep managed services stopped')
    current.rename(saved)
    shutil.copytree(incoming, current)
    for name in changes:
        shutil.copyfile(current / 'source' / name, tree / name)
    for name, digest in new['source_sha256'].items():
        if sha(tree / name) != digest:
            raise RuntimeError('Prepared tree not exact')
    evidence = {'status': 'PREPARED_REBUILD_REQUIRED', 'source_commit': new['source_commit'],
                'patch_sha256': new['patch_sha256'], 'corrected_files': changes}
    (BUNDLE / 'build_status.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
