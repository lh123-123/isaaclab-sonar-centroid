"""Install pinned official source archives into this task's isolated build only."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if root != Path('/home/liang/rov_firmware_build_20261004'):
        raise RuntimeError('Unexpected build root')
    manifest = json.loads((root / 'build_modules.json').read_text())
    installed = []
    for item in manifest:
        archive = root / 'archives' / f"{item['name']}-{item['sha']}.tar.gz"
        destination = root / 'source' / item['path']
        if not destination.resolve().is_relative_to(root / 'source' / 'modules'):
            raise RuntimeError('Module outside isolated source tree')
        previous = destination.with_name(destination.name + '.incomplete_git_clone')
        if previous.exists():
            raise RuntimeError('Previous archived module exists; refusing replacement')
        staging = root / 'extracted' / item['name']
        staging.mkdir(parents=True, exist_ok=False)
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            prefix = f"{item['name']}-{item['sha']}/"
            if any(not (m.name == prefix.rstrip('/') or m.name.startswith(prefix)) for m in members):
                raise RuntimeError('Unexpected archive layout')
            stream.extractall(staging, filter='data')
        unpacked = staging / prefix.rstrip('/')
        if destination.exists():
            destination.rename(previous)
        destination.parent.mkdir(parents=True, exist_ok=True)
        unpacked.rename(destination)
        installed.append(dict(item, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
        print(f"Installed {item['path']} at {item['sha']}", flush=True)
    with (root / 'module_archive_manifest.json').open('x') as stream:
        json.dump(installed, stream, indent=2)


if __name__ == '__main__':
    main()
