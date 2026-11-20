"""Repair task-owned extracted modules from pinned archives, then byte-verify."""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if root != Path('/home/liang/rov_firmware_build_20261004'):
        raise RuntimeError('Unexpected isolated build root')
    manifest = json.loads((root / 'build_modules.json').read_text())
    report = []
    for item in manifest:
        destination = root / 'source' / item['path']
        if not destination.resolve().is_relative_to(root / 'source/modules'):
            raise RuntimeError('Module path escaped isolated build')
        archive = root / 'archives' / f"{item['name']}-{item['sha']}.tar.gz"
        prefix = f"{item['name']}-{item['sha']}/"
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            if any(not (m.name == prefix.rstrip('/') or m.name.startswith(prefix)) for m in members):
                raise RuntimeError('Unexpected archive layout')
            if args.repair:
                subprocess.run(['tar', '-xzf', str(archive), '-C', str(destination),
                                '--strip-components=1'], check=True)
            count = 0
            for member in members:
                if not member.isfile():
                    continue
                path = destination / member.name[len(prefix):]
                if not path.resolve().is_relative_to(destination.resolve()):
                    raise RuntimeError('File outside module')
                expected = stream.extractfile(member).read()
                if path.read_bytes() != expected:
                    raise RuntimeError(f'Extracted file mismatch: {path}')
                count += 1
        report.append(dict(item, files_verified=count,
                           archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
        print(f"Verified {item['path']}: {count} files", flush=True)
    subprocess.run(['sync'], check=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
