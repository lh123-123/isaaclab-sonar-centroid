"""Thermal pacing for this task's compiler process group, never vehicle I/O."""
import argparse
import os
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    if bundle.parent != Path('/home/liang/rov_firmware_build_20261004') or not bundle.name.startswith('control_'):
        raise RuntimeError('Unexpected task build directory')
    proc = Path('/proc') / str(args.pid)
    def identity():
        command = (proc / 'cmdline').read_bytes().split(b'\0')
        if str(bundle / 'build_control_on_pi.py').encode() not in command:
            raise RuntimeError('Not the exact task build driver')
        fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
        return int(fields[2]), fields[19]  # process group and start time
    group, start = identity()
    if group <= 1 or group == os.getpgrp() or proc.stat().st_uid != os.getuid():
        raise RuntimeError('Unsafe process group/owner')
    paused = False
    try:
        while proc.exists():
            try:
                if identity() != (group, start):
                    raise RuntimeError('Build process identity changed')
            except FileNotFoundError:
                break
            temperature = int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000
            if not paused and temperature >= 82:
                os.killpg(group, signal.SIGSTOP)
                paused = True
                print(f'PAUSE compiler: {temperature:.1f} C', flush=True)
            elif temperature <= 74:
                # Also resumes a compiler that the operator paused before launch.
                os.killpg(group, signal.SIGCONT)
                if paused:
                    print(f'RESUME compiler: {temperature:.1f} C', flush=True)
                paused = False
            time.sleep(3)
    finally:
        if proc.exists() and identity() == (group, start):
            os.killpg(group, signal.SIGCONT)


if __name__ == '__main__':
    main()
