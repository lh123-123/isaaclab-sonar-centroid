#!/usr/bin/env python3
"""Read-only Linux memory/NIC sampler. Independent of FC, camera and actuators."""
import argparse
import json
from pathlib import Path
import re
import time


class Sampler:
    def __init__(self, interface='eth0', root=Path('/')):
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,32}', interface):
            raise ValueError('Invalid interface')
        self.interface, self.root, self.previous = interface, root, None

    def sample(self, now=None):
        now = time.monotonic() if now is None else now
        memory = {}
        for line in (self.root / 'proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            memory[key] = int(value.strip().split()[0])
        total, available = memory['MemTotal'], memory.get('MemAvailable')
        # MemAvailable includes reclaimable cache; do not call all cache "used".
        used = (100 * (total - available) / total
                if available is not None and total > 0 and 0 <= available <= total else None)
        nic = self.root / 'sys/class/net' / self.interface
        try:
            speed = int((nic / 'speed').read_text().strip())
            if speed <= 0 or (nic / 'carrier').read_text().strip() != '1':
                speed = None
        except (OSError, ValueError):
            speed = None
        counters = tuple(int((nic / 'statistics' / name).read_text())
                         for name in ('rx_bytes', 'tx_bytes'))
        rx = tx = None
        if self.previous is not None:
            stamp, old = self.previous
            dt = now - stamp
            if 0 < dt <= 5 and all(n >= o for n, o in zip(counters, old)):
                rx, tx = (8 * (n - o) / dt / 1e6 for n, o in zip(counters, old))
        self.previous = (now, counters)
        try:
            temp = int((self.root / 'sys/class/thermal/thermal_zone0/temp').read_text()) / 1000
        except (OSError, ValueError):
            temp = None
        return {'schema': 1, 'interface': self.interface, 'memory_percent': used,
                'memory_total_mb': total / 1024, 'temperature_c': temp,
                'link_mbps': speed, 'rx_mbps': rx, 'tx_mbps': tx,
                # Full duplex has independent RX/TX capacity. Do not add them.
                'link_percent': 100 * max(rx, tx) / speed if speed and rx is not None else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interface', default='eth0')
    parser.add_argument('--stream', action='store_true')
    args = parser.parse_args()
    sampler = Sampler(args.interface)
    while True:
        print(json.dumps(sampler.sample(), allow_nan=False), flush=True)
        if not args.stream:
            return
        time.sleep(1)


if __name__ == '__main__':
    main()
