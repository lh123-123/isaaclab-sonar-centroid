"""Read-only RTP/H264 reception test. Does not decode video or claim latency."""
import argparse
import ipaddress
import json
from pathlib import Path
import socket
import struct
import time


def rtp_packet(data):
    if len(data) < 12 or data[0] >> 6 != 2 or data[1] & 127 != 96:
        return None
    header = 12 + (data[0] & 15) * 4
    if len(data) < header:
        return None
    if data[0] & 16:
        if len(data) < header + 4:
            return None
        header += 4 + struct.unpack_from('!H', data, header+2)[0] * 4
    padding = data[-1] if data[0] & 32 else 0
    if data[0] & 32 and (padding == 0 or padding > len(data) - header):
        return None
    end = len(data) - padding
    if header >= end:
        return None
    sequence, timestamp, ssrc = struct.unpack_from('!HII', data, 2)
    payload = data[header:end]
    nal = payload[0] & 31
    types = set()
    if nal == 24:  # STAP-A
        offset = 1
        while offset + 2 <= len(payload):
            length = struct.unpack_from('!H', payload, offset)[0]
            offset += 2
            if length < 1 or offset + length > len(payload):
                return None
            types.add(payload[offset] & 31)
            offset += length
        if offset != len(payload) or not types:
            return None
    elif nal == 28:
        if len(payload) < 3:
            return None
        if payload[1] & 128:
            types.add(payload[1] & 31)
    else:
        types.add(nal)
    return sequence, timestamp, ssrc, bool(data[1] & 128), types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind', default='192.168.1.4')
    parser.add_argument('--source', default='192.168.1.3')
    parser.add_argument('--port', type=int, default=5600)
    parser.add_argument('--seconds', type=float, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for address in (args.bind, args.source):
        ipaddress.IPv4Address(address)
    if not 1 <= args.seconds <= 60 or not 1024 <= args.port <= 65535:
        raise ValueError('Invalid sample duration/port')
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite reception evidence')
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    udp.bind((args.bind, args.port))  # Do not steal/share the QGC video socket.
    udp.settimeout(0.5)
    start = time.monotonic()
    packets = frames = size = gaps = reordered = 0
    nal_types = set()
    previous = ssrc = None
    try:
        while time.monotonic() - start < args.seconds:
            try:
                data, address = udp.recvfrom(65535)
            except socket.timeout:
                continue
            if address[0] != args.source:
                continue
            packet = rtp_packet(data)
            if packet is None:
                continue
            seq, _stamp, stream, marker, types = packet
            if ssrc is None:
                ssrc = stream
            if stream != ssrc:
                continue
            if previous is not None:
                delta = (seq-previous) & 65535
                if 0 < delta < 32768:
                    gaps += delta-1
                else:
                    reordered += 1
            if previous is None or 0 < ((seq-previous) & 65535) < 32768:
                previous = seq
            packets += 1
            frames += int(marker)
            size += len(data)
            nal_types |= types
        duration = time.monotonic()-start
        report = {'duration_s': duration, 'packets': packets, 'frame_markers': frames,
                  'rtp_marker_fps': frames/duration, 'received_mbps': size*8/duration/1e6,
                  'sequence_gap_estimate': gaps, 'reordered_or_duplicate': reordered,
                  'h264_nal_types': sorted(nal_types), 'decoded_picture_verified': False,
                  'end_to_end_latency_ms': None}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report, indent=2))
        if packets == 0 or not {7, 8, 5} <= nal_types:
            raise SystemExit('No complete SPS/PPS/IDR reception evidence')
    finally:
        udp.close()


if __name__ == '__main__':
    main()
