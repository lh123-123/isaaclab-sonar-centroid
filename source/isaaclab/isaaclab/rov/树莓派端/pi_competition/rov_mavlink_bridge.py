#!/usr/bin/env python3
"""Bidirectional transparent bridge between a Pixhawk serial port and QGC UDP."""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import time

LOGGER = logging.getLogger("rov-mavlink-bridge")
RUNNING = True
MAX_UDP_PER_CYCLE = 32


def forward_datagrams(udp, pixhawk, allowed_host, budget=MAX_UDP_PER_CYCLE):
    """Bound each UDP burst so serial telemetry and shutdown cannot starve.

    Bytes remain transparent: QGC parameter/PID traffic is not filtered by type.
    A partial serial write is a transport failure, never reported as complete.
    """
    sent = ignored = 0
    for _ in range(budget):
        if not RUNNING:
            break
        try:
            payload, source = udp.recvfrom(65535)
        except BlockingIOError:
            break
        if source[0] != allowed_host:
            ignored += 1
            continue
        if payload:
            written = pixhawk.write(payload)
            if written != len(payload):
                raise OSError('Partial serial write; packet not confirmed complete')
            sent += written
    return sent, ignored


def stop(_signum: int, _frame: object) -> None:
    global RUNNING
    RUNNING = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True, help="Pixhawk serial device")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--gcs-ip", default="192.168.1.4")
    parser.add_argument("--udp-port", type=int, default=14550)
    parser.add_argument("--control-port", type=int, default=14551,
                        help="mirror Pixhawk telemetry to PC controller (0 disables)")
    parser.add_argument("--bind-ip", default="0.0.0.0")
    parser.add_argument("--status-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    import serial  # hardware dependency; transport unit tests require no port/library
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp.bind((args.bind_ip, args.udp_port))
    udp.setblocking(False)
    gcs_address = (args.gcs_ip, args.udp_port)
    telemetry_addresses = {gcs_address}
    if args.control_port:
        telemetry_addresses.add((args.gcs_ip, args.control_port))
    LOGGER.info(
        "UDP ready: bind=%s:%d, QGC=%s:%d",
        args.bind_ip,
        args.udp_port,
        args.gcs_ip,
        args.udp_port,
    )

    serial_to_qgc = 0
    qgc_to_serial = 0
    ignored_datagrams = 0
    last_status = time.monotonic()

    while RUNNING:
        try:
            with serial.Serial(
                args.serial,
                args.baud,
                timeout=0,
                write_timeout=1,
                exclusive=True,
            ) as pixhawk:
                LOGGER.info("Pixhawk connected: %s at %d baud", args.serial, args.baud)

                while RUNNING:
                    waiting = pixhawk.in_waiting
                    if waiting:
                        payload = pixhawk.read(min(waiting, 4096))
                        if payload:
                            for destination in telemetry_addresses:
                                udp.sendto(payload, destination)
                            serial_to_qgc += len(payload)

                    sent, ignored = forward_datagrams(udp, pixhawk, args.gcs_ip)
                    qgc_to_serial += sent
                    ignored_datagrams += ignored

                    now = time.monotonic()
                    if now - last_status >= args.status_seconds:
                        LOGGER.info(
                            "traffic: Pixhawk->QGC=%d bytes, QGC->Pixhawk=%d bytes, ignored=%d datagrams",
                            serial_to_qgc,
                            qgc_to_serial,
                            ignored_datagrams,
                        )
                        last_status = now
                    time.sleep(0.002)
        except (serial.SerialException, OSError) as exc:
            if RUNNING:
                LOGGER.error("Pixhawk link unavailable: %s; retrying in 2 seconds", exc)
                time.sleep(2)

    udp.close()
    LOGGER.info("Bridge stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
