#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authenticated, replay-resistant UDP control receiver."""

from __future__ import annotations

import json
import logging
import socket
import threading

from secure_protocol import (
    DEFAULT_MAX_PACKET_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_control_packet,
)


class CommandHandler:
    """Accept control only from the configured PC and authenticated session."""

    def __init__(
        self,
        host="0.0.0.0",
        port=5000,
        command_callback=None,
        error_callback=None,
        timeout=0.5,
        *,
        shared_key=None,
        allowed_ip=None,
        protocol_version=PROTOCOL_VERSION,
        max_packet_bytes=DEFAULT_MAX_PACKET_BYTES,
        allow_legacy_unsigned=False,
    ):
        self.host = host
        self.port = int(port)
        self.command_callback = command_callback
        self.error_callback = error_callback
        self.timeout = float(timeout)
        self.shared_key = shared_key
        self.allowed_ip = allowed_ip
        self.protocol_version = int(protocol_version)
        self.max_packet_bytes = int(max_packet_bytes)
        self.allow_legacy_unsigned = bool(allow_legacy_unsigned)
        self.running = False
        self.last_msg = None
        self.sock = None
        self.active_session = None
        self.last_seq = -1
        self._sequence_lock = threading.Lock()

        if not self.allow_legacy_unsigned and not self.shared_key:
            raise ValueError("shared_key is required when unsigned packets are disabled")
        if self.max_packet_bytes < 256 or self.max_packet_bytes > 65507:
            raise ValueError("max_packet_bytes must be in 256..65507")

    def _decode_legacy(self, data):
        try:
            msg = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"invalid legacy JSON: {exc}") from exc
        if not isinstance(msg, dict):
            raise ProtocolError("legacy command JSON must be an object")
        return msg, {
            "authenticated": False,
            "session": "legacy-unsigned",
            "seq": -1,
            "session_changed": False,
        }

    def process_datagram(self, data, addr):
        """Validate and dispatch one datagram. Returns True only if accepted."""
        source_ip = str(addr[0])
        if self.allowed_ip and source_ip != self.allowed_ip:
            logging.warning("control packet rejected from unexpected IP %s", source_ip)
            return False
        if not data or len(data) > self.max_packet_bytes:
            logging.warning(
                "control packet rejected: size %d outside 1..%d",
                len(data), self.max_packet_bytes,
            )
            return False

        try:
            if self.allow_legacy_unsigned and not self.shared_key:
                msg, metadata = self._decode_legacy(data)
            else:
                verified = decode_control_packet(
                    data,
                    self.shared_key,
                    protocol_version=self.protocol_version,
                    max_packet_bytes=self.max_packet_bytes,
                )
                with self._sequence_lock:
                    session_changed = verified.session != self.active_session
                    if not session_changed and verified.seq <= self.last_seq:
                        raise ProtocolError(
                            f"replayed/out-of-order seq {verified.seq}; last accepted {self.last_seq}"
                        )
                    self.active_session = verified.session
                    self.last_seq = verified.seq
                msg = verified.payload
                metadata = {
                    "authenticated": True,
                    "session": verified.session,
                    "seq": verified.seq,
                    "session_changed": session_changed,
                }

            metadata["source_ip"] = source_ip
            if self.command_callback:
                self.command_callback(msg, metadata)
            self.last_msg = msg
            return True
        except ProtocolError as exc:
            logging.warning("control packet rejected from %s: %s", source_ip, exc)
            return False
        except Exception as exc:
            logging.exception("authenticated command callback failed")
            if self.error_callback:
                self.error_callback("command_callback", exc)
            return False

    def _udp_listener(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind((self.host, self.port))
            logging.info(
                "secure UDP control listening on %s:%s (allowed PC %s)",
                self.host, self.port, self.allowed_ip,
            )
            self.sock.settimeout(0.2)
            while self.running:
                try:
                    data, addr = self.sock.recvfrom(self.max_packet_bytes + 1)
                    self.process_datagram(data, addr)
                except socket.timeout:
                    continue
                except OSError:
                    if not self.running:
                        break
                    raise
        except Exception as exc:
            logging.exception("secure UDP command listener stopped unexpectedly")
            if self.running and self.error_callback:
                self.error_callback("command_listener", exc)
        finally:
            self.running = False

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(
                target=self._udp_listener, daemon=True, name="SecureCommandUDP"
            ).start()

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass


def start_command_listener(callback, host="0.0.0.0", port=5000, **kwargs):
    handler = CommandHandler(
        host=host,
        port=port,
        command_callback=callback,
        **kwargs,
    )
    handler.start()
    return handler
