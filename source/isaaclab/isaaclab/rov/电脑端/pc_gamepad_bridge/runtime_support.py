"""Shared startup guards for the console bridge and compact competition panel."""

import socket
from dataclasses import dataclass


class SingleInstance:
    """OS-owned lock: automatically released on crash, no stale PID files."""

    def __init__(self, port=14559):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            self.socket.bind(("127.0.0.1", port))
            self.socket.listen(1)
        except OSError as exc:
            self.socket.close()
            raise RuntimeError("手柄程序已运行或本地14559端口被占用，请勿重复启动。") from exc

    def close(self):
        self.socket.close()


def neutral_input(state):
    return (max(abs(state.left_x), abs(state.left_y), abs(state.right_x),
                abs(state.right_y)) <= 0.04
            and max(state.left_trigger, state.right_trigger) <= 0.03
            and state.buttons == 0)


@dataclass
class ResumeGate:
    """Require fresh input at neutral for half a second after link loss/start."""
    ready: bool = False
    neutral_since: float | None = None

    def update(self, state, healthy, now):
        if not healthy:
            self.ready = False
            self.neutral_since = None
        elif not self.ready:
            if not neutral_input(state):
                self.neutral_since = None
            elif self.neutral_since is None:
                self.neutral_since = now
            elif now - self.neutral_since >= 0.5:
                self.ready = True
        return self.ready
