"""Read, apply, and independently verify the native roll/pitch tolerance."""

from __future__ import annotations

import math


PARAMETER = "ROV_T_ERR"
MAX_TOLERANCE_DEG = 5.0
PARAM_TYPE_REAL32 = 9


def parse_tolerance(value: str | float) -> float:
    try:
        tolerance = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("稳姿允许误差请输入0～5之间的角度") from exc
    if not math.isfinite(tolerance) or not 0.0 <= tolerance <= MAX_TOLERANCE_DEG:
        raise ValueError("稳姿允许误差必须在0～5°之间；0表示关闭容差")
    return tolerance


class AttitudeTolerance:
    def __init__(self, target_system: int = 1) -> None:
        self.target_system = int(target_system)
        self.actual: float | None = None
        self.phase: str | None = None
        self.notice = "等待飞控参数读回；新固件默认±0.5°"
        self.deadline = 0.0
        self.expected: float | None = None
        self.link = None

    @property
    def busy(self) -> bool:
        return self.phase is not None

    def read(self, link, now: float) -> None:
        if self.busy:
            raise ValueError("稳姿误差参数正在读写")
        self.link = link
        self.phase = "read"
        self.deadline = now + 4.0
        self.notice = "正在读取飞控稳姿允许误差…"
        link.mav.param_request_read_send(self.target_system, 1, PARAMETER.encode("ascii"), -1)

    def apply(self, link, modes, now: float, value: str | float, *, stopped: bool) -> None:
        requested = parse_tolerance(value)
        if self.busy:
            raise ValueError("稳姿误差参数正在读写")
        if self.actual is None:
            raise ValueError("尚未读到飞控 ROV_T_ERR；请确认已烧入支持此功能的新固件")
        if (not stopped or not modes.fresh(now) or modes.armed or modes.mode != 19
                or modes.pending is not None):
            raise ValueError("先上锁、切手动、停止手柄控制和电机测试，再应用稳姿允许误差")
        self.link = link
        self.expected = requested
        self.phase = "write"
        self.deadline = now + 4.0
        self.notice = f"正在写入±{requested:g}°，等待飞控应答…"
        link.mav.param_set_send(self.target_system, 1, PARAMETER.encode("ascii"),
                                requested, PARAM_TYPE_REAL32)

    def observe(self, message, now: float) -> None:
        if (message.get_type() != "PARAM_VALUE" or
                message.get_srcSystem() != self.target_system or
                message.get_srcComponent() != 1):
            return
        name = message.param_id
        if isinstance(name, bytes):
            name = name.decode("ascii", "ignore")
        if str(name).rstrip("\x00") != PARAMETER:
            return
        if not self.busy:
            return
        value = float(message.param_value)
        if message.param_type != PARAM_TYPE_REAL32 or not math.isfinite(value) or not 0 <= value <= 5:
            self.fail("飞控返回的稳姿误差参数类型或数值异常")
            return
        if self.phase == "write":
            if self.expected is None or abs(value - self.expected) > 0.001:
                self.fail(f"飞控未接受设定值，回传为±{value:g}°")
                return
            self.phase = "verify"
            self.deadline = now + 4.0
            self.notice = "飞控已应答，正在独立读回核验…"
            self.link.mav.param_request_read_send(self.target_system, 1,
                                                  PARAMETER.encode("ascii"), -1)
            return
        if self.phase == "verify" and (self.expected is None or abs(value - self.expected) > 0.001):
            self.fail(f"独立读回不匹配：±{value:g}°")
            return
        self.actual = value
        self.phase = None
        self.notice = f"飞控已读回：横滚/俯仰允许误差 ±{value:g}°"

    def tick(self, modes, now: float) -> None:
        if self.phase in ("write", "verify") and modes.fresh(now) and modes.armed:
            self.fail("读写期间飞控解锁，结果未确认")
        elif self.busy and now >= self.deadline:
            if self.phase == "read":
                self.fail("飞控未返回 ROV_T_ERR；当前固件可能尚不支持")
            else:
                self.fail("飞控参数写入或独立读回超时，结果未确认")

    def fail(self, reason: str) -> None:
        self.phase = None
        self.notice = reason
