"""中文字体配置与**真正有效**的缺字校验（给本目录的出图脚本共用）。

背景（踩过的坑，务必不要再犯）
----------------------------
1. matplotlib 的字体缓存（``~/.cache/matplotlib/fontlist-*.json``）可能陈旧。陈旧时
   ``findfont`` 会静默回退到 DejaVu Sans —— 而 DejaVu **没有中文字形**，图上中文直接消失。
   本机就踩了这个坑：缓存是 2026-01-19 的，Noto CJK 完全不在里面。

2. ``NotoSansCJK-Regular.ttc`` 是**字体集合**，matplotlib 的 ``addfont`` 只注册它的
   **第一个 face**，而第一个 face 是 **JP**。所以 ``"Noto Sans CJK SC"`` 这个名字在
   matplotlib 里**永远找不到**，只能用 ``"Noto Sans CJK JP"``。Noto Sans CJK 是泛 CJK
   字体，JP face 的汉字覆盖是完整的，只是少数汉字用日式字形。

3. **校验必须走 ``logging``，不能用 ``warnings``。** matplotlib 的字体回退和缺字提示都是
   ``logging.warning`` 发出的。早先版本只把 ``warnings`` 升级为异常，结果检查**必然通过**，
   等于没查 —— 那次"验证通过"是假的。``assert_cjk_ok()`` 同时做三件事：确认
   ``findfont`` 真的解析到含 CJK 的字体文件、捕获 ``logging`` 里的缺字告警、并用
   墨水像素数确认字形真的被画出来了。
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager as fm  # noqa: E402

# .ttc 集合；matplotlib 只会读到它的第一个 face（JP）。
_CJK_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# matplotlib 能通过名字找到的那个 face。SC 名字在 matplotlib 下不存在，写上只为
# 万一将来系统装了独立 SC 文件时能优先命中。
_CJK_FAMILY = "Noto Sans CJK JP"


def setup_cjk_font() -> str:
    """把 CJK 字体注册进 matplotlib 并设为默认 sans-serif。返回实际使用的 family 名。"""
    if Path(_CJK_TTC).is_file():
        # 即使缓存里没有，这一步也会当场把它加进来。对 .ttc 只生效于第一个 face。
        fm.fontManager.addfont(_CJK_TTC)
    resolved = fm.findfont(fm.FontProperties(family=_CJK_FAMILY), fallback_to_default=False)
    matplotlib.rcParams["font.sans-serif"] = [  # type: ignore[attr-defined]
        _CJK_FAMILY,
        "Noto Sans CJK SC",   # 将来若装了独立 SC 文件则优先
        "WenQuanYi Zen Hei",
        "DejaVu Sans",
    ]
    # 负号：CJK 字体的 U+2212 常常缺字形，必须改用 ASCII 连字符渲染。
    matplotlib.rcParams["axes.unicode_minus"] = False  # type: ignore[attr-defined]
    return resolved


class _GlyphCapture(logging.Handler):
    """捕获 matplotlib 的「缺字」告警。"""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if "missing" in msg.lower() or "not found" in msg.lower():
            self.messages.append(msg)


def assert_cjk_ok(sample: str, *, verbose: bool = True) -> None:
    """确认中文字形真的能渲染。三层检查，任一失败就抛异常。

    用法：在 ``savefig`` **之前**调用，传入图里出现过的一段中文。

    1. ``findfont`` 必须解析到含 "CJK" 的字体文件（而不是 DejaVu 回退）。
    2. 渲染期间 matplotlib 的 ``logging`` 不得报缺字/找不到字体。
    3. 渲染结果必须真的有墨水像素 —— 字形缺失时文字区域会几乎全白。
    """
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    resolved = setup_cjk_font()
    if "CJK" not in Path(resolved).name and "CJK" not in resolved:
        raise RuntimeError(f"findfont 没解析到 CJK 字体，回退到了 {resolved}")

    cap = _GlyphCapture()
    root = logging.getLogger("matplotlib")
    root.addHandler(cap)
    prev_level = root.level
    root.setLevel(logging.WARNING)
    try:
        fig = Figure(figsize=(6, 1.2), dpi=100)
        FigureCanvasAgg(fig)
        fig.text(0.02, 0.4, sample, fontsize=16)
        fig.canvas.draw()  # type: ignore[attr-defined]
        buf = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
    finally:
        root.removeHandler(cap)
        root.setLevel(prev_level)

    if cap.messages:
        raise RuntimeError("渲染期 matplotlib 报告了字体问题: " + " | ".join(cap.messages[:3]))

    # 墨水检测：背景是白的，字应该是深色。
    ink = int((buf[..., :3].min(axis=2) < 128).sum())
    if ink < 40:
        raise RuntimeError(f"文字区域墨水像素仅 {ink} 个 —— 字形很可能没画出来")

    if verbose:
        print(f"[cjk] 字体 OK -> {resolved}")
        print(f"[cjk] 墨水像素 {ink} 个，无缺字告警")
