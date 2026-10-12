"""RGB 叠帧 vs GT 光流叠帧：运动到底住在哪里？（中文版）

用真实的 4 帧叠帧（`logs/auv_v4_actor_frames_heading/step08_frame0..3.png`，来自一次
v4-cam 运行）。光流叠帧由**同一段运动**导出：先量出每帧里目标色块的质心，换算到 64×64
坐标，再用环境自己的公式做差分（`du = (u_t − u_{t-1}) / W`）。所以两块展示的是同一段轨迹，
唯一变量就是表示形式。

诚实说明：真实渲染是 512×512，而网络会降采样到 64×64，所以量到的运动被 512 网格量化到
1/8 像素。环境本身的 GT 光流是从 3D 目标位置解析算出来的、不经过像素，因此它的 `du` 是
平滑的。这张图是"能测到的最忠实版本"，不是逐位复现 —— 写在这里而不是藏起来。

    python3 scripts/make_flow_stack_vs_rgb_stack.py
"""

from __future__ import annotations

import sys

import math
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 中文字体：必须走这个模块。它的 addfont 不依赖 matplotlib 的字体缓存，并会
# 用 logging + 墨水像素做**真的**缺字校验（早先只查 warnings 是无效的）。
# 注意只能按 "Noto Sans CJK JP" 这个名字找 —— .ttc 集合里 matplotlib 只暴露第一个 face。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cjk_font import assert_cjk_ok, setup_cjk_font  # noqa: E402

setup_cjk_font()

OUT = Path("logs/auv_flow_compare")
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("logs/auv_v4_actor_frames_heading")
STEP = "step08"

W = H = 64
FX = W * 0.5 / math.tan(math.radians(90.0) * 0.5)      # 32.0 px
CX = CY = 0.5 * (W - 1)
TARGET_LEN = 0.8 * 0.8                                  # 0.64 m


def disp_width(s: str) -> int:
    """终端/等宽排版下的显示宽度：CJK 全角字符算 2 列。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def pad(s: str, width: int) -> str:
    """按显示宽度补空格。不能用 f-string 的 `:<n` —— 那按字符数算，中文会错位。"""
    return s + " " * max(0, width - disp_width(s))


frames512 = [np.array(Image.open(SRC / f"{STEP}_frame{i}.png")) for i in range(4)]
frames64 = [np.array(Image.fromarray(f).resize((W, H), Image.BILINEAR)) for f in frames512]


def orange_mask(a: np.ndarray) -> np.ndarray:
    r, g, b = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    return (r > 100) & (r > g) & (g > b)


centroids, radii = [], []
for a in frames512:
    m = orange_mask(a)
    ys, xs = np.where(m)
    centroids.append((xs.mean(), ys.mean()))
    radii.append(math.sqrt(m.sum() / math.pi))

print("=" * 84)
print(f"真实 RGB 叠帧  ({STEP}, 4 帧, 512x512 -> 网络看到 64x64)")
print("=" * 84)
print(f"{'帧':>4} {'质心@512':>20} {'质心@64':>18} {'半径@64':>10} {'目标像素@64':>12}")
for i, ((cx, cy), rr) in enumerate(zip(centroids, radii)):
    m64 = np.array(Image.fromarray((orange_mask(frames512[i]) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) > 127
    print(f"{i:>4} {f'({cx:7.1f},{cy:7.1f})':>20} {f'({cx/8:6.2f},{cy/8:6.2f})':>18} {rr/8:>10.2f} {m64.sum():>12}")

# --- 用实测运动生成光流叠帧，公式取自环境
u64 = np.array([c[0] for c in centroids]) / 8.0
v64 = np.array([c[1] for c in centroids]) / 8.0
du = np.concatenate([[0.0], np.diff(u64) / W])
dv = np.concatenate([[0.0], np.diff(v64) / H])
# 半径：环境用 rad = 0.5*L/dist*fx。这里把实测色块半径反推成距离，使画出的圆盘
# 与环境的同一关系式一致，而不是随手取个数。
dist_est = 0.5 * TARGET_LEN * FX / (np.array(radii) / 8.0)
rad_px = np.clip(0.5 * TARGET_LEN / dist_est * FX, 1.0, 12.0)

print()
print(f"{'帧':>4} {'u@64':>8} {'v@64':>8} {'du':>10} {'dv':>10} {'估计距离':>10} {'rad px':>8}")
for i in range(4):
    print(f"{i:>4} {u64[i]:>8.2f} {v64[i]:>8.2f} {du[i]:>+10.5f} {dv[i]:>+10.5f} {dist_est[i]:>9.2f}m {rad_px[i]:>8.2f}")

print()
print(f"  整个叠帧里最大 |du| : {np.abs(du).max():.5f}   (= 64 格中的 {np.abs(du).max()*W:.2f} px)")
print(f"  整个叠帧里最大 |dv| : {np.abs(dv).max():.5f}   (= 64 格中的 {np.abs(dv).max()*H:.2f} px)")


def flow_frame(u: float, v: float, rad: float, du_v: float, dv_v: float):
    ys = np.arange(H, dtype=np.float32).reshape(H, 1)
    xs = np.arange(W, dtype=np.float32).reshape(1, W)
    mask = (xs - u) ** 2 + (ys - v) ** 2 <= rad * rad
    return np.where(mask, du_v, 0.0).astype(np.float32), mask


flow_u, flow_masks = [], []
for i in range(4):
    fu, fm = flow_frame(u64[i], v64[i], rad_px[i], du[i], dv[i])
    flow_u.append(fu)
    flow_masks.append(fm)

# --- RGB 帧间差：卷积网络必须「自己学会算」的那个量
diffs = [np.abs(frames64[i].astype(np.float32) - frames64[i - 1].astype(np.float32)).mean(axis=2)
         for i in range(1, 4)]

# ================================================================ 出图
fig = plt.figure(figsize=(16.5, 11.0))
gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.0, 1.0, 0.62], hspace=0.36, wspace=0.13)

# ---------------- 第 1 行：RGB 叠帧
for i in range(4):
    ax = fig.add_subplot(gs[0, i])
    a = frames512[i].copy()
    m = orange_mask(frames512[i])
    ys, xs = np.where(m)
    if len(xs):
        ax.add_patch(plt.Rectangle((xs.min() - 8, ys.min() - 8), xs.max() - xs.min() + 16,
                                   ys.max() - ys.min() + 16, fill=False, ec="#00e5ff", lw=1.8))
    ax.imshow(a)
    ax.set_xticks([]); ax.set_yticks([])
    tag = "最旧" if i == 0 else ("最新" if i == 3 else f"t-{3-i}")
    ax.set_title(f"RGB 第 {i} 帧（{tag}）\n每帧 3 通道，共 12 通道", fontsize=9.5)
fig.text(0.5, 0.968, "1.  RGB 叠帧 —— 每帧一张照片。运动在哪？哪里都不在。",
         ha="center", fontsize=12.5, fontweight="bold")

# ---------------- 第 2 行：RGB 帧间差
for i in range(3):
    ax = fig.add_subplot(gs[1, i])
    im = ax.imshow(diffs[i], cmap="inferno", vmin=0, vmax=36)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"|第{i+1}帧 − 第{i}帧|   最大={diffs[i].max():.0f}", fontsize=9.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
ax = fig.add_subplot(gs[1, 3])
ax.axis("off")
ax.text(0.0, 0.95,
        "这一行不是观测的一部分。\n\n"
        "网络必须自己学会算出它 ——\n"
        "一次跨通道的相减，\n"
        "而这不是卷积擅长的事。\n\n"
        "（实测信噪比 5.6~13.5，\n"
        "说明信号是干净的，\n"
        "只是要花力气去得到。）",
        va="top", fontsize=9.5, color="#b03000")
fig.text(0.5, 0.718, "2.  网络必须自己从第 1 行「算出来」的东西（运动是隐含的、跨通道的）",
         ha="center", fontsize=12.5, fontweight="bold", color="#b03000")

# ---------------- 第 3 行：光流叠帧
du_max = max(np.abs(du).max(), 1e-6)
for i in range(4):
    ax = fig.add_subplot(gs[2, i])
    im = ax.imshow(flow_u[i], cmap="inferno", vmin=-du_max, vmax=du_max, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"光流第 {i} 帧\ndu = {du[i]:+.5f}", fontsize=9.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
fig.text(0.5, 0.458, "3.  GT 光流叠帧 —— 运动就是像素值本身。不需要做任何相减。",
         ha="center", fontsize=12.5, fontweight="bold", color="#0a6b2e")

# ---------------- 第 4 行：汇总
ax = fig.add_subplot(gs[3, :])
ax.axis("off")
maxpx = np.abs(du).max() * W
summary = "\n".join([
    "同一段 4 帧轨迹，两种表示形式",
    "─" * 100,
    "",
    f"{pad('形状', 12)}{pad('（12, 64, 64）uint8 = 49,152 字节', 40)}（8, 64, 64）float32 = 131,072 字节",
    f"{pad('每帧内容', 12)}{pad('一张照片（195 种颜色）', 40)}5 个数画成的一个圆盘",
    f"{pad('运动在哪', 12)}{pad('哪里都不在，必须自己推断', 40)}就在圆盘的像素值里",
    f"{pad('网络的活', 12)}{pad('找亮斑 → 估大小 → 自己学会做帧间减法', 40)}找到圆盘 → 读三个数",
    "─" * 100,
    f"注意：整段叠帧里最大的帧间位移只有 {maxpx:.2f} 像素（64×64 下）—— 亚像素漂移。"
    "RGB 一侧要从一张照片里把它恢复出来，光流一侧直接写着。",
])
ax.text(0.0, 1.0, summary, va="top", ha="left", fontsize=9, family="sans-serif")

assert_cjk_ok("同一段 4 帧轨迹，两种表示形式 —— 运动在哪，圆盘的像素值，帧间位移")
fig.savefig(OUT / "flow_stack_vs_rgb_zh.png", dpi=112, bbox_inches="tight", facecolor="white")
print(f"\n图已保存 -> {OUT / 'flow_stack_vs_rgb_zh.png'}")
