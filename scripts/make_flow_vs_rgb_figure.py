"""相机 RGB 帧 vs GT 光流栅格 —— 并排对比图（中文版）。

不启动 Isaac Sim：光流栅格是解析量（它就是"5 个数画进一个圆盘"），可以按
`auv_tracking_v6_flowvec_env._update_gt_flow_raster` 的公式精确复现。RGB 一侧用的是
v4-cam 真实 dump 出来的帧（`logs/auv_v4_actor_frames_heading/`），所以两块都是真数据，
只是来源不同。

    python3 scripts/make_flow_vs_rgb_figure.py

字体：见 `scripts/_cjk_font.py`。**不要**自己写 `rcParams` —— 本机 matplotlib 缓存陈旧时
`"Noto Sans CJK SC"` 这个名字找不到，会静默回退到没有中文字形的 DejaVu；而且 .ttc 集合
里 matplotlib 只暴露第一个 face（JP）。该模块的 `assert_cjk_ok()` 会做真正的缺字校验。
"""

from __future__ import annotations

import sys

import math
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

# ---------------------------------------------------------------- 环境常量（取自 env）
W = H = 64                          # cam_width / cam_height
HFOV_DEG = 90.0                     # _CAM_HFOV_DEG（不是 45；2026-09-29 修正）
FX = W * 0.5 / math.tan(math.radians(HFOV_DEG) * 0.5)   # = 32.0 px
CX = CY = 0.5 * (W - 1)             # = 31.5
TARGET_LEN = 0.8 * 0.8              # 0.8 * target_scale = 0.64 m


def flow_frame(bear: float, elev: float, dist: float, du: float, dv: float):
    """等价于 `_update_gt_flow_raster` 的标量版本。返回 (ch_u, ch_v, mask, rad, u, v)。"""
    u = CX - FX * math.tan(bear)
    v = CY - FX * math.tan(elev)
    rad = min(max(0.5 * TARGET_LEN / dist * FX, 1.0), 12.0)     # clamp(1, 12)
    ys = np.arange(H, dtype=np.float32).reshape(H, 1)
    xs = np.arange(W, dtype=np.float32).reshape(1, W)
    mask = (xs - u) ** 2 + (ys - v) ** 2 <= rad * rad
    ch_u = np.where(mask, du, 0.0).astype(np.float32)
    ch_v = np.where(mask, dv, 0.0).astype(np.float32)
    return ch_u, ch_v, mask, rad, u, v


def entropy_bits(x: np.ndarray) -> float:
    """像素值分布的香农熵，单位 bit/像素。"""
    _, counts = np.unique(x.ravel(), return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# 代表性稳态几何：目标在前方偏右、2 m 处，边靠近边横移。
DIST = 2.0
BEAR = math.radians(6.0)
ELEV = 0.0
REL_LATERAL = 0.6                   # m/s 相对横移
DU = FX * (REL_LATERAL * 0.2 / DIST) / W
DV = 0.0

ch_u, ch_v, mask, rad, u_px, v_px = flow_frame(BEAR, ELEV, DIST, DU, DV)

# ---------------------------------------------------------------- 真实 v4-cam RGB 帧
FRAME_DIR = Path("logs/auv_v4_actor_frames_heading")
rgb = np.array(Image.open(FRAME_DIR / "step01_frame0.png"))
rgb_gray = rgb.mean(axis=2)
# dump 出来的是 512x512（渲染分辨率）；网络看到的是它面积平均降采样到 64x64。
rgb64 = np.array(Image.fromarray(rgb).resize((64, 64), Image.BILINEAR))

r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
orange = (r > 100) & (r > g) & (g > b)

ent_rgb = entropy_bits(np.round(rgb64.mean(axis=2)).astype(int))
ent_flow = entropy_bits(ch_u)
n_colors = len(np.unique(rgb.reshape(-1, 3), axis=0))

print("=" * 78)
print("真实 v4-cam RGB 帧  (logs/auv_v4_actor_frames_heading/step01_frame0.png)")
print("=" * 78)
print(f"  尺寸                 : {rgb.shape}（渲染分辨率；网络看到 64x64）")
print(f"  唯一颜色数           : {n_colors}")
print(f"  R==G==B 像素占比     : {100 * np.mean((r == g) & (g == b)):6.2f} %   <- 无彩色的背景")
print(f"  橙色目标像素占比     : {100 * orange.mean():6.2f} %")
print(f"  取值范围（全通道）   : {rgb.min()} .. {rgb.max()}")
print(f"  熵（64x64 灰度）     : {ent_rgb:.3f} bit/像素")
print()
print("=" * 78)
print("GT 光流栅格  (解析复现, 64x64)")
print("=" * 78)
print(f"  目标像素位置 (u, v)  : ({u_px:.2f}, {v_px:.2f})   画面中心 = {CX:.1f}")
print(f"  圆盘半径 rad         : {rad:.2f} px   (dist={DIST} m, fx={FX:.1f} px)")
print(f"  圆盘面积             : {mask.sum()} px / {W * H}  ({100 * mask.mean():.2f} %)")
print(f"  圆盘内填充值         : du={DU:+.5f}  dv={DV:+.5f}   (已归一化, /W 和 /H)")
print(f"  ch0 不同值个数       : {len(np.unique(ch_u))}   (只有 0 和 du)")
print(f"  ch1 不同值个数       : {len(np.unique(ch_v))}   (只有 0 和 dv)")
print(f"  ch0 熵               : {ent_flow:.4f} bit/像素  (存储占 32 bit/像素)")
print(f"  ch0 冗余度           : {32 / max(ent_flow, 1e-9):.0f} 倍")
print(f"  两通道真实内容量     : {5 * 4} 字节 (5 个 float32)")

# ---------------------------------------------------------------- 出图
fig = plt.figure(figsize=(15.5, 9.4))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.72], hspace=0.40, wspace=0.26)

# --- 第 1 行：真实相机帧
ax = fig.add_subplot(gs[0, 0])
ax.imshow(rgb)
ax.set_title("A. 真实相机看到的画面（v4-cam）\n真实帧，512×512，195 种颜色", fontsize=10.5)
ax.axis("off")
ys_, xs_ = np.where(orange)
if len(xs_):
    ax.add_patch(plt.Rectangle((xs_.min(), ys_.min()), xs_.max() - xs_.min(), ys_.max() - ys_.min(),
                               fill=False, ec="#00e5ff", lw=1.6))
    ax.text(xs_.min(), max(ys_.min() - 16, 14), "目标（占 1.25% 像素）", color="#00e5ff", fontsize=9)

ax = fig.add_subplot(gs[0, 1])
ax.imshow(rgb64, interpolation="nearest")
ax.set_title("B. …降采样到网络实际拿到的\n64×64（橙色只有约 51 个像素）", fontsize=10.5)
ax.axis("off")

ax = fig.add_subplot(gs[0, 2])
ax.imshow(rgb_gray, cmap="gray", vmin=0, vmax=255)
ax.set_title("C. 同一帧的灰度图\n本来就有 74.9% 的像素是无彩色的", fontsize=10.5)
ax.axis("off")

# --- 第 2 行：光流栅格
ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(ch_u, cmap="inferno", vmin=-max(abs(DU), 1e-6), vmax=max(abs(DU), 1e-6),
               interpolation="nearest")
ax.set_title("D. GT 光流栅格 ch0 (du)\n一片纯色圆盘；其余 98% 精确为 0", fontsize=10.5)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(gs[1, 1])
ax.imshow(mask.astype(float), cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
ax.set_title(f"E. 圆盘掩码（rad={rad:.1f} px）\n亮 {mask.sum()} px / {W * H} = {100 * mask.mean():.1f}%", fontsize=10.5)
ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(gs[1, 2])
hard = np.zeros((H, W, 3), dtype=np.float32)
hard[mask] = [1.0, 0.35, 0.05]
ax.imshow(hard, interpolation="nearest")
ax.set_title("F. 光流叠帧 = K 个这样的圆盘\nK=4 → (8,64,64)，约 99% 是 0", fontsize=10.5)
ax.set_xticks([]); ax.set_yticks([])

# --- 第 3 行：直方图（真正的差别）
# 注意：两个直方图必须用独立坐标轴。早先版本把光流柱条叠在 ax.twinx() 的虚构 x 位置
# （-25 和 280），被共享的 0..255 x 范围直接裁掉，柱子完全看不见。两个子图，两套刻度。

ax = fig.add_subplot(gs[2, 0])
vals_gray = np.bincount(rgb64.mean(axis=2).astype(int).ravel(), minlength=256)[:256]
ax.fill_between(np.arange(256), vals_gray, color="#4a90d9", alpha=0.8)
ax.set_xlim(0, 255)
ax.set_xlabel("像素值（灰度 0..255）", fontsize=8.5)
ax.set_ylabel("像素数", fontsize=8.5)
ax.tick_params(labelsize=7)
ax.set_title(f"H. RGB 帧：值连续铺开\n熵 {ent_rgb:.2f} bit/像素", fontsize=10)

ax = fig.add_subplot(gs[2, 1])
ax.bar([0, 1], [(ch_u == 0).sum(), (ch_u == DU).sum()],
       color=["#7a7a7a", "#e2503a"], alpha=0.9, width=0.55)
ax.set_xticks([0, 1])
ax.set_xticklabels(["0\n（背景）", f"{DU:+.3f}\n（圆盘）"], fontsize=8)
ax.set_ylim(0, (H * W) * 1.18)
ax.set_ylabel("像素数", fontsize=8.5)
ax.tick_params(axis="y", labelsize=7)
ax.set_title(f"I. 光流 ch0：只有两个值\n熵 {ent_flow:.4f} bit/像素（冗余约 {32 / max(ent_flow, 1e-9):.0f} 倍）", fontsize=10)

ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
info = (
    "每帧的信息量\n"
    "（这才是真正的差别）\n"
    "\n"
    "RGB 帧\n"
    f"  {rgb.shape[2]} 通道 × 64×64 uint8 = 12,288 字节\n"
    f"  {n_colors} 种颜色，连续渐变\n"
    f"  熵 {ent_rgb:.2f} bit/像素\n"
    "\n"
    "GT 光流栅格\n"
    f"  2 通道 × 64×64 float32 = 32,768 字节\n"
    "  每通道只有 2 个值\n"
    f"  熵 {ent_flow:.4f} bit/像素\n"
    f"  冗余约 {32 / max(ent_flow, 1e-9):.0f} 倍\n"
    "\n"
    "GT 光流向量（v6）\n"
    "  5 个数 = 20 字节\n"
    "  同样内容，不画成图\n"
)
ax.text(0.0, 1.0, info, va="top", ha="left", fontsize=9)

fig.suptitle("相机 RGB 帧 vs GT 光流 —— 像素层面到底差在哪", fontsize=14, y=0.975)
assert_cjk_ok("相机 RGB 帧 vs GT 光流 —— 像素层面到底差在哪，圆盘掩码，熵，冗余，字节")
fig.savefig(OUT / "flow_vs_rgb_zh.png", dpi=115, bbox_inches="tight", facecolor="white")
print(f"\n图已保存 -> {OUT / 'flow_vs_rgb_zh.png'}")
