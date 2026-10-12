"""Plot the Direct-v6-flowvec ep_rew_mean curve (Chinese labels).

Reads the newest TensorBoard run under logs/sb3/Isaac-AUV-Tracking-Direct-v6-flowvec and writes
a PNG so the collapse-and-recovery shape is visible rather than buried in a table.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing import event_accumulator as ea  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cjk_font import assert_cjk_ok, setup_cjk_font  # noqa: E402

font = setup_cjk_font()
plt.rcParams["axes.unicode_minus"] = False
print("font:", font)

runs = sorted(glob.glob("logs/sb3/Isaac-AUV-Tracking-Direct-v6-flowvec/*/SAC_*"))
if not runs:
    raise SystemExit("no v6-flowvec run found")

acc = ea.EventAccumulator(runs[-1], size_guidance={ea.SCALARS: 0})
acc.Reload()
tags = acc.Tags()["scalars"]

rew = acc.Scalars("rollout/ep_rew_mean")
steps = [e.step for e in rew]
vals = [e.value for e in rew]
lens = [e.value for e in acc.Scalars("rollout/ep_len_mean")] if "rollout/ep_len_mean" in tags else None

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(steps, vals, "o-", color="#1f77b4", lw=2, ms=5, label="ep_rew_mean")

# Peak / trough / recovery annotations
peak_i = vals.index(max(vals))
ax.annotate(
    f"峰值 {max(vals):+.0f}\n@{steps[peak_i]//1000}k",
    xy=(steps[peak_i], vals[peak_i]),
    xytext=(steps[peak_i] - 60_000, max(vals) + 130),
    fontsize=11,
    color="#2ca02c",
    arrowprops=dict(arrowstyle="->", color="#2ca02c"),
)
trough_i = vals.index(min(vals))
ax.annotate(
    f"崩塌谷底 {min(vals):+.0f}\n@{steps[trough_i]//1000}k",
    xy=(steps[trough_i], vals[trough_i]),
    xytext=(steps[trough_i] - 40_000, min(vals) - 150),
    fontsize=11,
    color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728"),
)
ax.annotate(
    "恢复后平台 ~+430",
    xy=(steps[-1], vals[-1]),
    xytext=(steps[-1] - 190_000, vals[-1] - 230),
    fontsize=11,
    color="#ff7f0e",
    arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
)

ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.set_xlabel("训练步数 (transitions)", fontsize=12)
ax.set_ylabel("ep_rew_mean", fontsize=12)
ax.set_title("Direct-v6-flowvec 训练奖励曲线（SAC, 8 env, gradient_steps=1）", fontsize=13)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=11)

if lens is not None:
    ax2 = ax.twinx()
    ax2.plot(steps, lens, "--", color="#9467bd", lw=1.4, alpha=0.75, label="ep_len_mean")
    ax2.set_ylabel("ep_len_mean", fontsize=12, color="#9467bd")
    ax2.tick_params(axis="y", labelcolor="#9467bd")
    ax2.set_ylim(170, 202)
    ax2.legend(loc="upper right", fontsize=11)

out = Path("logs/auv_flow_compare/v6_reward_curve_zh.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=140)
print("saved:", out.resolve())

assert_cjk_ok("峰值 崩塌 恢复 训练奖励曲线")
