#!/bin/bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Test1 单 checkpoint 里程碑评估 + 是否继续训练建议
#
# 用法:
#   ./scripts/eval_test1_milestone.sh <checkpoint.pt> [step] [run_logdir]
#
# 例:
#   ./scripts/eval_test1_milestone.sh \
#     logs/skrl/residual_rl_test1_ppo_v3.2_wrench/2026-06-09_19-28-04_ppo_torch/checkpoints/agent_100000.pt \
#     100000 \
#     logs/skrl/residual_rl_test1_ppo_v3.2_wrench/2026-06-09_19-28-04_ppo_torch

set -euo pipefail

CHECKPOINT="${1:?usage: eval_test1_milestone.sh <checkpoint.pt> [step] [run_logdir]}"
STEP="${2:-0}"
RUN_LOGDIR="${3:-}"
TAG="$(basename "$(dirname "$(dirname "$CHECKPOINT")")")"
STEP_K=$((STEP / 1000))
SUMMARY="logs/residual_rl_eval/milestone_${TAG}_${STEP_K}k.json"
REPORT="logs/residual_rl_eval/milestones_report.jsonl"
mkdir -p logs/residual_rl_eval

echo "========== Milestone eval @ ${STEP} =========="
echo "  checkpoint: $CHECKPOINT"

./scripts/test1_residual_rl.sh --headless --episodes 1 --seed 42 --algorithm PPO \
    --summary "$SUMMARY" \
    --checkpoint "$CHECKPOINT"

python3 - "$SUMMARY" "$STEP" "$RUN_LOGDIR" "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

summary_path, step_s, run_logdir, report_path = sys.argv[1:5]
step = int(step_s)
data = json.loads(Path(summary_path).read_text())
pd = data["pd_only"]
rl = data.get("residual_rl") or {}
run_name = Path(run_logdir).name if run_logdir else Path(data.get("checkpoint", "")).parent.parent.name

tb_reward = None
if run_logdir and Path(run_logdir).exists():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(run_logdir)
        ea.Reload()
        scalars = ea.Scalars("Reward / Total reward (mean)")
        tb_reward = next((x.value for x in scalars if x.step == step), None)
        if tb_reward is None and scalars:
            # nearest logged step
            nearest = min(scalars, key=lambda x: abs(x.step - step))
            tb_reward = nearest.value
    except Exception as exc:
        print(f"[warn] TB read failed: {exc}")

PD_FINAL = 0.450
V2_FINAL = 0.159
TARGET_FINAL = 0.18

final_err = rl.get("final_pos_err_m")
prev_final = None
report_file = Path(report_path)
if report_file.exists():
    for line in report_file.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("run") == run_name and row["step"] < step:
            prev_final = row.get("final_pos_err_m")

verdict = "CONTINUE"
reasons = []

if final_err is None:
    verdict = "STOP"
    reasons.append("eval 无 RL 结果")
elif final_err <= TARGET_FINAL:
    verdict = "TARGET_MET"
    reasons.append(f"final_pos_err={final_err:.3f}m ≤ 目标 {TARGET_FINAL}m（接近 v2 {V2_FINAL}m）")
elif final_err >= PD_FINAL:
    verdict = "STOP"
    reasons.append(f"final_pos_err={final_err:.3f}m ≥ PD {PD_FINAL}m，残差未带来收益")
elif prev_final is not None and final_err > prev_final + 0.05:
    verdict = "STOP"
    reasons.append(f"相对上一里程碑退化 {prev_final:.3f}→{final_err:.3f}m（>5cm）")
else:
    beat_pd = PD_FINAL - final_err
    reasons.append(f"final_pos_err={final_err:.3f}m，优于 PD {beat_pd:.3f}m（{100*beat_pd/PD_FINAL:.0f}%）")
    if final_err <= V2_FINAL + 0.05:
        reasons.append(f"已接近 v2 水平（{V2_FINAL}m）")
    else:
        reasons.append(f"距 v2 最佳 {V2_FINAL}m 还差 {final_err - V2_FINAL:.3f}m")

if tb_reward is not None:
    if tb_reward < -100 and verdict == "CONTINUE":
        verdict = "CAUTION"
        reasons.append(f"TB mean reward={tb_reward:+.1f} 偏低，建议密切观察")
    elif tb_reward >= 150:
        reasons.append(f"TB mean reward={tb_reward:+.1f} 健康（v2 约 +220~+240）")
    else:
        reasons.append(f"TB mean reward={tb_reward:+.1f}")

record = {
    "run": run_name,
    "tag": Path(summary_path).stem,
    "step": step,
    "checkpoint": data.get("checkpoint"),
    "final_pos_err_m": final_err,
    "mean_pos_err_m": rl.get("mean_pos_err_m"),
    "success_rate": rl.get("success_rate"),
    "mean_u_rl_norm": rl.get("mean_u_rl_norm"),
    "tb_mean_reward": tb_reward,
    "pd_final_pos_err_m": pd.get("final_pos_err_m"),
    "verdict": verdict,
    "reasons": reasons,
    "summary_json": str(Path(summary_path).resolve()),
}
with report_file.open("a") as f:
    f.write(json.dumps(record) + "\n")

print("\n========== Milestone Report ==========")
print(f"  step:           {step}")
print(f"  final_pos_err:  {final_err:.3f} m  (PD {PD_FINAL}, v2 {V2_FINAL})")
if tb_reward is not None:
    print(f"  TB reward:      {tb_reward:+.1f}")
print(f"  success:        {rl.get('success_rate', 0):.0%}")
print(f"  mean |residual|: {rl.get('mean_u_rl_norm', 0):.3f}")
print(f"  >>> Verdict:    {verdict}")
for r in reasons:
    print(f"      - {r}")
print(f"  report:         {report_file.resolve()}")
print("======================================\n")
PY
