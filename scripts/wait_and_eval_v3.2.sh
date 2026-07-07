#!/bin/bash
# Wait for v3.2 resume run to finish, then eval vs v2.
set -euo pipefail
cd /home/lh/IsaacLab

RUN_DIR="logs/skrl/residual_rl_test1_ppo_v3.2_wrench/2026-06-09_20-49-49_ppo_torch"
CKPT="$RUN_DIR/checkpoints/agent_500000.pt"
V2_CKPT="logs/skrl/residual_rl_test1_ppo_v2/2026-06-08_14-08-37_ppo_torch/checkpoints/agent_500000.pt"
LOG="logs/skrl/v3.2_wait_eval.log"
POLL=120

echo "[wait_eval] watching $CKPT" | tee "$LOG"

while [[ ! -f "$CKPT" ]]; do
  prog="$(grep -oP '\d+/500000' logs/skrl/v3.2_train.log 2>/dev/null | tail -1 || echo '?/500000')"
  if ! pgrep -f "train.py.*Test1-Train" >/dev/null; then
    step="${prog%%/*}"
    if [[ -n "${step:-}" && "$step" -lt 500000 ]] 2>/dev/null; then
      echo "[wait_eval] $(date +%H:%M:%S) train died early at $prog" | tee -a "$LOG"
      CKPT="$(ls -t "$RUN_DIR/checkpoints"/agent_*.pt 2>/dev/null | head -1)"
      [[ -n "$CKPT" ]] && break
      exit 1
    fi
  fi
  echo "[wait_eval] $(date +%H:%M:%S) progress=$prog" | tee -a "$LOG"
  sleep "$POLL"
done

[[ -f "$CKPT" ]] || { echo "[wait_eval] no checkpoint"; exit 1; }
echo "[wait_eval] found $CKPT" | tee -a "$LOG"

# ensure train stopped before eval
pkill -f "train.py.*Test1-Train" 2>/dev/null || true
sleep 10

export TERM=xterm
./scripts/test1_residual_rl.sh --headless --episodes 1 --seed 42 --algorithm PPO \
  --summary logs/residual_rl_eval/compare_v3.2_final_500k.json \
  --checkpoint "$CKPT" 2>&1 | tee -a "$LOG" | tail -20

./scripts/test1_residual_rl.sh --headless --episodes 1 --seed 42 --algorithm PPO \
  --residual-mode thruster --delta 0.15 \
  --summary logs/residual_rl_eval/compare_v2_final_500k.json \
  --checkpoint "$V2_CKPT" 2>&1 | tee -a "$LOG" | tail -20

python3 <<'PY' | tee -a "$LOG"
import json
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text())

v32 = load("logs/residual_rl_eval/compare_v3.2_final_500k.json")
v2 = load("logs/residual_rl_eval/compare_v2_final_500k.json")
pd = v32["pd_only"]

rows = [
    ("PD-only", pd),
    ("v2 @ 500k", v2["residual_rl"]),
    ("v3.2 @ 500k", v32["residual_rl"]),
]
print("\n========== Final Comparison (seed=42, 1ep) ==========")
print(f"{'方案':<14} {'final_err':>10} {'mean_err':>10} {'success':>8} {'|res|':>8}")
for name, r in rows:
    print(f"{name:<14} {r['final_pos_err_m']:>10.3f} {r['mean_pos_err_m']:>10.3f} {r['success_rate']:>8.0%} {r['mean_u_rl_norm']:>8.3f}")
out = {"pd_only": pd, "v2_500k": v2["residual_rl"], "v3.2_500k": v32["residual_rl"]}
Path("logs/residual_rl_eval/compare_final_v2_vs_v3.2.json").write_text(json.dumps(out, indent=2))
print("Saved: logs/residual_rl_eval/compare_final_v2_vs_v3.2.json")
PY

echo "[wait_eval] done $(date)" | tee -a "$LOG"
