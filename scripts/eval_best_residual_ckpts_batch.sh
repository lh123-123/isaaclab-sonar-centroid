#!/usr/bin/env bash
# Sequential headless PD vs PPO for the stronger residual ckpts (data export).
set -uo pipefail
cd "$(dirname "$0")/.."
export TERM="${TERM:-xterm-256color}"
mkdir -p logs/centroid_track
OUT="${OUT:-logs/centroid_track/best_ckpts_compare_$(date +%Y%m%d_%H%M%S).log}"

run_one() {
  local tag="$1" ckpt="$2" blend="$3" dsw="$4" dsu="$5"
  echo ""
  echo "############################################################"
  echo "# ${tag}  α=${blend}  Δsway=±${dsw}  Δsurge=±${dsu}"
  echo "############################################################"
  CKPT="${ckpt}" ./scripts/eval_centroid_track_auv_residual_ppo_headless.sh \
    --gt_blend "${blend}" \
    --max_residual_sway "${dsw}" \
    --max_residual_surge "${dsu}" \
    --tag "${tag}" \
    --print_every 50
  local rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    echo "[batch] FAILED ${tag} rc=${rc}"
    return "${rc}"
  fi
  echo "[batch] OK ${tag}"
}

{
  echo "=== Best residual ckpt comparison $(date -Is) ==="

  run_one \
    "A+B_alpha1_beamhold_stationnoise" \
    "logs/skrl/sonar_centroid_track/2026-07-14_12-25-30_ppo_torch_auv_residual_ppo_sine_beamhold_stationnoise/checkpoints/best_agent.pt" \
    1.0 0.25 0.50

  run_one \
    "C_alpha05_Cblend05_res045_070" \
    "logs/skrl/sonar_centroid_track/2026-07-14_14-22-55_ppo_torch_auv_residual_ppo_AB_Cblend05_res045_070/checkpoints/best_agent.pt" \
    0.5 0.45 0.70

  run_one \
    "C_alpha03_Cblend03_res045_070" \
    "logs/skrl/sonar_centroid_track/2026-07-14_15-29-12_ppo_torch_auv_residual_ppo_AB_Cblend03_res045_070/checkpoints/best_agent.pt" \
    0.3 0.45 0.70

  echo ""
  echo "=== ALL DONE $(date -Is) ==="
} 2>&1 | tee "${OUT}"

echo "[batch] wrote ${OUT}"
