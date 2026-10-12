#!/usr/bin/env bash
# Overnight: train1 → eval1 → recommend reward → train2 → eval2.
#
#   nohup ./scripts/overnight_auv_v4_cam_pipeline.sh --num_envs 8 --headless \
#     > /tmp/auv_v4_overnight.log 2>&1 &
#
# Morning checklist:
#   tail -n 80 /tmp/auv_v4_overnight.log
#   cat logs/auv_tracking/overnight_v4_cam_*/REPORT.md
set -euo pipefail
cd "$(dirname "$0")/.."

export TERM="${TERM:-xterm-256color}"
export PYTHONUNBUFFERED=1

NUM_ENVS=8
# Camera SAC is ~36 transitions/s after fast NCC; 2×1M is a long overnight.
# Override with MAX_ITERS=150000 for a short smoke run.
MAX_ITERS="${MAX_ITERS:-1000000}"
# KF source for overnight: gt = geometric polar (no cam meas into KF). cam = appearance S1.
KF_MEAS="${KF_MEAS:-gt}"
HEADLESS_ARGS=(--headless)
FORWARD=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_envs) NUM_ENVS="$2"; FORWARD+=("$1" "$2"); shift 2 ;;
    --num_envs=*) NUM_ENVS="${1#*=}"; FORWARD+=("$1"); shift ;;
    --max_iterations) MAX_ITERS="$2"; shift 2 ;;
    --max_iterations=*) MAX_ITERS="${1#*=}"; shift ;;
    --no-headless) HEADLESS_ARGS=(); shift ;;
    *) FORWARD+=("$1"); shift ;;
  esac
done
has_ne=0
has_mi=0
has_kf=0
for a in "${FORWARD[@]+"${FORWARD[@]}"}"; do
  [[ "$a" == "--num_envs" || "$a" == --num_envs=* ]] && has_ne=1
  [[ "$a" == "--max_iterations" || "$a" == --max_iterations=* ]] && has_mi=1
  [[ "$a" == env.kf_measurement=* ]] && has_kf=1
done
if [[ "$has_ne" -eq 0 ]]; then
  FORWARD=(--num_envs "$NUM_ENVS" "${FORWARD[@]+"${FORWARD[@]}"}")
fi
if [[ "$has_mi" -eq 0 ]]; then
  FORWARD+=(--max_iterations "${MAX_ITERS}")
fi
if [[ "$has_kf" -eq 0 ]]; then
  FORWARD+=("env.kf_measurement=${KF_MEAS}")
fi

# Watchdog progress regex needs the comma-formatted total matching the bar.
TOTAL_STEPS_FMT="$(python3 -c "print(f'{int(\"${MAX_ITERS}\"):,}')")"
export TOTAL_STEPS="${TOTAL_STEPS_FMT}"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_ROOT="logs/auv_tracking/overnight_v4_cam_${STAMP}"
mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/pipeline.log"
REPORT="${OUT_ROOT}/REPORT.md"
RUN_LOG1="${OUT_ROOT}/train1_watchdog.log"
RUN_LOG2="${OUT_ROOT}/train2_watchdog.log"
EVAL1_JSON="${OUT_ROOT}/eval1.json"
EVAL2_JSON="${OUT_ROOT}/eval2.json"
OVERRIDES="${OUT_ROOT}/reward_overrides.txt"
REW_NOTE="${OUT_ROOT}/reward_recommend.md"
TASK_LOG_ROOT="logs/sb3/Isaac-AUV-Tracking-Direct-v4-cam"

exec > >(tee -a "${LOG}") 2>&1

echo "===== overnight v4-cam pipeline ${STAMP} ====="
echo "OUT_ROOT=${OUT_ROOT}"
echo "MAX_ITERS=${MAX_ITERS} TOTAL_STEPS=${TOTAL_STEPS} KF_MEAS=${KF_MEAS}"
echo "args → ${FORWARD[*]} ${HEADLESS_ARGS[*]:-}"

latest_model_zip() {
  local since="$1"
  python3 - <<PY
from pathlib import Path
root = Path("${TASK_LOG_ROOT}")
since = int("${since}")
best = None
best_m = -1.0
for p in root.glob("20*/model.zip"):
    try:
        m = p.stat().st_mtime
    except OSError:
        continue
    if m >= since - 5 and m > best_m:
        best_m = m
        best = p
if best is None:
    raise SystemExit("no model.zip found after session_start")
print(best.resolve())
PY
}

kill_clean() {
  ./scripts/kill_isaac_lab_processes.sh || true
  sleep 3
}

# ---------- phase 1: train ----------
echo ""
echo "===== PHASE 1 train (defaults) $(date -Is) ====="
kill_clean
SESSION1="$(date +%s)"
export RUN_LOG="${RUN_LOG1}"
export SESSION_START_EPOCH="${SESSION1}"
set +e
./scripts/watchdog_train_auv_tracking_sb3_sac_v4.sh "${FORWARD[@]}" "${HEADLESS_ARGS[@]+"${HEADLESS_ARGS[@]}"}"
rc1=$?
set -e
echo "[pipeline] train1 exit=${rc1}"
kill_clean
CKPT1="$(latest_model_zip "${SESSION1}")"
echo "[pipeline] ckpt1=${CKPT1}"
echo "${CKPT1}" >"${OUT_ROOT}/ckpt1.txt"

# ---------- phase 2: eval ----------
echo ""
echo "===== PHASE 2 eval1 $(date -Is) ====="
set +e
./scripts/eval_auv_tracking_v4_cam.sh \
  --checkpoint "${CKPT1}" \
  --episodes 20 \
  --seed 66 \
  --summary "${EVAL1_JSON}" \
  "${HEADLESS_ARGS[@]+"${HEADLESS_ARGS[@]}"}" \
  "env.kf_measurement=${KF_MEAS}"
rc_e1=$?
set -e
echo "[pipeline] eval1 exit=${rc_e1}"
kill_clean

# ---------- phase 3: recommend reward ----------
echo ""
echo "===== PHASE 3 recommend reward $(date -Is) ====="
python3 scripts/lib/recommend_v4_cam_reward.py \
  --eval_json "${EVAL1_JSON}" \
  --out_overrides "${OVERRIDES}" \
  --out_report "${REW_NOTE}"

REW_ARGS=()
if [[ -s "${OVERRIDES}" ]]; then
  mapfile -t REW_ARGS <"${OVERRIDES}"
  echo "[pipeline] second train overrides: ${REW_ARGS[*]}"
else
  echo "[pipeline] no reward override — train2 uses same weights (fresh seed run / confirm)"
fi

# ---------- phase 4: train2 ----------
echo ""
echo "===== PHASE 4 train2 $(date -Is) ====="
kill_clean
SESSION2="$(date +%s)"
export RUN_LOG="${RUN_LOG2}"
export SESSION_START_EPOCH="${SESSION2}"
set +e
./scripts/watchdog_train_auv_tracking_sb3_sac_v4.sh \
  "${FORWARD[@]}" \
  "${HEADLESS_ARGS[@]+"${HEADLESS_ARGS[@]}"}" \
  ${REW_ARGS[@]+"${REW_ARGS[@]}"}
rc2=$?
set -e
echo "[pipeline] train2 exit=${rc2}"
kill_clean
CKPT2="$(latest_model_zip "${SESSION2}")"
echo "[pipeline] ckpt2=${CKPT2}"
echo "${CKPT2}" >"${OUT_ROOT}/ckpt2.txt"

# ---------- phase 5: eval2 ----------
echo ""
echo "===== PHASE 5 eval2 $(date -Is) ====="
set +e
./scripts/eval_auv_tracking_v4_cam.sh \
  --checkpoint "${CKPT2}" \
  --episodes 20 \
  --seed 66 \
  --summary "${EVAL2_JSON}" \
  "${HEADLESS_ARGS[@]+"${HEADLESS_ARGS[@]}"}" \
  "env.kf_measurement=${KF_MEAS}"
rc_e2=$?
set -e
echo "[pipeline] eval2 exit=${rc_e2}"
kill_clean

# ---------- report ----------
{
  echo "# Overnight v4-cam pipeline ${STAMP}"
  echo ""
  echo "| phase | result |"
  echo "|---|---|"
  echo "| train1 | exit=${rc1} ckpt=\`${CKPT1}\` |"
  echo "| eval1 | exit=${rc_e1} json=\`${EVAL1_JSON}\` |"
  echo "| reward | see \`${REW_NOTE}\` overrides=\`${OVERRIDES}\` |"
  echo "| train2 | exit=${rc2} ckpt=\`${CKPT2}\` |"
  echo "| eval2 | exit=${rc_e2} json=\`${EVAL2_JSON}\` |"
  echo ""
  echo "## Eval1"
  echo ""
  echo '```'
  python3 -c "import json;from pathlib import Path;d=json.loads(Path('${EVAL1_JSON}').read_text());print(f\"ep_ret={d.get('ep_ret_mean'):.1f}±{d.get('ep_ret_std'):.1f} med_|a|={d.get('med_abs_bearing_deg'):.1f} mean_r={d.get('mean_gt_r'):.2f} frac_r>3={d.get('frac_r_gt3'):.1%} lost={d.get('lost_rate'):.1%}\")" 2>/dev/null || echo "(eval1 missing)"
  echo '```'
  echo ""
  echo "## Reward note"
  echo ""
  cat "${REW_NOTE}" 2>/dev/null || echo "(none)"
  echo ""
  echo "## Eval2"
  echo ""
  echo '```'
  python3 -c "import json;from pathlib import Path;d=json.loads(Path('${EVAL2_JSON}').read_text());print(f\"ep_ret={d.get('ep_ret_mean'):.1f}±{d.get('ep_ret_std'):.1f} med_|a|={d.get('med_abs_bearing_deg'):.1f} mean_r={d.get('mean_gt_r'):.2f} frac_r>3={d.get('frac_r_gt3'):.1%} lost={d.get('lost_rate'):.1%}\")" 2>/dev/null || echo "(eval2 missing)"
  echo '```'
  echo ""
  echo "Finished $(date -Is)"
} >"${REPORT}"

echo ""
echo "===== DONE $(date -Is) ====="
echo "REPORT → ${REPORT}"
exit 0
