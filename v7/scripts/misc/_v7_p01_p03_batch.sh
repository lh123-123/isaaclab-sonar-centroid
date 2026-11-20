#!/bin/bash
# P0-1 + P0-3 in one GPU session — one Isaac instance at a time, self-clearing.
#
#   P0-3  calibrate the ruler: how much of visodom's mm-level accuracy survives a wrong
#         f_px / Z(0) / camera mount. Produces the interval the paper can actually defend.
#   P0-1  same-configuration baseline tables: the LQR-vs-RL comparison has been mixing a
#         spawn-DR LQR arm with a DR-off RL arm. Runs both sides so the table cannot be
#         attacked on "you compared two different tasks".
#
# Order matters: P0-3 first, because if visodom turns out to be a pure scale ladder, the
# nominal P0-1 numbers are the optimistic end of a range and should be reported as such.
#
#   ./v7/scripts/misc/_v7_p01_p03_batch.sh              # full batch
#   ./v7/scripts/misc/_v7_p01_p03_batch.sh --dry        # print the plan, launch nothing
#
# Knobs: NUM_ENVS, STEPS, JOB_TIMEOUT, MIN_FREE_MB, LOG
#
# Two hard-won constraints from the 2026-11-15 session, both enforced below:
#   * an 8 GB card carries ONE Isaac instance — a second fails to build the PhysX scene;
#   * Isaac does not exit once tiled cameras are alive, so every job is followed by a
#     forced kill and a GPU wait, or the next job starves.
#
set -u

cd "$(dirname "$0")/../../.." || exit 1
REPO=$(pwd)

NUM_ENVS=${NUM_ENVS:-4}
STEPS=${STEPS:-470}            # 470 > 2x225 steps = two full 15 s episodes with margin
JOB_TIMEOUT=${JOB_TIMEOUT:-2700}
MIN_FREE_MB=${MIN_FREE_MB:-6000}   # headroom a second Isaac instance needs; NOT a "card is idle" test
LOG=${LOG:-$HOME/v7_p01_p03_$(date +%Y%m%d_%H%M%S).log}

DRY=0
for a in "$@"; do
  case "$a" in
    --dry|--dry-run) DRY=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

# A dry run must be previewable on a busy card, so it writes nothing and checks nothing.
SAYLOG=$LOG
if [ "$DRY" -eq 1 ]; then
  SAYLOG=/dev/null
fi

# Non-interactive ssh starts without conda; isaaclab.sh shells out to it.
if ! command -v python >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate isaaclab
  fi
fi

say() { printf '%s\n' "$*" | tee -a "$SAYLOG"; }

section() {
  say ""
  say "================================================================================"
  say "$*"
  say "================================================================================"
}

gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr -d ' ' || echo 0
}

own_procs() { pgrep -f 'debug_auv_v7' 2>/dev/null | wc -l | tr -d ' '; }

# Refuse to start unless the card can carry one more Isaac instance.
#
# The test is remaining HEADROOM, not total occupancy: this box runs training at the same
# time, so "GPU already busy" would be true forever. Deliberately does NOT kill anything —
# if something holds the GPU it may be someone's run, and that is not this script's to stop.
abort_if_busy() {
  local procs free_mb
  procs=$(own_procs)
  free_mb=$(gpu_free_mb)
  if [ "$procs" -gt 0 ]; then
    say "ABORT: a previous measurement is still alive (${procs} match 'debug_auv_v7')."
    say "       pgrep -af debug_auv_v7   # identify it; this script will not kill it."
    exit 1
  fi
  if [ "${free_mb:-0}" -lt "$MIN_FREE_MB" ]; then
    say "ABORT: only ${free_mb} MiB free (< ${MIN_FREE_MB}). Not enough for a second Isaac"
    say "       instance on this card. Check with: nvidia-smi"
    exit 1
  fi
  say "pre-flight ok: ${free_mb} MiB free, no leftover debug_auv_v7 processes"
  say "               (a concurrent training run is fine as long as it fits alongside)"
}

wait_gpu() {
  local waited=0 free_mb procs
  if [ "$DRY" -eq 1 ]; then
    return 0
  fi
  while :; do
    free_mb=$(gpu_free_mb)
    procs=$(own_procs)
    if [ "${free_mb:-0}" -ge "$MIN_FREE_MB" ] && [ "$procs" -eq 0 ]; then
      return 0
    fi
    if [ "$waited" -ge 120 ]; then
      say "  still not enough after ${waited}s (free=${free_mb} MiB, procs=${procs}); clearing our leftovers"
      pkill -9 -f 'debug_auv_v7' 2>/dev/null || true
      sleep 10
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

run_job() {
  local name="$1"
  shift
  wait_gpu
  section "[$name]  $(date '+%H:%M:%S')"
  say "\$ $*"
  if [ "$DRY" -eq 1 ]; then
    say "  (dry run — not executed)"
    return 0
  fi
  mkdir -p "$(dirname "$LOG")"
  timeout --signal=SIGKILL "$JOB_TIMEOUT" "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    say "  [ok] $name"
  elif [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    say "  [TIMEOUT ${JOB_TIMEOUT}s] $name — Isaac hung; continuing, next job self-clears"
  else
    say "  [exit $rc] $name"
  fi
  # Isaac holds the GPU after printing results; a chained run never reaches its next job.
  pkill -9 -f 'debug_auv_v7' 2>/dev/null || true
  sleep 5
}

say "v7 P0-1 + P0-3 batch  $(date '+%Y-%m-%d %H:%M:%S')"
say "  repo      : $REPO"
say "  num_envs  : $NUM_ENVS"
say "  steps     : $STEPS  (~2 episodes per point)"
say "  timeout   : ${JOB_TIMEOUT}s per job"
say "  needs     : ${MIN_FREE_MB} MiB GPU headroom per job (safe alongside a training run)"
say "  log       : $LOG"

if [ "$DRY" -eq 0 ]; then
  abort_if_busy
else
  say ""
  say "pre-flight checks skipped (--dry): nothing below will be launched"
fi

# ---------------------------------------------------------------------------------
# P0-3 — calibrate the ruler. Every point is pinned, so a row is a point, not a sample.
# ---------------------------------------------------------------------------------
run_job "P0-3a focal x range grid" \
  ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_visodom_dr.py --headless \
  --num_envs "$NUM_ENVS" --steps "$STEPS" --episodes 2 \
  --f_errs 0.90,0.95,1.00,1.05,1.10 --z_errs 0.90,1.00,1.10

# The mount axis is baked into the wall mesh, so it cannot be swept inside one process.
run_job "P0-3b mount tilt +-2deg" \
  ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_visodom_dr.py --headless \
  --num_envs "$NUM_ENVS" --steps "$STEPS" --episodes 2 \
  --f_errs 1.0 --z_errs 1.0 --tilt_deg 2.0

# ---------------------------------------------------------------------------------
# P0-1 — the same baseline table under both DR settings. The pair is the deliverable:
# one alone cannot answer "which task was the RL arm trained on".
# ---------------------------------------------------------------------------------
run_job "P0-1a current sweep, DR ON" \
  ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_baseline.py --headless \
  --num_envs "$NUM_ENVS" --steps "$STEPS" \
  --currents 0.2,0.3,0.45 --dir_deg 17 --controllers p,pid,lqr

run_job "P0-1b current sweep, DR OFF" \
  ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_baseline.py --headless \
  --num_envs "$NUM_ENVS" --steps "$STEPS" \
  --currents 0.2,0.3,0.45 --dir_deg 17 --controllers p,pid,lqr --no_dr

run_job "P0-1c current step, DR OFF" \
  ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_baseline.py --headless \
  --num_envs "$NUM_ENVS" --steps 240 \
  --currents 0.3 --ramp_s 0 --step_at 5 --step_to 0.45 --flip_deg 180 \
  --controllers p,pid,lqr --no_dr

run_job "P0-1d blind 5-10s, DR OFF" \
  ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_baseline.py --headless \
  --num_envs "$NUM_ENVS" --steps "$STEPS" \
  --currents 0.3 --dir_deg 17 --controllers p,pid,lqr \
  --blind_at 5 --blind_len 5 --no_dr

# ---------------------------------------------------------------------------------
# Summary — pulled from the log that the jobs already wrote.
# ---------------------------------------------------------------------------------
section "SUMMARY  (full output in $LOG)"

echo ""
echo "P0-3  visodom accuracy under a mis-calibrated ruler"
echo "      lat_scale = z_err/f_err; compare the printed slope with mean travel."
grep -E '^\[visodom-dr\]' "$LOG" 2>/dev/null | grep -v 'no episode log'

echo ""
echo "P0-1  baseline tables — compare 'cur=' rows of job a (DR ON) vs job b (DR OFF)"
grep -E '^\[baseline\] cur=' "$LOG" 2>/dev/null

echo ""
echo "Read it like this:"
echo "  P0-3  slope ~= mean travel  -> pure scale ladder, one calibration constant fixes it"
echo "        slope >> mean travel  -> structural error (plane assumption), no constant helps"
echo "  P0-1  b ~= a  -> the DR-off RL arm (10.7 mm) is comparable to this table"
echo "        b <  a  -> LQR won on the harder task too: the conclusion is conservative"
echo "        b >  a  -> something is off (spawn DR helping the controller?); investigate"
echo ""
echo "Report nominal AND the worst calibration point, never nominal alone."
