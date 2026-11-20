#!/usr/bin/env bash
# P0-3 (N@150k gate) then P0-2 (blind-cue comparison) as one Isaac-at-a-time queue.
#
#   setsid nohup bash v7/scripts/eval/queue_v7_p02_p03.sh >/dev/null 2>&1 &
#
# It WAITS for the box to go idle (no train.py, no eval, no baseline), so it can be armed
# while N is still training: the monitor stops N at 150 k, the monitor's own 100-episode
# eval runs, and only then does this queue start. Nothing here may run beside another Isaac
# process -- one instance saturates the GPU and sharing it invalidates both.
#
# Order matters. P0-3 first: if N150k does not clear the gate at least as well as K150k,
# the blind comparison is being run on a checkpoint nobody will put in the paper.
#
#   P0-3   eval N150k        2 seeds x 100 episodes             gate table
#   P0-2   eval K150k clean  2 seeds x 100 episodes             the reference
#   P0-2   eval K150k blind  2 seeds x 100 episodes             visodom dropped 4-9 s
#   P0-2   baseline clean    p,pid,lqr @ DR distribution        same estimator
#   P0-2   baseline blind    same + --blind_at 4 --blind_len 5  same dropout
#
# The dropout is 5 s starting at t=4 s of a 15 s episode on BOTH sides, and both sides hold
# their last good estimate and return to the live one afterwards. If that ever diverges, the
# ratio below measures nothing (see the P0-2 note in v7/docs/待办_论文实验清单.md).
#
# Env overrides:  EPISODES=100  NUM_ENVS=4  SEEDS="1234 777"  RUN_DIR=...  PY=...
#                 SKIP_P03=1 (jump straight to the blind comparison)

set -u

ROOT="${ROOT:-$HOME/IsaacLab}"
PY="${PY:-/home/lh/miniconda3/envs/isaaclab/bin/python}"
EPISODES="${EPISODES:-100}"
NUM_ENVS="${NUM_ENVS:-4}"
SEEDS="${SEEDS:-1234 777}"
BLIND_AT="${BLIND_AT:-4.0}"
BLIND_LEN="${BLIND_LEN:-5.0}"
BASE_EP="${BASE_EP:-10}"
BASE_STEPS="${BASE_STEPS:-2250}"
TASK="Isaac-AUV-Tracking-Direct-v7-flow-hover"
CKPT_ROOT="$ROOT/logs/sb3/$TASK"
OUTDIR="$ROOT/v7/logs"
OUT="$OUTDIR/queue_p02_p03.txt"
K_CKPT="$ROOT/v7/checkpoints/best_v7_K150k.zip"
BUSY="train.py|eval_v7_checkpoint.py|debug_auv_v7_baseline.py"

mkdir -p "$OUTDIR"
say() { echo "$(date '+%F %T') $*" | tee -a "$OUT"; }

pick_run_dir() {
  if [ -n "${RUN_DIR:-}" ] && [ -d "${RUN_DIR:-}" ]; then printf "%s" "$RUN_DIR"; return; fi
  ls -td "$CKPT_ROOT"/*/ 2>/dev/null | head -1
}

# N's checkpoint at ~150 k, or the first one above it. The gate is read at 150 k (K's best
# window); M had already collapsed by 125 k, so "latest" is not a safe default here the way
# it is in the monitor.
pick_n_ckpt() {
  local run; run="$(pick_run_dir)"
  [ -n "$run" ] || return 0
  ls "$run"model_*_steps.zip 2>/dev/null \
    | sed 's/.*model_\([0-9]*\)_steps.zip/\1 &/' | sort -n \
    | awk '$1 >= 150000 {print $2; exit}'
}

run_eval() {  # $1 tag, $2 checkpoint, $3 seed, rest: hydra overrides
  local tag="$1" ckpt="$2" seed="$3"; shift 3
  local json="$OUTDIR/p0_${tag}_s${seed}.json" log="$OUTDIR/p0_${tag}_s${seed}.log"
  say "EVAL $tag seed=$seed"
  "$PY" "$ROOT/v7/scripts/eval/eval_v7_checkpoint.py" \
    --task "$TASK" --checkpoint "$ckpt" \
    --episodes "$EPISODES" --num_envs "$NUM_ENVS" --seed "$seed" \
    --action_delay 3 --headless --out "$json" "$@" > "$log" 2>&1
  if [ $? -ne 0 ] || [ ! -f "$json" ]; then
    say "EVAL FAILED $tag seed=$seed (see $log)"; return 1
  fi
  grep -aE "GATE RESULT|disp_spawn_m  mean|peak_m       mean|late_drift   mean" "$log" \
    | sed "s/^/    [$tag s$seed] /" | tee -a "$OUT"
}

run_baseline() {  # $1 tag, rest: extra args
  local tag="$1"; shift
  local log="$OUTDIR/p0_${tag}.log"
  say "BASELINE $tag"
  "$PY" "$ROOT/v7/scripts/debug/debug_auv_v7_baseline.py" --headless \
    --controllers p,pid,lqr --sample_current --episodes "$BASE_EP" --steps "$BASE_STEPS" \
    "$@" > "$log" 2>&1
  if [ $? -ne 0 ]; then say "BASELINE FAILED $tag (see $log)"; return 1; fi
  grep -aE "^\[baseline\] cur=" "$log" | sed "s/^/    [$tag] /" | tee -a "$OUT"
}

# ---- wait for an idle box -------------------------------------------------------------
say "queue armed: task=$TASK episodes=$EPISODES seeds=[$SEEDS] root=$ROOT"
while pgrep -af "$BUSY" | grep -v queue_v7_p02_p03 >/dev/null 2>&1; do
  say "busy (training or another eval alive), waiting 120 s"
  sleep 120
done
say "box idle, starting"

# ---- P0-3: the gate on N ---------------------------------------------------------------
if [ "${SKIP_P03:-0}" != "1" ]; then
  N_CKPT="$(pick_n_ckpt)"
  if [ -z "$N_CKPT" ] || [ ! -f "$N_CKPT" ]; then
    say "P0-3 SKIPPED: no N checkpoint >= 150 k under $(pick_run_dir)"
  else
    say "P0-3 checkpoint: $N_CKPT"
    for s in $SEEDS; do run_eval N150k "$N_CKPT" "$s"; done
  fi
fi

# ---- P0-2: the comparison --------------------------------------------------------------
if [ ! -f "$K_CKPT" ]; then
  say "P0-2 ABORTED: $K_CKPT missing (weights live outside git; rsync them first)"
  exit 1
fi
for s in $SEEDS; do
  run_eval Kclean "$K_CKPT" "$s"
  run_eval KblindVO "$K_CKPT" "$s" \
    "env.blind_at_s=$BLIND_AT" "env.blind_len_s=$BLIND_LEN"
done
run_baseline base_clean
run_baseline base_blind --blind_at "$BLIND_AT" --blind_len "$BLIND_LEN"

# ---- the decision, computed rather than eyeballed ---------------------------------------
"$PY" - "$OUTDIR" <<'PY' | tee -a "$OUT"
import glob, json, os, statistics as st, sys

d = sys.argv[1]

def vals(tag, key):
    out = []
    for f in sorted(glob.glob(os.path.join(d, f"p0_{tag}_s*.json"))):
        out += [e[key] for e in json.load(open(f))["episodes"]]
    return out

def med(tag, key):
    v = vals(tag, key)
    return st.median(v) if v else float("nan")

print("\n=== P0-2 decision table (medians over both seeds) ===")
print(f"{'run':<12}{'n':>5}{'end mm':>10}{'peak mm':>10}{'drift mm/s':>12}")
for tag in ("Kclean", "KblindVO"):
    print(f"{tag:<12}{len(vals(tag,'peak_m')):>5}{med(tag,'disp_spawn_m')*1e3:>10.1f}"
          f"{med(tag,'peak_m')*1e3:>10.1f}{med(tag,'late_drift_mps')*1e3:>12.1f}")

clean, blind = med("Kclean", "peak_m"), med("KblindVO", "peak_m")
if clean == clean and blind == blind and clean > 0:
    r = blind / clean
    print(f"\npeak degradation = {r:.2f}x  (P 1.56x | PID 1.81x | LQR 4.43x)")
    if r < 1.56:
        v = "CLAIM HOLDS -> paper main figure"
    elif r < 2.0:
        v = "WEAK -> 'not worse than PID'"
    elif r > 4.43:
        v = "CLAIM DEAD -> framework B"
    else:
        v = "BETWEEN PID AND LQR -> report both, no superiority claim"
    print(f"verdict: {v}")
print("\nnote: report the absolute peak too. The ratio is unfair across controllers: P's own")
print("peak is already ~0.6 m, so the same metre of damage is 1.56x for P and 4.43x for LQR.")
print("baseline numbers are in p0_base_clean.log / p0_base_blind.log (peak_disp_m).")
PY

say "queue done"
