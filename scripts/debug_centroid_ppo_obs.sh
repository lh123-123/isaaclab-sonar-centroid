#!/usr/bin/env bash
# Verify PPO 4D obs ↔ Phase-2 green/yellow col convention (headless).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib/ensure_single_isaac_process.sh
TERM=xterm ./isaaclab.sh -p scripts/debug_centroid_ppo_obs.py --headless "$@"
