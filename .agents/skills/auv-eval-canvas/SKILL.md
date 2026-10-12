---
name: auv-eval-canvas
description: Evaluate trained AUV checkpoints on the server then build a canvas report on this machine. Use when the user trained a model and says 测评, 评测, 跑完测一下, 出看板, canvas, 对比, or gives checkpoint paths to evaluate.
disable-model-invocation: true
---

# AUV eval + canvas report

One command flow: server-side eval (reuse `auv-20ep-eval` protocol) → copy summary JSONs back → write one `.canvas.tsx` locally.

## 1. Collect eval targets

Ask only what is missing (one question, sensible defaults):
- checkpoint absolute paths on the server + short TAG each (e.g. `v5_500k`), default TAG = filename without extension
- server SSH target, default: current project convention (ask once, reuse for all)
- episodes, default 20; envs default 1; always `--headless`, `rew_scale_smooth 0`, calm water

If user gave ckpts in chat, skip asking.

## 2. Server eval (do not reinvent)

Per checkpoint, on the server, follow skill `auv-20ep-eval`:

```bash
CKPT=<abs path> TAG=<tag>
./scripts/eval_auv_tracking_vision_v1.sh \
  --checkpoint "$CKPT" \
  --episodes 20 --num_envs 1 --headless \
  --dump /tmp/eval_${TAG}_ep20.npz \
  --summary /tmp/eval_${TAG}_ep20.json
```

Rules: `pgrep -af 'isaaclab|train.py --task Isaac-AUV|play_auv'` first; occupied without kill permission → stop and ask. Never run 2 Isaac jobs at once; serialize checkpoints. Never change eval script flags beyond ckpt/episodes/dump/summary.

Summary JSON per-ep arrays (keys): `auv_spd`, `gt_r`, `a_v_sat`, `a_w_sat`, `da`, `speed_err`, `ep_ret`, gated-only `gate`. Report mean±std n=episodes. Ranking = speed vs 0.45 + gt_r + saturation; `ep_ret` appendix only.

## 3. Bring results back

```bash
scp <server>:/tmp/eval_<TAG>_ep20.json /tmp/eval_<TAG>_ep20.json
scp <server>:/tmp/eval_<TAG>_ep20.npz /tmp/eval_<TAG>_ep20.npz   # optional, for per-step plots
```

Verify each JSON parses: `python3 -c "import json;print(json.load(open('/tmp/eval_${TAG}_ep20.json')).keys())"`.

## 4. Write the canvas (local)

Read skill `canvas` first, then read the reference layout:
`~/.cursor/projects/<workspace>/canvases/auv-ep20-eval-20260831.canvas.tsx` — copy its structure (stats row, per-ep LineCharts for spd/gt_r/av_sat/ret, BarCharts for mean saturation + speed/distance, gated gate chart, summary Table, 怎么读 section), replace data with new TAGs. Embed all numbers inline, no fetch.

Filename: `/home/lh/.cursor/projects/home-lh-IsaacLab/canvases/auv-eval-<date>-<tags>.canvas.tsx` (kebab-case, e.g. `auv-eval-20261010-v5-500k.canvas.tsx`). If user says 更新/同一张 → edit that file instead.

Per-episode series: prefer per-ep means from each summary JSON; if JSON only has mean±std, plot BarCharts + Table only, omit LineCharts (never render empty charts).

Every plot: title with exact metric, both axis labels with units, legend with TAG names, caption with source file + protocol (`1 env · headless · deterministic · 平静水 · smooth=0 · n=20 · cruise 0.45 m/s`).

Verdict Callout on top: who is closest to 0.45, who lost tracking (any ep `gt_r`>4), saturation comparison. `ep_ret` only in table/appendix.

Reply with a markdown link to the `.canvas.tsx` file + one line on how to open it beside chat.
