---
name: auv-20ep-eval
description: Runs Direct-v1 20-episode headless eval (speed, gt_r, saturation; not ep_ret). Use when evaluating or comparing G0/G1/A1/hybrid/sym checkpoints, or when the user types /auv-20ep-eval, 评测, or 20 局.
disable-model-invocation: true
---

# AUV 20-episode eval

Direct-v1 对照协议。主指标是车速和跟距，不是回报。

## Do not

- Start a second Isaac while train/eval/play is running. Kill first only if the user said 先 kill / kill.
- Use `rew_scale_smooth` other than 0.
- Treat `ep_ret` as the ranking metric.
- Open GUI unless the user asked for GUI.
- Invent a new eval script; use `scripts/eval_auv_tracking_vision_v1.sh`.

## Protocol

| 项 | 值 |
|---|---|
| envs | 1 |
| mode | `--headless` · deterministic |
| water | 平静水（不要开 `ocean_current`） |
| smooth | `--rew_scale_smooth 0`（脚本已默认） |
| episodes | 20 个满局（约 3980 步） |
| 目标 cruise | ~0.45 m/s |

## Steps

1. `pgrep -af 'isaaclab|train.py --task Isaac-AUV|play_auv'` — 有占用且用户没说 kill → 停下来问。
2. 用户说 kill → `./scripts/kill_isaac_lab_processes.sh`，再确认干净。
3. 跑：

```bash
CKPT=<absolute path to model_*_steps.zip>
TAG=<short name e.g. a1_200k>
./scripts/eval_auv_tracking_vision_v1.sh \
  --checkpoint "$CKPT" \
  --episodes 20 \
  --num_envs 1 \
  --headless \
  --dump /tmp/eval_${TAG}_ep20.npz \
  --summary /tmp/eval_${TAG}_ep20.json
```

4. 读 `/tmp/eval_${TAG}_ep20.json`。报告 mean±std（n=20）：

   - `auv_spd`（对 0.45）
   - `gt_r`（跟距；叶瓣量程 4 m；单局 >~4 标跟丢）
   - `a_v_sat` / `a_w_sat`（|a|>0.9）
   - `da`、`speed_err`
   - gated 才报 `gate`
   - `ep_ret` 只当附录

5. 用户说补 canvas → 更新 `auv-ep20-eval-20260831.canvas.tsx`，不要新开一张除非他们要求。

## Read the numbers

- Critic 吃图（sym）→ `a_v_sat` 高、车速 ~1.1。
- gated 压 bang-bang；`speed=0.06` 钉跟距（G1 vs G0）。
- 多条 `ep_ret` 都在 600 出头，分不出跟没跟好。
