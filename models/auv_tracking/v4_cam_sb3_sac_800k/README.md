# Direct-v4-cam SAC — frozen at 800k

- **File**: `model.zip` (= `model_800000_steps.zip`, md5 `d9cbcf0de5dd955cf0d15831377a1e35`)
- **Run**: `logs/sb3/Isaac-AUV-Tracking-Direct-v4-cam/2026-09-24_19-57-01` (seed 66, trained to 1M, 8 envs)
- **Task**: `Isaac-AUV-Tracking-Direct-v4-cam` — v0 task semantics, camera observation, **second WarpAUV** as target
- **Frozen**: 2026-09-28
- **Train command**: see `command.txt`

## Why 800k and not the final 1M

Chosen on a 20-full-episode headless eval of 4 checkpoints from the same run (1 env, deterministic,
seed 66). Full numbers in `eval/sweep_4ckpt.json`, plot in `eval/sweep_4ckpt.png`.

| ckpt | `ep_ret` | `mean_gt_r` | `med\|α\|` | `frac r>3m` | `lost` | `da_p90` | `a_w_sat` |
|---|---|---|---|---|---|---|---|
| 400k | 222 ± 362 | 2.95 m | 31.6° | 20.8 % | 18.8 % | 1.75 | 36.6 % |
| 600k | 448 ± 293 | 2.71 m | 31.3° | 27.7 % | 12.5 % | 1.65 | 14.8 % |
| **800k** | **569 ± 244** | **2.15 m** | **13.6°** | **8.8 %** | **0.3 %** | **1.22** | **7.3 %** |
| 1M | 409 ± 285 | 3.68 m | 29.0° | 34.3 % | 17.6 % | 1.42 | 9.9 % |

800k wins on every geometric metric, not just return. `lost_rate` 0.3 % means it held the target
for essentially the entire eval (one episode, 7 % of it). Its `mean_gt_r` of 2.15 m is the
initialisation distance — it simply keeps station at spawn range.

**The 1M regression is real, not a reward-scale artifact.** Same run, so the uncertainty term
(`0.2·(−log det Σ)`) is directly comparable. `mean_gt_r` 2.15 → 3.68 m and `lost` 0.3 → 17.6 %.

Two distinct failure shapes, worth not conflating:

- **800k's worst episode** (`ret −399`): `med|α|` 23.3°, `lost` 7 %. A control-quality dip; it
  stayed alive and mostly on target.
- **1M's two worst episodes** (`ret −281`, `−253`): `med|α|` only 5–6° — pointing accurately —
  but `mean_gt_r` **12.5 / 13.3 m**, `frac>3m` 95 %, `lost` 80 %, and `ep_len` 159/139 (early
  done). It drove off and could not reacquire. Note the `rad.clamp(min=1.0)` in
  `_update_gt_flow_image`: at ~12.8 m the GT flow disk is a single pixel, so the actor's motion
  cue is gone exactly in that regime. "Accurate but out of range" is consistent with that.

`ep_ret` alone would have ranked 400k ≈ 1M and hidden all of this — the metrics are what select
800k. Note also 400k→600k: return doubles while `med|α|` does not move at all (31.6° → 31.3°), so
that gain is the uncertainty term, not tracking. Do not use `ep_ret` as the ranking metric here.

## Config

`params/env.yaml` / `params/env.pkl` are the **training-time snapshot** (written by
`train.py` → `dump_yaml`/`dump_pickle`). That is the authoritative record of what this policy was
trained under; the task source can drift later.

`params/agent.yaml`:

```yaml
policy: ...agents.custom_sac_policy.CustomSACPolicy   # asymmetric actor/critic
features_extractor_class: ...agents.cam_encoder.MidCNNExtractor
features_extractor_kwargs: {features_dim: 512, use_sparse_flow: false}
net_arch: {pi: [512, 256, 256], qf: [512, 256]}
share_features_extractor: false
learning_rate: 1e-4   buffer_size: 15000   batch_size: 256
tau: 0.005   gamma: 0.99   train_freq: 1   gradient_steps: 1
learning_starts: 1000   ent_coef: auto   target_entropy: auto
action_noise_std: 0.05   normalize_input: false   seed: 66
```

Key task fields (full list in `params/env.yaml`):

| Field | Value |
|---|---|
| `kf_measurement` | **`cam`** — first 1M-camera-measurement run (prior v4-cam runs were `gt`) |
| `decimation` / `episode_length_s` | 24 / 40 s → env dt **0.2 s**, 200-step cap |
| obs `image` | **(12, 64, 64) uint8** = `frame_stack=4` × **RGB** |
| obs `flow_image` | **(2, 64, 64) float32** — see the caveat below |
| obs `state` | 4-D KF belief `[r, α, log det Σ, observed]` → **critic only** |
| camera | `TiledCamera` on `LookFollowCamera0` (`spawn=None`), 256×256 → area-downsample to 64×64; **HFOV and `sensor_fov_deg` both 90°** |
| target | second WarpAUV, `target_speed 0.5` constant (`cruise_spread 1.0`, `peak_factor 0`, `max_accel 0`), `target_init_dist 2.0/2.0`, `target_init_ang 0.0/0.0` |
| dynamics | `dyn_time_constant 0.05`, **`thruster_first_order_lag false`** |
| action | `action_scale_v 1.5`, `action_scale_w 0.5`, `action_v ∈ [−1.0, 1.0]` (reverse allowed) |
| reward | **only** `rew_scale_uncertainty_mean 0.2`; range/bearing/lag/speed/σ/smooth all 0. Penalties: range 0.2, bearing 0.3, lost 0.5, Δaction 0.02 |
| water | calm (`ocean_current.enabled false`) |

### Caveat: the actor is not purely visual

The CNN's first conv is `Conv2d(14, 32, 5, 5)`: 12 RGB channels **+ 2 `flow_image` channels**. And
`flow_image` is *ground truth* — `_update_gt_flow_image` projects `_target_pos` through the pinhole
model to `(u, v)` and stamps the true inter-frame pixel displacement on a disk there, whose radius
scales with `1/dist`. So the actor's input encodes the **GT bearing** (disk `x`) and the **GT range**
(disk radius). `use_sparse_flow: false` disables only the sparse-flow branch; the flow-image concat
is unconditional.

This is fine for what this checkpoint is — a closed-loop controller trained against a specific
observation contract — but **do not describe it as camera-only / vision SAC**, and do not carry it
into a claim about image-based tracking, without replacing that channel with a non-GT estimate.

### Architecture caveat

`MidCNNExtractor` at the time of training ends with `concat(GAP, spatial-softmax)`, giving
`trunk.0.weight` shape `(512, 384)` = 512 → 3·128. An earlier revision ended in
`AdaptiveAvgPool2d(1)` + `Linear(128, 512)`. **The latter is what sits in the git index for
`agents/cam_encoder.py`** (status `AM`), because the working tree was edited after staging. A
commit made without re-staging would ship an encoder that cannot load this checkpoint. See
`SOURCE_PIN.md`.

## Eval

```bash
./scripts/eval_auv_tracking_v4_cam.sh \
  --checkpoint models/auv_tracking/v4_cam_sb3_sac_800k/model.zip \
  --episodes 20 --num_envs 1 --headless \
  --summary /tmp/eval_v4_800k.json
```

Expected: `ep_ret ≈ 569 ± 244`, `mean_gt_r ≈ 2.14 m`, `lost ≈ 0.3 %`, `da_p90 ≈ 1.22`.

`eval/sweep_4ckpt.png` was produced from `eval/sweep_4ckpt.json`. The sweep JSON and the `.npz`
per-step dump (`rew`, `abs_bearing_deg`, `gt_r`, `lost`, `a_v`, `a_w`, `da`, `done`) are kept for
provenance.
