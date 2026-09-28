# SOURCE_PIN — what code `model.zip` needs

`model_*.zip` stores weights only, never the environment config and never the encoder code. This
file records the code state this checkpoint was trained against, because **the git index does not
reproduce it**.

## Fingerprints (working tree at freeze time, 2026-09-28)

`md5` of the files that decide this checkpoint's behaviour:

| File | md5 |
|---|---|
| `.../isaac_auv_tracking/auv_tracking_v4_cam_env.py` | `8458e67c1b44f0278c21a97b4cf7dd8d` |
| `.../isaac_auv_tracking/auv_tracking_env.py` (base) | `cfc7d1dfdb7e20d69e71c4f430f6fe9a` |
| `.../isaac_auv_tracking/auv_tracking_cam_env.py` | `7ff9cbcc3a5b0fd9ecb11d5a8dd3aec9` |
| `.../isaac_auv_tracking/agents/cam_encoder.py` | `270e58e34445846976c1f641585fbd5f` |
| `.../isaac_auv_tracking/agents/custom_sac_policy.py` | `99aa1697e0bc2528687b771724ac5250` |
| `.../isaac_auv_tracking/cam_meas.py` | `f523ab6ed81efd94edfa8363d7a135a9` |
| `.../isaac_auv_tracking/agents/sb3_sac_v4_cam_cfg.yaml` | `cce2c4287440bd3246adf470952e548b` |

Repo HEAD at freeze time: `239a19e` ("Add optional ocean currents to AUV tracking (default off).")

### These are working-tree states, not commits

`auv_tracking_v4_cam_env.py`, `cam_meas.py`, `agents/sb3_sac_v4_cam_cfg.yaml` are **untracked**
(`??`) at HEAD. `agents/cam_encoder.py` is `AM` — staged once, then edited again, so the **index
holds an older revision**. Do not treat a plain `git commit` as a snapshot of the above.

## Encoder shape guard (how to tell the right revision)

```python
# the loaded policy MUST show exactly these:
actor.features_extractor.cnn.0.weight    -> (32, 14, 5, 5)   # 12 RGB + 2 flow channels
actor.features_extractor.trunk.0.weight  -> (512, 384)       # GAP(128) + soft-argmax(2*128)
```

If `trunk.0.weight` is `(512, 128)`, that is the old GAP-only encoder; it will fail to load. All four
checkpoints of run `2026-09-24_19-57-01` (400k/600k/800k/1M) have the correct shapes, verified.

## Environment config

Use `params/env.yaml` / `params/env.pkl` — the training-time dump. Do **not** trust the live task
defaults to still match; re-verify with a diff if the source has moved.

At freeze time the live defaults were checked against the training snapshot and differed in only
4 fields, none of them affecting rollouts:

| field | training | live | note |
|---|---|---|---|
| `seed` | 66 | `None` | eval script sets `env_cfg.seed = 66` explicitly |
| `observation_space.image.dtype` | float32 | uint8 | declared value; eval script re-declares as float32 to load a 12-channel Box |
| `observation_space.image.low_repr` | 0.0 | 0 | string repr |
| `observation_space.image.high_repr` | 255.0 | 255 | string repr |

Reproduce the check with `scripts`-external helper: load `params/env.pkl` inside Isaac and compare
against `parse_env_cfg(task)` (the pickle needs `omni.*` importable, so it cannot be loaded in a
bare Python).

## Known doc drift to fix when touching these files

- `agents/sb3_sac_v4_cam_cfg.yaml` header says "stacked **grayscale** frames (K=4 -> **4ch**)".
  Reality is **RGB → 12 channels** (`observation_space.image` shape `(12, 64, 64)`), and the
  "89 % of pixels have R==G==B / don't use RGB" argument in that comment is stale — the target is
  now orange-tinted and the stack is deliberately kept in colour.
- `.cursor/rules/auv-tracking-holo-design.mdc` still describes v4-cam as 45° HFOV with
  `target_init_dist [1.5, 2.5]`. Current v4-cam is **90°** (matching `sensor_fov_deg`, which the
  GT `observed` gate requires) with the target pinned at 2.0 m dead ahead.
