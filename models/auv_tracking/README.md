# Isaac-AUV-Tracking SB3 SAC checkpoints

| Tag | Task | Policy | Meas | Checkpoint |
|-----|------|--------|------|------------|
| **v0** | `Isaac-AUV-Tracking-Direct-v0` | MLP on belief | Geometric GT → KF | `v0_gt_sb3_sac/model.zip` |
| **v1 (MLP)** | `Isaac-AUV-Tracking-Direct-v0` | MLP on belief | OceanSim sonar → KF | `v1_sonar_sb3_sac/model.zip` |
| **v1 vision best** | `Isaac-AUV-Tracking-Direct-v1` | CustomSAC (Actor=sonar img) | OceanSim → KF + image | `v1_vision_sb3_sac_best/model.zip` |

**Naming:** folder `v1_sonar_sb3_sac` is still **Direct-v0** (MLP). Gym **`Isaac-AUV-Tracking-Direct-v1`** is vision SAC. See `v1_vision_sb3_sac_best/README.md` for why later steps underperformed the early peak.

Packaged checkpoints were trained in **calm water** (`ocean_current.enabled=false`, the default). Enabling currents changes dynamics without changing obs/network interfaces.

## Play (Direct-v0 sonar MLP)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py \
  --task Isaac-AUV-Tracking-Direct-v0 \
  --agent sb3_cfg_entry_point \
  --algorithm SAC \
  --num_envs 1 \
  --checkpoint models/auv_tracking/v1_sonar_sb3_sac/model.zip \
  --headless
```

## Play (Direct-v1 vision best)

```bash
./isaaclab.sh -p scripts/play_auv_tracking_vision_v1.py \
  --checkpoint models/auv_tracking/v1_vision_sb3_sac_best/model.zip \
  --num_envs 1 --steps 400 --headless --enable_cameras
```

For GUI play of v0 MLP, use `sb3/play.py` and drop `--headless`.
