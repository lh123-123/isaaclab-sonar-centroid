# Isaac-AUV-Tracking SB3 SAC checkpoints

| Tag | Meas | Checkpoint | Source run |
|-----|------|------------|------------|
| **v0** | Geometric GT `(r, α)` → KF | `v0_gt_sb3_sac/model.zip` | `logs/sb3/.../2026-08-03_19-05-06` |
| **v1** | OceanSim sonar centroid → KF | `v1_sonar_sb3_sac/model.zip` | `logs/sb3/.../2026-08-09_11-00-22` @ 2,099,920 steps |

## Play (v1 example)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py \
  --task Isaac-AUV-Tracking-Direct-v0 \
  --agent sb3_cfg_entry_point \
  --algorithm SAC \
  --num_envs 1 \
  --checkpoint models/auv_tracking/v1_sonar_sb3_sac/model.zip \
  --headless
```

For GUI, drop `--headless`. AUV-Tracking auto-enables cameras for sonar.
