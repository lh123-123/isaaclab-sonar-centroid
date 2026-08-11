# Direct-v1 vision SAC — best checkpoint

- **File**: `model.zip` (copied from `model_250000_steps.zip`)
- **Run**: `logs/sb3/Isaac-AUV-Tracking-Direct-v1/2026-08-10_12-24-58`
- **Why this one**: Among saved checkpoints (every 250k), nearest TensorBoard `ep_rew_mean` was highest (~593 @ 240k). Absolute TB peak (~648 @ 380k) had no zip; play eval of this ckpt: ep_ret ≈ 663 (2 episodes, len 199).
- **Task**: `Isaac-AUV-Tracking-Direct-v1` (Actor=sonar image, Critic=belief state)

## Play

```bash
./isaaclab.sh -p scripts/play_auv_tracking_vision_v1.py \
  --checkpoint models/auv_tracking/v1_vision_sb3_sac_best/model.zip \
  --num_envs 1 --steps 400 --headless --enable_cameras
```

## Why later training looked worse

1. **Peak then plateau**: Return climbed fast to ~648 by ~380k, then oscillated ~500–570 through 3.5M — no new high. Extra steps overfit / redistributed policy without improving mean return.
2. **Entropy collapsed**: `ent_coef` fell from ~0.9 → ~0.1 early and stayed low — less exploration; policy locked into a mode that survives (ep_len≈198) but does not keep raising reward.
3. **Checkpoint spacing missed the peak**: Saves every 250k; best TB point (~380k) was never snapshotted. Later zips (1M–3.25M) sit on the flatter, lower band (play of 3.25M: mean ep_ret ~502, high variance).
4. **Vision + belief critic asymmetry**: Actor may keep fitting image noise/clutter while critic Q drifts (occasional critic_loss spikes); survival stays good while return quality wobbles.
5. **Not collapse to failure**: Late models still finish episodes; they are “good enough survivors,” not broken — just worse average return than the early peak.

**Practical**: Prefer this `best` zip for eval/demo; resume/fine-tune from it with lower LR or earlier stopping if continuing training.
