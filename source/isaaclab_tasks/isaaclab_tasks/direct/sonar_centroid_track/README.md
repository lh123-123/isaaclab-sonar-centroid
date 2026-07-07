# Sonar Centroid Track

Polar 质心追踪 + 固定点原点 + 离散 PPO `{左, 不动, 右}`。

- 方案：[PROJECT_PLAN.md](./PROJECT_PLAN.md)

## 任务 ID

| ID | 用途 |
|----|------|
| `Isaac-SonarCentroidTrack-Debug-Direct-v0` | GUI / Phase 1 debug |
| `Isaac-SonarCentroidTrack-Discrete-PPO-Direct-v0` | 离散 PPO 训练 |
| `Isaac-SonarCentroidTrack-Eval-Direct-v0` | 评估 |

## 命令

```bash
# 离线模块（无需 Isaac Sim）
python scripts/debug_centroid_track_modules.py --layer all

# Phase 1 质心 sweep（sim，运动学 teleport，可选连续 LK）
TERM=xterm ./isaaclab.sh -p scripts/debug_centroid_track_validation.py --headless \
  --output_dir logs/centroid_track/phase1_validation

# 真实速度移动 + 单次 teach + 无 re-teach（验证红点能否跟住）
TERM=xterm ./isaaclab.sh -p scripts/debug_centroid_track_motion.py --headless \
  --steps 240 --vy 0.12 --motion ramp --output_dir logs/centroid_track/phase1_motion_track

# 离散 PPO 训练（纯图像 obs 4D + round-trip 目标 0→+1.3→0）
./scripts/train_centroid_track_ppo.sh

# 或手动指定迭代次数
TERM=xterm ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-SonarCentroidTrack-Discrete-PPO-Direct-v0 \
  --headless --enable_cameras --num_envs 8 --max_iterations 2000 --seed 42
```

## 目录

```
modules/     M0–M10
metrics/     M11 validation
agents/      skrl_ppo_discrete_cfg.yaml
```
