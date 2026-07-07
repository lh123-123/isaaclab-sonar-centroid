# 声纳质心追踪（Sonar Centroid Tracking）

> Polar 原图 + 固定点原点 + 亮斑质心检测 + 离散 PPO `{左, 不动, 右}`  
> 模块 M0–M11 已落地；Env / Gym / skrl yaml 已注册；Phase 3 待完整训练。

---

## 0. 文档与状态

| 文档 | 用途 |
|------|------|
| **本文** | 方案、接口约定、实施路线 |
| `modules/*.py` | 可独立 debug 的实现 |
| `scripts/debug_centroid_track_modules.py` | 离线逐模块冒烟（无需 Isaac Sim） |

| 组件 | 状态 |
|------|------|
| M0–M11 模块代码（除光流） | ✅ |
| Env + Gym 注册（Debug / PPO / Eval） | ✅ MS0 |
| 离线 debug | ✅ `--layer all` |
| Phase 1 质心 sweep | ✅ 列 mean ~1.5 px |
| 固定点 teach（`fixed_point.npz`） | ✅ reset settle + ROI 亮斑标定 |
| Phase 2 纯图像 P 控制（round_trip） | ✅ MS3：\|d_col\| ~1 px，\|sway−target\| ~0.10 m |
| Phase 3 离散 PPO 脚手架 + 冒烟 | ✅ 4D obs + MLP 64-64；完整训练 🔲 MS4 |
| 光流 M6 | ❌ 已移除；改用每帧 `SonarGtDetector` + ROI |

**脚本入口**

| 阶段 | 命令 |
|------|------|
| Phase 1 | `scripts/debug_centroid_track_validation.py` |
| Phase 2 P 控制 | `scripts/eval_centroid_track_p_control.py` |
| Phase 3 PPO | `scripts/train_centroid_track_ppo.sh` |

---

## 1. 目标与成功判据

### 1.1 五项任务

| # | 任务 | 说明 |
|---|------|------|
| 1 | 场景 | 长方块 AUV + 小球 + OceanSim FLS（复用 `sonar_flow_hover/shared_platform`） |
| 2 | 感知 | Polar 亮斑 → `p_sonar_abs`（每帧 GT 检测，ROI 围绕固定点） |
| 3 | 固定点 | reset settle 后 `p_fixed_abs` → 工作坐标原点 `(0,0)` |
| 4 | 质心验证 | Phase 1：`p_perc_abs ≈ p_sonar_abs`（绝对坐标） |
| 5 | 闭环 / RL | Phase 2：手调 P；Phase 3：离散 PPO 学 `{左, 停, 右}` |

### 1.2 量化判据（Lateral 1D，仅 sway）

```
Phase 1  PASS  mean(|p_perc_abs.col − p_sonar_abs.col|) < 2.5 px 且 p90 < 3.0 px
Phase 2  OK    |green_col − yellow_col| mean < ~2 px；|sway − target| < 0.15 m（纯图像 P）
Phase 3  MS4   PPO checkpoint：success_rate > 70%
运行时 SUCCESS  ||p_sonar|| < 3 px  ∧  |robot_sway − target_dy| < 0.15 m  ∧  valid_ratio > 0.2
```

**经验关系（Phase 2，vy=0.28）**：稳态 ~1 px 列差 ≈ ~0.14 m 横向滞后（非普适常数）。

---

## 2. 场景配置

| 组件 | 参数 |
|------|------|
| 机器人 | Cuboid 0.6×0.35×0.25 m，body-Y 速度控制 |
| 目标 | 球 r=0.5 m，初始前方 ~3 m |
| 声呐 | Polar binning **110×61**（OceanSim FLS），max_range≈15 m |
| 工作高度 | Z = 10 m |
| 控制步长 | decimation=5，dt=0.02 → **0.1 s/step** |

---

## 3. 坐标系（仅 Polar）

### 3.1 图像

- **唯一工作图像**：OceanSim polar，预处理 `log + median`（可 skip median）
- **检测**：`SonarGtDetector` 亮斑质心；标定/跟踪均可用 **ROI 围绕 `p_fixed`**
- **Azimuth**：不做 boresight 列 mask（中心线球会偏）

### 3.2 双层坐标

| 层 | 符号 | 用途 |
|----|------|------|
| 绝对 | `p_abs=(col,row)` | GT 检测、教示存盘、overlay 绿/黄点 |
| 固定点 | `p=(u,v)=p_abs−p_fixed_abs` | 观测、奖励、控制 |

```python
p_rel = to_fixed_frame(p_abs, p_fixed_abs)
u_norm, v_norm = p_rel.col / W, p_rel.row / H
```

### 3.3 控制方向约定（+Y = 右移 = action 2）

```
u > 0  绿点在黄点右侧  →  robot 应右移 (action=2) 使 u→0
u < 0  绿点在黄点左侧  →  robot 应左移 (action=0) 使 u→0
```

Phase 2 连续 P（eval 脚本）：

```
Δcol = green_col − yellow_col   （绝对图像列，px）
vy   = kff·target_vy + kp·(Δcol / W)     # 纯图像，无位置 P
```

---

## 4. 质心链路（当前）

| 链路 | 模块 | 输出 | 用途 |
|------|------|------|------|
| ① 图像 GT | M3 `sonar_gt` | `p_sonar_abs` | **观测 / 奖励 / 控制 / 标定** |
| ② 几何 | M5 `geometry_perception` | `p_perc_abs` | Phase 1 验证、debug 蓝点 |
| ~~③ 光流~~ | ~~M6~~ | — | **已移除** |

```
Phase 1 验证：|p_perc_abs − p_sonar_abs|
Phase 2+ 控制：minimize ||p_sonar||（固定点坐标）≡ 绿黄 col 差
```

**固定点存盘** `logs/centroid_track/fixed_point.npz`：

```
col0, row0          # float32，绝对坐标（黄点）
mode                # "sonar_gt"
polar_snapshot      # 可选
```

---

## 5. 离散 PPO（Phase 3）

### 5.1 动作 `Discrete(3)`

| action | 含义 | body-Y 速度 |
|--------|------|-------------|
| 0 | 左移 | −0.30 m/s |
| 1 | 不动 | 0 |
| 2 | 右移 | +0.30 m/s |

实现：`modules/actions.discrete_to_lateral_velocity`

### 5.2 观测 4D（纯图像，`image_only_obs=True`）

```
[u_norm, v_norm, valid_ratio, prev_act∈{−1,0,+1}]
```

- 无 `target_dy` / `robot_lat`（与 Phase 2 纯图像 P 一致）
- 7D 全量观测仍可在 Debug cfg 使用，**PPO 训练默认 4D**

### 5.3 奖励（`modules/rewards.py`）

```
r = −1.0·|u|/W − 0.3·|v|/H − 0.02·action_switch − 10·safety
```

可选 shaping（未实现）：`u>0 → action=2` 小额正奖励。

### 5.4 网络（`agents/skrl_ppo_discrete_cfg.yaml`）

```
Box(4) → RunningStandardScaler → MLP [64, ELU, 64, ELU]
         ├─ CategoricalMixin → π(a|s), 3 类
         └─ DeterministicMixin → V(s)
```

| 项 | 值 |
|----|-----|
| 共享 trunk | `separate: False` |
| PPO | rollouts=64, lr=3e-4, γ=0.95, entropy=0.01 |
| reward scale | `rewards_shaper_scale: 0.1` |
| timesteps | 256000（或 `--max_iterations 2000`） |

### 5.5 训练环境与任务 ID

| ID | 用途 |
|----|------|
| `Isaac-SonarCentroidTrack-Discrete-PPO-Direct-v0` | PPO 训练 |
| `Isaac-SonarCentroidTrack-Eval-Direct-v0` | play / 评估（4D + triangle，与训练一致） |
| `Isaac-SonarCentroidTrack-Debug-Direct-v0` | GUI debug |

**PPO cfg 要点**（`SonarCentroidTrackDiscretePpoCfg`）：

- `lateral_target_mode = "triangle"`：**0 → +1.3 → 0 → −1.3 → 0**，`vy=0.28 m/s`，跑完即结束
- `episode_length_s ≈ 18.4`（184 step × 0.1 s，4 段匀速）
- `immediate_lost_terminate = True`：小球离开声纳图（检测失败）**当步 reset**
- `num_envs = 8`，`spawn_radius = 0.10 m`

```bash
./scripts/train_centroid_track_ppo.sh
# 或
TERM=xterm ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-SonarCentroidTrack-Discrete-PPO-Direct-v0 \
  --headless --enable_cameras --num_envs 8 --max_iterations 2000
```

---

## 6. 模块架构（M0–M11）

```
M0  constants          M7  centroid_bus
M1  coordinates        M8  observation
M2  sonar_preprocess   M9  actions
M3  sonar_gt           M10 rewards
M4  fixed_point        M11 validation (metrics/)
M5  geometry_perception
M6  target_motion      （round_trip / sine 轨迹，env + eval 共用）
──────────────────────────────────────
Env sonar_centroid_track_env
Tracker SonarCentroidTracker  （每帧 M3 + M4，非光流）
```

| 模块 | 文件 | 说明 |
|------|------|------|
| M3 | `sonar_gt.py` | 亮斑质心；支持 ROI |
| M4 | `fixed_point.py` | 黄点标定 / 存盘 |
| M8 | `observation.py` | 4D 或 7D |
| — | `target_motion.py` | round_trip / sine 运动学 |

**仿真排错**

| 现象 | 查 |
|------|-----|
| GUI 跑飞、sway 2× | `sonar_pipeline` 勿在控制环调 `app.update()`（已修） |
| 小球出视场 | `immediate_lost_terminate`：检测失败即 terminated → reset |
| 检测锁错亮点 | 启用 ROI 围绕 `p_fixed` |
| headless vs GUI 不一致 | 同上 + 单进程约定 |

---

## 7. 实施路线

```
Phase 0  Env 脚手架     ✅ env + cfg + Gym + skrl yaml
    ↓
Phase 1  质心验证       ✅ p_perc vs p_sonar
    ↓
Phase 2  固定点 + P 控制 ✅ round_trip 纯图像 P；GUI/headless 一致
    ↓
Phase 3  离散 PPO        🔲 脚手架就绪，待完整训练 → MS4
    ↓
Phase 4  鲁棒性         🔲 sin 目标、随机 spawn、课程学习
```

| 里程碑 | 交付 | 验收 | 状态 |
|--------|------|------|------|
| **MS0** | env reset + polar | GUI 见亮斑 | ✅ |
| **MS1** | validation sweep | Phase 1 PASS | ✅ |
| **MS2** | fixed_point.npz | 黄点标定稳定 | ✅ |
| **MS3** | P 控制基线 | \|sway_err\| < 0.15 m，\|d_col\| ~1 px | ✅ |
| **MS4** | PPO checkpoint | success_rate > 70% | 🔲 |

---

## 8. Env 每步流程（当前）

```python
reset:
    布置 robot / target
    pipeline.warmup()
    fixed.calibrate_from_polar(polar)   # M4，黄点

step(action):
    _update_lateral_target_motion()     # round_trip 或 sine
    vy = discrete_to_lateral_velocity(action)
    apply_sway(vy); sim.step()
    polar = pipeline.ensure_updated()
    p_sonar = gt.detect_abs_roi(polar, p_fixed)   # M3
    obs = obs_builder.build(tracker, ...)         # M8，4D
    reward = compute_reward(tracker, ...)         # M10
```

---

## 9. 风险

| 风险 | 对策 |
|------|------|
| 纯图像 P 稳态 lag | PPO 学 bang-bang 脉宽；或 Phase 4 加 sin 课程 |
| 离散 ±0.30 m/s 量化 | 目标 vy=0.28，靠 stay 动作调占空比 |
| 奖励尺度偏小 | 可调 `rewards_shaper_scale` 或有符号 u 奖励 |
| GUI 双步物理 | 已去掉 pipeline 内 `app.update()` |
| 单进程 | `./scripts/kill_isaac_lab_processes.sh` |

---

## 10. 与 sonar_flow_hover

| 本项目 | 旧项目 |
|--------|--------|
| Polar + 固定点 + 亮斑质心 | cart / 光流 / IBVS |
| 离散 PPO 3 动作，4D 图像 obs | 连续 SAC / 72D |
| round_trip / triangle 移动目标 | sin 横向目标（Phase 4） |

**复用**：`shared_platform`、`geometry`、`beacon_cart`、预处理

---

## 附录：公式

```
p_rel = p_sonar_abs − p_fixed_abs
u_norm = p_rel.col / W
Δcol   = p_sonar_abs.col − p_fixed_abs.col    # overlay 绿−黄（px）
vy_p   = kff·target_vy + kp·(Δcol/W)          # Phase 2 连续 P
range_m = (H−1−row)/H · max_range
```

---

*v1.2 | 2026-07-06 | PPO：triangle 0→+A→0→−A→0；出图即 reset；episode≈18.4s*
