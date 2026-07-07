# ResidualRL — WarpAUV 残差强化学习

基于 [WarpAUV_ResidualRL 方案](/home/lh/Downloads/WarpAUV_ResidualRL_方案.md) 的初始实现，在 Isaac Lab Direct 工作流中扩展 `isaac-auv-env`。

## 核心思路

```
u_total = clip(u_base + clip(u_rl, ±δ))
           ↑              ↑
      故意调差的 PD    SAC 残差 (6 推进器 PWM)
```

- **基础控制器** (`base_controller.py`)：增益偏低、带系统性偏置的 PD，给 RL 留出补偿空间
- **残差策略**：SAC 输出 6 维有界残差（默认 δ=0.3）
- **奖励**：位置/姿态误差平方惩罚 + 残差/基础动作惩罚 + 稀疏成功奖励
- **课程训练**：4 个阶段逐步加入域随机化（推进器偏置 → 水流 → 全随机化）

## 依赖

需要先安装/链接 WarpAUV 环境：

```bash
# 若尚未链接
cd source/isaaclab_tasks/isaaclab_tasks/direct/
ln -s <isaac-auv-env_Path> isaac-auv-env
```

## 训练

```bash
# 阶段 1：无扰动，学习基本残差补偿
./scripts/train_residual_rl_sac.sh --num_envs 512 --headless

# 阶段 2–4：逐步增加扰动
./scripts/train_residual_rl_sac.sh --task Isaac-ResidualRL-Stage2-Direct-v0 --num_envs 512 --headless
./scripts/train_residual_rl_sac.sh --task Isaac-ResidualRL-Stage3-Direct-v0 --num_envs 512 --headless
./scripts/train_residual_rl_sac.sh --task Isaac-ResidualRL-Stage4-Direct-v0 --num_envs 512 --headless
```

TensorBoard：`tensorboard --logdir logs/skrl/residual_rl_sac/`

## 注册环境

| Gym ID | 说明 |
|--------|------|
| `Isaac-ResidualRL-Direct-v0` | 阶段 1（默认） |
| `Isaac-ResidualRL-Stage2-Direct-v0` | 推进器偏置/增益随机化 |
| `Isaac-ResidualRL-Stage3-Direct-v0` | + 水平水流 |
| `Isaac-ResidualRL-Stage4-Direct-v0` | + 附加质量/浮力全随机化 |

## 目录结构

```
residual_rl/
├── __init__.py                 # Gym 注册
├── base_controller.py          # 故意调差的 PD 基础控制器
├── residual_rl_env.py          # 继承 WarpAUVEnv，残差叠加逻辑
├── residual_rl_env_cfg.py      # 环境/奖励/课程配置
├── reward.py                   # 残差 RL 奖励函数
├── agents/
│   └── skrl_sac_cfg.yaml       # SAC 超参
└── README.md
```

## 观测 / 动作

| 项目 | 维度 | 说明 |
|------|------|------|
| 观测 | 23 | WarpAUV 17 维 + 上一步残差动作 6 维 |
| 动作 | 6 | 残差推进器 PWM，clip 到 ±δ |

## Test 1 — 定点到达

固定任务：起点 `(0, 0, 8)` → 目标 `(5, 0, 8)`（env 坐标系；方案中的 `(5,0,-8)` 对应 WarpAUV 深度 `z=8`）。

### 训练

```bash
./scripts/train_residual_rl_test1.sh --num_envs 512 --headless
```

Gym ID: `Isaac-ResidualRL-Test1-Train-v0`

### 评估（PD vs SAC）

```bash
# 仅 PD 基线
./scripts/test1_residual_rl.sh --headless --episodes 10

# PD + SAC 对比
./scripts/test1_residual_rl.sh --headless --episodes 10 \
    --checkpoint logs/skrl/residual_rl_sac/<run>/checkpoints/best_agent.pt
```

Gym ID: `Isaac-ResidualRL-Test1-Direct-v0`

输出：
- `logs/residual_rl_eval/test1_<timestamp>.csv` — 每 episode 明细
- `logs/residual_rl_eval/test1_<timestamp>.json` — 汇总（**final_pos_err_m** 为主要指标）

## 评估与回放（通用）

### 批量对比（PD-only vs SAC）

```bash
# 仅 PD 基线（零残差）
./scripts/eval_residual_rl.sh --headless --episodes 20

# PD vs SAC（需 checkpoint）
./scripts/eval_residual_rl.sh --headless --episodes 20 --stage 3 \
    --checkpoint logs/skrl/residual_rl_sac/<run>/checkpoints/best_agent.pt
```

结果 CSV 默认写入 `logs/residual_rl_eval/`。

### GUI 回放

```bash
./scripts/play_residual_rl.sh --checkpoint logs/skrl/residual_rl_sac/<run>/checkpoints/best_agent.pt
./scripts/play_residual_rl.sh --headless --eval_debug --checkpoint <path>
```

## 后续工作

- [x] Test 1 定点到达（固定目标 + PD vs SAC 评估脚本）
- [ ] Test 2 轨迹跟踪（矩形/圆形路径）
- [ ] Test 3 抗扰测试（分级水流）
- [ ] 推进器映射精细化
- [ ] RSL-RL PPO 备选训练配置
