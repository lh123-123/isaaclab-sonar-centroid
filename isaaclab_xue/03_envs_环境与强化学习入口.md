# 03 环境：Manager 式 MDP、自定义项与策略推理

对应目录：`scripts/tutorials/03_envs/`。扩展阅读：`docs/source/tutorials/03_envs/`（`create_manager_base_env.rst`、`run_rl_training.rst`、`policy_inference_in_usd.rst` 等）。

---

## 核心抽象

- **`ManagerBasedEnv`**：组合 **Scene**（通常来自 `InteractiveSceneCfg`）、**Action / Observation / Event** 等 **Manager**，构成 MDP；`env.step(action)` 内部按 `decimation` 多次推进物理。
- **`ManagerBasedRLEnv`**：在之上增加奖励、终止、truncation，用于 RL。
- **配置模式**：`@configclass` 嵌套 `ActionsCfg`、`ObservationsCfg`、`EventCfg`；观测用 **`ObsGroup`** + **`ObsTerm(func=mdp.xxx, params=...)`**；动作用 **`mdp.JointEffortActionCfg`** 等或自定义 `ActionTerm`。

---

## `create_cartpole_base_env.py`

- 继承 `CartpoleSceneCfg`（来自 `isaaclab_tasks`），定义：
  - **Actions**：`JointEffortActionCfg` 作用在 `slider_to_cart`。
  - **Observations**：`joint_pos_rel`、`joint_vel_rel` 拼成 `policy` 组。
  - **Events**：`startup` 随机杆质量；`reset` 用 `reset_joints_by_offset` 重置小车与杆关节。
- `CartpoleEnvCfg(ManagerBasedEnvCfg)`：`decimation=4`、`sim.dt=0.005`（200 Hz 物理 → 50 Hz 环境，与注释一致）。
- 主循环：`env.reset()`、`env.step(joint_efforts)`，观测字典键 **`policy`**。

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/create_cartpole_base_env.py --num_envs 32
```

---

## `run_cartpole_rl_env.py`

- 直接使用任务包里的 **`CartpoleEnvCfg`**（完整 RL 配置），环境类换为 **`ManagerBasedRLEnv`**。
- `env.step` 返回 `obs, rew, terminated, truncated, info`。

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32
```

---

## `create_cube_base_env.py`

- **自定义 `ActionTerm`**：`CubeActionTerm` / `CubeActionTermCfg`，在 `process_actions` 与 `apply_actions` 中实现 **PD 跟踪目标位置**（动作为目标位置 3 维）。
- **自定义观测函数**：`base_position(env, asset_cfg)` 返回立方体根位置相对 `env_origins`。
- **Scene**：`TerrainImporterCfg(terrain_type="plane")`；`RigidObjectCfg` 立方体 **`disable_gravity=True`**，悬浮 PD 控制。
- **Event**：
  - `reset`：`reset_root_state_uniform`。
  - **`prestartup`**：`randomize_rigid_body_scale`、`randomize_visual_color` —— **USD 级随机化**；注释说明需 **`replicate_physics=False`**，否则无法每环境独立改尺度/材质。
- `CubeEnvCfg`：`scene=MySceneCfg(..., replicate_physics=False)`，`render_interval` 与 `decimation` 对齐说明在 `__post_init__`。

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/create_cube_base_env.py --num_envs 32
```

---

## `create_quadruped_base_env.py`

- **地形**：`TerrainImporterCfg(terrain_type="generator", terrain_generator=ROUGH_TERRAINS_CFG, ...)`。
- **机器人**：`ANYMAL_C_CFG`；**高度扫描**：`RayCasterCfg` 挂 `Robot/base`，`mesh_prim_paths=["/World/ground"]`。
- **观测**：`mdp.height_scan` + `SceneEntityCfg("height_scanner")`，噪声与 clip；`constant_commands` 提供恒速指令张量。
- **传感器频率**：`__post_init__` 里设置 `height_scanner.update_period = decimation * sim.dt`（与策略控制频率对齐）。
- **策略**：从 **`ISAACLAB_NUCLEUS_DIR`** 加载 JIT **`policy.pt`**，`policy(obs["policy"])` 驱动 `env.step(action)`。

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/create_quadruped_base_env.py --num_envs 32
```

---

## `policy_inference_in_usd.py`

- **目的**：在**现成的 USD 场景**（仓库）里跑训练好的 **H1**  locomotion 策略。
- CLI：**`--checkpoint`** 指向 JIT 导出的 `.pt`；用 `omni.client.read_file` 读入后 `torch.jit.load`。
- 配置基于 **`H1RoughEnvCfg_PLAY()`**，`num_envs=1`，`curriculum=None`；地形改为 **`TerrainImporterCfg(terrain_type="usd", usd_path=...Simple_Warehouse...)`**。
- **`device == "cpu"`** 时 `env_cfg.sim.use_fabric = False`（避免 Fabric 与 CPU 组合问题）。

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/policy_inference_in_usd.py --checkpoint /path/to/jit/checkpoint.pt
```

---

## 小结

| 脚本 | 环境类 | 重点 |
|------|--------|------|
| `create_cartpole_base_env` | `ManagerBasedEnv` | Events、观测组、decimation |
| `run_cartpole_rl_env` | `ManagerBasedRLEnv` | 任务包完整 Cartpole RL |
| `create_cube_base_env` | `ManagerBasedEnv` | 自定义 ActionTerm、`prestartup` USD 随机化、`replicate_physics=False` |
| `create_quadruped_base_env` | `ManagerBasedEnv` | 生成地形、RayCaster、JIT 策略 |
| `policy_inference_in_usd` | `ManagerBasedRLEnv` | 外部 USD + 用户 checkpoint |
