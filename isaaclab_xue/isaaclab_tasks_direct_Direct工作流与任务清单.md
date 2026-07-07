# `isaaclab_tasks.direct`：Direct 工作流与任务清单（不含本地项目 `lh_2`）

源码根：`source/isaaclab_tasks/isaaclab_tasks/direct/`。  
本文覆盖该目录下**除 `lh_2/`（你的工程）以外**的包与注册环境，并归纳写 Direct 环境的通用模式。

---

## 1. Direct 与 Manager 的区别（一句话）

- **Manager**：观测/动作/事件用配置项 + `mdp` 函数拼装，环境类多为 `ManagerBasedRLEnv`。
- **Direct**：在一个 `DirectRLEnv` 子类里**手写** `_setup_scene`、`_pre_physics_step`、`_apply_action`、`_get_observations`、`_get_rewards`、终止与重置逻辑；适合高度定制、与仿真耦合紧的任务。

---

## 2. `DirectRLEnv` 与子类要实现什么

基类：`isaaclab.envs.DirectRLEnv`（`source/isaaclab/isaaclab/envs/direct_rl_env.py`）。配置：`DirectRLEnvCfg`（含 `sim`、`decimation`、`episode_length_s`、`scene: InteractiveSceneCfg`、观测/动作空间维度等）。

**必须实现（抽象方法）**：

- `_pre_physics_step(actions)`：每个 **环境步** 开始时处理动作（缩放、缓存）。
- `_apply_action()`：每个 **物理子步** 把动作写到仿真（如 `set_joint_effort_target`）。
- `_get_observations()`：返回字典（常见键 `policy`，shape `(num_envs, obs_dim)`）。
- `_get_rewards()`：返回 `(num_envs,)` 标量奖励张量。

另有终止条件、重置、`state_space`（非对称 critic）等，按任务覆盖。

**典型 `_setup_scene` 模式**（以 `cartpole_env.py` 为例）：

1. 实例化 `Articulation` / `RigidObject` 等。
2. 生成地面等：`spawn_ground_plane` 或 `terrain.class_type(cfg)`。
3. `self.scene.clone_environments(copy_from_source=False)`。
4. **`device == "cpu"`** 时调用 `self.scene.filter_collisions(...)`（教程注释：CPU 需显式过滤）。
5. `self.scene.articulations["cartpole"] = self.cartpole`（名称与 cfg 中一致）。
6. 灯光等 `DomeLightCfg().func(...)`。

**向量环境约定**：`is_vector_env = True`；首次 `reset` 后主要由 `step` 内处理子环境终止重置（见基类 docstring）。

---

## 3. Gymnasium 注册与训练入口

各包 `__init__.py` 使用 `gymnasium.register(...)`，常见 `kwargs`：

- `env_cfg_entry_point`：字符串 `"package.module:CfgClass"` 或 **Cfg 类对象**（如 Factory）。
- `rsl_rl_cfg_entry_point`、`rl_games_cfg_entry_point`、`skrl_cfg_entry_point`、`sb3_cfg_entry_point`、`skrl_amp_cfg_entry_point` 等：供 `isaaclab_tasks` 启动脚本解析。

`disable_env_checker=True` 避免 Gym 对向量维度的误报。

---

## 4. 子包一览（**不含 `lh_2`**）

| 子目录 | 注册的 Gym `id`（摘要） | 环境类 / 说明 |
|--------|-------------------------|----------------|
| `ant` | `Isaac-Ant-Direct-v0` | `AntEnv`；继承共享的 `locomotion/LocomotionEnv` 思路（目标点势能、关节力矩） |
| `anymal_c` | `Isaac-Velocity-Flat-Anymal-C-Direct-v0`，`...-Rough-...` | `AnymalCEnv` + `AnymalCFlatEnvCfg` / `AnymalCRoughEnvCfg` |
| `cartpole` | `Isaac-Cartpole-Direct-v0`，`...-RGB-Camera-...`，`...-Depth-Camera-...` | `CartpoleEnv` / `CartpoleCameraEnv` |
| `cartpole_showcase/cartpole` | `Isaac-Cartpole-Showcase-{Box,Discrete,MultiDiscrete,Dict,Tuple}-...-Direct-v0` 等 | 展示 **不同 Gym 动作/观测空间组合** 的 Cartpole |
| `cartpole_showcase/cartpole_camera` | `Isaac-Cartpole-Camera-Showcase-...` | 带相机观测的空间展示 |
| `cart_double_pendulum` | `Isaac-Cart-Double-Pendulum-Direct-v0` | 双摆小车；支持 skrl IPPO/MAPPO 等入口 |
| `quadcopter` | `Isaac-Quadcopter-Direct-v0` | `QuadcopterEnv` |
| `humanoid` | `Isaac-Humanoid-Direct-v0` | `HumanoidEnv` |
| `humanoid_amp` | `Isaac-Humanoid-AMP-{Dance,Run,Walk}-Direct-v0` | `HumanoidAmpEnv`；**AMP 配置以 skrl 为主** |
| `franka_cabinet` | `Isaac-Franka-Cabinet-Direct-v0` | `FrankaCabinetEnv` |
| `factory` | `Isaac-Factory-{PegInsert,GearMesh,NutThread}-Direct-v0` | 同一 `FactoryEnv`，不同 `FactoryTask*Cfg` |
| `forge` | `Isaac-Forge-{PegInsert,GearMesh,NutThread}-Direct-v0` | 同一 `ForgeEnv`，与 Factory 类似的装配任务变体 |
| `automate` | `Isaac-AutoMate-Assembly-Direct-v0`，`...-Disassembly-...` | `AssemblyEnv` / `DisassemblyEnv`；工业拆装线风格 |
| `shadow_hand` | `Isaac-Repose-Cube-Shadow-...-Direct-v0`（多种）、`...-Vision-...` | 多条目共用 `inhand_manipulation/InHandManipulationEnv` + 本包 `shadow_hand_env_cfg`；含 OpenAI FF/LSTM、视觉 Play 配置 |
| `allegro_hand` | `Isaac-Repose-Cube-Allegro-Direct-v0` | 同上，`AllegroHandEnvCfg` |
| `shadow_hand_over` | `Isaac-Shadow-Hand-Over-Direct-v0` | `ShadowHandOverEnv`；双手协作类 |
| `isaac-auv-env` | `Isaac-WarpAUV-Direct-v1` | `WarpAUVEnv`；AUV + Warp 相关动力学，目录内常含独立 `.git`，视为**嵌套子项目** |
| `locomotion` | （无 `gym.register`） | `LocomotionEnv`：**基类/共享逻辑**，供 Ant 等复用 |
| `inhand_manipulation` | （无独立 register） | `InHandManipulationEnv`：**共享手部操作环境实现**，由 Shadow / Allegro 注册引用 |

---

## 5. 与你自己的 `lh_2` 对比（便于对齐习惯）

`lh_2` 注册 `Isaac-Lh-2-Direct-v0`，结构与其他 Direct 包一致：`Lh2Env` + `Lh2EnvCfg` + `agents/` 下各库 YAML / Runner。**本文刻意不写 `lh_2` 业务细节**；你可在同目录对照 `cartpole` / `anymal_c` 的 `_setup_scene` 与 `step` 节奏。

---

## 6. 阅读顺序建议（想快速吃透 Direct）

1. `cartpole/cartpole_env.py`：最短、模式最全。
2. `locomotion/locomotion_env.py` + `ant/ant_env.py`：移动机器人势能奖励与地形。
3. `anymal_c/anymal_c_env.py` + `*_env_cfg.py`：四足 + 平坦/粗糙配置切换。
4. `inhand_manipulation/inhand_manipulation_env.py` + `shadow_hand/shadow_hand_env_cfg.py`：高维动作与接触丰富任务。
5. `factory/factory_env.py` 或 `forge/forge_env.py`：多任务共用同一 Env 类、按 Cfg 切换。

---

## 7. 扩展阅读

- 基类 API：`source/isaaclab/isaaclab/envs/direct_rl_env.py`、`direct_rl_env_cfg.py`。
- 多智能体 Direct：`DirectMARLEnv`（如未来需要 PettingZoo 风格）。
