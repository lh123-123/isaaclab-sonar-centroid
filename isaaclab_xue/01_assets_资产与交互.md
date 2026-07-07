# 01 资产：关节体、刚体、软体、吸盘与自定义机器人

对应目录：`scripts/tutorials/01_assets/`。文档见 `docs/source/tutorials/01_assets/*.rst`。

---

## 多实例与路径通配

教程普遍模式：

1. 为每个环境创建 `Xform` 原点（如 `/World/Origin0` …），`translation` 为该局地原点在世界系中的偏移。
2. 资产 `prim_path` 使用 **`/World/Origin.*/Robot`** 或 **`/World/Origin.*/Cone`** 等形式，一次匹配多个环境实例。
3. 重置时：`root_state = asset.data.default_root_state.clone()`，然后 **`root_state[:, :3] += origins`**（张量 `origins` 与 `sim.device` 一致），再 `write_root_pose_to_sim` / `write_root_velocity_to_sim`，否则所有实例会叠在世界原点。
4. 写完仿真后调用 **`asset.reset()`** 清内部缓存；循环末尾 **`asset.update(sim_dt)`** 刷新 `data`。

下一章 `InteractiveScene` 用 `scene.env_origins` 替代手写 `origins`。

---

## `run_articulation.py` — `Articulation`

- 使用 `isaaclab_assets.CARTPOLE_CFG`，改 `prim_path` 匹配多 Origin。
- 控制：`set_joint_effort_target` + `write_data_to_sim`；随机力矩扰动示例。
- 与 `create_scene.py` 逻辑平行，但用手写字典 `scene_entities` 管理实体。

```bash
./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py
```

---

## `run_rigid_object.py` — `RigidObject`

- `RigidObjectCfg`：`spawn=ConeCfg(...)` 带刚体与碰撞；`init_state` 默认。
- 重置时在原点上加 `math_utils.sample_cylinder` 随机位置。
- 仅 `write_data_to_sim`（无力控），打印 `root_pos_w`。

```bash
./isaaclab.sh -p scripts/tutorials/01_assets/run_rigid_object.py
```

---

## `run_deformable_object.py` — `DeformableObject`

- `MeshCuboidCfg` + `DeformableBodyPropertiesCfg` / `DeformableBodyMaterialCfg`。
- 状态在 **节点（nodal）** 空间：`default_nodal_state_w`、`write_nodal_state_to_sim`；`transform_nodal_pos` 配合随机位姿。
- **运动学目标**：`nodal_kinematic_target`，第 4 维约束标志（0 约束 / 1 自由）；`write_nodal_kinematic_target_to_sim` 可对部分顶点施加牵引示例。

```bash
./isaaclab.sh -p scripts/tutorials/01_assets/run_deformable_object.py
```

---

## `run_surface_gripper.py` — `SurfaceGripper` + `Articulation`

- **重要**：脚本要求 **`--device cpu`**（Surface Gripper 当前仅 CPU）。
- `SurfaceGripperCfg.prim_expr` 指向 USD 里已存在的 gripper Prim（如 `.../picker_head/SurfaceGripper`）。
- 命令 `set_grippers_command`：区间映射 Opening / Idle / Closing；`update` 后读 `state`（-1 Open，0 Closing，1 Closed）。
- 与机械臂同一循环：`write_data_to_sim` 在 gripper 侧调用。

```bash
./isaaclab.sh -p scripts/tutorials/01_assets/run_surface_gripper.py --device cpu
```

---

## `add_new_robot.py` — 自定义 `ArticulationCfg` + `InteractiveScene`

- 内联定义 **`JETBOT_CONFIG`**、**`DOFBOT_CONFIG`**：`UsdFileCfg` 指向 Nucleus 机器人 + `ImplicitActuatorCfg` 等。
- `InteractiveSceneCfg` 子类中：`ground`、`dome_light`、两台机器人用 `replace(prim_path="{ENV_REGEX_NS}/Jetbot")` 等。
- 循环里：`scene["Jetbot"]` / `scene["Dofbot"]`，重置时加 `scene.env_origins`，最后 **`scene.write_data_to_sim()`**、**`scene.update(sim_dt)`**。

```bash
./isaaclab.sh -p scripts/tutorials/01_assets/add_new_robot.py --num_envs 1
```

---

## 小结

| 脚本 | 资产类型 | 要点 |
|------|-----------|------|
| `run_articulation` | 关节体 | 力矩控制、多实例原点偏移 |
| `run_rigid_object` | 刚体 | `RigidObjectCfg`、采样重置 |
| `run_deformable_object` | 软体 | nodal 状态与 kinematic target |
| `run_surface_gripper` | 吸盘 + 臂 | **CPU**、`prim_expr` |
| `add_new_robot` | 自定义 USD 机器人 | `ArticulationCfg` + `InteractiveScene` |
