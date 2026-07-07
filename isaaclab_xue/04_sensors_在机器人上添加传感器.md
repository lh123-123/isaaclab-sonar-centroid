# 04 传感器：在场景与机器人上添加传感器

对应官方路径：`scripts/tutorials/04_sensors/`；详细说明见 `docs/source/tutorials/04_sensors/add_sensors_on_robot.rst`。

---

## 1. 核心概念

- **传感器与资产**：资产（Articulation 等）描述机器人的物理形态；传感器提供对环境的观测（外感受）与本体感受，通常**更新频率低于或等于**物理步长，由各自的 `update_period`（秒）控制。
- **基类**：各传感器继承 `SensorBase`，通过对应的 `*Cfg` 配置类创建。
- **附着方式**：
  - **会创建 USD Prim** 的传感器（如 USD 相机）：在 `prim_path` 下生成相机 Prim，并相对父 Prim 做位姿偏移。
  - **虚拟传感器**（如 RayCaster）：不新建“传感器 Prim”，而是**挂在一个已有 Prim 上**，用该 Prim 的位姿表示传感器原点，在 Warp 里对指定网格做射线检测。
  - **接触传感器**：基于 PhysX 接触报告，作用在**刚体**上；需在资产侧开启接触报告（如 `activate_contact_sensors=True`）。
- **数据读取**：仿真 `play` 后句柄与缓冲区才就绪，务必在创建 `InteractiveScene` 后调用 `sim.reset()`。循环里对场景调用 `scene.update(sim_dt)`，再读 `scene["传感器名"].data`。

---

## 2. 推荐入门：`add_sensors_on_robot.py`（InteractiveScene）

在 `InteractiveSceneCfg` 子类里像定义 `ground`、`robot` 一样定义传感器字段，多环境时用 `{ENV_REGEX_NS}` 作为路径前缀。

### 2.1 USD 相机 `CameraCfg`

- **路径示例**：`{ENV_REGEX_NS}/Robot/base/front_cam` — 环境命名空间 + 机器人 + 父链路 `base` + 相机 Prim 名 `front_cam`。
- **要点**：
  - `spawn`：`PinholeCameraCfg` 或 `FisheyeCameraCfg`。
  - `offset`：相对父 Prim 的平移与四元数，`convention` 常用 `"ros"`（与机器人习惯一致）。
  - `data_types`：如 `"rgb"`、`"distance_to_image_plane"`（深度）、`"normals"` 等 Replicator 支持的类型。
- **运行**：必须加 `--enable_cameras`（相机依赖 Omniverse 渲染管线）。

```bash
./isaaclab.sh -p scripts/tutorials/04_sensors/add_sensors_on_robot.py --num_envs 2 --enable_cameras
```

- **读数据**：`scene["camera"].data.output["rgb"]`、`output["distance_to_image_plane"]` 等。

### 2.2 高度扫描 / RayCaster `RayCasterCfg`

- **用途**：教程里用作**地形高度扫描**：从机器人 `base` 向上偏移发射射线网格，打到地面网格得到高度。
- **要点**：
  - `pattern_cfg`：`GridPatternCfg(resolution=..., size=[...])` 定义射线网格。
  - `ray_alignment="yaw"`：只随 yaw 对齐，忽略 roll/pitch，适合“只要高度”的地面感知。
  - `mesh_prim_paths`：射线检测目标（如 `["/World/defaultGroundPlane"]`）。
  - `debug_vis=True`：可在视口里看到击中点（如红点）。
- **读数据**：如 `scene["height_scanner"].data.ray_hits_w`（世界系击中点，最后一维常含高度相关信息，具体 shape 以 API 为准）。

### 2.3 接触力 `ContactSensorCfg`

- **路径**：可用正则匹配多个 Prim，例如 `"{ENV_REGEX_NS}/Robot/.*_FOOT"` 匹配四足四个脚。
- **要点**：
  - `update_period=0`：每个物理步都更新。
  - `history_length`：保留最近若干步的接触历史。
  - `debug_vis=True`：便于调试可视化。
- **读数据**：如 `scene["contact_forces"].data.net_forces_w`。

### 2.4 仿真循环中与传感器相关的顺序

与教程一致：重置时写 root/joint 状态 → `scene.reset()` → 每步 `write_data_to_sim()` → `sim.step()` → `scene.update(sim_dt)` → 打印或消费传感器 `data`。

---

## 3. 同目录下的最小示例脚本（非 InteractiveScene）

这些脚本用手动创建实体 + `Sensor` 类，便于单独理解某一类传感器。

| 脚本 | 传感器类 | 说明 |
|------|-----------|------|
| `run_ray_caster.py` | `RayCaster` | 多实例 `Origin*/ball` 上挂射线网格，对 `/World/ground` 求交；`update(..., force_recompute=True)`。 |
| `run_usd_camera.py` | `Camera` | USD + Replicator；多种 `data_types`（rgb、深度、法线、语义/实例分割等）；`set_world_poses_from_view` 或 ROS 四元数设姿；可选 `--draw` 点云、`--save` 写盘。 |
| `run_ray_caster_camera.py` | `RayCasterCamera` | **Warp 射线**模拟相机，对**静态网格**有效；`PinholeCameraPatternCfg`；无 RGB 光栅，主要是几何类 `distance_*`、`normals` 等。 |
| `run_frame_transformer.py` | `FrameTransformer` | 从源 Prim 查询多组目标坐标系位姿（可用 `offset` 定义用户帧）；用于 FK/多连杆相对位姿，非“物理传感器”但属于 `sensors` 模块。 |

运行命令（摘自官方文档）：

```bash
./isaaclab.sh -p scripts/tutorials/04_sensors/run_frame_transformer.py
./isaaclab.sh -p scripts/tutorials/04_sensors/run_ray_caster.py
./isaaclab.sh -p scripts/tutorials/04_sensors/run_ray_caster_camera.py
./isaaclab.sh -p scripts/tutorials/04_sensors/run_usd_camera.py --enable_cameras
```

---

## 4. USD 相机 vs RayCast 相机（经验对比）

| 维度 | `Camera`（USD） | `RayCasterCamera` |
|------|-----------------|-------------------|
| 实现 | Replicator / 渲染 | Warp 射线打静态网格 |
| RGB / 真实渲染 | 支持 | 不支持（教程侧重深度/法线等） |
| 动态物体 / 全场景 | 适合完整渲染管线 | 仅配置中的 `mesh_prim_paths` |
| 适用场景 | 视觉、分割、真实感 | 快速几何深度、确定性、偏地形/静态 |

---

## 5. 实操检查清单

1. 相机类脚本：是否加了 `--enable_cameras`。
2. 多环境：路径是否含 `{ENV_REGEX_NS}`，与 `InteractiveSceneCfg` 的 `num_envs` 一致。
3. 接触力：机器人 USD/Spawner 是否已开启 contact reporting。
4. 射线类：`mesh_prim_paths` 是否指向场景中实际存在的碰撞/可视化网格。
5. 创建场景后是否 `sim.reset()`，循环中是否调用 `scene.update` 或传感器的 `update`。

---

## 6. 源码索引（仓库内）

- 主教程：`scripts/tutorials/04_sensors/add_sensors_on_robot.py`
- 文档：`docs/source/tutorials/04_sensors/add_sensors_on_robot.rst`
- 传感器实现：`isaaclab/sensors/`（阅读 `CameraCfg`、`RayCasterCfg`、`ContactSensorCfg` 等定义可扩展自定义）
