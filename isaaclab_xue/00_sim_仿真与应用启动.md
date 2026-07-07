# 00 仿真基础：空场景、启动、生成 Prim、渲染与日志

对应目录：`scripts/tutorials/00_sim/`。官方文档片段在 `docs/source/tutorials/00_sim/`（`create_empty.rst`、`launch_app.rst`、`spawn_prims.rst`）。

---

## 共同模式

1. **必须先启动 Kit**：`AppLauncher.add_app_launcher_args(parser)` → `AppLauncher(args_cli)` → 得到 `simulation_app`；业务代码写在 launch 之后（避免在未初始化 Kit 时 import 仿真相关模块）。
2. **仿真上下文**：`SimulationCfg` → `SimulationContext`；`sim.reset()` 后再进主循环；每帧 `sim.step()`。
3. **视口**：`sim.set_camera_view(eye, target)` 设置观察视角。

---

## `create_empty.py`

- 仅创建 `SimulationContext`，无额外 Prim。
- 用途：最小可运行骨架，确认 Isaac Lab 能正常起仿真循环。

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

---

## `launch_app.py`

- 在 `design_scene()` 里用 **spawn cfg 的 `.func(prim_path, cfg, ...)`** 放置地面、平行光、立方体。
- 自定义 CLI：`--size` 控制立方体边长；`--width` / `--height` 影响视口与生成图像（见脚本注释中的 SimulationApp 文档链接）。
- `SimulationCfg(dt=0.01, device=args_cli.device)` 与设备一致。

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/launch_app.py
```

---

## `spawn_prims.py`

- **地面**：`GroundPlaneCfg().func(...)`。
- **灯光**：`DistantLightCfg`，可传 `translation`。
- **几何体**：`ConeCfg`（纯视觉）、带 `rigid_props` / `mass_props` / `collision_props` 的刚体锥、`MeshCuboidCfg` + 可变形体材质生成软体方块。
- **外部 USD**：`UsdFileCfg(usd_path=...)` + `func` 实例化桌子等。
- 可先 `prim_utils.create_prim("/World/Objects", "Xform")` 再在其下组织子 Prim。

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py
```

---

## `set_rendering_mode.py`

（脚本头部 docstring 仍写 “spawn prims”，实际内容是渲染预设。）

- `RenderCfg(rendering_mode="performance"|…, carb_settings={...})` 传入 `SimulationCfg(render=render_cfg)`。
- CLI 的 `--rendering_mode` 会覆盖代码里的预设（见注释）。
- 示例加载 Nucleus 医院场景 USD，用于观察画质/性能差异。

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/set_rendering_mode.py
```

---

## `log_time.py`

- 配合 Docker 教程：在 `logs/docker_tutorial/log.txt` 中按物理步写入仿真时间 `sim_time`（累加 `sim.get_physics_dt()`）。
- 演示在容器内工作目录通常为 `/workspace/isaaclab` 时日志落盘位置。

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py
```

---

## 小结

| 脚本 | 重点 |
|------|------|
| `create_empty` | 最小循环 |
| `launch_app` | `SimulationCfg` + 简单 `design_scene` |
| `spawn_prims` | 多种 `*Cfg().func` 生成场景 |
| `set_rendering_mode` | `RenderCfg` 与大型 USD 场景 |
| `log_time` | 仿真时间与文件日志 |
