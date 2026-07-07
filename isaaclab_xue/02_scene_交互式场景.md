# 02 场景：`InteractiveScene` 与配置类

对应脚本：`scripts/tutorials/02_scene/create_scene.py`。文档：`docs/source/tutorials/02_scene/create_scene.rst`。

---

## 目的

把 01 里手写的 `scene_entities` 字典换成 **`InteractiveScene` + `InteractiveSceneCfg`**：在配置类里声明地面、灯光、机器人等，由场景统一生成、索引和更新。

---

## 要点

1. **配置类**：`@configclass class CartpoleSceneCfg(InteractiveSceneCfg)`。
2. **资产条目**：如 `ground = AssetBaseCfg(prim_path=..., spawn=GroundPlaneCfg())`；关节体用 `ArticulationCfg`，例如 `CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")`。
3. **多环境**：构造时 `CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)`；机器人路径必须含 **`{ENV_REGEX_NS}`**。
4. **运行时访问**：`scene["cartpole"]`（键为配置字段名，与 `replace` 后的资产名一致；此处 cfg 字段叫 `cartpole`）。
5. **循环**：
   - 重置：根状态加 **`scene.env_origins`**，关节加噪声，`scene.reset()`。
   - 每步：`scene.write_data_to_sim()` → `sim.step()` → `scene.update(sim_dt)`。
6. 与 01 的 `run_articulation` 相比，唯一结构性变化是**场景管理与原点**由 `InteractiveScene` 接管。

```bash
./isaaclab.sh -p scripts/tutorials/02_scene/create_scene.py --num_envs 32
```

---

## 与后续章节的关系

- **04 传感器**、**05 控制器**：继续在 `InteractiveSceneCfg` 上增加 `CameraCfg`、`ContactSensorCfg` 或机器人条目。
- **03 环境**：`ManagerBasedEnv` 内部仍使用基于 `InteractiveSceneCfg` 的 scene 配置（或 tasks 包中的 scene cfg）。
