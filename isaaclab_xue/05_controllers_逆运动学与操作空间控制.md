# 05 控制器：微分 IK 与操作空间控制（OSC）

对应目录：`scripts/tutorials/05_controllers/`。文档：`docs/source/tutorials/05_controllers/run_diff_ik.rst`、`run_osc.rst`。

---

## `run_diff_ik.py` — `DifferentialIKController`

- **场景**：`InteractiveSceneCfg` 含地面、灯光、桌子 USD、机械臂（**`--robot franka_panda`** 或 **`ur10`**）。
- **控制器**：`DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")`；`set_command` 目标位姿 `[x,y,z,qx,qy,qz,qw]`（per-env 可同一目标）。
- **Jacobian**：`robot.root_physx_view.get_jacobians()`；**固定基座**时 `ee_jacobi_idx = body_ids[0] - 1`（根不在 Jacobian 行里）；移动基座用 `body_ids[0]`。
- **位姿在根系**：`subtract_frame_transforms` 将末端世界位姿变到 root frame，与 IK 内部一致。
- **输出**：`joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)`，再 `set_joint_position_target(..., joint_ids=...)`。
- **`SceneEntityCfg.resolve(scene)`**：解析 `joint_names`、`body_names` 到索引。
- 可视化：`VisualizationMarkers` 显示当前 EE 与目标帧。

```bash
./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py --robot franka_panda --num_envs 128
```

---

## `run_osc.py` — `OperationalSpaceController`

- **场景**：倾斜墙（`CuboidCfg` + `kinematic_enabled=True`）、`ContactSensorCfg` 挂在墙上测接触；Franka，部分关节刚度/阻尼置 0、`disable_gravity` 等用于教程设定。
- **OSC 配置**：`target_types=["pose_abs", "wrench_abs"]`，阻抗 `variable_kp`，任务空间轴向选择 `motion_control_axes_task` / `contact_wrench_control_axes_task`；`nullspace_control="position"` 用软限位中心作零空间目标。
- **状态更新**（`update_states`）：
  - Jacobian、广义质量阵、重力补偿来自 `root_physx_view`。
  - Jacobian 从世界系转到根/体轴：`matrix_from_quat(quat_inv(root_quat))` 乘子块。
  - 接触力：更新 `contact_forces`，对历史 `net_forces_w_history` 做均值与 max 简化成 EE 受力（教程式简化）。
- **控制循环**：`osc.compute(...)` 得到 **`joint_efforts`**，`set_joint_effort_target`；重置时 `osc.reset()`、`set_command` 与任务系转换 `convert_to_task_frame`。
- 渲染：`sim.step(render=True)` 保证 GUI 下可视化。

```bash
./isaaclab.sh -p scripts/tutorials/05_controllers/run_osc.py --num_envs 128
```

---

## 小结

| 脚本 | 控制器 | 执行器接口 | 依赖动力学 |
|------|--------|------------|------------|
| `run_diff_ik` | 微分 IK | 关节位置目标 | Jacobian |
| `run_osc` | 操作空间 | 关节力矩 | Jacobian + 质量矩阵 +（示例）接触 |

两者都依赖 **`InteractiveScene`** 与正确的 **`SceneEntityCfg` / 体索引**；多环境并行时命令与缓冲区 shape 为 `num_envs`。
