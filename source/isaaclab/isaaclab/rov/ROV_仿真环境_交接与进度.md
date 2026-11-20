# ROV 仿真环境 — 进度交接与后续计划

> 生成时间：2026-10-30  
> 目的：在上下文过长时保留完整进度，便于新对话直接续接。  
> 当前阶段：**已完成真机模型 STEP→USD 转换与几何/推进器数据提取**；仿真环境与强化学习训练**尚未开始实现**。

---

## 0. TL;DR

- 真机模型是 `GS-ROV_drl.STEP`（**不是**旧的 `Robocup-ROV`）。
- 已转 USD，并提取了包围盒、体积、质心、8 个推进器位置与推力轴、6×8 推力分配矩阵。
- 已把数据写成 JSON + Markdown，工具脚本齐全，可重复生成。
- 下一步：基于 `isaac-auv-env` 的流体力学/推进器模块，搭建 8 推进器 ROV 的 Isaac Lab 直接 RL 环境，先做定深悬停与位置移动，再扩展抓取，最终 sim2real。

---

## 1. 本轮已完成

| # | 事项 | 状态 |
|---|---|---|
| 1 | `GS-ROV_drl.STEP` → `GS-ROV_drl.usd`（Hoops 转换器） | ✅ |
| 2 | 整机包围盒 / 体积 / 质心统计 | ✅ |
| 3 | 8 个推进器位置（点云 AABB 中心） | ✅ |
| 4 | 8 个推进器推力轴（面法向协方差法） | ✅ |
| 5 | 6×8 推力分配矩阵 TAM | ✅ |
| 6 | 机器可读数据 `GS-ROV_drl_data.json` | ✅ |
| 7 | 人读文档 `GS-ROV_真机模型_仿真数据.md` | ✅ |
| 8 | 标注旧 `Robocup-ROV` 文档为非真机 | ✅ |
| 9 | 调研 Isaac Lab 现有可复用模块 | ✅ |
| 10 | 仿真环境 / RL 训练实现 | ❌ 未开始 |

---

## 2. 真机关键数据

来源：`GS-ROV_drl.STEP` 转换后的 USD 几何实测。

### 2.1 整机

| 参数 | 数值 |
|---|---|
| 包围盒 (m) | 0.56227 × 0.42860 × 0.29500 |
| 包围盒 min (m) | (-0.26429, -0.04718, 0.81300) |
| 包围盒 max (m) | ( 0.29798,  0.38142, 1.10800) |
| 闭合实体总体积 | 8.3426 L |
| 体积质心 COM (m) | (0.01708, 0.16763, 0.95957) |
| Mesh / 三角面 | 62 / 211,148 |
| 单位 / 上轴 | metersPerUnit=0.001，Z-up |

> ⚠️ 模型原点在 CAD 原点 (0,0,0)，几何位于 z≈0.81–1.11 m。**建议后续生成以 COM 为原点的居中 USD**（计划中的下一步），否则刚体/推进器/重力参考点容易错位。

### 2.2 质量相关（关键缺口）

| 参数 | 状态 |
|---|---|
| 整机质量 | **待称重 [MISS]** |
| 惯性张量 | **待 CAD 导出/扭摆 [MISS]** |
| 实际排水 / 浮力材浮力 | **待排水法实测 [MISS]**（8.34 L 为实体体积和，非严格排水） |

---

## 3. 推进器布局与推力分配

### 3.1 布局结论

**4 个垂直推进器（中部矩形）+ 4 个水平矢量推进器（外侧四角，45° 斜置）**，共 8 个。

- 垂直 4 台位于 z ≈ +62.8 mm（质心上方）。
- 水平 4 台位于 z ≈ -1.2 mm（质心高度），成两对平行：4/5 沿 (+X,+Y)，6/7 沿 (+X,-Y)。
- 两对水平推进器正交，可覆盖全平面平移；对内差动产生偏航。

### 3.2 位置与推力轴（相对 COM，米；轴为世界/机体系单位向量）

| # | 类型 | 相对 COM 位置 | 推力轴 |
|---|---|---|---|
| 1 | 垂直 | (-0.11689, 0.16213, 0.06277) | (0, -0.0126, 0.9999) |
| 2 | 垂直 | (-0.11689, -0.16315, 0.06277) | (0, 0.0126, 0.9999) |
| 3 | 垂直 | (0.11461, -0.16315, 0.06277) | (0, 0.0126, 0.9999) |
| 4 | 水平 | (0.23041, -0.14957, -0.00123) | (0.7071, 0.7070, -0.0126) |
| 5 | 水平 | (-0.24170, 0.14856, -0.00123) | (0.7071, 0.7070, 0.0126) |
| 6 | 水平 | (0.23041, 0.14855, -0.00123) | (0.7070, -0.7071, -0.0126) |
| 7 | 水平 | (-0.24170, -0.14958, -0.00123) | (0.7070, -0.7071, 0.0126) |
| 8 | 垂直 | (0.11461, 0.16213, 0.06277) | (0, -0.0126, 0.9999) |

### 3.3 推力分配矩阵 TAM（6×8，单位推力）

行 = `[Fx, Fy, Fz, Mx, My, Mz]`，列 = 推进器 1..8；`wrench = TAM · thrust`。

```
Fx:  [ 0.000  0.000  0.000  0.7071  0.7071  0.7070  0.7070  0.000 ]
Fy:  [-0.0126 0.0126 0.0126  0.7070  0.7070 -0.7071 -0.7071 -0.0126]
Fz:  [ 0.9999 0.9999 0.9999 -0.0126  0.0126 -0.0126  0.0126  0.9999]
Mx:  [ 0.1629 -0.1639 -0.1639  0.0028  0.0027 -0.0027 -0.0028  0.1629]
My:  [ 0.1169  0.1169 -0.1146  0.0020  0.0022  0.0020  0.0022 -0.1146]
Mz:  [ 0.0015 -0.0015  0.0014  0.2687 -0.2759 -0.2679  0.2767 -0.0014]
```

### 3.4 推进器规格（ROVMAKER，产品手册）

| 参数 | 数值 |
|---|---|
| 额定电压 / 范围 | 24 V / 12–26 V（3S–6S） |
| 最大功率 / 电流 | 300 W / 15 A |
| 最大正/反向推力 | 19.61 N / 17.65 N（2.0 / 1.8 kgf） |
| PWM 范围 / 中立 | 1000–2000 µs / 1500 µs |
| 飞控输出端点 | 1100 / 1900 µs |
| 单台重量 | 200 g |
| 外形 | Ø64 × 101 mm |
| 耐深 | 300 m |
| 系统功率/电流限 | 480 W / 20 A |

> 8 台理论满推 2.4 kW / 120 A，远超电源 600 W 与软件限流 480 W → **动作空间必须做功率归一化**。

---

## 4. 关键结论与风险

1. **推力轴方向符号未定**：几何只给无符号轴；正/反转需结合实物电机转向与 `MOT_n_DIRECTION`。TAM 采用「垂直 +Z 为正、水平 +X 分量为正」的统一约定。
2. **质心为体积质心**，非配重/电池/浮力材调整后的真实重心。
3. **体积 8.34 L 非严格排水**，实心建模零件会高估；需排水法。
4. **模型原点远离质心**（z≈0.96 m），实现刚体前应居中到 COM。
5. 旧 `Robocup-ROV` 数据（旧 md）与真机不同，训练一律用 GS-ROV。

---

## 5. 后续实现计划

### 阶段 A：可复用模块调研（已完成，记录在此）

Isaac Lab 版本：**2.2.1**；conda 环境：**isaaclab**；工作目录：`/home/lh/IsaacLab`。

仓库内已有高质量水下模块，**优先复用**：

| 模块 | 路径 | 可复用内容 |
|---|---|---|
| 流体力学 | `source/isaaclab_tasks/isaaclab_tasks/direct/isaac-auv-env/rigid_body_hydrodynamics.py` | 浮力 / 二次阻力 / 线性粘性力（MuJoCo 流体模型） |
| 推进器动力学 | `.../isaac-auv-env/thruster_dynamics.py` | 一阶响应 + `thrust = k·|u|·u` 转换 |
| 参考环境 | `.../isaac-auv-env/warpauv_env.py` | `_compute_dynamics` 完整流程、`set_external_force_and_torque` 用法 |
| 其它水下任务 | `.../isaac_auv_tracking/`、`.../lh_2/`、`.../nonar2/`、`.../sonar_flow_hover/` | 观测/奖励/域随机化/声呐与视觉任务范式 |

> 注：`isaac-auv-env` 目录名带连字符，不能直接 `import`；**建议把上面两个物理解本复制进新任务包**，避免导入问题。

### 阶段 B：搭建 ROV 环境包（下一步）

建议新建：

```
source/isaaclab_tasks/isaaclab_tasks/direct/gs_rov/
  __init__.py            # gym.register("Isaac-GSrov-Direct-v0", ...)
  gs_rov_env_cfg.py      # 场景/刚体/域随机化/动作观测奖励配置
  gs_rov_env.py          # DirectRLEnv：8 推进器动作 + 流体力学 + 任务逻辑
  gs_rov_hydrodynamics.py# 复制自 isaac-auv-env 并适配
  gs_rov_thrusters.py    # 8 推进器（一阶响应 + 转换函数 + 位置/轴）
  agents/rsl_rl_ppo_cfg.py
  README.md
```

**刚体设置要点**：
1. 生成**以 COM 为原点的居中 USD**（`models/GS-ROV_drl_centered.usd`，把几何平移 -COM）。
2. RigidBodyProperties：mass / inertia（先给占位值并做域随机化，等实测再替换）。
3. 8 推进器：位置用 `center_rel_com_m`，方向用 `axis_world`（居中后即机体系方向）。
4. 推进器力：`thrust_i = k·|a_i|·a_i`，`F_i = thrust_i · u_i`，`τ = Σ r_i × F_i`。
5. 流体：浮力（volume，COB≈COM）、二次阻力 + 线性粘性（用质量/惯量推断等效半尺寸）。
6. 功率约束：对 8 个指令加 `Σ|a_i|^3·P_max ≤ P_limit` 的归一化或惩罚。

### 阶段 C：任务与 MDP

任务（按优先级）：
1. **定深悬停**：目标深度 + 姿态稳定，奖励深度误差/姿态/角速度/动作平滑。
2. **移动到指定位置**：目标 3D 位置，奖励位置误差 + 收敛速度 + 平滑。
3. **抓取物体**：先做靠近-对准-夹爪开合（夹爪行程 0–120 mm），再联合控制。

观测（初始版）：root 位置/姿态/线速度/角速度、深度、目标相对位置、上一步动作。  
域随机化：质量、体积/浮力、阻尼、质心偏移、推进器推力系数、传感器噪声。

### 阶段 D：训练与 sim2real

- 训练入口：复用 `scripts/reinforcement_learning/rsl_rl/train.py`，`--task Isaac-GSrov-Direct-v0`。
- 日志：TensorBoard / rsl_rl 自带；play 脚本导出轨迹 npz（参考仓库已有 commit）。
- sim2real：按实测质量/排水/推力曲线替换占位参数；先仿真定深悬停，再实物低速验证。

---

## 6. 待实测数据清单

| 数据 | 用途 | 获取方法 |
|---|---|---|
| 整机质量 | 刚体 mass | 称重 |
| 惯性张量 | 刚体 inertia | CAD 导出 / 扭摆 |
| 实际排水体积 | 浮力 | 排水法 |
| 浮力材真实体积/位置 | 浮力/浮心 | 几何+称重 |
| 推力-电流曲线 | 推进器模型 | 水下测功（PWM 1100→1900 扫描） |
| 推进器正方向 ↔ MOT_n_DIRECTION | 控制符号 | 实物低推力验水流 |
| 水动力阻尼系数 | 阻力模型 | 拖曳/CFD/系统辨识 |
| 传感器噪声 | 观测噪声 | 数据手册/静止数据 |

---

## 7. 环境与运行命令

```bash
# 环境
cd /home/lh/IsaacLab
conda activate isaaclab          # Isaac Lab 2.2.1

# 重新转换 STEP -> USD
conda run -n isaaclab python source/isaaclab/isaaclab/rov/tools/cad_to_usd.py \
  "/home/lh/Desktop/rov jixie/ROV/GS-ROV_drl.STEP" \
  "/home/lh/Desktop/rov jixie/ROV/GS-ROV_drl.usd"

# 重新生成几何/推进器/TAM 数据
conda run -n isaaclab python source/isaaclab/isaaclab/rov/tools/build_rov_data.py \
  "source/isaaclab/isaaclab/rov/models/GS-ROV_drl.usd" \
  "source/isaaclab/isaaclab/rov/models/GS-ROV_drl_data.json"

# 查看单台推力轴
conda run -n isaaclab python source/isaaclab/isaaclab/rov/tools/thruster_normal_world.py \
  "source/isaaclab/isaaclab/rov/models/GS-ROV_drl.usd"
```

---

## 8. 文件索引

| 路径 | 说明 |
|---|---|
| `/home/lh/Desktop/rov jixie/ROV/GS-ROV_drl.STEP` | 真机源文件 |
| `.../rov/models/GS-ROV_drl.usd` | 真机 USD（仓库副本） |
| `.../rov/models/GS-ROV_drl_data.json` | 几何+推进器+TAM（机器可读） |
| `.../rov/GS-ROV_真机模型_仿真数据.md` | 真机数据文档 |
| `.../rov/ROV_实机与仿真数据_20261028.md` | 旧 Robocup 数据（仅供对照，已标注非真机） |
| `.../rov/tools/cad_to_usd.py` | STEP→USD |
| `.../rov/tools/build_rov_data.py` | 位置/轴/TAM/JSON |
| `.../rov/tools/analyze_rov.py` | 包围盒/体积/COM |
| `.../rov/tools/thruster_normal_world.py` | 逐台推力轴（面法向协方差） |
| `.../rov/tools/` 其余脚本 | 中间分析（parts/PCA/detail 等，可保留作溯源） |

---

## 9. 新对话续接提示（可直接粘贴）

> 请先阅读 `source/isaaclab/isaaclab/rov/ROV_仿真环境_交接与进度.md` 和 `GS-ROV_真机模型_仿真数据.md`。  
> 目标：在 Isaac Lab 2.2.1 中，基于真机 `GS-ROV_drl.usd` 与 `GS-ROV_drl_data.json`（8 推进器 + TAM），复用 `isaac-auv-env` 的 `rigid_body_hydrodynamics.py` / `thruster_dynamics.py`，搭建 `gs_rov` 直接 RL 环境。  
> 第一步：生成以 COM 为原点的居中 USD；第二步：实现环境包与定深悬停任务并 smoke test；第三步：扩展到位置移动与抓取；第四步：按实测数据做 sim2real。

---

## 10. 续接记录（2026-10-30，GS-ROV v7 框架）

- 新增轻量代理环境 `auv_tracking_v7_flow_hover_gsrov_env.py`（`Isaac-AUV-Tracking-Direct-v7-flow-hover-gsrov`），机器人换为 GS-ROV 代理（box + 质量/体积占位），相机 prim 程序化生成，沿用 v7 光流零化 + 纹理门控 + 深度/姿态奖励。
- 生成以 COM 为原点的居中 USD：`models/GS-ROV_drl_centered.usd`（平移 -COM，附加 RigidBody/Mass API 与 LookFollowCamera0）；因 CAD 过重，仿真暂用 box 代理，居中 USD 保留作后续真机可视化/实验。
- 新增调试脚本 `v7/scripts/debug/debug_auv_v7_flow_hover_gsrov.py/.sh`，smoke test 通过（obs 契约、flow/texture、reward 有限）。
- 训练方案沿用 v7 SAC（`sb3_sac_v7_flow_hover_cfg.yaml`），策略/观测/奖励契约不变；后续按真机质量/排水/推力曲线替换占位参数。
- 正式实验场景仍为 `view_textured_ground.py`（水下场景），待将 v7 环境与该场景集成。
