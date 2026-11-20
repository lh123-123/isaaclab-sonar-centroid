# v7 域随机化（DR）清单与缺口

> 2026-11-17 整理。本文是**现状清单 + 缺口台账**，只回答两个问题：DR 都做了什么、还缺什么。
> 设计动机与历史取舍见 [`方案设计_v7_光流RGB视觉悬停.md §10.2`](方案设计_v7_光流RGB视觉悬停.md)，不要在两处重复维护。

**源码契约**（所有行号以 `source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/auv_tracking_v7_flow_hover_gsrov_env.py` 为准）：

| 位置 | 内容 |
|---|---|
| L351–441 | 全部 `dr_*` 与 `train_action_delay_*` 的 cfg 声明 |
| L1622–1678 | `_sample_domain_randomization(env_ids, per_episode)` —— 唯一的抽样入口 |
| L1688–1745 | `_apply_action` —— 推力增益、附加阻力的注入点 |
| L1452–1458 | `_camera_rgb` —— 相机增益与噪声的注入点 |
| L897 / L2104 | 两个调用点：`__init__` 走 `per_episode=False`，`_reset_idx` 走 `per_episode=True` |

---

## 一、一句话结论

**已接线 11 个旋钮，但其中 4 个在 `2026-11-16_23-57-37` 这批检查点里是空操作，1 个是"声明了但从没用过"的死旋钮。**
真正致命的缺口不在"随机得不��多"，而在**三条物理链路上一条都没覆盖**：传感器侧延迟（44.5 ms 完全未建模）、图像域（浑浊/光照/散射）、执行器时域（推进器阶跃响应 + 死区 + 电压跌落）。

---

## 二、抽样 Horizon 的设计（先理解这个，才看得懂表）

分两个 horizon，理由写在 `_sample_domain_randomization` 的 docstring 里：

- **`per_episode=True`（每次 reset 重抽）** —— 物理上每次下潜都会变的东西：水、出生位姿。
- **`per_episode=False`（只在 `__init__` 抽一次，整轮恒定）** —— 物理上一台机器只有一套的东西：相机标定/增益、电池、推进器装配偏差。**这就是"预随机"。**

这个划分本身是对的（真机一次下水 = 一台设备一套标定），但它带来一个已知副作用，见 §四的"可被适应"问题。

---

## 三、已实现的 DR —— 逐项台账

### 3.1 每回合重抽（水 + 位姿）

| 旋钮 | 范围 | 注入点 | 日志键 | 备注 |
|---|---|---|---|---|
| `dr_current_speed_range` | **(0.20, 0.45) m/s** | L1702–1704 | `dr_cur_speed` | 直接构造速度向量，**完全覆盖** `ocean_current` 基础场 |
| `dr_current_dir_full_circle` | True → **±180° 全圆** | L1645 | `dr_cur_deg` | 打开时 `dr_current_dir_range_deg` 被架空 |
| `dr_current_dir_range_deg` | 20.0 | L1648–1650 | — | **当前无效**，仅 `full_circle=False` 时生效 |
| `dr_current_vert_frac` | 0.10 → **±0.045 m/s** @0.45 | L1655 | — | 竖直分量 = ±frac × 速度 |
| `dr_spawn_xy_range` | **±0.12 m**（每水平轴） | `_reset_idx` | — | 首帧不再正对名义位置 |
| `dr_spawn_yaw_range_deg` | **±8°** | `_reset_idx` | — | anchor 匹配在 yaw≠0 时本就脆弱 |
| `dr_spawn_depth_range` | **±0.04 m** | `_reset_idx` | — | 深度 PI 会拉回 |
| `dr_visodom_ep_scale_rel` | ±5%（默认 **0.0 = 关**） | L1658–1660 | — | **新加，未启用**（见 §4.1） |
| `dr_visodom_bias_m` | ±20 mm/轴（默认 **0.0 = 关**） | L1661–1663 | — | **新加，未启用**（见 §4.1） |

> ⚠️ **`ocean_current` 基础场在 DR 打开时不生效。** L1692–1706 直接由 `_dr_cur_dir * _dr_cur_speed + _dr_cur_vz` 构造速度向量，源码注释写明理由："a fixed direction/strength would just add a second, contradictory source of truth"。所以 play 脚本的 `--current` 参数**只在 `dr_enable=False` 时才有意义**。

### 3.2 整轮一次（硬件 + 图像 + 标定）= 预随机

| 旋钮 | 范围 | 注入点 | 状态 | 备注 |
|---|---|---|---|---|
| `dr_thrust_gain_range` | **(0.90, 1.05)** | L1738 `_pwm.mul_` | ✅ 生效 | 作用在归一化杆量上，推力 ∝ pwm² → 实际 ±10%~21% |
| `dr_drag_gain_range` | **(0.0, 3.0)** N/(m/s) | L1744 `thrust -= g·v_w` | ✅ 生效 | 附加**线性**阻力，跨度为 0–3 倍，很宽 |
| `dr_cam_brightness_range` | (0.92, 1.08) | L1456 | ✅ 生效 | 曝光/增益 |
| `dr_cam_noise_std_range` | (0.004, 0.010) | L1458 | ✅ 生效 | 每帧注入，std×255，在 **render 分辨率**；flow/template/mask/策略看到同一帧 |
| `dr_visodom_focal_rel_range` | **(1.0, 1.0)** | L1682 | ❌ **空操作** | 乘 1.0 = 没做；代码默认值已改成 (0.90, 1.10) |
| `dr_visodom_range_rel_range` | **(1.0, 1.0)** | L1683 | ❌ **空操作** | 同上 |
| `dr_cam_mount_deg_range` | 0.0 | L1197 → L1235 | ❌ **关** | 这是"墙/相机倾斜"，等价于破坏 fronto-parallel 假设；因烘焙进几何，单独在 `_setup_scene` 里抽 |
| `dr_texture_choice` | False | **无** | 💀 **死旋钮** | 全文件只有 L396 一处声明，**没有任何读取点**。想用必须先接线 |

### 3.3 时间常数类（严格说不是 DR，但等效地决定任务分布）

| 项 | 值 | 备注 |
|---|---|---|
| `train_action_delay_fixed_steps` | 训练用 **3 步 = 200 ms**（默认 -1 = 关） | 15 Hz × 3 = 200 ms，整回合恒定；实机链路 201 ms |
| `train_action_delay_max_steps` | **0 = 关** | D 臂（0–3 步随机）已证伪，别开（见 §五） |
| `current_ramp_s` | 见 cfg | 洋流有爬升，不是一步到位 |
| 固定量（不随机） | `plane_distance=2.0 m`、`trim_pitch_rad=-0.3488`(≈-20°)、`kf_measurement=gt`、`episode_length_s=15` | 见 §四缺口 |

### 3.4　2026-11-18 新增：传感器 / 水 / 几何三块补齐

上一批"预随机"只覆盖了水、姿态、推力、阻力、相机增益噪声，而**传感器与几何侧一个都没随机**。以下 6 项全部落地，并由 §八 的 `debug_auv_v7_dr_coverage.py` 审计（本地与服务器均 47/47 PASS）。

| 旋钮 | 范围 | Horizon | 注入点 | 解决的是哪个"没处理好" |
|---|---|---|---|---|
| `dr_imu_yaw_bias_deg` | ±3° | **每环境一次** | `_sensed_yaw()` | 去旋转与 anchor 一直用**真值四元数**（免费信息）；零偏会直接转歪整条 visodom 积分 |
| `dr_imu_yaw_noise_deg` | 0–0.3° | 每帧 | `_sensed_yaw()` | 每 sim step 缓存一次，保证同一步内 flow 去旋转与 anchor 用同一个 yaw |
| `dr_drag_aniso_range` | ±30% × 3 轴 | **每环境一次** | `_apply_action` | 原来只有**标量** `dr_drag_gain`，等于假设三轴附加质量相同（方形壳体不成立） |
| `dr_cam_mount_deg_range` | **0.0 → 1.5°** | **每环境一次** | `_spawn_cam_mount_dr` → 墙面网格 | 代码路径早就有，默认 0 = 关；开了才真正破坏 fronto-parallel 假设 |
| `dr_texture_choice` | **False → True** | **每环境一次** | `_dr_texture_for_env()` | 原来是**死旋钮**（只声明、无读取点）；现在按 `cfg.seed + 7919·ei` 确定性抽海床，可复现 |
| `dr_turbidity_range` | (0.0, 0.6) | **每回合** | `_camera_rgb` | 对比度压制 + 3×3 软化。**这是"看得到但信息全是废的"**：浑浊会让 flow RMS 虚低，策略会误判"我停住了" |

顺带修掉一个结构问题：`dr_spawn_depth_range` 原来是在 `_reset_idx` 里**内联抽样**、不进统一采样器，也不留状态，审计和日志都看不见。现已提升为 `_dr_spawn_depth` 状态张量。

**注意**：`dr_visodom_focal/range/ep_scale/bias` 四项仍是 `(1.0,1.0)` / `0.0`（no-op）。这是 11-17 实测后**主动关闭**的——注入误差 15–58 mm 与要优化的 50–88 mm 同阶，奖励被噪声支配。审计脚本用 canary 机制（仅在测试里强制非零）证明它们**接线完好**，不会因为"测不出效果"而漏掉。



按"是否阻断部署"排序。P0 = 不改就会让 sim 结果在实机上不成立。

### P0-1　传感器侧延迟完全未建模

训练只覆盖了**执行侧** 200 ms（`train_action_delay_fixed_steps=3`）。实机全链路 201 ms 里还有**传感器侧 44.5 ms**（曝光 + 传输 + Pi 解码），这部分**一步都没有进观测链** —— 策略看到的是"当前帧"，实机看到的是"半帧之前的世界"。

- 影响：所有基于图像运动量的线索（光流、visodom 增量）都会系统性滞后；在 15 Hz 下 44.5 ms ≈ **0.67 步**。
- 补法（最小可行）：给图像队列加 1 步管线延迟（obs 用 N-1 帧的图，状态仍用当前），总死区变成 4 步 ≈ 267 ms；或按实测相机链路单独测一个数再定。
- 判据：加入后 `disp_spawn_m` 退化 ≤ 1.2× 才算可接受；否则说明 200 ms 死区本身就不够。
- ⚠️ 顺序：先把 §4.1 的估计器 DR 修好再动这个，否则两个变量混在一起无法归因。

### P0-2　估计器校准 DR 在这批检查点里没生效

`dr_visodom_focal_rel_range` / `dr_visodom_range_rel_range` 在 `params/env.yaml` 里是 **(1.0, 1.0)**，且 `dr_visodom_ep_scale_rel` / `dr_visodom_bias_m` **整个键都不存在** —— 说明训练启动时那批"根因修复"代码还没写进去。

- 后果：`visodom_err_m` 量的是"积分器自己的误差"，同时假装底下的尺子是完美的。
- 补法：把四个旋钮全开（f/range ±10% 每环境；±5% 尺度 + ±20 mm 偏置**每回合**），从 E 臂 200k 续训。
- 判据（三条同时满足）：① `disp_spawn_m` 后段 ≤ 前段 × 1.2；② `ep_rew_mean` 不再冻结在 548–550；③ 40 回合正式评估 ≤ 36.1 mm。

### P0-3　线索"硬失效"没有随机

现在的 `present` 门控靠**纹理像素数阈值**，只能模拟"看不到纹理"，模拟不了"看得到但全是废信息"：

- 浑浊/后向散射 → 对比度塌陷 → **光流 RMS 虚低**，策略会误判成"我停住了"（这是本任务最危险的单点失效）。
- 遮挡（悬浮物、管线、鱼）、纯黑帧、运动模糊。
- 补法：`dr_turbidity`（低通模糊 + 雾化对比度）、随机遮挡块、`dr_blur_motion`。
- 判据：这些开起来后，`ablate_*` 的**峰值退化倍数**是论文唯一还站得住的主张的直接证据（见 §六）。

### P1　图像域差距（sim2real 的主要来源）

| # | 缺什么 | 为什么重要 | 建议旋钮 |
|---|---|---|---|
| P1-1 | 浑浊度 / 后向散射 | 低对比度直接造假"零光流"，比噪声更致命 | `dr_turbidity_range`（模糊 σ + 对比度压缩） |
| P1-2 | 光照 / 曝光 / 短曝光运动模糊 | 实机要锁短曝光；训练里没有 | `dr_exposure_range` + `dr_motion_blur_len` |
| P1-3 | 纹理选择 | `dr_texture_choice` 是死旋钮；固定纹理 = 模板匹配风险未覆盖（实测跳变率 0–2.0%） | 接线 `dr_texture_choice`，或 `dr_texture_scale/rot` |
| P1-4 | 相机内参/畸变/vignetting/rolling shutter | 全景畸变会让 `du = a + kx` 拟合串味 | `dr_lens_distortion`、`dr_vignette` |
| P1-5 | 分辨率与压缩不一致 | 训练 640×360 → flow 176×320；实机 960×540 + **MJPEG**（8×8 块伪影） | 训练里加 JPEG 压缩旋钮最省事 |

### P2　执行器与动力学

| # | 缺什么 | 现状 | 建议 |
|---|---|---|---|
| P2-1 | **推进器阶跃响应** | 假设 100 ms，**从未实测** | 先上台架测，再按实测设 `dr_thruster_tau_range`（这是单项收益最大的台架动作） |
| P2-2 | 死区 / ESC 非线性 / 电压跌落 | `thruster_deadband` 存在但**固定不随机** | `dr_deadband_range`、`dr_battery_sag` |
| P2-3 | 阻力各向异性 / 附加质量 | `dr_drag_gain` 是**标量**，只做线性附加阻力 | `dr_drag_aniso`（前后/侧向/垂向分别缩放） |
| P2-4 | 浮力 / 重心偏差 | `trim_pitch_rad=-0.3488` 是**写死的常数** | `dr_trim_pitch/roll_range` |

### P3　环境与感知后端

| # | 缺什么 | 现状 | 建议 |
|---|---|---|---|
| P3-1 | 站位距离（standoff） | `plane_distance` **固定 2.0 m** | visodom 误差 ∝ Z，且实测 0.87 m 外扩时误差 586 mm → `dr_plane_distance_range` |
| P3-2 | 洋流时间演化 | `time_modulation=false`，只有 ramp | 突变工况是待测主张之一 → `dr_current_step_change` |
| P3-3 | 深度量测 | `kf_measurement=gt`（真值） | 实机是压力传感器 → `dr_kf_noise/bias` |
| P3-4 | yaw 来源 | 去旋转用**仿真真值四元数** | 实机是 IMU/罗盘 → `dr_imu_yaw_drift`（否则"纯视觉"的说法不成立） |

---

## 五、不要加的（已实测证伪，别重复踩）

| 做法 | 结论 | 依据 |
|---|---|---|
| 随机动作延迟 0–3 步 | ❌ **失败** | D 臂平台 130–165 mm，250k 后崩到 300 mm（p90 1376）。每步延迟不同 → 同一观测对应不同"已生效动作" → **部分可观测**，是信息不足不是训练不足 |
| 逐帧 visodom 噪声 ±3 mm | 不值得 | 光流积分本身已带噪声，加一层只抬方差，不改变**可被利用的偏置**；尺度误差已让误差随位移线性增长（Δ = Δpx·Z/f），形状已经对了 |
| 只开"每环境一次"的标定 DR | ❌ 不够 | 整轮恒定的偏置正是策略能适应掉的东西（策略会停在让读数归零的位置）。必须配**每回合**重抽的尺度 + 加性偏置 |
| 用 `(1.0, 1.0)` "假装开了 DR" | ❌ 自欺 | 乘 1.0 是数学上的 no-op，但日志和文档会让人以为已经注入 —— 本批检查点就吃了这个亏 |

---

## 六、"DR 做得对不对"的验收指标

DR 本身没有分数，只能通过这三条间接验收：

1. **饱和判据**：`ep_rew_mean` 是否冻结在 548–550（满分 571）。冻结 = 奖励分辨不出 5 mm 与 38 mm，DR 再花哨也没意义。
2. **单调性判据**：`disp_spawn_m` 对 `dr_cur_speed` 是否单调上涨。现在 RL 是 26→38 mm（1.4×）而 LQR 是 3.3→4.9 mm —— **说明策略部分随流漂，DR 没逼出真正抵消扰动的解**。
3. **退化倍数判据**（唯一还站得住的主张）：`ablate_flow` / `ablate_template` / `ablate_visodom` 分别清零后的**峰值退化倍数**。基线 P 1.56× / PID 1.81× / LQR 4.43×；**RL 侧至今未测**。注意 `debug_v7_ablation.py` 里的 `ALL PASS` 只是结构验证（确认开关把图像/状态/奖励项归零了），**不是**退化倍数。

---

## 七、补充顺序建议（单变量原则，一次只改一个）

| Step | 改什么 | 为什么排这个位置 | 判据 |
|---|---|---|---|
| 1 | 开四个 visodom 校准旋钮（P0-2） | 唯一已定位的根因，且改动最小、可单变量归因 | §四 P0-2 的三条判据 |
| 1b | 若仍退化 → 单独改 `sigma_visodom_fine_m: 0.1 → 0.05` | 保持单变量，不跟 Step 1 混 | 同上 |
| 2 | 传感器侧延迟 1 步（P0-1） | 决定 200 ms 死区到底够不够 | 退化 ≤ 1.2× |
| 3 | 浑浊度 + 曝光（P1-1/2） | 直接冲击光流线索，是"优雅退化"证据的来源 | `present` 有效率下降但 `disp_spawn_m` 不爆 |
| 4 | 线索丢失消融（不是 DR，但是验收） | 三条主张里唯一站得住的 | RL < 1.56× 成立；> 4.43× 全灭 |
| 5 | 台架测推进器阶跃 → 再设 DR（P2-1） | 未实测的假设，单项收益大于所有软件优化 | 实测值替换 100 ms 假设 |

---

## 八、复现 / 快速核对

```bash
# 1) 看清某个检查点到底用了哪套 DR（唯一可信来源）
python - <<'PY'
import yaml
p = "logs/sb3/Isaac-AUV-Tracking-Direct-v7-flow-hover/2026-11-16_23-57-37/params/env.yaml"
c = yaml.safe_load(open(p))
for k, v in c.items():
    if k.startswith(("dr_", "train_action_delay")) or k == "ocean_current":
        print(f"{k} = {v}")
PY

# 2) 带 UI 复现训练条件（--visodom_cal_dr off 才对齐这批检查点）
./isaaclab.sh -p v7/scripts/play/play_auv_v7_flow_hover_ui.py \
  --checkpoint logs/sb3/Isaac-AUV-Tracking-Direct-v7-flow-hover/2026-11-16_23-57-37/model_100000_steps.zip \
  --visodom_cal_dr off --action_delay_fixed 3
```

**核对清单**（拿到任何检查点先跑这个）：

| 检查 | 通过标准 |
|---|---|
| `dr_visodom_focal_rel_range` / `range_rel_range` | 不是 (1.0, 1.0) |
| `dr_visodom_ep_scale_rel` / `dr_visodom_bias_m` | 键存在且 > 0 |
| `train_action_delay_fixed_steps` | = 3（或实机实测对应值） |
| `dr_current_speed_range` | (0.20, 0.45) |
| `dr_spawn_*` | 0.12 / 8.0 / 0.04 |
| `ocean_current.type` | 无所谓（DR 打开时被覆盖） |

---

## 九、维护约定

- 新增 DR 旋钮 → 必须先在本表 §三 登记（键名、范围、抽样 horizon、注入点行号），再改代码。
- **`(1.0, 1.0)` / `0.0` 的默认值必须在表里标成"空操作 / 关"**，避免下次再出现"以为开了其实没开"。
- 声明了但没接线读的键要在表里标 💀（`dr_texture_choice` 就是例子）。
- 不要覆盖已有的日志键语义。
