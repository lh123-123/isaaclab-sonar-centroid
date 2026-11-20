# 方案设计：全局光流零化悬停 DRL（Direct-v7-flow-hover-gsrov）

版本：2026-11-03 训练实现对接版（以部署侧《方案_v3_对接版_20261103.md》为准）
适用任务：`Isaac-AUV-Tracking-Direct-v7-flow-hover` 与 `-v7-flow-hover-gsrov`（两者现均指向同一 GS-ROV 实现）
实现文件：`auv_tracking_v7_flow_hover_gsrov_env.py`
动作与动力学：8 推进器 GS-ROV 布局（`thruster_layout="gsrov8"`）

> 本文描述当前 GS-ROV v7 训练实现。`Isaac-AUV-Tracking-Direct-v7-flow-hover` 与 `-v7-flow-hover-gsrov` 两个注册任务现在都指向 `auv_tracking_v7_flow_hover_gsrov_env.py`。

---

## 0. 实现状态与部署契约

| 项 | 当前 v7 GS-ROV 实现 |
|---|---|
| 相机 / 渲染 | 仿真 tiled camera 渲染 640×360（`render_height/width`）；光流在 **176×320** 上计算；策略图 64×64 |
| 相机 HFOV | 81.65° |
| 策略频率 | `decimation=8`、`sim.dt=1/120 s` → **15 Hz**（66.7 ms/步） |
| 光流 | OpenCV Farneback `(pyr_scale=.5, levels=3, winsize=21, iterations=2, poly_n=5, poly_sigma=1.2, flags=0)` @ 176×320 |
| policy image | `(11,64,64)`：RGB(3)、du、dv、flow magnitude、texture mask、anchor 偏移(2)、visodom(2) |
| state | `(17,)`（仅 critic 用）|
| action | `(3,)`：surge、sway、yaw，范围 `[-1,1]` |
| 执行器 | 8 个推进器，4 vertical + 4 horizontal，`_pwm.shape=(num_envs,8)` |
| heave | 策略动作不输出 heave；深度由 PID 以 `starting_depth` 保持 |
| sway | 已接入：`actions[:,1]` → 横移目标，PID 经 4 水平推进器 Fy 分量消减侧漂 |

策略频率已统一为 **15 Hz**（`decimation=8`、`sim.dt=1/120`），早期文档里的 20 Hz / `decimation=6` 作废。
20 Hz 数值不代表端到端部署时延已通过验证。部署文档中的 `L_camera` 仍待实测；若延迟约束不满足，训练频率需据实调整。

---

## 1. 观测契约

```python
image = Box(low=-1, high=1, shape=(11, 64, 64), dtype=float32)   # actor 唯一输入
state = Box(shape=(17,), dtype=float32)                          # critic only（非对称 AC）
```

| 通道 | 内容 | 处理 |
|---|---|---|
| 0–2 | tiled camera RGB | `/255`，范围 `[0,1]` |
| 3 | `du / 320` | Farneback 位移下采样至 64×64 后，按 **flow-res 宽(320)** 归一化，clip `[-1,1]` |
| 4 | `dv / 176` | 同上，按 **flow-res 高(176)** 归一化，clip `[-1,1]` |
| 5 | `sqrt(du_norm²+dv_norm²)` | 由归一化后的 du/dv 计算 |
| 6 | texture mask | render 分辨率算梯度阈值，再 adaptive max pool 至 64×64，值为 0/1 |
| 7–8 | 首帧模板偏移 `du/margin_u`, `dv/margin_v` | 广播成常数通道，±1 = 匹配窗边缘（≈±0.9 m @ Z=2 m） |
| 9–10 | visodom 位移 `x/σ`, `y/σ` | 广播成常数通道，σ=`sigma_visodom_m`=1.0 m；跨模板盆地仍继续计数 |

`frame_stack=1`，无历史帧堆叠。Farneback 由 CPU OpenCV 执行；部署侧 OpenCL 速度数据不能直接视作训练端耗时。`state` **17 维**（见 §1 表与 §10.3；早期文档写的"九维"作废），其中 `flow_rms` 是 64×64 降采样流场上纹理区域的像素 RMS；它不是部署侧原始 1920×1080 流 RMS 的同量纲直接值。

首次取得帧（或 reset 后首帧）无前序帧可计算光流，flow 与 flow_rms 置零。

---

## 2. 动作与推进器

策略动作 shape 为 `(3,)`，语义 `[surge, sway, yaw]`，范围 `[-1,1]`。GS-ROV PID 三个维度都参与控制：surge→Fx、sway→Fy、yaw→Mz，全部经 `BatchPIDController.gsrov8` 分配到四个水平推进器；heave 由深度 PID 独立维持。三维接口与部署动作契约对齐。

v7 不是逐推进器 8 维策略动作。策略输出 3 维控制量，经 `BatchPIDController` 的 GS-ROV 分配矩阵转换为八路归一化推进器命令，再由动力学模型计算推力。推进器顺序为 CAD 几何顺序 `[V1,V2,V3,H1,H2,H3,H4,V4]`。部署输出若需映射到飞控 MAIN 输出，须按 `GSROV_ACTION_TO_MAIN` 排列，不能假定 CAD 顺序即 MAIN 顺序。

GS-ROV 推力模型为带 deadband 的有符号二次曲线 `T = T_max * u * |u|`；额定推力参数当前来自数据表，电池电压默认 24 V。部署杆量到 PWM 的 `1500 + 400u → [1100,1900] µs` 是飞控接口契约；仿真这里由连续归一化命令与推力曲线近似，不在策略动作预处理处直接生成微秒 PWM。

---

## 3. 奖励、门控与终止

奖励结构：纹理存在时以 flow RMS 高斯项为主，叠加 alive、深度、姿态项并惩罚动作变化。配置初值：`rew_flow=1.0`、`rew_alive=0.3`、`rew_depth=0.1`、`rew_att=0.1`、`rew_action=0.01`、`sigma_flow=0.05`（**口径见下方复核说明；旧文档"20 Hz 下 std≈0.047，取 1×std"已不可复现**；仍可通过 `env.sigma_flow=...` 进行消融）、`sigma_depth=1.5 m`、`sigma_att=0.3 rad`。

`present = texture_count >= min_texture_px`，当前 `min_texture_px=20`。无纹理连续 50 个策略步（**15 Hz 下约 3.3 s**；早期文档按 20 Hz 写 2.5 s，作废）终止；深度偏离 `starting_depth` 超过 1.5 m、roll/pitch 合成姿态误差超过 0.3 rad 或越界也终止；最大 episode 长度 **15 s / 225 步**。无纹理时 flow/alive 奖励被门控为零，但深度/姿态项仍然计算。`min_texture_px=20`；`flow_mag_min_px`/`flow_mag_rel_frac` 已移除（无使用处）。

> **`sigma_flow=0.05` 的标定口径已复核（2026-11-16），结论：值不改，理由要改。**
> 本文档旧版称 0.05 来自"20 Hz 下实测 std≈0.047"，**该值在当前配置下无法复现**。现状（15 Hz、DR 开、覆盖完整 10 s 洋流爬坡、225 步）：flow_rms std = 0.150 px → 归一化 **0.0023**（稳态段 0.0026），比 0.047 小 20 倍。方向也不符 —— 15 Hz 每帧位移比 20 Hz 大 4/3，flow 只会更大。旧值对应的漂移速度约 0.49 m/s，而当前零动作漏速仅 0.053 m/s（≈0.33 px，与实测 0.29 px 吻合）。
>
> **不要把 sigma 改成 1× 噪声 std（0.0023）**：那会让 `exp(−(x/0.0023)²)` 在漂移超过 ~2 cm/s 时就饱和，工作区间内没有梯度。0.05 实际定义的是**约 0.52 m/s 的盆宽**：守位 0.99、0.30 m/s 随流 0.71、0.45 m/s 随流 0.47，能区分"顶住"和"随流漂"，是合理取值。
>
> 另需记住的既有结论：零动作漏速（5 cm/s）与完美悬停（5 mm）在 flow 项上**都是 0.99**，光流项分辨不出这个量级 —— 细梯度由 visodom 位置项承担，这也是当初引入它的原因。`calibrate_v7_sigma_flow.py` 的 `max(0.05, ...)` 下限会把任何低于 0.05 的测量钳到 0.05，**看它的 suggested 值没有信息量**，要看打印的原始 std。

纹理 mask 是基于灰度梯度阈值 `8.0` 的训练仿真门控，不等同于部署端 `present` 的实测绝对阈值。实际池底噪声、噪声底与 flow 阈值应待部署侧现场统计后再标定。

---

## 4. 频率与分辨率注意事项

- 本实现 `sim.dt=1/120 s`、`decimation=8`，策略频率为 `120/8=15 Hz`（66.7 ms/步）。早期文档写的 `decimation=6 / 20 Hz / 20.83 Hz` 全部作废。
- 图像策略输入固定 64×64，光流在 **176×320** 上计算（render 640×360 先行面积下采样），最后插值到 64×64 给策略。**不要把 64×64 当作光流估计分辨率**，也不要把 render 分辨率当成 1280×720（那是 tiled camera 的 spawn 配置，实际 `render_width/height=640×360`）。
- 当前光流实现使用核心 OpenCV API `cv2.calcOpticalFlowFarneback`（参数 pyr_scale=0.5, levels=3, winsize=21, iterations=2, poly_n=5, poly_sigma=1.2, flags=0）。该函数随 `opencv-python` / `opencv-python-headless` 一起提供，无需 contrib 包；返回 `(H,W,2)` 后转置为 `(2,H,W)` 通道优先。
- 部署方案给出的 GPU/OpenCL 耗时与此 CPU OpenCV 路径不同，训练性能须单独实测。

---

## 5. 与旧设计稿的差异

| 旧稿内容 | 当前 GS-ROV v7 |
|---|---|
| WarpAUV 代理 | GS-ROV CAD hull / 8 推进器动力学 |
| RAFT | Farneback OpenCV |
| 256×256 渲染、HFOV 90° | **640×360** 渲染（光流在 176×320 上算）、HFOV **81.65°** |
| frame stack=2、state 10 维 | 单帧 **11 通道**图像、state **17 维** |
| 15 步纹理丢失终止 | 50 步（**15 Hz** 下约 3.3 秒） |
| 六推进器 | 八推进器；策略仍输出三维而非八路 motor action |


---

## 6. 训练验收清单

1. 单 env reset + step，确认 observation `image=(N,11,64,64)`、`state=(N,17)`、action `(N,3)`。
2. 确认环境 `num_thrusters == 8` 且 `_pwm.shape == (N,8)`；检查八个命令通道有限且位于 `[-1,1]`。
3. 检查控制方向：surge/yaw/sway 输出均应作用于 horizontal 组（Fx/Fy/Mz）；深度/姿态由 vertical 组控制。
4. 检查相邻两帧 Farneback 输出、纹理 mask 计数、RMS 单位及奖励门控；reset 首帧 flow 应为零。
5. 仿真相机与真机帧并排确认 16:9、HFOV 与纹理尺度；部署延迟、池底噪声、present 阈值另行测量后更新参数。


---

## 7. sigma_flow 消融与标定

- 默认：`sigma_flow=0.05`。**复核（2026-11-16）**：现行配置实测归一化 std = 0.0023（0.150 px），该值若直接用作 sigma 会让奖励在 ~2 cm/s 漂移处就饱和，因此不作数值修改；0.05 对应的实际语义是**约 0.52 m/s 的盆宽**。脚本的 `max(0.05, …)` 下限会把低于 0.05 的测量钳到 0.05，**suggested 值没有信息量，要看原始 std**。
- 重新标定：

```bash
./isaaclab.sh -p v7/scripts/bench/calibrate_v7_sigma_flow.py --num_envs 1 --steps 60 --random
```

取输出的 `suggested sigma_flow`。
- 消融覆盖（hydra，无需改 yaml）：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
  --task Isaac-AUV-Tracking-Direct-v7-flow-hover \
  --agent sb3_cfg_entry_point --algorithm SAC \
  env.sigma_flow=0.1 --num_envs 16 --headless
```

- 流 RMS 单位为 64×64 网格像素；reward 中归一化为 `flow_rms/64`，因此 sigma_flow 是归一化后的无量纲值。

---

## 8. 第一帧视觉锚定（2026-11-10）：悬停目标从"动得慢"改为"回到第一帧"

### 动机

原 v7 的目标是帧间光流置零 + 漏积分漂移惩罚 —— 本质是"被动减速"。漏积分 τ=3 s 会遗忘，
287k 模型还用持续 yaw 把积分洗成 0（drift_mag≈0.018 却漂了 2.8 m）。目标导向悬停需要一个
**不遗忘、有方向、真机可用**的"离第一帧多远"信号。

### 信号（`_update_anchor_state`，每 policy step 一次）

- **捕获**：reset 后第一张有纹理的帧（flow-res 176×320 灰度中心裁块 96×160）作模板；
  裁块 std<1.0（暗墙）拒绝捕获、下一帧重试。
- **匹配**：`cv2.matchTemplate(TM_CCOEFF_NORMED)`，一次给出：
  - `_anchor_off_px (du,dv)`：场景相对第一帧的像素位移（body 系，+u=场景右移=ROV 左移）；
  - `_anchor_ncc ∈ [-1,1]`：图像相似度（=1 即"和第一帧一样"）。
- **换算**：`d = |px|·Z/f_px`（f_px=185 @320 宽、HFOV 81.65°，Z=plane_distance−x）。

### 实测（debug_auv_v7_drift_flow.py，零动作 + medium 洋流，2026-11-10 本机）

GT 漂移 vs 视觉 d：0.12/0.15 → 0.23/0.27 → 0.31/0.30 → 0.39/0.34（±0.05 m 内贴合）；
ncc 随漂移单调衰减 1.0→0.75→0.60→0.51→0.42→0.32。**有效盆地 ≈ ±0.4 m**（匹配窗口 ±80/±40 px
@ flow-res）；出盆后 ncc<0.35、偏移成噪声但符号大体仍指向回程，全局拉回由 GT 锚点项承担。
深度变化混入图像位移属预期语义："不像第一帧"包含深度/姿态分量。

### 观测/奖励变更

- image 9 通道不变：drift 广播(2) → **anchor 偏移广播(2)**（±1 = 匹配窗口边缘）。
- state 13 → **15**：尾部 drift(2) 替换为 anchor 偏移(2)，追加 sim=(ncc+1)/2、anchor yaw 误差。
  （**日期记录**：随后 §10.2 的 visodom 又追加 2 维，现行 state 为 **17 维**。）
- 奖励：`+ rew_anchor_vis·exp(-d_vis²/σ_vis²)`（0.3，σ_vis=0.2 ≈ 半盆地）；
  `+ rew_sim·(ncc+1)/2·present`（0.2）；`+ rew_yaw_anchor·exp(-yaw_err²/0.3²)`（0.1）；
  `rew_drift 0.3→0.1`（退化为盆地内去噪；**已于 2026-11-15 整项删除，见 §10.1**）；`rew_anchor（GT）0.3→0.15`（保留：平面纹理跨
  episode 不变，纯视觉奖励存在 alias 作弊后门；GT 项不可欺）。
- 日志新增 `Episode_Reward/anchor_disp_m`、`anchor_sim`；`disp_end_m` 仍是端点 GT 指标。

### 不兼容与消费者

- **旧 checkpoint 全部作废**（image 通道语义 + state 维度变了）：含 2026-11-10 的
  `model_100000_steps.zip`。
- `MidCNNExtractor`/`CustomSACPolicy` 从 space 读维度，无需改；`debug_auv_v7_drift_int.py`
  改读 `env._drift_int`（state[11:13] 现在是 anchor 偏移）；`debug_auv_v7_drift_flow.py`
  增加 anchor/ncc/GT 对照列作为锚点信号的回归验证。

### 验收

```bash
# 1) 锚点信号回归：anchor d 必须贴 GT（±0.4 m 内），ncc 单调
TERM=xterm-256color ./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_drift_flow.py --headless --num_envs 1 --steps 300
# 2) 冒烟训练 50k：看 anchor_sim 上升、anchor_disp_m 收窄
# 3) 全量 2M 后再下结论（v0 教训：<200k 别判生死）
```

---

## 9. 垂向动作通道 + 深度环整定（2026-11-11）

### 动机

30 万步模型实测：水平悬停 0.129 ± 0.013 m，但深度稳定沉 0.43 m——深度环是纯 P（kp=5 N/m）
对 ~2 N 负浮力的稳态误差，导致 anchor_sim 只到 0.75（画面缩放变了）。

### 变更

- 动作保持 **3-D** `[surge, sway, yaw]`。曾试加第 4 维深度目标偏移（±1 → ±0.5 m），但在
  深度环修好后第 4 维零可学信号（设定值平移加速不了积分器），只贡献探索噪声，已撤掉。
  18 个零动作脚本随之回到 3 维。
- 深度环 v7 本地整定（`__init__` post-hoc，不动 batch_pid 类默认，其它任务不受影响）：
  `kp 5→12, ki 0→1.5, kd 3→6, heave clamp 3→6 N`。
- **Routh 稳定性是硬约束**：加 ki 后特征方程 `m·s³+kd·s²+kp·s+ki`，稳定需 `kd·kp > m_eff·ki`。
  原增益下 `kd·kp=15`，m_eff≈26 kg（含垂向附加质量），任何有用 ki 都不稳——首版 ki=4 实测
  ±7 cm 极限环不收敛。先抬 kp/kd 把上限拉到 72，ki=1.5 才有 1.8x 裕度。
- 实测（零动作 220 步）：初沉至死区边 −6 cm，积分 ~12 s 清偏差，episode 末 z=+1.2 cm，
  单调收敛无振荡。P 独下垂从 0.42 m 降到 0.17 m，再由 I 清零。
- batch_pid：积分仅当 `ki_depth != 0` 时累加（其它任务零开销）；`depth_int_limit` 1→4 m·s
  （ki=1.5 下需 1.33 m·s 才能顶住 2 N）。reset 清积分状态。
- 所有 v7 零动作脚本（18 个）动作张量 3→4 维。

### 不兼容

- 旧 checkpoint（3-D 动作）全部作废，SAC.load 会直接拒绝。需重训。

### 终态（2026-11-11 定稿）

动作 3-D 不变；深度由本地整定的 PI 环（kp 12 / ki 1.5 / kd 6 / 限幅 6 N）自主保持出生深度。
零动作实测：初沉至死区边 −6 cm，积分 ~12 s 清偏差，episode 末 z=+1.2 cm，单调收敛。
策略的活儿只剩水平（洋流）——这正是第一帧锚定任务本来的自由度。

---

## 10. 2026-11-15：删除 drift 项 + 域随机化（以本节为最新契约）

### 10.1 删除"慢漂移"奖励项及其积分器

四组留一消融（各 170k–1.6M 步）的结论：`no_drift` 组**最优窗口 `disp_end_m` = 0.0107 m，是四组里最好的**（`no_flow` 0.0212 / `no_visodom` 0.0165 / `no_template` 0.0136）。该奖励项（`rew_drift` 0.1）冗余：首帧 anchor 偏移已经**绝对**度量同一种漂移，而 visodom 位置项提供了它当初被发明出来要补的速度尺度梯度。

- 移除：`rew_drift`、`sigma_drift`、`drift_tau_s`、`ablate_drift`、`_drift_int`、`_drift_leak`、`_drift_yaw_prev`、`_ep_drift_sum`、日志 `drift_mag`、以及每步的去旋转+积分（热路径）。
- **光流本身全部保留**：图像通道 du/dv/mag、`rew_flow=1.0`、visodom 积分器、`flow_rms` 状态量、texture mask 一律不动。
- 副作用：`scripts/debug_auv_v7_drift_int.py` 删除；`debug_auv_v7_ablation.py` / `expand_vs_anchor.py` / `flow_decompose.py` 里的相关引用改为 visodom。

### 10.2 域随机化（`dr_enable=True`，默认开启）

按物理性质分成两种采样horizon：水与位姿**每回合**重采样；硬件/图像**每环境一次**（一台相机、一组电池不会每潜重抽）。

| 旋钮 | 范围 | 采样 | 说明 |
|---|---|---|---|
| `dr_current_speed_range` | 0.20–0.45 m/s | 每回合 | 取代原来固定的 0.3 m/s |
| `dr_current_dir_full_circle` | True（±180°） | 每回合 | 取代固定的 [1,0.3,0] |
| `dr_current_vert_frac` | ≤10% 速度 | 每回合 | 池内洋流近似水平 |
| `dr_thrust_gain_range` | 0.90–1.05 | 每环境 | 作用在归一化杆量上，推力 ∝ g² ≈ ±10% |
| `dr_drag_gain_range` | 0–3.0 N/(m/s) | 每环境 | 附加线性阻力，扰动用阻尼而非缩放浮力 |
| `dr_cam_brightness_range` | 0.92–1.08 | 每环境 | 曝光/增益 |
| `dr_cam_noise_std_range` | 0.004–0.010 | 每帧 | 读出噪声，在 render 分辨率注入，flow/template/mask 与策略看到同一帧 |
| `dr_spawn_xy_range` | ±0.12 m | 每回合 | 首帧不再正对名义位置 |
| `dr_spawn_yaw_range_deg` | ±8° | 每回合 | |
| `dr_spawn_depth_range` | ±0.04 m | 每回合 | 深度 PI 会把它拉回 |

新增日志：`dr_cur_speed`、`dr_cur_deg`（与 `disp_end_m` 同曲线看，才知道策略是否只会应付顺流）。

### 10.3 部署契约修正（本文档前几节已同步）

| 项 | 旧文档（错） | 实际训练（以此为准） |
|---|---|---|
| 策略频率 | 20 Hz（decimation=6） | **15 Hz**（decimation=8, dt=1/120） |
| 光流分辨率 | 1280×720 原生 | **176×320**（render 640×360 下采样后计算） |
| du/dv 归一化分母 | 1280 / 720 | **320 / 176** |
| policy image | (7,64,64) | **(11,64,64)** |
| state | (9,) | **(17,)，且只喂 critic** |

最后一条最容易被忽略：`CustomSACPolicy` 是非对称 AC，**actor 只吃 image**。所以实机推理**不需要 DVL、不需要 GT 速度/深度**，只需要相机 + IMU yaw（anchor/visodom 的去旋转用）。

---

## 11. 2026-11-16：日志键修订、相机几何现状、经典基线对照

### 11.1 日志键：不要再用 `disp_end_m` 衡量驻留误差

加了出生点 DR（`dr_spawn_xy_range=±0.12 m`）之后，`disp_end_m`（相对 **env origin**）测的主要是出生偏移本身，下限 ≈0.05–0.09 m，永远压不到 0.01 量级。实测 `disp_end_m 0.839` vs `disp_spawn_m 0.789`，差的 0.05 m 正好是出生偏移。

- **`disp_spawn_m`（相对本回合出生点）是唯一诚实的位置指标**
- `disp_end_m` 保留旧语义不动，供前四轮消融继续可比（不要就地重定义已写出的键）
- `visodom_err_m` 改为对比出生点，之前混入了偏移
- 新增 **`anchor_valid_frac`**：模板实际有效的步数占比。它为 0.5 时 `anchor_sim` 天然只有一半，**不先看这个键会误判成"匹配质量差"**
- 另有 `disp_m` 是回合内**时间均值**（同样含 origin 偏移），也不是末态

### 11.2 相机几何：内参已完成，仍缺外参

实机内参已标定（`相机/camera_parameters_refit.json`，水下棋盘格 29 视角）：

```
fx=1111.28  fy=1162.44  cx=992.66  cy=689.26   @1920×1080
k1=0.1649  k2=-0.0899  p1=0.0581  p2=0.0131  k3=0
RMS 重投影 0.543 px，HFOV 81.646°，VFOV 49.83°
```

仿真已按 `_CAM_HFOV_DEG=81.65`、`_CAM_APERTURE_MM=20.955` 对齐，**这条不再是缺口**。注意 `camera_parameters.json`（旧版 k3=1.80）已被 refit 取代：旧版把边角径向增益放大到 +130%，无畸变视场会塌到 ~36°，不要用。

仍然缺的三项：

| 项 | 现状 | 影响 |
|---|---|---|
| **外参（相机↔机体位姿）** | 未实测，仿真按机头前方朝 +X 假设 | 光流方向与机体运动的对应、模板偏移的米制换算 |
| **像素长宽比** | 仿真按 fx=fy（正方形像素）；实机 `fy/fx=1.046` | 垂直方向差 4.6% |
| **畸变** | 仿真纯针孔，实机 k1=0.165/k2=−0.09 | 画面边缘光流偏差（二阶） |

**另有一项结构性问题**：实机画面**上部约 52% 是水面倒影**（`camera_parameters_refit.json` 的 `water_surface` 字段），不是景物。v7 的光流、纹理掩码、首帧模板全都假设整幅画面是静态场景；倒影会随水面波动产生与艇位移无关的大光流，污染 `flow_rms`，并可能让模板捕获到半张倒影。当前域随机化（`dr_cam_*` 只扰亮度与噪声）**完全没有覆盖这项**。

### 11.3 经典基线对照（同环境、同 DR、同估计器）

基线在 `v7/scripts/debug/debug_auv_v7_baseline.py`，P / PID / LQR 只读 `_visodom_m()` + 偏航角，不碰模板。

**洋流扫描**（`disp_spawn_m`, mm）：

| 洋流 | P | PID | LQR |
|---|---|---|---|
| 0.20 m/s | 29.8 | 13.5 | **3.3** |
| 0.30 m/s | 42.6 | 15.9 | **3.3** |
| 0.45 m/s | 63.3 | 22.3 | **4.9** |

**洋流突变**（5 s 时 0.30→0.45 m/s 并反向 180°）：P 64.8→141.6、PID 15.9→62.7、**LQR 3.5→1.7（无退化）**。

**线索丢失**（5–10 s 断光流，末态 / 峰值，mm）：

| | 末态 live | 末态 blind | 峰值 live | 峰值 blind |
|---|---|---|---|---|
| P | 41.9 | 47.5 | 107.0 | 166.8 |
| PID | 17.3 | 23.3 | 84.3 | 152.3 |
| **LQR** | 4.0 | 1.6 | **12.4** | **54.9** |

**两条必须记住的结论**：

1. **末态位移指标看不见可恢复的瞬态**。LQR 在盲区结束 5 s 后完全拉回（1.6 mm），但峰值 excursion 从 12.4 mm 恶化到 54.9 mm（4.4×）。评估线索丢失必须报**峰值**。
2. 恒定与突变工况下**经典 LQR 全面优于 RL**（3.3–4.9 mm vs RL 最优 10.7 mm；突变下 LQR 甚至无退化）。v7 目前唯一还站得住的差异是**线索冗余下的优雅退化**，而这一条**尚未在策略上测过**。

因此本节之后，v7 的价值主张不能再说"RL 比经典控制更准/更稳"，只能落在"多路视觉线索让系统在单路失效时保持精度"，并以观测层 dropout 的对照实验为判据。

**披露义务**：基线用了 GT 偏航角（`euler_xyz_from_quat`），RL actor 是纯图像输入；实机 IMU 能给偏航，不算作弊，但论文须写明。基线增益一次性给定、未针对任何工况调参。

