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
| 相机 / 渲染 | 1280×720，部署输入分辨率；仿真 tiled camera 配置为 1280×720 |
| 相机 HFOV | 81.65° |
| 策略频率 | `decimation=6`、`sim.dt=1/120 s` → 20 Hz（50 ms/步） |
| 光流 | OpenCV Farneback `(pyr_scale=.5, levels=3, winsize=21, iterations=2, poly_n=5, poly_sigma=1.2, flags=0)` |
| policy image | `(7,64,64)`：RGB(3)、du、dv、flow magnitude、texture mask |
| state | `(9,)`：depth, roll, pitch, yaw, v_surge, v_sway, v_heave, yaw_rate, flow_rms |
| action | `(3,)`：surge、sway、yaw，范围 `[-1,1]` |
| 执行器 | 8 个推进器，4 vertical + 4 horizontal，`_pwm.shape=(num_envs,8)` |
| heave | 策略动作不输出 heave；深度由 PID 以 `starting_depth` 保持 |
| sway | 已接入：`actions[:,1]` → 横移目标，PID 经 4 水平推进器 Fy 分量消减侧漂 |

策略 20 Hz 是训练配置，不代表端到端部署时延已通过验证。部署文档中的 `L_camera` 仍待实测；若延迟约束不满足，训练频率需据实调整。

---

## 1. 观测契约

```python
image = Box(low=-1, high=1, shape=(7, 64, 64), dtype=float32)
state = Box(shape=(9,), dtype=float32)
```

| 通道 | 内容 | 处理 |
|---|---|---|
| 0–2 | tiled camera RGB | `/255`，范围 `[0,1]` |
| 3 | `du / 1280` | Farneback 位移下采样至 64×64 后，按原始相机宽度归一化，clip `[-1,1]` |
| 4 | `dv / 720` | 同上，按原始相机高度归一化，clip `[-1,1]` |
| 5 | `sqrt(du_norm²+dv_norm²)` | 由归一化后的 du/dv 计算 |
| 6 | texture mask | render 分辨率算梯度阈值，再 adaptive max pool 至 64×64，值为 0/1 |

`frame_stack=1`，无历史帧堆叠。Farneback 由 CPU OpenCV 执行；部署侧 OpenCL 速度数据不能直接视作训练端耗时。`state` 九维顺序如上，最后一维 `flow_rms` 是 64×64 降采样流场上纹理区域的像素 RMS；它不是部署侧原始 1280×720 流 RMS 的同量纲直接值。

首次取得帧（或 reset 后首帧）无前序帧可计算光流，flow 与 flow_rms 置零。

---

## 2. 动作与推进器

策略动作 shape 为 `(3,)`，语义 `[surge, sway, yaw]`，范围 `[-1,1]`。GS-ROV PID 三个维度都参与控制：surge→Fx、sway→Fy、yaw→Mz，全部经 `BatchPIDController.gsrov8` 分配到四个水平推进器；heave 由深度 PID 独立维持。三维接口与部署动作契约对齐。

v7 不是逐推进器 8 维策略动作。策略输出 3 维控制量，经 `BatchPIDController` 的 GS-ROV 分配矩阵转换为八路归一化推进器命令，再由动力学模型计算推力。推进器顺序为 CAD 几何顺序 `[V1,V2,V3,H1,H2,H3,H4,V4]`。部署输出若需映射到飞控 MAIN 输出，须按 `GSROV_ACTION_TO_MAIN` 排列，不能假定 CAD 顺序即 MAIN 顺序。

GS-ROV 推力模型为带 deadband 的有符号二次曲线 `T = T_max * u * |u|`；额定推力参数当前来自数据表，电池电压默认 24 V。部署杆量到 PWM 的 `1500 + 400u → [1100,1900] µs` 是飞控接口契约；仿真这里由连续归一化命令与推力曲线近似，不在策略动作预处理处直接生成微秒 PWM。

---

## 3. 奖励、门控与终止

奖励结构：纹理存在时以 flow RMS 高斯项为主，叠加 alive、深度、姿态项并惩罚动作变化。配置初值：`rew_flow=1.0`、`rew_alive=0.3`、`rew_depth=0.1`、`rew_att=0.1`、`rew_action=0.01`、`sigma_flow=0.05`（由 `scripts/calibrate_v7_sigma_flow.py` 在 20 Hz 零/小随机动作下测得 flow_rms_n std≈0.047，取 1×std 并截断 `[0.05,0.5]`；仍可通过 `env.sigma_flow=...` 进行消融）、`sigma_depth=1.5 m`、`sigma_att=0.3 rad`。

`present = texture_count >= min_texture_px`，当前 `min_texture_px=20`。无纹理连续 50 个策略步（约 2.5 s@20Hz）终止；深度偏离 `starting_depth` 超过 1.5 m、roll/pitch 合成姿态误差超过 0.3 rad 或越界也终止；最大 episode 长度由父配置决定。无纹理时 flow/alive 奖励被门控为零，但深度/姿态项仍然计算。`min_texture_px=20`；`flow_mag_min_px`/`flow_mag_rel_frac` 已移除（无使用处）。

纹理 mask 是基于灰度梯度阈值 `8.0` 的训练仿真门控，不等同于部署端 `present` 的实测绝对阈值。实际池底噪声、噪声底与 flow 阈值应待部署侧现场统计后再标定。

---

## 4. 频率与分辨率注意事项

- 本实现 `sim.dt=1/120 s`、`decimation=6`，策略频率为 `120/6=20 Hz`，与部署方案 20.83 Hz 接近但不完全相同。
- 图像策略输入固定 64×64，但光流在 1280×720 原生分辨率计算后才下采样；不要把 64×64 当作光流估计分辨率。
- 当前光流实现使用核心 OpenCV API `cv2.calcOpticalFlowFarneback`（参数 pyr_scale=0.5, levels=3, winsize=21, iterations=2, poly_n=5, poly_sigma=1.2, flags=0）。该函数随 `opencv-python` / `opencv-python-headless` 一起提供，无需 contrib 包；返回 `(H,W,2)` 后转置为 `(2,H,W)` 通道优先。
- 部署方案给出的 GPU/OpenCL 耗时与此 CPU OpenCV 路径不同，训练性能须单独实测。

---

## 5. 与旧设计稿的差异

| 旧稿内容 | 当前 GS-ROV v7 |
|---|---|
| WarpAUV 代理 | GS-ROV CAD hull / 8 推进器动力学 |
| RAFT | Farneback OpenCV |
| 256×256 渲染、HFOV 90° | 1280×720、HFOV 81.65° |
| frame stack=2、state 10 维 | 单帧 7 通道图像、state 9 维 |
| 15 步纹理丢失终止 | 50 步（20 Hz 下约 2.5 秒） |
| 六推进器 | 八推进器；策略仍输出三维而非八路 motor action |


---

## 6. 训练验收清单

1. 单 env reset + step，确认 observation `image=(N,7,64,64)`、`state=(N,9)`、action `(N,3)`。
2. 确认环境 `num_thrusters == 8` 且 `_pwm.shape == (N,8)`；检查八个命令通道有限且位于 `[-1,1]`。
3. 检查控制方向：surge/yaw/sway 输出均应作用于 horizontal 组（Fx/Fy/Mz）；深度/姿态由 vertical 组控制。
4. 检查相邻两帧 Farneback 输出、纹理 mask 计数、RMS 单位及奖励门控；reset 首帧 flow 应为零。
5. 仿真相机与真机帧并排确认 16:9、HFOV 与纹理尺度；部署延迟、池底噪声、present 阈值另行测量后更新参数。


---

## 7. sigma_flow 消融与标定

- 默认：`sigma_flow=0.05`（由 `scripts/calibrate_v7_sigma_flow.py` 标定：20 Hz 零/小随机动作下 flow_rms_n std≈0.047，取 1×std 并截断 `[0.05,0.5]`）。
- 重新标定：

```bash
./isaaclab.sh -p scripts/calibrate_v7_sigma_flow.py --num_envs 1 --steps 60 --random
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
- 奖励：`+ rew_anchor_vis·exp(-d_vis²/σ_vis²)`（0.3，σ_vis=0.2 ≈ 半盆地）；
  `+ rew_sim·(ncc+1)/2·present`（0.2）；`+ rew_yaw_anchor·exp(-yaw_err²/0.3²)`（0.1）；
  `rew_drift 0.3→0.1`（退化为盆地内去噪）；`rew_anchor（GT）0.3→0.15`（保留：平面纹理跨
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
TERM=xterm-256color ./isaaclab.sh -p scripts/debug_auv_v7_drift_flow.py --headless --num_envs 1 --steps 300
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
