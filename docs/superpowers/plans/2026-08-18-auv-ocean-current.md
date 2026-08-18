# 洋流（Isaac-AUV-Tracking）实现计划

> **给代理执行者：** 必用子技能：`superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐步实现。步骤用复选框（`- [ ]`）跟踪。

**目标：** 为 Isaac-AUV-Tracking 增加**可选、可配置**的洋流：提供与 Holo 同语义的 `set_ocean_currents` API，用水动力相对速度施加；默认平静水。

**架构：** 批量化 torch 场函数算出世界系水流速度，经 `set_ocean_currents` 写入每 env 缓冲。`_compute_dynamics` 用现有 `HydrodynamicForceModels`，以 \(v_{\mathrm{rel}} = v_{\mathrm{auv}} - v_{\mathrm{current}}\)。可选 `affect_target` 把同一洋流叠到运动学目标球上。Direct-v1 继承动力学，**不改** obs/网络。

**技术栈：** PyTorch、Isaac Lab `DirectRLEnv`、现有 WarpAUV 水动力（`rigid_body_hydrodynamics.py`）。

**设计规格：** `docs/superpowers/specs/2026-08-18-auv-ocean-current-design.md`

## 全局约束

- 默认 `ocean_current.enabled=false`（平静水；已有 checkpoint 行为不变）。
- 第一期不改 obs、reward、done、KF 接口。
- Isaac 世界系（Z 向上，`starting_depth≈8`）；**不要**照搬 Holo 的 `z>0 ⇒ 零流` 或 UE 的 Y 取反。
- 相对速度只喂进**现有**密度/粘性力；不要移植 Holo 的 \(C_d A\) 二次阻力公式。
- `affect_target` 默认 `false`；为 true 时在**目标**位置采样洋流。
- 用 `clip_speed`（默认 `5.0`，对齐 Holo）裁剪 `‖v_current‖`。
- 仅在用户明确要求时提交 git（计划执行中途不要自动 commit）。

## 文件一览

| 文件 | 职责 |
|------|------|
| 新建 `.../isaac_auv_tracking/ocean_current.py` | 场类型 + 工厂 + 速度裁剪 |
| 新建 `source/isaaclab_tasks/test/test_ocean_current.py` | 纯 torch 单测（放在包外，避免 pytest 拉 Isaac） |
| 修改 `.../auv_tracking_env.py` | 配置、缓冲、API、动力学、步进接线、目标叠加 |
| 修改 `.../batch_lqr.py` | `BatchHoloTargetMotion.step` 可选 `current_vel_xy` |
| 修改 `.cursor/rules/auv-tracking-holo-design.mdc` | 注明洋流可选、默认关闭 |

---

### 任务 1：批量化洋流场模块

**文件：**
- 新建：`source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/ocean_current.py`
- 测试：`source/isaaclab_tasks/test/test_ocean_current.py`（勿放在 `isaaclab_tasks` 包内，否则 pytest collect 会 import `omni`）

**接口：**
- 产出：
  - `LEVEL_STRENGTH: dict[str, float]`：`off=0.0`，`mild=0.15`，`medium=0.4`，`strong=0.8`
  - `clip_current_velocity(v: Tensor[N,3], clip_speed: float) -> Tensor[N,3]`
  - `resolve_strength(strength: float | None, level: str | None) -> float`
  - `OceanCurrentField` 协议 / 可调用对象：`(pos_w[N,3], t: Tensor|float) -> Tensor[N,3]`
  - `create_ocean_current_field(cfg) -> OceanCurrentField | None`  
    其中 `cfg` 为含 `enabled`、`type`、`strength`、`level`、`params`、`clip_speed` 的命名空间或 dict

- [x] **步骤 1：先写会失败的单测**

```python
# test_ocean_current.py
import torch
from isaaclab_tasks.direct.isaac_auv_tracking.ocean_current import (
    clip_current_velocity,
    create_ocean_current_field,
    resolve_strength,
    LEVEL_STRENGTH,
)

def test_resolve_strength_level():
    assert resolve_strength(None, "mild") == LEVEL_STRENGTH["mild"]
    assert resolve_strength(0.5, None) == 0.5
    assert resolve_strength(0.5, "strong") == 0.5  # 显式 strength 优先

def test_clip():
    v = torch.tensor([[3.0, 4.0, 0.0]])  # 模长 5
    out = clip_current_velocity(v, clip_speed=2.5)
    assert torch.allclose(out.norm(dim=-1), torch.tensor([2.5]), atol=1e-5)

def test_uniform_batch_ignores_holo_depth_sign():
    # Isaac z≈8 仍应有流（不同于 Holo z>0 → 0）
    class Cfg:
        enabled = True
        type = "uniform"
        strength = 0.5
        level = None
        clip_speed = 5.0
        params = {"direction": [1.0, 0.0, 0.0]}
    field = create_ocean_current_field(Cfg())
    pos = torch.tensor([[0.0, 0.0, 8.0], [1.0, 0.0, 8.0]])
    v = field(pos, 0.0)
    assert v.shape == (2, 3)
    assert torch.allclose(v[:, 0], torch.tensor([0.5, 0.5]), atol=1e-5)

def test_disabled_returns_none():
    class Cfg:
        enabled = False
        type = "uniform"
        strength = 1.0
        level = None
        clip_speed = 5.0
        params = {}
    assert create_ocean_current_field(Cfg()) is None
```

- [x] **步骤 2：跑测试 — 预期 ImportError / FAIL**

```bash
cd /home/lh/IsaacLab
python -m pytest source/isaaclab_tasks/test/test_ocean_current.py -v
```

预期：FAIL（模块尚未存在）

- [x] **步骤 3：实现 `ocean_current.py`**

至少实现：

- `uniform`：`v = strength * normalize(direction)`（默认方向 `[1,0,0]`）；可选 Holo 式 `time_modulation`。
- `vortex`：绕 `center` 的水平切向场（默认 xy=`[0,0]`，z 用 params 传入的 `starting_depth` 或 `[0,0,8]`）；按 `strength` 缩放；**不要** `if z>0: return 0`。
- `sine_wave`：行波；振幅=`strength`。
- `random`：由 `pos`、`t`、`seed` 决定的确定性噪声（每 env 可复现）；振幅=`strength` 或 `params.max_velocity`。
- 返回的闭包内部始终调用 `clip_current_velocity`。
- `create_ocean_current_field`：若未 `enabled` → `None`。

参数名尽量对齐 Holo（uniform 的 `velocity`：若存在 `params.velocity` 则用之并忽略 strength×direction；否则用 `strength * direction`）。

- [x] **步骤 4：再跑测试 — 预期全部 PASS**

```bash
python -m pytest source/isaaclab_tasks/test/test_ocean_current.py -v
```

预期：全部 PASS

- [ ] **步骤 5：仅在用户要求时 commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/ocean_current.py \
        source/isaaclab_tasks/test/test_ocean_current.py
# 仅当用户要求时 git commit
```

---

### 任务 2：配置 + 缓冲 + `set_ocean_currents` / `get_ocean_currents`

**文件：**
- 修改：`source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/auv_tracking_env.py`

**接口：**
- 依赖：任务 1 的 `create_ocean_current_field`、`clip_current_velocity`
- 产出：
  - `AUVTrackingEnvCfg.ocean_current_*` 字段（或嵌套 `@configclass OceanCurrentCfg`）
  - `AUVTrackingEnv.set_ocean_currents(env_ids, velocity)`
  - `AUVTrackingEnv.get_ocean_currents(env_ids) -> Tensor`
  - 缓冲 `self._ocean_current_vel: Tensor[N,3]`

- [x] **步骤 1：增加配置（默认平静水）**

在 `AUVTrackingEnvCfg` 其它旋钮附近（水动力属性约第 187 行之后）：

```python
@configclass
class OceanCurrentCfg:
    enabled: bool = False
    type: str = "uniform"  # uniform | vortex | sine_wave | random
    strength: float = 0.3
    level: str | None = None  # off|mild|medium|strong；None → 用 strength
    affect_target: bool = False
    clip_speed: float = 5.0
    # 类型相关；方向默认 +x
    params: dict = None  # __post_init__ 里 default_factory，或 field(default_factory=dict)

# 挂在 AUVTrackingEnvCfg 上：
ocean_current: OceanCurrentCfg = OceanCurrentCfg()
```

若 Isaac configclass 对嵌套 dict 不友好，可展平：

```python
ocean_current_enabled: bool = False
ocean_current_type: str = "uniform"
ocean_current_strength: float = 0.3
ocean_current_level: str | None = None
ocean_current_affect_target: bool = False
ocean_current_clip_speed: float = 5.0
# 第一期 uniform 可用 ocean_current_direction: tuple = (1.0, 0.0, 0.0)
```

优先嵌套 `OceanCurrentCfg`（若仓库已有类似嵌套）；否则第一期展平，其它场类型参数在支持的前提下走 `params`。

- [x] **步骤 2：在 `AUVTrackingEnv.__init__` 中初始化缓冲与场（`super().__init__` 之后）**

```python
self._ocean_current_vel = torch.zeros((self.num_envs, 3), device=self.device)
self._ocean_current_field = create_ocean_current_field(self.cfg.ocean_current)
```

- [x] **步骤 3：实现 API**

```python
def set_ocean_currents(self, env_ids: torch.Tensor | Sequence[int] | None, velocity: torch.Tensor) -> None:
    """Holo 同语义 API：为 env 设置世界系水流速度 [m/s]。"""
    if env_ids is None:
        env_ids = self._robot._ALL_INDICES  # 或 torch.arange(self.num_envs, device=self.device)
    env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
    v = torch.as_tensor(velocity, device=self.device, dtype=torch.float32)
    if v.ndim == 1:
        v = v.view(1, 3).expand(env_ids.numel(), 3)
    assert v.shape == (env_ids.numel(), 3)
    v = clip_current_velocity(v, float(self.cfg.ocean_current.clip_speed))
    self._ocean_current_vel[env_ids] = v

def get_ocean_currents(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> torch.Tensor:
    if env_ids is None:
        return self._ocean_current_vel.clone()
    env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
    return self._ocean_current_vel[env_ids].clone()
```

`env_ids` 处理方式与本文件 reset 中已有模式一致。

- [x] **步骤 4：Reset 时清零洋流（下一步会由场函数再写入）**

在 `_reset_idx` 其它 reset 之后：

```python
self._ocean_current_vel[env_ids] = 0.0
```

- [ ] **步骤 5：不做完整训练的 API 冒烟**（可选）

```python
# 小段 debug 里构造 env 后断言，或非 Isaac 单测 mock —
# 优先：由任务 3 的 smoke 覆盖。
```

- [ ] **步骤 6：仅在用户要求时 commit**

---

### 任务 3：每步评估场 + 相对速度动力学

**文件：**
- 修改：`source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/auv_tracking_env.py`（`_apply_action`、`_compute_dynamics`）

**接口：**
- 依赖：`_ocean_current_field`、`set_ocean_currents`、`_ocean_current_vel`
- 产出：AUV 经密度/粘性路径感受 \(v_{\mathrm{rel}}\)

- [x] **步骤 1：在 `_apply_action` 开头用场函数刷新 AUV 洋流**

```python
def _apply_action(self) -> None:
    if self._ocean_current_field is not None:
        t = self.episode_length_buf.to(dtype=torch.float32) * float(self.sim.cfg.dt)
        # 有绝对时间更好；第一期用 episode 时间即可
        v = self._ocean_current_field(self._robot.data.root_pos_w, t)
        self.set_ocean_currents(None, v)

    self._target_state[:] = self.target_motion.step(...)
    ...
```

若 `set_ocean_currents(None, v)` 不好用，可：

```python
self._ocean_current_vel[:] = clip_current_velocity(v, clip)
```

但仍优先走 API，保持单一写入路径。

- [x] **步骤 2：改 `_compute_dynamics`，线速度用相对速度**

只替换线速度参数：

```python
v_rel = self._robot.data.root_lin_vel_w - self._ocean_current_vel
df, dt_, vf, vt_ = self.force_calc.calculate_density_and_viscosity_forces(
    self._robot.data.root_quat_w,
    v_rel,  # 原先是 root_lin_vel_w
    self._robot.data.root_ang_vel_w,
    ...
)
```

浮力与推进器不变。角速度不变（Holo 也是把洋流当平动水流速度用于阻力）。

- [x] **步骤 3：关闭时的健全性**

`enabled=false` 时 `_ocean_current_field is None` 且缓冲为 0 → 与改前受力路径等价（允许多一次减零；或 field 为 None 时跳过减法以求位级一致）：

```python
lin_vel = self._robot.data.root_lin_vel_w
if self._ocean_current_field is not None or torch.any(self._ocean_current_vel != 0):
    lin_vel = lin_vel - self._ocean_current_vel
```

- [x] **步骤 4：Headless 冒烟（同一时刻仅一个 Isaac 进程）**

```bash
./scripts/kill_isaac_lab_processes.sh
# 临时：cfg 里 ocean_current.enabled=True, type=uniform, strength=0.5
# 或写约 20 行的 scripts/smoke_auv_ocean_current.py：Direct-v0、2 env、50 步
```

最小冒烟脚本（需要则新建）：

`scripts/smoke_auv_tracking_ocean_current.py` — AppLauncher，`gym.make` Direct-v0，覆盖 cfg `ocean_current.enabled=True`、`strength=0.5`，随机动作步进 100 次，打印 `get_ocean_currents` 平均模长后退出。

预期：不崩溃；uniform 下 `‖v_current‖ ≈ 0.5`。

- [ ] **步骤 5：仅在用户要求时 commit**

---

### 任务 4：可选 `affect_target` 运动学叠加

**文件：**
- 修改：`source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/batch_lqr.py`（`BatchHoloTargetMotion.step`）
- 修改：`source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/auv_tracking_env.py`（`_apply_action`）

**接口：**
- 依赖：目标位置处的场；cfg `affect_target`
- 产出：开启时目标 XY 随流漂移

- [x] **步骤 1：扩展 `step` 签名**

```python
def step(
    self,
    state: torch.Tensor,
    dt: float,
    agent_xy: torch.Tensor | None = None,
    current_vel_xy: torch.Tensor | None = None,  # [N,2] 世界系 m/s
) -> torch.Tensor:
```

在算完 `pos_new = pos + vel_new * dt`（现有逻辑）之后，若 `current_vel_xy is not None`：

```python
pos_new = pos_new + current_vel_xy * dt
# 可选：vel_new = vel_new + current_vel_xy  # 第一期优先只漂位置（YAGNI）
```

叠加后**再**做工作区 clamp（把现有 clamp 挪到叠加之后，或再 clamp 一次）。

- [x] **步骤 2：在 `_apply_action` 接线**

```python
current_xy = None
if self._ocean_current_field is not None and self.cfg.ocean_current.affect_target:
    # 目标世界坐标：用已同步的目标，或 [target_xy, depth]
    tgt_pos = self._robot.data.root_pos_w.clone()  # 错误示例 — 不要用机器人位姿
    tgt_pos = torch.zeros((self.num_envs, 3), device=self.device)
    tgt_pos[:, :2] = self._target_state[:, :2]
    tgt_pos[:, 2] = self.cfg.starting_depth
    t = self.episode_length_buf.to(dtype=torch.float32) * float(self.sim.cfg.dt)
    v_tgt = self._ocean_current_field(tgt_pos, t)
    current_xy = v_tgt[:, :2]

self._target_state[:] = self.target_motion.step(
    self._target_state,
    dt=float(self.cfg.sim.dt),
    agent_xy=self._robot.data.root_pos_w[:, :2],
    current_vel_xy=current_xy,
)
```

说明：AUV 的 `set_ocean_currents` 仍用 AUV 位置采样；目标叠加用目标位置（规格要求）。

- [x] **步骤 3：无完整 Isaac 的单元级检查（可选）**

在 CPU 上用很小的 `N` 实例化 `BatchHoloTargetMotion`；带 `current_vel_xy=ones` 步进；断言相对无流基线，位置约多移 `dt`。

- [ ] **步骤 4：仅在用户要求时 commit**

---

### 任务 5：文档 / 规则说明 + 收尾

**文件：**
- 修改：`.cursor/rules/auv-tracking-holo-design.mdc`（洋流小节）
- 修改：`docs/superpowers/specs/2026-08-18-auv-ocean-current-design.md` — 勾选 Approval 审阅项
- 可选：`models/auv_tracking/README.md` 短注 — “洋流默认关闭”

- [x] **步骤 1：更新 cursor 规则**

在「允许例外 / 底座差异」中，把笼统的「不要海流」换成：

```markdown
- **洋流（可选）**：默认关闭（平静水）。启用见 `AUVTrackingEnvCfg.ocean_current`；
  Holo 同语义 `set_ocean_currents` + 相对速度进 WarpAUV 水动力；`affect_target` 默认 false。
```

- [x] **步骤 2：规格清单**

在设计文档中将 `- [x] Spec file review` 勾上。

- [ ] **步骤 3：仅在用户要求时 commit**

---

## 规格覆盖对照

| 规格条目 | 任务 |
|----------|------|
| Holo 同语义 `set/get_ocean_currents` | 任务 2 |
| 水动力相对速度 | 任务 3 |
| 批量化场模块 + 类型 | 任务 1 |
| 默认关闭 | 任务 2 配置 |
| `strength` / `level` | 任务 1–2 |
| `affect_target` | 任务 4 |
| Isaac 坐标系（无 Holo z 符号 / UE Y） | 任务 1 测试 |
| clip_speed 5 | 任务 1–2 |
| 不改 obs/reward/KF | 全部（全局约束） |
| Smoke | 任务 3 |
| 文档 / 规则 | 任务 5 |

## 占位符 / 一致性自检

- API 命名一致：`set_ocean_currents`、`get_ocean_currents`、`_ocean_current_vel`。
- 配置：优先嵌套 `OceanCurrentCfg`；若 Isaac configclass 不接受 `dict`，任务 2 展平 uniform 旋钮，其它类型额外参数仅在支持时放 `params` — 实现者选定一种，并在 `create_ocean_current_field` 中一致使用。
- 步骤中无 TBD。

---

## 执行交接

计划已保存至 `docs/superpowers/plans/2026-08-18-auv-ocean-current.md`。

**两种执行方式：**

1. **Subagent-Driven（推荐）** — 每任务一个新子代理，任务间复查  
2. **Inline Execution** — 本会话连续执行，设检查点  

选哪种？
