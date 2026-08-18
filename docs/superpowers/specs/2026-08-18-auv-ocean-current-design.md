# Ocean currents for Isaac-AUV-Tracking (design)

**Date:** 2026-08-18  
**Status:** draft for review  
**Task:** `Isaac-AUV-Tracking-Direct-v0` / `Direct-v1` (shared env dynamics)  
**Decision:** Approach A — relative water velocity into existing hydrodynamics, with a Holo-like `set_ocean_currents` API.

## Goal

Add **optional, configurable ocean currents** so training/eval can run calm water (default) or graded current strength, without changing obs/reward/KF interfaces in phase 1.

## Non-goals (phase 1)

- Obstacles / LOS / rangefinder
- Putting current into policy obs or KF state
- Changing reward / done semantics
- Copying HoloOcean’s Unreal `OceanCurrentsCommand` wire protocol
- Replacing WarpAUV drag with Holo’s exact \(-\tfrac12\rho\|v\|^2 C_d A\) formula
- Full CFD / spatial grid field in PhysX

## HoloOcean reference (what we align to)

### RL layer (`RL_AUV_tracking`)

1. Config `ocean_current.enabled / type / params` (+ optional visualization).
2. Each tick, before `ocean.act`:  
   `v = field(agent_xyz, t)` → clip speed → `ocean.set_ocean_currents('auv0', v)`.
3. Field library: `uniform` / `vortex` / `random` / `sine_wave` in `current_fields.py`.
4. Default application target: **agent only** (`auv0`).

### Engine layer (`HolodeckBuoyantAgent`)

Currents are a **per-agent water velocity**. Drag uses relative velocity:

\[
v_{\mathrm{rel}} = v_{\mathrm{auv}} - v_{\mathrm{current}}
\]

then quadratic drag on \(v_{\mathrm{rel}}\).  
`set_ocean_currents` only updates stored `Ocean_Current_Velocity_*`; it does not implement a global fluid sim.

**Alignment:** same **API semantics** (set water velocity on agent) and same **relative-velocity drag idea**.  
**Not aligned:** UE command channel, UE Y-axis flip, Holo depth sign conventions, Holo’s exact drag coefficients.

## Isaac platform deltas (must drive implementation)

| Topic | Holo | Isaac integration | Implication |
|-------|------|-------------------|-------------|
| Apply path | `set_ocean_currents` → engine | Forces in `_compute_dynamics` | API writes buffer; dynamics reads it |
| Hydro model | Quadratic drag on \(v_{\mathrm{rel}}\) | `HydrodynamicForceModels` density + viscosity | Pass `root_lin_vel_w - v_current` into existing calls |
| Parallelism | Few envs | Many envs | Batched field eval on GPU |
| Depth / frame | Underwater \(z<0\), `fix_depth≈-15`; UE Y flip | `starting_depth≈8`, Z-up | Isaac world frame only; do not copy Holo “\(z>0\Rightarrow 0\)” or UE Y negate |
| Target | Often another AUV | Kinematic tracking **ball** | Optional `affect_target`: kinematic velocity overlay, not buoyant-agent API |
| Task defaults | Some configs enable currents | Calm water was intentional | **`enabled=false` by default** |

## Design

### 1. Public API (Holo-like)

On the Direct env (v0 base; v1 inherits):

```text
set_ocean_currents(env_ids, velocity)
  # velocity: (3,) broadcast or (len(env_ids), 3), world-frame m/s
  # stores into self._ocean_current_vel[env_ids], then clip

get_ocean_currents(env_ids) -> Tensor[N,3]
```

- Semantically matches Holo: “set water velocity at this vehicle.”
- Unit tests / debug can set constant `[1,0,0]` like Holo `test_currents.py`.
- Training path: field function → `set_ocean_currents` every step (not ad-hoc force injection elsewhere).

Optional later (not required phase 1): `set_target_ocean_currents` if target needs a separate vector; until then, when `affect_target=true`, reuse the same `v` sampled at target position (or at AUV position — **prefer sample at each body’s position**).

### 2. Dynamics (Approach A)

In `_compute_dynamics`:

```text
v_cur = self._ocean_current_vel          # [N,3], zeros if disabled
v_rel = root_lin_vel_w - v_cur
# existing density + viscosity forces use v_rel instead of root_lin_vel_w
# buoyancy / thrusters unchanged
```

When `enabled=false`, buffer stays zero → identical to today’s calm water.

### 3. Field module

New: `ocean_current.py` under `isaac_auv_tracking/` (torch, batched).

- Types (names aligned with Holo): `uniform`, `vortex`, `sine_wave`, `random`.
- Signature: `(pos_w[N,3], t[N|scalar], params) -> v[N,3]`.
- Parameters adapted to Isaac frames (centers near `starting_depth`, not Holo `-10`).
- Factory from cfg: `create_ocean_current_field(cfg) -> callable | None`.

### 4. Config (`AUVTrackingEnvCfg`)

Conceptual fields:

```yaml
ocean_current:
  enabled: false                 # default: calm water
  type: uniform                  # uniform | vortex | sine_wave | random
  strength: 0.3                  # primary intensity knob [m/s] (scales / maps into type params)
  level: null                    # optional preset: off|mild|medium|strong → strength
  affect_target: false           # default AUV-only; true → also kinematic target
  clip_speed: 5.0                # match Holo max clip
  params: {}                     # type-specific (direction, center, period, seed, …)
```

Suggested level map (tunable): `off=0`, `mild=0.15`, `medium=0.4`, `strong=0.8` m/s.

CLI / train scripts may expose `--ocean_current_level` later; not required for core env merge.

### 5. Who is affected

| Flag | AUV | Target ball |
|------|-----|-------------|
| `enabled=false` | no | no |
| `enabled=true`, `affect_target=false` (default) | relative-velocity hydro | unchanged motion |
| both true | relative-velocity hydro | add \(v_{\mathrm{current}}(\mathrm{pos_{tgt}},t)\) into kinematic target update |

### 6. Step order

1. (Optional) evaluate field at AUV (and target) positions.
2. `set_ocean_currents` for AUV buffer.
3. Existing control → thrusters → `_compute_dynamics` with \(v_{\mathrm{rel}}\).
4. If `affect_target`, overlay current on target motion for this step.

### 7. Observability / logging (phase 1)

- TensorBoard or episode info: mean `\|v_current\|` (optional, low priority).
- Debug draw of vector field: **optional / defer** (no Holo `draw_debug_vector_field` equivalent required).

### 8. Compatibility

- Default off → existing SB3 / DrQ checkpoints and calm-water semantics unchanged.
- Direct-v1 inherits dynamics; no network / obs change.

## Implementation sketch (for later plan)

1. `ocean_current.py` + unit tests (CPU/GPU) for field shapes and clip.
2. Buffer + `set/get_ocean_currents` on `AUVTrackingEnv`.
3. Wire \(v_{\mathrm{rel}}\) in `_compute_dynamics`.
4. Optional target overlay behind `affect_target`.
5. Cfg defaults + short headless smoke (enabled uniform mild, 1–2 envs).
6. Docs: how to enable levels; note differences vs Holo.

## Risks

- Strength scales may not feel like Holo m/s because drag model differs — tune via `strength` / levels empirically.
- `random` / `vortex` need stable seeds per env for reproducibility.
- Target overlay changes relative tracking difficulty a lot — keep default off.

## Approval

- [x] Approach A + Holo-like API (user)
- [x] Default AUV-only, `affect_target` configurable (user)
- [x] Account for Isaac vs Holo deltas (user)
- [x] Spec file review
