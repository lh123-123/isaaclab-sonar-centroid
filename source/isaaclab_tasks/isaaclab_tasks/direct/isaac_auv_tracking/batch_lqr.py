"""HoloOcean-style target motion for kinematic ball (open water).

Mirrors v0 AgentAuvTarget + RRT_2d + LQR chase character:
  - RRT to random goal (no obstacles → collision checks always pass)
  - densified polyline waypoints
  - advance when ||pos - wp|| < 0.3 m
  - replan when path exhausted
  - first-order velocity lag toward current waypoint at ~0.5 m/s
  - desired velocity at waypoint is toward wp (kinematic stand-in for LQR)
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _rrt_path_open_water(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    lo_xy: np.ndarray,
    hi_xy: np.ndarray,
    step_size: float = 2.0,
    max_iter: int = 2000,
    densify_ds: float = 0.25,
) -> np.ndarray:
    """Build open-water RRT path (XY), densify along polyline. Returns (M, 2)."""
    start = np.asarray(start_xy, dtype=np.float64).reshape(2)
    goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    lo = np.asarray(lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(hi_xy, dtype=np.float64).reshape(2)

    tree = [start.copy()]
    parent = [-1]
    finish = [False]
    dist_cost = [0.0]

    def _steer(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        d = dst - src
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return src.copy()
        if n > step_size:
            d = d * (step_size / n)
        return src + d

    # Bias some samples toward goal (helps open-water convergence)
    for _ in range(max_iter):
        if np.random.rand() < 0.1:
            sample = goal.copy()
        else:
            sample = np.random.uniform(lo, hi)
        dists = np.linalg.norm(np.stack(tree, axis=0) - sample[None, :], axis=1)
        nearest = int(np.argmin(dists))
        new_pt = _steer(tree[nearest], sample)
        new_pt = np.clip(new_pt, lo, hi)
        tree.append(new_pt)
        parent.append(nearest)
        dist_cost.append(dist_cost[nearest] + float(np.linalg.norm(new_pt - tree[nearest])))
        reach = float(np.linalg.norm(new_pt - goal)) <= step_size
        finish.append(reach)
        if sum(finish) >= 10:
            break
    else:
        # Fallback: straight line start → goal
        tree = [start, goal]
        parent = [-1, 0]
        finish = [False, True]
        dist_cost = [0.0, float(np.linalg.norm(goal - start))]

    connecting = [i for i, f in enumerate(finish) if f]
    if not connecting:
        # Connect nearest tree node to goal
        dists = np.linalg.norm(np.stack(tree, axis=0) - goal[None, :], axis=1)
        nearest = int(np.argmin(dists))
        tree.append(goal.copy())
        parent.append(nearest)
        connecting = [len(tree) - 1]
        dist_cost.append(dist_cost[nearest] + float(dists[nearest]))

    idx = min(connecting, key=lambda i: dist_cost[i])
    path_idx = [idx]
    while parent[path_idx[-1]] >= 0:
        path_idx.append(parent[path_idx[-1]])
    path_idx.reverse()
    nodes = np.stack([tree[i] for i in path_idx] + [goal], axis=0)

    # Densify polyline (~Holo Bezier resampling character, without bezier dep)
    densified = [nodes[0]]
    for a, b in zip(nodes[:-1], nodes[1:]):
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length < 1e-9:
            continue
        n_steps = max(1, int(math.ceil(length / densify_ds)))
        for k in range(1, n_steps + 1):
            densified.append(a + seg * (k / n_steps))
    return np.stack(densified, axis=0).astype(np.float32)


class BatchHoloTargetMotion:
    """Batched Holo v0 target XY motion for a kinematic sphere."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        cruise_speed: float = 0.5,
        waypoint_reach_dist: float = 0.3,
        rrt_step_size: float = 2.0,
        tau_vel: float = 0.7,
        max_speed: float = 0.6,
        soft_start_s: float = 3.0,
        easy_s: float = 5.0,
        leash_range: float = 3.2,
        max_path_points: int = 512,
        workspace_half_xy: tuple[float, float] = (20.0, 20.0),
        workspace_margin: float = 1.0,
        cruise_speed_spread: float = 1.0,
        speed_peak_factor: float = 0.0,
        max_accel: float = 0.0,
        bearing_limit_rad: float = math.pi,
    ):
        self.num_envs = num_envs
        self.device = device
        self.cruise_speed = float(cruise_speed)
        # --- per-episode speed variation (Holo's RRT_2d runs at a constant 0.5 m/s) ---
        # Cruise is resampled on reset from U[cruise_speed, cruise_speed * spread]; spread <= 1
        # pins it to cruise_speed, which is the Holo-faithful setting. Note cruise_speed = 0 stays
        # frozen at any spread, so the "park the target" debug scripts keep working.
        self.cruise_spread = max(float(cruise_speed_spread), 1.0)
        # Speed ceiling as a multiple of the episode's own cruise speed. <= 0 falls back to the
        # fixed `max_speed`, which is the pre-existing behaviour (0.6 = 1.2 x 0.5).
        self.speed_peak_factor = float(speed_peak_factor)
        # Acceleration ceiling in m/s^2. <= 0 disables it, leaving the first-order lag as the only
        # inertia stand-in (Holo's behaviour).
        self.max_accel = float(max_accel)
        # ±limit about the agent's heading. pi (180°) is the full circle, i.e. Holo.
        self.bearing_limit = float(bearing_limit_rad)
        self.waypoint_reach_dist = float(waypoint_reach_dist)
        self.rrt_step_size = float(rrt_step_size)
        self.tau_vel = float(tau_vel)
        self.max_speed = float(max_speed)
        self.soft_start_s = float(soft_start_s)
        self.easy_s = float(easy_s)
        self.leash_range = float(leash_range)
        self.max_path_points = int(max_path_points)
        self.workspace_half_xy = workspace_half_xy
        self.workspace_margin = workspace_margin

        self.env_centers = torch.zeros(num_envs, 2, device=device)
        self.path = torch.zeros(num_envs, max_path_points, 2, device=device)
        self.path_len = torch.ones(num_envs, dtype=torch.long, device=device)
        self.path_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.yaw = torch.zeros(num_envs, device=device)
        # Seconds since last reset (for cruise soft-start / easy leash)
        self.age_s = torch.zeros(num_envs, device=device)
        # Per-episode speed (see cruise_speed_spread / speed_peak_factor). Initialised at the
        # Holo-faithful constant so any step() before the first reset behaves exactly as before.
        self.cruise = torch.full((num_envs,), self.cruise_speed, device=device)
        self.max_speed_ep = torch.full((num_envs,), self.max_speed, device=device)
        self._refresh_speed_episode(torch.arange(num_envs, device=device))

    def _bounds(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hx = self.workspace_half_xy[0] - self.workspace_margin
        hy = self.workspace_half_xy[1] - self.workspace_margin
        lo = self.env_centers[env_ids] - torch.tensor([hx, hy], device=self.device)
        hi = self.env_centers[env_ids] + torch.tensor([hx, hy], device=self.device)
        return lo, hi

    def _clamp_heading(self, ang: torch.Tensor, agent_yaw: torch.Tensor) -> torch.Tensor:
        """Keep a heading inside ±limit of the agent.

        A path that would leave the cone is turned to the agent's own heading, not
        parked on the ±limit boundary. The boundary is only for a heading that is
        already inside the cone.
        """
        delta = torch.atan2(torch.sin(ang - agent_yaw), torch.cos(ang - agent_yaw))
        inside = delta.abs() <= self.bearing_limit
        return torch.where(inside, ang, agent_yaw)

    def _sample_goals(
        self,
        env_ids: torch.Tensor,
        avoid_xy: torch.Tensor,
        *,
        near_xy: torch.Tensor | None = None,
        near_radius: float | None = None,
        agent_yaw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lo, hi = self._bounds(env_ids)
        n = env_ids.shape[0]
        if (
            self.bearing_limit < math.pi - 1e-3
            and near_xy is not None
            and agent_yaw is not None
        ):
            ang = agent_yaw + (torch.rand(n, device=self.device) - 0.5) * (2.0 * self.bearing_limit)
            if near_radius is not None and near_radius > 1e-3:
                rad = (0.4 + 0.6 * torch.rand(n, device=self.device)) * float(near_radius)
            else:
                rad = 4.0 + 8.0 * torch.rand(n, device=self.device)
            goal = near_xy + torch.stack([torch.cos(ang) * rad, torch.sin(ang) * rad], dim=-1)
            return torch.max(torch.min(goal, hi), lo)
        if near_xy is not None and near_radius is not None and near_radius > 1e-3:
            # Easy phase: short hops inside leash disk around agent
            ang = torch.rand(n, device=self.device) * (2 * math.pi)
            rad = (0.4 + 0.6 * torch.rand(n, device=self.device)) * float(near_radius)
            goal = near_xy + torch.stack([torch.cos(ang) * rad, torch.sin(ang) * rad], dim=-1)
            goal = torch.max(torch.min(goal, hi), lo)
            return goal
        u = torch.rand(n, 2, device=self.device)
        goal = lo + u * (hi - lo)
        # Prefer goals not too close to current pos (encourage travel)
        dist = torch.norm(goal - avoid_xy, dim=-1)
        too_close = dist < 3.0
        if too_close.any():
            ang = torch.rand(int(too_close.sum()), device=self.device) * (2 * math.pi)
            goal[too_close] = avoid_xy[too_close] + torch.stack(
                [torch.cos(ang), torch.sin(ang)], dim=-1
            ) * 8.0
            goal[too_close] = torch.max(torch.min(goal[too_close], hi[too_close]), lo[too_close])
        return goal

    def _write_path(self, env_id: int, pts: np.ndarray) -> None:
        m = min(len(pts), self.max_path_points)
        self.path[env_id].zero_()
        self.path[env_id, :m] = torch.as_tensor(pts[:m], device=self.device)
        self.path_len[env_id] = m
        self.path_idx[env_id] = 0

    def _plan_envs(
        self,
        env_ids: torch.Tensor,
        start_xy: torch.Tensor,
        *,
        agent_xy: torch.Tensor | None = None,
        agent_yaw: torch.Tensor | None = None,
        leash: bool = False,
    ) -> None:
        lo, hi = self._bounds(env_ids)
        if self.bearing_limit < math.pi - 1e-3 and agent_xy is not None and agent_yaw is not None:
            radius = self.leash_range if leash else None
            goals = self._sample_goals(
                env_ids, start_xy, near_xy=agent_xy, near_radius=radius, agent_yaw=agent_yaw
            )
        elif leash and agent_xy is not None:
            goals = self._sample_goals(
                env_ids, start_xy, near_xy=agent_xy, near_radius=self.leash_range
            )
        else:
            goals = self._sample_goals(env_ids, start_xy)
        starts_np = start_xy.detach().cpu().numpy()
        goals_np = goals.detach().cpu().numpy()
        lo_np = lo.detach().cpu().numpy()
        hi_np = hi.detach().cpu().numpy()
        for i, env_id in enumerate(env_ids.tolist()):
            pts = _rrt_path_open_water(
                starts_np[i],
                goals_np[i],
                lo_np[i],
                hi_np[i],
                step_size=self.rrt_step_size,
            )
            self._write_path(env_id, pts)
            # Face first segment, same as Holo teleport: atan2(dy, dx) of the path.
            if pts.shape[0] >= 2:
                d = pts[min(1, len(pts) - 1)] - pts[0]
            else:
                d = goals_np[i] - starts_np[i]
            yaw = math.atan2(float(d[1]), float(d[0]))
            if self.bearing_limit < math.pi - 1e-3 and agent_yaw is not None:
                ref = float(agent_yaw[i])
                delta = math.atan2(math.sin(yaw - ref), math.cos(yaw - ref))
                if abs(delta) > self.bearing_limit:
                    yaw = ref
            self.yaw[env_id] = yaw

    def _refresh_speed_episode(self, env_ids: torch.Tensor) -> None:
        """Resample this episode's cruise speed and speed ceiling for ``env_ids``.

        Cruise is uniform in ``[cruise_speed, cruise_speed * cruise_spread]``; with the default
        spread of 1.0 this is a no-op and reproduces Holo's constant 0.5 m/s exactly. The ceiling
        follows the episode's own cruise (``speed_peak_factor``) so a slow episode cannot burst to
        a fast one's peak; ``speed_peak_factor <= 0`` keeps the fixed ``max_speed`` instead.

        Called from ``reset()`` only, so it never fires mid-episode: the whole episode runs at one
        cruise speed, which is what makes the soft start / leash schedule below meaningful.
        """
        n = len(env_ids)
        lo = self.cruise_speed
        hi = self.cruise_speed * self.cruise_spread
        if hi > lo:
            self.cruise[env_ids] = lo + torch.rand(n, device=self.device) * (hi - lo)
        else:
            self.cruise[env_ids] = lo
        if self.speed_peak_factor > 0.0:
            self.max_speed_ep[env_ids] = self.cruise[env_ids] * self.speed_peak_factor
        else:
            self.max_speed_ep[env_ids] = self.max_speed

    def reset(
        self,
        env_ids: torch.Tensor,
        init_state: torch.Tensor,
        centers_xy: torch.Tensor,
        agent_xy: torch.Tensor | None = None,
        agent_yaw: torch.Tensor | None = None,
    ) -> None:
        """Reset motion for envs. init_state: (N,4) [x,y,vx,vy] — vx,vy ignored (start at rest)."""
        self.env_centers[env_ids] = centers_xy
        self.age_s[env_ids] = 0.0
        # One cruise speed per episode; sampled here so the whole episode is self-consistent.
        # Everything below is unchanged from v0: the soft-start ramp, the easy leash and the
        # first-waypoint heading all behave exactly as before, just toward this episode's cruise.
        self._refresh_speed_episode(env_ids)
        # Start in easy leash so the first path stays measurable
        near = agent_xy if agent_xy is not None else init_state[:, :2]
        self._plan_envs(
            env_ids, init_state[:, :2], agent_xy=near, agent_yaw=agent_yaw, leash=self.easy_s > 1e-6
        )

    def current_waypoint(self) -> torch.Tensor:
        idx = torch.minimum(self.path_idx, self.path_len - 1).clamp(min=0)
        return self.path[torch.arange(self.num_envs, device=self.device), idx]

    def step(
        self,
        state: torch.Tensor,
        dt: float,
        agent_xy: torch.Tensor | None = None,
        agent_yaw: torch.Tensor | None = None,
        current_vel_xy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate one physics tick. state (N,4) → next (N,4)."""
        pos = state[:, :2]
        vel = state[:, 2:4]
        easy = (self.easy_s > 1e-6) & (self.age_s < self.easy_s)

        # Advance waypoint indices (Holo: dis < 0.3 → next path point)
        wp = self.current_waypoint()
        dist = torch.norm(wp - pos, dim=-1)
        reached = dist < self.waypoint_reach_dist
        if reached.any():
            ids = reached.nonzero(as_tuple=False).squeeze(-1)
            self.path_idx[ids] += 1

        # Replan when path exhausted (past last point)
        exhausted = self.path_idx >= self.path_len
        if exhausted.any():
            ids = exhausted.nonzero(as_tuple=False).squeeze(-1)
            self.path_idx[ids] = torch.clamp(self.path_len[ids] - 1, min=0)
            if agent_xy is not None and easy.any():
                easy_ids = ids[easy[ids]]
                hard_ids = ids[~easy[ids]]
                if easy_ids.numel() > 0:
                    yaw_easy = None if agent_yaw is None else agent_yaw[easy_ids]
                    self._plan_envs(
                        easy_ids,
                        pos[easy_ids],
                        agent_xy=agent_xy[easy_ids],
                        agent_yaw=yaw_easy,
                        leash=True,
                    )
                if hard_ids.numel() > 0:
                    yaw_hard = None if agent_yaw is None else agent_yaw[hard_ids]
                    self._plan_envs(
                        hard_ids, pos[hard_ids], agent_xy=agent_xy[hard_ids] if agent_xy is not None else None,
                        agent_yaw=yaw_hard, leash=False,
                    )
            else:
                yaw_ids = None if agent_yaw is None else agent_yaw[ids]
                xy_ids = None if agent_xy is None else agent_xy[ids]
                self._plan_envs(ids, pos[ids], agent_xy=xy_ids, agent_yaw=yaw_ids, leash=False)

        wp = self.current_waypoint()
        diff = wp - pos
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)
        dir_unit = diff / dist
        # Soft-start: desired cruise ramps 0 → cruise_speed over soft_start_s after reset
        if self.soft_start_s > 1e-6:
            ramp = (self.age_s / self.soft_start_s).clamp(0.0, 1.0)
        else:
            ramp = torch.ones(self.num_envs, device=self.device)
        cruise = self.cruise * ramp
        v_des = dir_unit * cruise.unsqueeze(-1)

        # Easy leash: if target drifts beyond leash_range from agent, pull back inward
        if agent_xy is not None and easy.any():
            rel = pos - agent_xy
            dist_a = torch.norm(rel, dim=-1).clamp(min=1e-6)
            too_far = easy & (dist_a > self.leash_range)
            if too_far.any():
                inward = -rel / dist_a.unsqueeze(-1) * cruise.unsqueeze(-1)
                v_des = torch.where(too_far.unsqueeze(-1), inward, v_des)

        if self.bearing_limit < math.pi - 1e-3 and agent_xy is not None and agent_yaw is not None:
            rel = pos - agent_xy
            dist_a = torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-6)
            bear = torch.atan2(rel[:, 1], rel[:, 0]) - agent_yaw
            bear = torch.atan2(torch.sin(bear), torch.cos(bear))
            outside = bear.abs() > self.bearing_limit
            if outside.any():
                held = agent_yaw + bear.clamp(-self.bearing_limit, self.bearing_limit)
                desired = agent_xy + torch.stack([torch.cos(held), torch.sin(held)], dim=-1) * dist_a
                pull = desired - pos
                pull = pull / torch.norm(pull, dim=-1, keepdim=True).clamp(min=1e-6)
                v_des = torch.where(outside.unsqueeze(-1), pull * cruise.unsqueeze(-1), v_des)

        # Travel direction stays inside the same cone as the position bearing, so the hull
        # does not face more than ±limit away from the agent while the body slides elsewhere.
        if self.bearing_limit < math.pi - 1e-3 and agent_yaw is not None:
            spd_des = torch.norm(v_des, dim=-1, keepdim=True)
            des_ang = torch.atan2(v_des[:, 1], v_des[:, 0])
            held = self._clamp_heading(des_ang, agent_yaw)
            v_des = torch.stack([torch.cos(held), torch.sin(held)], dim=-1) * spd_des

        # First-order lag (ROV / LQR inertia stand-in)
        alpha = 1.0 - math.exp(-dt / max(self.tau_vel, 1e-3))
        vel_new = vel + (v_des - vel) * alpha
        # Acceleration ceiling: bounds |dv| per tick. Applied *before* the speed cap so the cap
        # stays authoritative. This does not flatten the cold start -- during the soft start the
        # ramp already holds acceleration to cruise / soft_start_s (<= 0.33 m/s^2 at cruise 1.0),
        # i.e. well under the default 1.0 -- so the catch-up window at episode start is unchanged.
        if self.max_accel > 0.0:
            dv = vel_new - vel
            dv_norm = torch.norm(dv, dim=-1, keepdim=True).clamp(min=1e-6)
            vel_new = vel + dv * torch.clamp(self.max_accel * dt / dv_norm, max=1.0)
        spd = torch.norm(vel_new, dim=-1, keepdim=True).clamp(min=1e-6)
        # Cap also scales with soft-start so peak does not jump early
        max_spd = self.max_speed_ep.unsqueeze(-1) * ramp.unsqueeze(-1).clamp(min=1e-3)
        vel_new = vel_new * torch.clamp(max_spd / spd, max=1.0)
        self.age_s = self.age_s + float(dt)

        pos_new = pos + vel_new * dt
        if current_vel_xy is not None:
            pos_new = pos_new + current_vel_xy * dt

        # Workspace clamp (after current overlay)
        hx = self.workspace_half_xy[0] - self.workspace_margin
        hy = self.workspace_half_xy[1] - self.workspace_margin
        lo = self.env_centers - torch.tensor([hx, hy], device=self.device)
        hi = self.env_centers + torch.tensor([hx, hy], device=self.device)
        pos_new[:, 0].clamp_(lo[:, 0], hi[:, 0])
        pos_new[:, 1].clamp_(lo[:, 1], hi[:, 1])

        # Soft positional leash (prevents overshoot leaving the disk)
        if agent_xy is not None and easy.any():
            rel = pos_new - agent_xy
            dist_a = torch.norm(rel, dim=-1).clamp(min=1e-6)
            scale = torch.clamp(self.leash_range / dist_a, max=1.0)
            scale = torch.where(easy, scale, torch.ones_like(scale))
            pos_new = agent_xy + rel * scale.unsqueeze(-1)

        if self.bearing_limit < math.pi - 1e-3 and agent_xy is not None and agent_yaw is not None:
            rel = pos_new - agent_xy
            dist_a = torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-6)
            bear = torch.atan2(rel[:, 1], rel[:, 0]) - agent_yaw
            bear = torch.atan2(torch.sin(bear), torch.cos(bear))
            outside = bear.abs() > self.bearing_limit
            if outside.any():
                held = agent_yaw + bear.clamp(-self.bearing_limit, self.bearing_limit)
                snapped = agent_xy + torch.stack([torch.cos(held), torch.sin(held)], dim=-1) * dist_a
                pos_new = torch.where(outside.unsqueeze(-1), snapped, pos_new)
                vel_new = torch.where(outside.unsqueeze(-1), v_des, vel_new)

        # Yaw along velocity / path, then the same ±limit about the agent's heading.
        moving = torch.norm(vel_new, dim=-1) > 1e-3
        yaw = torch.where(moving, torch.atan2(vel_new[:, 1], vel_new[:, 0]), self.yaw)
        if self.bearing_limit < math.pi - 1e-3 and agent_yaw is not None:
            yaw = self._clamp_heading(yaw, agent_yaw)
        self.yaw = yaw

        next_state = torch.zeros_like(state)
        next_state[:, :2] = pos_new
        next_state[:, 2:4] = vel_new
        return next_state


# Back-compat alias used by older imports / comments
class BatchKinematicTargetLQR(BatchHoloTargetMotion):
    """Deprecated name — Holo RRT target motion."""

    def __init__(self, num_envs, device, dt=0.2, cruise_speed=0.5, **kwargs):
        reach = kwargs.pop("waypoint_reach_dist", 0.3)
        workspace_half_xy = kwargs.pop("workspace_half_xy", (20.0, 20.0))
        workspace_margin = kwargs.pop("workspace_margin", 1.0)
        for k in ("l_p", "l_v", "r_a", "max_speed_factor"):
            kwargs.pop(k, None)
        super().__init__(
            num_envs=num_envs,
            device=device,
            cruise_speed=cruise_speed,
            waypoint_reach_dist=reach,
            workspace_half_xy=workspace_half_xy,
            workspace_margin=workspace_margin,
            **kwargs,
        )
