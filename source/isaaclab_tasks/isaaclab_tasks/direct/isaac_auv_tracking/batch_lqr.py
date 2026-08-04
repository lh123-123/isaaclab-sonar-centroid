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
        max_path_points: int = 512,
        workspace_half_xy: tuple[float, float] = (20.0, 20.0),
        workspace_margin: float = 1.0,
    ):
        self.num_envs = num_envs
        self.device = device
        self.cruise_speed = float(cruise_speed)
        self.waypoint_reach_dist = float(waypoint_reach_dist)
        self.rrt_step_size = float(rrt_step_size)
        self.tau_vel = float(tau_vel)
        self.max_speed = float(max_speed)
        self.max_path_points = int(max_path_points)
        self.workspace_half_xy = workspace_half_xy
        self.workspace_margin = workspace_margin

        self.env_centers = torch.zeros(num_envs, 2, device=device)
        self.path = torch.zeros(num_envs, max_path_points, 2, device=device)
        self.path_len = torch.ones(num_envs, dtype=torch.long, device=device)
        self.path_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.yaw = torch.zeros(num_envs, device=device)

    def _bounds(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hx = self.workspace_half_xy[0] - self.workspace_margin
        hy = self.workspace_half_xy[1] - self.workspace_margin
        lo = self.env_centers[env_ids] - torch.tensor([hx, hy], device=self.device)
        hi = self.env_centers[env_ids] + torch.tensor([hx, hy], device=self.device)
        return lo, hi

    def _sample_goals(self, env_ids: torch.Tensor, avoid_xy: torch.Tensor) -> torch.Tensor:
        lo, hi = self._bounds(env_ids)
        u = torch.rand(env_ids.shape[0], 2, device=self.device)
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

    def _plan_envs(self, env_ids: torch.Tensor, start_xy: torch.Tensor) -> None:
        lo, hi = self._bounds(env_ids)
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
            # Face first segment (Holo reset yaw toward first path point)
            if pts.shape[0] >= 2:
                d = pts[min(1, len(pts) - 1)] - pts[0]
            else:
                d = goals_np[i] - starts_np[i]
            self.yaw[env_id] = math.atan2(float(d[1]), float(d[0]))

    def reset(self, env_ids: torch.Tensor, init_state: torch.Tensor, centers_xy: torch.Tensor) -> None:
        """Reset motion for envs. init_state: (N,4) [x,y,vx,vy] — vx,vy ignored (start at rest)."""
        self.env_centers[env_ids] = centers_xy
        self._plan_envs(env_ids, init_state[:, :2])

    def current_waypoint(self) -> torch.Tensor:
        idx = torch.minimum(self.path_idx, self.path_len - 1).clamp(min=0)
        return self.path[torch.arange(self.num_envs, device=self.device), idx]

    def step(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        """Integrate one physics tick. state (N,4) → next (N,4)."""
        pos = state[:, :2]
        vel = state[:, 2:4]

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
            self._plan_envs(ids, pos[ids])

        wp = self.current_waypoint()
        diff = wp - pos
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)
        dir_unit = diff / dist
        v_des = dir_unit * self.cruise_speed

        # First-order lag (ROV / LQR inertia stand-in)
        alpha = 1.0 - math.exp(-dt / max(self.tau_vel, 1e-3))
        vel_new = vel + (v_des - vel) * alpha
        spd = torch.norm(vel_new, dim=-1, keepdim=True).clamp(min=1e-6)
        vel_new = vel_new * torch.clamp(self.max_speed / spd, max=1.0)

        pos_new = pos + vel_new * dt

        # Workspace clamp
        hx = self.workspace_half_xy[0] - self.workspace_margin
        hy = self.workspace_half_xy[1] - self.workspace_margin
        lo = self.env_centers - torch.tensor([hx, hy], device=self.device)
        hi = self.env_centers + torch.tensor([hx, hy], device=self.device)
        pos_new[:, 0].clamp_(lo[:, 0], hi[:, 0])
        pos_new[:, 1].clamp_(lo[:, 1], hi[:, 1])

        # Yaw along velocity / path
        moving = torch.norm(vel_new, dim=-1) > 1e-3
        self.yaw = torch.where(
            moving,
            torch.atan2(vel_new[:, 1], vel_new[:, 0]),
            self.yaw,
        )

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
        for k in ("l_p", "l_v", "r_a", "max_accel", "max_speed_factor"):
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
