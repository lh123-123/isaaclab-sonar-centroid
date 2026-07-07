"""Module C: EKF belief estimation — world-frame state (design doc §3.3, /home/lh/1 style)."""

from __future__ import annotations

import torch

from isaaclab.utils.math import wrap_to_pi


def observation_noise(target_range: torch.Tensor) -> torch.Tensor:
    """Range-dependent 2x2 observation covariance for [azimuth, range].

    Larger noise (2° base) + higher process noise prevents EKF covariance
    saturation, matching /home/lh/1 behavior.
    """
    sigma_r = 0.05 * target_range + 0.1
    sigma_a = torch.deg2rad(2.0 + 0.0 * target_range)
    n = target_range.shape[0]
    r = torch.zeros(n, 2, 2, device=target_range.device, dtype=target_range.dtype)
    r[:, 0, 0] = sigma_a * sigma_a
    r[:, 1, 1] = sigma_r * sigma_r
    return r


class BeliefEstimator:
    """Batched extended-Kalman filter for target position in **world** frame.

    State: [target_wx, target_wy, target_vwx, target_vwy]  (world coordinates).
    Observation: [azimuth, range] in sonar body frame.
    Agent position and yaw are supplied at each update step.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        process_noise: float = 0.01,
        lost_inflation: float = 1.02,
        lost_inflation_speed: float = 0.05,
        max_position_variance: float = 25.0,
        max_velocity_variance: float = 4.0,
    ):
        self.num_envs = num_envs
        self.device = device
        self.process_noise = process_noise
        self.lost_inflation = lost_inflation
        self.lost_inflation_speed = lost_inflation_speed
        self.max_position_variance = max_position_variance
        self.max_velocity_variance = max_velocity_variance

        self.x = torch.zeros(num_envs, 4, device=device)  # wx, wy, vwx, vwy
        self.p = torch.eye(4, device=device).unsqueeze(0).repeat(num_envs, 1, 1) * 1.0

    def _cap_covariance(self) -> None:
        """Cap covariance at upper bounds only (no floor)."""
        self.p = self.p.clone()
        self.p[:, 0, 0] = torch.clamp(self.p[:, 0, 0], max=self.max_position_variance)
        self.p[:, 1, 1] = torch.clamp(self.p[:, 1, 1], max=self.max_position_variance)
        self.p[:, 2, 2] = torch.clamp(self.p[:, 2, 2], max=self.max_velocity_variance)
        self.p[:, 3, 3] = torch.clamp(self.p[:, 3, 3], max=self.max_velocity_variance)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.x[env_ids] = 0.0
        self.p[env_ids] = torch.eye(4, device=self.device)

    def predict(self, dt: float) -> None:
        """CV predict in world frame."""
        f = torch.eye(4, device=self.device)
        f[0, 2] = dt
        f[1, 3] = dt
        f_b = f.unsqueeze(0).expand(self.num_envs, -1, -1)
        self.x = torch.bmm(f_b, self.x.unsqueeze(-1)).squeeze(-1)
        # Standard CV discrete process noise
        q = self.process_noise
        dt2 = dt * dt
        dt3 = dt2 * dt
        q_mat = torch.zeros(4, 4, device=self.device, dtype=self.x.dtype)
        q_mat[0, 0] = q * dt3 / 3.0
        q_mat[1, 1] = q * dt3 / 3.0
        q_mat[2, 2] = q * dt
        q_mat[3, 3] = q * dt
        q_mat[0, 2] = q * dt2 / 2.0
        q_mat[2, 0] = q * dt2 / 2.0
        q_mat[1, 3] = q * dt2 / 2.0
        q_mat[3, 1] = q * dt2 / 2.0
        self.p = torch.bmm(torch.bmm(f_b, self.p), f_b.transpose(1, 2)) + q_mat
        self._cap_covariance()

    def _apply_lost_inflation(self, mask: torch.Tensor, dt: float = 0.1) -> None:
        """Increase position uncertainty when target is not detected (world frame)."""
        if not mask.any():
            return
        boost_diag = torch.full((4,), self.lost_inflation_speed * dt, device=self.device)
        boost = torch.diag(boost_diag).unsqueeze(0).expand(mask.sum(), -1, -1)
        self.p[mask] = self.p[mask] + boost
        self._cap_covariance()

    # ------------------------------------------------------------------
    # Observation model — operates on body-frame relative position
    # ------------------------------------------------------------------

    @staticmethod
    def _h_body(rel_body: torch.Tensor) -> torch.Tensor:
        """Predicted observation [azimuth, range] from body-frame relative position (N,2)."""
        px, py = rel_body[:, 0], rel_body[:, 1]
        bearing = torch.atan2(py, px)
        dist = torch.sqrt(px * px + py * py + 1.0e-8)
        return torch.stack([bearing, dist], dim=-1)

    @staticmethod
    def _h_jacobian_body(rel_body: torch.Tensor) -> torch.Tensor:
        """Jacobian ∂[azimuth, range]/∂rel_body at a body-frame relative position (N,2)."""
        px, py = rel_body[:, 0], rel_body[:, 1]
        r2 = px * px + py * py + 1.0e-8
        r = torch.sqrt(r2)
        h = torch.zeros(rel_body.shape[0], 2, 4, device=rel_body.device, dtype=rel_body.dtype)
        h[:, 0, 0] = -py / r2
        h[:, 0, 1] = px / r2
        h[:, 1, 0] = px / r
        h[:, 1, 1] = py / r
        return h  # cols 2,3 remain zero (velocity does not affect observation)

    def update(
        self,
        azimuth: torch.Tensor,
        target_range: torch.Tensor,
        confidence: torch.Tensor,
        detected: torch.Tensor,
        *,
        agent_pos_xy: torch.Tensor | None = None,
        agent_yaw: torch.Tensor | None = None,
        conf_threshold: float = 0.15,
        dt: float = 0.1,
    ) -> None:
        """EKF update with body-frame sonar observations and world-frame agent pose."""
        valid = detected & (confidence >= conf_threshold) & torch.isfinite(target_range)
        if not valid.any():
            self._apply_lost_inflation(torch.ones(self.num_envs, dtype=torch.bool, device=self.device), dt=dt)
            return

        z = torch.stack(
            [
                azimuth,
                torch.clamp(target_range, min=1.0e-3),
            ],
            dim=-1,
        )
        z_cart = torch.stack(
            [target_range * torch.cos(azimuth), target_range * torch.sin(azimuth)],
            dim=-1,
        )

        # Precompute body→world rotation for each env
        # R(yaw) = [[cos, -sin], [sin, cos]]
        ca = torch.cos(agent_yaw)
        sa = torch.sin(agent_yaw)
        # world→body rotation: R(-yaw) = [[ca, sa], [-sa, ca]]

        for idx in valid.nonzero(as_tuple=False).flatten():
            i = int(idx.item())
            if not valid[i]:
                continue

            # ---- initialise state from first detection (body → world) ----
            if torch.allclose(self.x[i, :2], torch.zeros(2, device=self.device)):
                self.x[i, 0] = agent_pos_xy[i, 0] + ca[i] * z_cart[i, 0] - sa[i] * z_cart[i, 1]
                self.x[i, 1] = agent_pos_xy[i, 1] + sa[i] * z_cart[i, 0] + ca[i] * z_cart[i, 1]

            # ---- predicted body-frame relative position ----
            rel_wx = self.x[i, 0] - agent_pos_xy[i, 0]
            rel_wy = self.x[i, 1] - agent_pos_xy[i, 1]
            # R(-yaw) = [[ca, sa], [-sa, ca]]
            rel_bx = ca[i] * rel_wx + sa[i] * rel_wy
            rel_by = -sa[i] * rel_wx + ca[i] * rel_wy
            rel_body = rel_bx, rel_by

            # predicted observation
            rbody = torch.tensor([[rel_body[0], rel_body[1]]], device=self.device)
            h = self._h_body(rbody)[0]
            y = z[i] - h
            y[0] = wrap_to_pi(y[0])

            # Jacobian chain: H = ∂h/∂state = ∂h/∂rel_body × ∂rel_body/∂state
            # ∂rel_body/∂rel_world = R(-yaw),  ∂rel_world/∂state = [I₂, 0₂]
            hj = self._h_jacobian_body(rbody)[0]  # (2,4)
            # H[:, :2] = hj[:, :2] @ R(-yaw)
            H = torch.zeros(2, 4, device=self.device)
            H[:, 0] = ca[i] * hj[:, 0] - sa[i] * hj[:, 1]
            H[:, 1] = sa[i] * hj[:, 0] + ca[i] * hj[:, 1]
            # H[:, 2:] already zero (velocity columns)

            r = observation_noise(target_range[i: i + 1])[0]
            s = H @ self.p[i] @ H.T + r
            k = self.p[i] @ H.T @ torch.linalg.inv(s)
            self.x[i] = self.x[i] + (k @ y.unsqueeze(-1)).squeeze(-1)
            self.p[i] = (torch.eye(4, device=self.device) - k @ H) @ self.p[i]
            self.p[i] = 0.5 * (self.p[i] + self.p[i].transpose(0, 1))
            self._cap_covariance()

        lost = ~valid
        self._apply_lost_inflation(lost, dt=dt)

    # ------------------------------------------------------------------
    # World-frame state accessors
    # ------------------------------------------------------------------

    @property
    def mean_xy(self) -> torch.Tensor:
        """Target world position estimate (N, 2)."""
        return self.x[:, :2]

    @property
    def velocity_xy(self) -> torch.Tensor:
        """Target world velocity estimate (N, 2)."""
        return self.x[:, 2:4]

    @property
    def uncertainty_xy(self) -> torch.Tensor:
        """Diagonal position variances in world frame (N, 2)."""
        return torch.stack([self.p[:, 0, 0], self.p[:, 1, 1]], dim=-1)

    @property
    def position_covariance(self) -> torch.Tensor:
        """2×2 position block of the EKF covariance (world frame)."""
        return self.p[:, :2, :2]

    @property
    def position_cov_det(self) -> torch.Tensor:
        """det(Σ_xy) — rotation invariant, usable as reward signal."""
        return torch.linalg.det(self.position_covariance)

    # ------------------------------------------------------------------
    # Body-frame state (available for policy obs / rewards)
    # ------------------------------------------------------------------

    def state_in_body(
        self,
        agent_pos_xy: torch.Tensor,
        agent_yaw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return position and velocity in sonar body frame.

        Args:
            agent_pos_xy: (N, 2) agent world position.
            agent_yaw: (N,) agent yaw angle.

        Returns dict with keys ``position``, ``velocity`` (body frame, N,2).
        """
        rel_xy = self.mean_xy - agent_pos_xy  # (N, 2)
        ca = torch.cos(agent_yaw)
        sa = torch.sin(agent_yaw)
        # R(-yaw)
        bx = ca * rel_xy[:, 0] + sa * rel_xy[:, 1]
        by = -sa * rel_xy[:, 0] + ca * rel_xy[:, 1]
        pos_body = torch.stack([bx, by], dim=-1)

        vx_w = self.velocity_xy[:, 0]
        vy_w = self.velocity_xy[:, 1]
        vbx = ca * vx_w + sa * vy_w
        vby = -sa * vx_w + ca * vy_w
        vel_body = torch.stack([vbx, vby], dim=-1)

        return {"position": pos_body, "velocity": vel_body}
