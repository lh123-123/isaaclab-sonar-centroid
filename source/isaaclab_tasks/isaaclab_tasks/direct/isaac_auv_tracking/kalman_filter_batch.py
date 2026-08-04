"""
Batch Kalman Filter for target tracking (PyTorch GPU version).

Ported from RL_AUV_tracking/auv_control/estimation/kf.py
Supports batched computation across parallel environments on GPU.

State: [x, y, vx, vy] (4D) per target
Observation: [r, alpha] (2D polar coordinates relative to agent)
"""

from __future__ import annotations

import torch


def _cv_process_noise(dim: int, dt: float, const_q: float, device: torch.device) -> torch.Tensor:
    """Constant-velocity process noise W used by HoloOcean v0.

    W = const_q * [[dt^3/3 I, dt^2/2 I],
                   [dt^2/2 I, dt I]]
    """
    I2 = torch.eye(2, device=device)
    W = torch.zeros(dim, dim, device=device)
    W[0:2, 0:2] = (dt**3 / 3.0) * I2
    W[0:2, 2:4] = (dt**2 / 2.0) * I2
    W[2:4, 0:2] = (dt**2 / 2.0) * I2
    W[2:4, 2:4] = dt * I2
    return const_q * W


class BatchKalmanFilter:
    """Extended Kalman Filter for target position estimation, batched on GPU."""

    def __init__(
        self,
        num_envs: int,
        dim: int = 4,
        device: torch.device = "cuda",
        dt: float = 0.2,
        const_q: float = 0.5,
        sensor_r_sd: float = 0.2,
        sensor_b_sd: float = 0.01,
    ):
        self.num_envs = num_envs
        self.dim = dim
        self.device = device
        self.dt = dt
        self.sensor_r_sd = sensor_r_sd
        self.sensor_b_sd = sensor_b_sd

        self.A = torch.eye(dim, device=device)
        self.A[0, 2] = dt
        self.A[1, 3] = dt
        self.W = _cv_process_noise(dim, dt, const_q, device)

        self.state = torch.zeros(num_envs, dim, device=device)
        self.cov = torch.zeros(num_envs, dim, dim, device=device)
        # Holo KFbelief.limit: [xmin,ymin,vxmin,vymin] .. [xmax,ymax,vxmax,vymax]
        self.limit_lo = torch.full((num_envs, dim), -1.0e6, device=device)
        self.limit_hi = torch.full((num_envs, dim), 1.0e6, device=device)

    def set_limits(self, env_ids: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
        """lo/hi: (N, dim) absolute state bounds (world frame)."""
        self.limit_lo[env_ids] = lo
        self.limit_hi[env_ids] = hi

    def _clip_state(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.state = torch.maximum(torch.minimum(self.state, self.limit_hi), self.limit_lo)
        else:
            self.state[env_ids] = torch.maximum(
                torch.minimum(self.state[env_ids], self.limit_hi[env_ids]), self.limit_lo[env_ids]
            )

    def reset(self, env_ids: torch.Tensor, init_state: torch.Tensor, init_cov: float):
        self.state[env_ids] = init_state
        self.cov[env_ids] = torch.eye(self.dim, device=self.device) * init_cov
        self._clip_state(env_ids)

    def predict(self):
        self.state = self.state @ self.A.T
        cov_temp = self.A @ self.cov
        cov = cov_temp @ self.A.T + self.W
        cov = 0.5 * (cov + cov.transpose(1, 2))
        self.cov = cov + 1e-6 * torch.eye(self.dim, device=self.device)
        self._clip_state()

    def update(
        self,
        z_t: torch.Tensor,
        agent_pos: torch.Tensor,
        agent_yaw: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ):
        """EKF update with polar observation (matches HoloOcean KF H construction)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        N = env_ids.shape[0]
        if N == 0:
            return

        state = self.state[env_ids]
        cov = self.cov[env_ids]

        # Body-frame predicted measurement (for innovation)
        diff = state[:, :2] - agent_pos
        cos_yaw = torch.cos(agent_yaw)
        sin_yaw = torch.sin(agent_yaw)
        dx = cos_yaw * diff[:, 0] + sin_yaw * diff[:, 1]
        dy = -sin_yaw * diff[:, 0] + cos_yaw * diff[:, 1]
        r_pred = torch.sqrt(dx**2 + dy**2 + 1e-10)
        alpha_pred = torch.atan2(dy, dx)

        # HoloOcean v0 Jacobian: global-frame range/bearing derivatives / r
        # H = [[dx, dy, 0, 0], [-dy/r, dx/r, 0, 0]] / r  with global diff
        r_global = torch.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2 + 1e-10)
        H = torch.zeros(N, 2, 4, device=self.device)
        H[:, 0, 0] = diff[:, 0] / r_global
        H[:, 0, 1] = diff[:, 1] / r_global
        H[:, 1, 0] = -diff[:, 1] / (r_global**2)
        H[:, 1, 1] = diff[:, 0] / (r_global**2)
        # Equivalent to v0's "/ r" form for row0; row1 already / r^2

        innov = torch.zeros(N, 2, device=self.device)
        innov[:, 0] = z_t[:, 0] - r_pred
        innov[:, 1] = torch.atan2(torch.sin(z_t[:, 1] - alpha_pred), torch.cos(z_t[:, 1] - alpha_pred))

        R = torch.zeros(N, 2, 2, device=self.device)
        R[:, 0, 0] = self.sensor_r_sd**2
        R[:, 1, 1] = self.sensor_b_sd**2

        H_cov = torch.bmm(H, cov)
        S = torch.bmm(H_cov, H.transpose(1, 2)) + R
        cov_Ht = torch.bmm(cov, H.transpose(1, 2))
        K = torch.bmm(cov_Ht, torch.linalg.inv(S))

        self.state[env_ids] = state + torch.bmm(K, innov.unsqueeze(-1)).squeeze(-1)
        I = torch.eye(self.dim, device=self.device).expand(N, -1, -1)
        cov_new = torch.bmm(I - torch.bmm(K, H), cov)
        cov_new = 0.5 * (cov_new + cov_new.transpose(1, 2))
        # Keep covariance numerically PSD
        eye = torch.eye(self.dim, device=self.device)
        self.cov[env_ids] = cov_new + 1e-6 * eye
        self._clip_state(env_ids)

    def get_observation(self, target_pos: torch.Tensor, agent_pos: torch.Tensor, agent_yaw: torch.Tensor):
        """Generate noisy body-frame polar observation (constant sensor noise like v0)."""
        diff = target_pos - agent_pos
        cos_yaw = torch.cos(agent_yaw)
        sin_yaw = torch.sin(agent_yaw)
        dx = cos_yaw * diff[:, 0] + sin_yaw * diff[:, 1]
        dy = -sin_yaw * diff[:, 0] + cos_yaw * diff[:, 1]
        r = torch.sqrt(dx**2 + dy**2 + 1e-10)
        alpha = torch.atan2(dy, dx)
        z = torch.stack(
            [r + torch.randn_like(r) * self.sensor_r_sd, alpha + torch.randn_like(alpha) * self.sensor_b_sd],
            dim=-1,
        )
        return z, r, alpha


class BatchUnscentedKalmanFilter:
    """Unscented Kalman Filter for target tracking, batched on GPU."""

    def __init__(
        self,
        num_envs: int,
        dim: int = 4,
        device: torch.device = "cuda",
        dt: float = 0.2,
        const_q: float = 0.5,
        sensor_r_sd: float = 0.2,
        sensor_b_sd: float = 0.01,
        kappa: float = 1.0,
    ):
        self.num_envs = num_envs
        self.dim = dim
        self.device = device
        self.dt = dt
        self.sensor_r_sd = sensor_r_sd
        self.sensor_b_sd = sensor_b_sd

        self.A = torch.eye(dim, device=device)
        self.A[0, 2] = dt
        self.A[1, 3] = dt
        self.W = _cv_process_noise(dim, dt, const_q, device)

        self.kappa = kappa
        n = dim
        self.n_sigma = 2 * n + 1
        lam = kappa
        self.Wm = torch.zeros(self.n_sigma, device=device)
        self.Wc = torch.zeros(self.n_sigma, device=device)
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam)
        self.Wm[1:] = 1.0 / (2 * (n + lam))
        self.Wc[1:] = 1.0 / (2 * (n + lam))

        self.state = torch.zeros(num_envs, dim, device=device)
        self.cov = torch.zeros(num_envs, dim, dim, device=device)
        self.limit_lo = torch.full((num_envs, dim), -1.0e6, device=device)
        self.limit_hi = torch.full((num_envs, dim), 1.0e6, device=device)

    def set_limits(self, env_ids: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
        self.limit_lo[env_ids] = lo
        self.limit_hi[env_ids] = hi

    def _clip_state(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.state = torch.maximum(torch.minimum(self.state, self.limit_hi), self.limit_lo)
        else:
            self.state[env_ids] = torch.maximum(
                torch.minimum(self.state[env_ids], self.limit_hi[env_ids]), self.limit_lo[env_ids]
            )

    def reset(self, env_ids: torch.Tensor, init_state: torch.Tensor, init_cov: float):
        self.state[env_ids] = init_state
        self.cov[env_ids] = torch.eye(self.dim, device=self.device) * init_cov
        self._clip_state(env_ids)

    def _sigma_points(self):
        n = self.dim
        batch = self.state.shape[0]
        scale = n + self.kappa
        cov_scaled = scale * self.cov + 1e-8 * torch.eye(n, device=self.device).expand(batch, -1, -1)
        L = torch.linalg.cholesky(cov_scaled)
        sigmas = torch.zeros(batch, self.n_sigma, n, device=self.device)
        sigmas[:, 0, :] = self.state
        for i in range(n):
            sigmas[:, i + 1, :] = self.state + L[:, :, i]
            sigmas[:, i + 1 + n, :] = self.state - L[:, :, i]
        return sigmas

    def predict(self):
        sigmas = self._sigma_points()
        sigmas_pred = sigmas @ self.A.T
        state_pred = (self.Wm * sigmas_pred.transpose(0, 1)).sum(dim=-1).transpose(0, 1)
        diff = sigmas_pred - state_pred.unsqueeze(1)
        cov_pred = torch.zeros(self.state.shape[0], self.dim, self.dim, device=self.device)
        for i in range(self.n_sigma):
            d = diff[:, i, :].unsqueeze(-1)
            cov_pred += self.Wc[i] * torch.bmm(d, d.transpose(1, 2))
        self.state = state_pred
        self.cov = cov_pred + self.W
        self._clip_state()

    def update(
        self,
        z_t: torch.Tensor,
        agent_pos: torch.Tensor,
        agent_yaw: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        N = env_ids.shape[0]
        if N == 0:
            return

        full_state = self.state
        full_cov = self.cov
        self.state = full_state[env_ids].clone()
        self.cov = full_cov[env_ids].clone()
        try:
            sigmas = self._sigma_points()
            sigma_pos = sigmas[:, :, :2]
            diff = sigma_pos - agent_pos.unsqueeze(1)
            cos_yaw = torch.cos(agent_yaw).unsqueeze(1)
            sin_yaw = torch.sin(agent_yaw).unsqueeze(1)
            dx = cos_yaw * diff[:, :, 0] + sin_yaw * diff[:, :, 1]
            dy = -sin_yaw * diff[:, :, 0] + cos_yaw * diff[:, :, 1]
            r = torch.sqrt(dx**2 + dy**2 + 1e-10)
            alpha = torch.atan2(dy, dx)
            z_sigmas = torch.stack([r, alpha], dim=-1)

            z_pred = (self.Wm * z_sigmas.transpose(0, 1)).sum(dim=-1).transpose(0, 1)
            z_diff = z_sigmas - z_pred.unsqueeze(1)
            z_diff[:, :, 1] = torch.atan2(torch.sin(z_diff[:, :, 1]), torch.cos(z_diff[:, :, 1]))

            Pzz = torch.zeros(N, 2, 2, device=self.device)
            for i in range(self.n_sigma):
                d = z_diff[:, i, :].unsqueeze(-1)
                Pzz += self.Wc[i] * torch.bmm(d, d.transpose(1, 2))

            state_diff = sigmas - self.state.unsqueeze(1)
            Pxz = torch.zeros(N, self.dim, 2, device=self.device)
            for i in range(self.n_sigma):
                sd = state_diff[:, i, :].unsqueeze(-1)
                zd = z_diff[:, i, :].unsqueeze(-1)
                Pxz += self.Wc[i] * torch.bmm(sd, zd.transpose(1, 2))

            R = torch.zeros(N, 2, 2, device=self.device)
            R[:, 0, 0] = self.sensor_r_sd**2
            R[:, 1, 1] = self.sensor_b_sd**2
            S = Pzz + R
            K = torch.bmm(Pxz, torch.linalg.inv(S))

            innov = z_t - z_pred
            innov[:, 1] = torch.atan2(torch.sin(innov[:, 1]), torch.cos(innov[:, 1]))
            self.state = self.state + torch.bmm(K, innov.unsqueeze(-1)).squeeze(-1)
            # Standard UKF covariance update: P <- P - K S K^T
            self.cov = self.cov - torch.bmm(torch.bmm(K, S), K.transpose(1, 2))
            self.cov = 0.5 * (self.cov + self.cov.transpose(1, 2))

            full_state[env_ids] = self.state
            full_cov[env_ids] = self.cov
            full_state[env_ids] = torch.maximum(
                torch.minimum(full_state[env_ids], self.limit_hi[env_ids]), self.limit_lo[env_ids]
            )
        finally:
            self.state = full_state
            self.cov = full_cov

    def get_observation(self, target_pos: torch.Tensor, agent_pos: torch.Tensor, agent_yaw: torch.Tensor):
        return BatchKalmanFilter.get_observation(self, target_pos, agent_pos, agent_yaw)
