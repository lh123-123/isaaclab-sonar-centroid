# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from typing import Tuple, Optional


class SonarProcessor:
    """
    声呐数据处理类
    包含阈值分割、DBSCAN聚类、目标提取功能
    """
    
    def __init__(
        self,
        image_height: int = 128,
        image_width: int = 256,
        threshold: float = 0.3,
        min_samples: int = 3,
        eps: float = 2.0,
        device: str = "cuda:0"
    ):
        """
        Args:
            image_height: 声呐图像高度
            image_width: 声呐图像宽度
            threshold: 强度阈值 (0-1)
            min_samples: DBSCAN最小样本数
            eps: DBSCAN邻域半径
            device: 计算设备
        """
        self.image_height = image_height
        self.image_width = image_width
        self.threshold = threshold
        self.min_samples = min_samples
        self.eps = eps
        self.device = device
        
    def threshold_segmentation(self, sonar_image: torch.Tensor) -> torch.Tensor:
        """
        阈值分割，过滤低强度噪声
        
        Args:
            sonar_image: 声呐图像 (B, H, W) 或 (H, W)
            
        Returns:
            binary_mask: 二值掩码
        """
        if sonar_image.dim() == 2:
            sonar_image = sonar_image.unsqueeze(0)
            
        binary_mask = (sonar_image > self.threshold).float()
        return binary_mask
    
    def extract_target_polar(
        self, 
        sonar_image: torch.Tensor,
        max_range: float = 15.0,
        hori_fov: float = 90.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从声呐图像提取目标极坐标
        
        Args:
            sonar_image: 声呐图像 (H, W)
            max_range: 最大探测距离 (m)
            hori_fov: 水平视场角 (deg)
            
        Returns:
            distance: 目标距离
            angle: 目标角度
            intensity: 目标强度
        """
        binary_mask = self.threshold_segmentation(sonar_image)
        
        nonzero_indices = torch.nonzero(binary_mask.squeeze(0), as_tuple=False)
        
        if nonzero_indices.shape[0] == 0:
            return (
                torch.tensor(max_range * 2, device=self.device),
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device)
            )
        
        intensities = sonar_image[nonzero_indices[:, 0], nonzero_indices[:, 1]]
        
        center_y = nonzero_indices[:, 0].float().mean()
        center_x = nonzero_indices[:, 1].float().mean()
        
        range_per_pixel = max_range / self.image_height
        angle_per_pixel = hori_fov / self.image_width
        
        distance = (self.image_height - center_y) * range_per_pixel
        angle = (center_x - self.image_width / 2) * angle_per_pixel
        angle = torch.deg2rad(angle)
        
        intensity = intensities.mean()
        
        return distance, angle, intensity
    
    def process_sonar_batch(
        self,
        sonar_images: torch.Tensor,
        max_range: float = 15.0,
        hori_fov: float = 90.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量处理声呐图像
        
        Args:
            sonar_images: 声呐图像批次 (B, H, W)
            max_range: 最大探测距离
            hori_fov: 水平视场角
            
        Returns:
            distances: 距离张量 (B,)
            angles: 角度张量 (B,)
            intensities: 强度张量 (B,)
        """
        batch_size = sonar_images.shape[0]
        distances = torch.zeros(batch_size, device=self.device)
        angles = torch.zeros(batch_size, device=self.device)
        intensities = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            d, a, intensity = self.extract_target_polar(
                sonar_images[i], max_range, hori_fov
            )
            distances[i] = d
            angles[i] = a
            intensities[i] = intensity
            
        return distances, angles, intensities


class EKFSonarTracker:
    """
    2D扩展卡尔曼滤波器
    用于声呐目标跟踪
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        process_noise: float = 0.1,
        measurement_noise: float = 0.2,
        device: str = "cuda:0"
    ):
        """
        Args:
            state_dim: 状态维度 [x, y, vx, vy]
            process_noise: 过程噪声
            measurement_noise: 测量噪声
            device: 计算设备
        """
        self.state_dim = state_dim
        self.device = device
        
        self.state = torch.zeros(state_dim, device=device)
        self.covariance = torch.eye(state_dim, device=device) * 1.0
        
        self.Q = torch.eye(state_dim, device=device) * process_noise
        self.R = torch.eye(2, device=device) * measurement_noise
        
        self.initialized = False
        
    def initialize(self, x: float | torch.Tensor, y: float | torch.Tensor):
        """初始化滤波器状态"""
        if isinstance(x, torch.Tensor):
            x = x.item()
        if isinstance(y, torch.Tensor):
            y = y.item()
        self.state = torch.tensor([x, y, 0.0, 0.0], device=self.device)
        self.covariance = torch.eye(self.state_dim, device=self.device) * 1.0
        self.initialized = True
        
    def predict(self, dt: float):
        """
        预测步骤 (CV匀速模型)
        
        Args:
            dt: 时间步长
        """
        F = torch.eye(self.state_dim, device=self.device)
        F[0, 2] = dt
        F[1, 3] = dt
        
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.Q
        
    def update(self, z_range: float | torch.Tensor, z_angle: float | torch.Tensor) -> torch.Tensor:
        """
        更新步骤 (极坐标观测)
        
        Args:
            z_range: 测量距离
            z_angle: 测量角度
            
        Returns:
            更新后的状态
        """
        if not self.initialized:
            x = float(z_range) * np.cos(float(z_angle))
            y = float(z_range) * np.sin(float(z_angle))
            self.initialize(x, y)
            return self.state
            
        x, y = self.state[0], self.state[1]
        
        pred_range = torch.sqrt(x**2 + y**2)
        pred_angle = torch.atan2(y, x)
        
        z = torch.tensor([z_range, z_angle], device=self.device)
        z_pred = torch.tensor([pred_range, pred_angle], device=self.device)
        
        H = torch.zeros((2, self.state_dim), device=self.device)
        r = pred_range + 1e-6
        H[0, 0] = x / r
        H[0, 1] = y / r
        H[1, 0] = -y / (r**2)
        H[1, 1] = x / (r**2)
        
        S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ torch.inverse(S)
        
        innovation = z - z_pred
        if innovation[1] > torch.pi:
            innovation[1] -= 2 * torch.pi
        elif innovation[1] < -torch.pi:
            innovation[1] += 2 * torch.pi
            
        self.state = self.state + K @ innovation
        self.covariance = (torch.eye(self.state_dim, device=self.device) - K @ H) @ self.covariance
        
        return self.state
    
    def get_state(self) -> torch.Tensor:
        """获取当前状态"""
        return self.state.clone()
    
    def get_position(self) -> Tuple[float, float]:
        """获取当前位置估计"""
        return self.state[0].item(), self.state[1].item()
    
    def get_velocity(self) -> Tuple[float, float]:
        """获取当前速度估计"""
        return self.state[2].item(), self.state[3].item()


class EKFBatchTracker:
    """
    批量EKF跟踪器
    用于多环境并行处理
    """
    
    def __init__(
        self,
        num_envs: int,
        state_dim: int = 4,
        process_noise: float = 0.1,
        measurement_noise: float = 0.2,
        device: str = "cuda:0"
    ):
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.device = device
        
        self.states = torch.zeros(num_envs, state_dim, device=device)
        self.covariances = torch.eye(state_dim, device=device).unsqueeze(0).repeat(num_envs, 1, 1)
        
        self.Q = torch.eye(state_dim, device=device) * process_noise
        self.R = torch.eye(2, device=device) * measurement_noise
        
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
    def initialize_batch(self, positions: torch.Tensor):
        """
        批量初始化
        
        Args:
            positions: 初始位置 (N, 2)
        """
        self.states[:, :2] = positions
        self.states[:, 2:] = 0.0
        self.covariances = torch.eye(self.state_dim, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.initialized[:] = True
        
    def predict_batch(self, dt: float):
        """批量预测"""
        F = torch.eye(self.state_dim, device=self.device)
        F[0, 2] = dt
        F[1, 3] = dt
        
        self.states = (F @ self.states.T).T
        self.covariances = F @ self.covariances @ F.T + self.Q
        
    def update_batch(
        self, 
        ranges: torch.Tensor, 
        angles: torch.Tensor
    ) -> torch.Tensor:
        """
        批量更新
        
        Args:
            ranges: 测量距离 (N,)
            angles: 测量角度 (N,)
            
        Returns:
            更新后的状态 (N, 4)
        """
        x = self.states[:, 0]
        y = self.states[:, 1]
        
        pred_range = torch.sqrt(x**2 + y**2 + 1e-6)
        pred_angle = torch.atan2(y, x)
        
        z = torch.stack([ranges, angles], dim=1)
        z_pred = torch.stack([pred_range, pred_angle], dim=1)
        
        r = pred_range + 1e-6
        H = torch.zeros(self.num_envs, 2, self.state_dim, device=self.device)
        H[:, 0, 0] = x / r
        H[:, 0, 1] = y / r
        H[:, 1, 0] = -y / (r**2)
        H[:, 1, 1] = x / (r**2)
        
        S = torch.bmm(H, torch.bmm(self.covariances, H.transpose(1, 2))) + self.R
        K = torch.bmm(self.covariances, torch.bmm(H.transpose(1, 2), torch.inverse(S)))
        
        innovation = z - z_pred
        innovation[:, 1] = torch.where(
            innovation[:, 1] > torch.pi,
            innovation[:, 1] - 2 * torch.pi,
            innovation[:, 1]
        )
        innovation[:, 1] = torch.where(
            innovation[:, 1] < -torch.pi,
            innovation[:, 1] + 2 * torch.pi,
            innovation[:, 1]
        )
        
        self.states = self.states + torch.bmm(K, innovation.unsqueeze(2)).squeeze(2)
        
        I_KH = torch.eye(self.state_dim, device=self.device) - torch.bmm(K, H)
        self.covariances = torch.bmm(I_KH, self.covariances)
        
        return self.states
    
    def reset_idx(self, env_ids: torch.Tensor, positions: torch.Tensor):
        """重置指定环境的状态"""
        self.states[env_ids, :2] = positions
        self.states[env_ids, 2:] = 0.0
        self.covariances[env_ids] = torch.eye(self.state_dim, device=self.device)
        self.initialized[env_ids] = True
