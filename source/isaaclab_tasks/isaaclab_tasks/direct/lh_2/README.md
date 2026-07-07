# LH-2 水下目标追踪任务

## 📋 任务概述

这是一个基于Isaac Lab的水下机器人目标追踪强化学习任务。机器人使用高保真OceanSim声呐传感器来追踪移动的目标物体。

### 核心特性
- ✅ **高保真声呐**: 基于Oculus M370s/MT370s/MD370s真实参数
- ✅ **GPU加速**: 使用Warp kernels进行声呐图像生成
- ✅ **多环境并行**: 支持4个环境同时训练
- ✅ **OceanSim集成**: 完整的OceanSim扩展声呐支持

## 🚀 快速开始

### 方法1: 使用便捷脚本（推荐）

```bash
# 基础训练（4个环境）
./scripts/train_lh_2.sh

# 自定义训练（8个环境，CUDA设备0）
./scripts/train_lh_2.sh --num_envs 8 --device cuda:0

# 完整训练配置
./scripts/train_lh_2.sh \
  --num_envs 8 \
  --device cuda:0 \
  --max_iterations 50000 \
  --video
```

### 方法2: 手动配置环境变量

```bash
# 设置OceanSim扩展路径
export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser

# 启动训练
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lh-2-Direct-v0 \
  --num_envs 4
```

### 方法3: 通过命令行参数

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lh-2-Direct-v0 \
  --num_envs 4 \
  --kit_args "--ext-folder=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser/isaacsim.oceansim"
```

## 📊 声呐传感器配置

### OceanSim声呐参数

```python
OceanSimSonarSensorCfg(
    min_range=0.5,           # 最小探测距离（米）
    max_range=15.0,           # 最大探测距离（米）
    hori_fov=90.0,            # 水平视场角（度）
    vert_fov=45.0,            # 垂直视场角（度）
    angular_res=0.5,           # 角度分辨率（度）
    hori_res=256,              # 水平分辨率（像素）
    height=128,                # 图像高度（像素）
    width=256,                 # 图像宽度（像素）
    beam_count=64,              # 波束数量
    include_unlabelled=True,     # 包含未标记物体
    binning_method="sum",        # 强度累积方法
    normalizing_method="range",  # 归一化方法
    attenuation=0.1,            # 距离衰减系数
    gau_noise_param=0.2,       # 高斯噪声参数
    ray_noise_param=0.05,      # 瑞利噪声参数
    intensity_offset=0.0,       # 强度偏移
    intensity_gain=1.0,          # 强度增益
)
```

### 观测空间（32,973维）

```python
观测 = {
    'policy': torch.cat([
        sonar_image_flat,         # 32,768维 [4 × 128 × 256]
        sonar_beam_flat,         # 192维 [4 × 64 × 3]
        relative_pos_local,       # 2维 [x, y]
        distance,                # 1维
        heading_error,           # 1维
        robot_vx,               # 1维
        robot_vy,               # 1维
        robot_yaw,              # 1维
        robot_yaw_rate,         # 1维
        ekf_x,                  # 1维
        ekf_y,                  # 1维
        sonar_range_obs,        # 1维
        sonar_angle_obs,        # 1维
        detection_flag,          # 1维
    ], dim=-1)
}
```

## 🎯 奖励函数

```python
奖励 = 距离奖励 + 航向奖励 + 动作平滑奖励 + 碰撞惩罚 + 声呐探测奖励

其中：
- 距离奖励: exp(-distance / 5.0) × 10.0
- 航向奖励: cos(heading_error) × 5.0
- 动作平滑惩罚: sum((action - prev_action)²) × -0.1
- 碰撞惩罚: -10.0 (当distance < 1.0)
- 声呐探测奖励: 1.0 (当探测到目标)
```

## 📁 文件结构

```
lh_2/
├── lh_2_env.py                    # 主环境类
├── lh_2_env_cfg.py                # 环境配置
├── oceansim_sonar_sensor.py       # OceanSim声呐传感器包装
├── oceansim_sonar_sensor_cfg.py   # 声呐传感器配置
└── README.md                       # 本文件
```

## 🔧 故障排除

### 问题1: OceanSim扩展未找到

**错误信息**:
```
ImportError: OceanSim extension not found. Please install OceanSim extension from:
https://github.com/NVIDIA-OceanSim/OceanSim
```

**解决方案**:
确保设置了正确的环境变量：
```bash
export ISAACSIM_EXTENSIONS_FOLDER=/home/lh/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extsUser
```

### 问题2: 声呐传感器初始化失败

**错误信息**:
```
Warning: Failed to create sonar sensor 0: [错误详情]
```

**解决方案**:
1. 检查OceanSim扩展路径是否正确
2. 确认Isaac Sim能够访问扩展目录
3. 查看完整错误信息以诊断问题

### 问题3: CUDA内存不足

**错误信息**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
减少环境数量：
```bash
./scripts/train_lh_2.sh --num_envs 2
```

## 📚 参考资料

- [OceanSim GitHub](https://github.com/NVIDIA-OceanSim/OceanSim)
- [Oculus M370s声呐规格](https://www.marineelectronics.com/)
- [Isaac Lab文档](https://isaac-sim.github.io/IsaacLab/main/)

## 📄 许可证

Copyright (c) 2022-2025, The Isaac Lab Project Developers

SPDX-License-Identifier: BSD-3-Clause
