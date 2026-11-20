# v4-cam 第三轮训练（0929 Reward 重塑）：训练记录与评估

> 状态：训练中（2026-09-28 22:32 启动，目标 1M 步，8 env / headless / kf=gt）

## 1. 本轮改了什么（相对 A/B 对比的 frozen 800k）

| # | 改动 | 值 | 来源 |
|---|------|----|------|
| 1 | 本体 surge 上限 | ∓1.0 → ∓1.5 m/s | 对齐 HoloOcean v1（用户选 A） |
| 2 | range 权重 + sigma | 0.05→0.10，sigma 0.6→0.45 | P0：B 版距离放到 2.70m 是理性交易，用更强的 range 梯度拉回 r\*=2（err 0.7m 处 0.068→0.24/step，对标 uncertainty 上限 0.339） |
| 3 | bearing 线性尾（新增） | \|α\|>15° 后 0.02/deg | P0：压住 30s 后漂移段，二次型在边缘太平 |
| 4 | yaw 饱和惩罚（新增） | \|a_w\|>0.9 时 0.02/step | P1：AB 证明饱和在 Δa=0.02/0.10 下都是 25%，是主动选择不是抖动，单独立项 |
| 5 | 目标 RRT | 0.5 / 0.6 不动 | 用户选择 |

保留：uncertainty 0.02、clip 0.0（9-28 rebalance 不动）。
推迟到下一轮：方案 A（9 通道 obs，state 降维到 r/α/logdet/observed/lost）——改 obs 形状会废 ckpt 兼容且与 reward 改动混杂，单独一轮做。

旧 ckpt 全废（动力学 + reward 双变），本轮从头训。

## 2. 训练配置

- 脚本：`scripts/watchdog_train_auv_tracking_sb3_sac_v4.sh --num_envs 8 --max_iterations 1000000 env.kf_measurement=gt --headless`
- 单跑 1M（不用 overnight pipeline：它的 phase 3 会按 `recommend_v4_cam_reward.py` 自动覆盖 reward，和手调权重打架）
- 日志：`logs/sb3/Isaac-AUV-Tracking-Direct-v4-cam/2026-09-28_22-19-41/SAC_1`，watchdog 600s stall 熔断
- 注意：replay buffer 6.88GB > 可用 3.45GB 的 warning（懒分配，跑不满则无事；OOM 则 watchdog 重启 + 降 buffer 到 15000）

## 3. 评估（待填：与 A/B 同口径，20 回合）

| 指标 | A (tight Δa=0.02) | B (Δa=0.10) | 本轮 |
|------|-------------------|-------------|------|
| succ / 20 | 20 | 20 | |
| 中位距离均值 | 1.51m | 2.70m | |
| P90 距离 | 2.00m | 3.54m | |
| 中位 \|bearing\| 均值 | 6.08° | 4.58° | |
| P90 bearing | 9.4° | 10.2° | |
| 30s 后 in-FOV | 100% | 71.7% | |
| yaw 饱和占比 | ~25% | ~25% | |
| ep_ret（当前 reward 下） | — | — | |

## 5. 方案 A 已落地（2026-09-29，修正版）：1 帧 RGB + K-1 个 GT flow 叠帧

- obs：`image` = 当帧 RGB（3ch，uint8）；`flow_image` = K-1 个 GT flow 帧叠起来（2*(K-1)=6ch，float32），首层卷积共 9 输入（K=4）
- 保留 GT flow：本轮目的就是验证"GT 光流到底行不行"，去 GT 就测不出这个了；可部署化是下一轮的事（估计 flow 替代 GT）
- reset 时 flow 栈清零（无运动历史），`_gt_px_ok=False` 使首帧位移为零
- encoder/yaml/eval 无需结构改动（`MidCNNExtractor` 自动拼接 `flow_image`）
- **旧 ckpt 全废**（首层卷积 14→9 通道），需从头训

- 距离中位回 2.2m 内 **且** 方位保持 <6° → 锁定权重，训 3M
- 否则 → 下一轮上方案 A（9 通道 obs）
