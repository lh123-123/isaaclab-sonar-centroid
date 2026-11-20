# v7 单一入口

> 整理日期：2026-11-17
> **以后 v7 相关的东西只看这一个目录。** 包内代码（环境本体、agent 配置、任务注册）位置固定、不能动，下面单独列出。

---

## 目录结构

```
IsaacLab/v7/
├── README.md            ← 你在这里（总索引）
├── docs/                ← 所有文档（8 个）
│   ├── README.md                     文档汇总 + 阅读顺序
│   ├── canvas_v7_延迟预算_1117.md     ★ 延迟预算与优化清单（当前结论）
│   ├── 方案设计_v7_光流RGB视觉悬停.md  v7 立项原始设计
│   ├── 方案对比_v7_悬停.md             候选方案横向对比
│   ├── 数据需求_视觉悬停DRL.md         训练数据需求
│   └── *.html                          可视化看板（3 个）
├── scripts/             ← 42 个 v7 脚本，按用途分类
│   ├── debug/           28 个  诊断/探针
│   ├── eval/             3 个  检查点评估
│   ├── play/             3 个  交互回放（UI）
│   ├── bench/            4 个  基准/标定/性能
│   ├── train/            2 个  训练启动与队列
│   └── misc/             2 个  批次脚本、运行检查
└── logs/                ← 49 个实验日志（原在仓库根目录，散落的 .log）
```

---

## 包内代码（**位置固定，不可移动**）

这些文件被 Python import / 任务注册引用，动了就找不到任务：

| 文件 | 作用 |
|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/auv_tracking_v7_flow_hover_gsrov_env.py` | **v7 环境本体**（约 1945 行），观测/奖励/DR/延迟全在这 |
| `source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/template_anchor.py` | 首帧模板匹配算法 |
| `source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/agents/sb3_sac_v7_flow_hover_cfg.yaml` | SAC 超参与网络结构 |
| `source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/__init__.py` | 任务注册（`Isaac-AUV-Tracking-Direct-v7-flow-hover`） |
| `scripts/reinforcement_learning/sb3/train.py` | 训练入口（含 `--action_delay_fixed` / `--action_delay_max`） |

---

## 常用命令（路径已更新）

```bash
# 训练（固定 200 ms 死区）
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
    --task Isaac-AUV-Tracking-Direct-v7-flow-hover --agent sb3_cfg_entry_point \
    --algorithm SAC --flow_method farneback --num_envs 8 --headless \
    --max_iterations 125000 --action_delay_fixed 3

# 评估某检查点（含延迟扫描）
./isaaclab.sh -p v7/scripts/eval/eval_v7_checkpoint.py ...

# 经典基线（P / PID / LQR，支持洋流扫描 / 突变 / 线索丢失）
./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_baseline.py ...

# 偏航链路诊断
./isaaclab.sh -p v7/scripts/debug/debug_auv_v7_yaw.py ...

# 教程式回放 UI
./isaaclab.sh -p v7/scripts/play/play_auv_v7_flow_hover_ui.py ...
```

> 脚本内部的路径引用已全部改写为 `v7/scripts/<类别>/<文件名>`。
> 历史文档里出现的 `scripts/debug_auv_v7_xxx.py` 旧写法均已失效，按上表映射即可。

---

## 训练产物在哪

| 位置 | 内容 |
|---|---|
| `~/IsaacLab/logs/sb3/Isaac-AUV-Tracking-Direct-v7-flow-hover/<时间戳>/` | 32 个 run 目录，含 `model_*.zip` 检查点与 TensorBoard |
| `lab:~/auv_v7_ablate/logs/` | 19 个训练日志（服务器端） |
| `lab:~/auv_v7_ablate/backups/` | 2 个 run 备份（twoscale / delay_dr，179 MB） |
| `lab:~/auv_v7_ablate/sweeps/` | 扫描结果 json |

**关键检查点：**

| 臂 | 检查点 | 性能 |
|---|---|---|
| B（基线，无延迟训练） | `2026-11-15_21-25-34/model_960000_steps.zip` | 30.8 mm |
| C（双尺度） | `backups/twoscale_run_backup/model_200000_steps.zip` | 21.8 mm（该臂最好阶段） |
| E（固定 200 ms） | `2026-11-16_23-57-37/model_650000_steps.zip`（最好在 200k） | 200 ms 下 36.1 mm |

---

## ⚠️ 本次整理中的损失（需要你决定怎么补）

整理时我把三份文档误删了，**且无法恢复**（不在 git HEAD、不在不可达 blob、不在 IDE 本地历史、全盘无副本）：

| 丢失文档 | 大小 | 内容 |
|---|---|---|
| `canvas_v7_系统全貌.md` | 21,021 字节 | **主文档**：任务定义、11 通道观测 + 17 维 state 逐项、动作→推力链、三个视觉线索的完整数学、奖励权重、终止条件、DR 两种抽样周期、日志可信度说明、风险清单 |
| `canvas_v7_ab_reward_1116.md` | 14,050 字节 | 奖励锐度 A/B 实验记录、预注册判据、监控与评估协议 |
| `canvas_v7_unified_1115_baseline.md` | 9,251 字节 | 服务器统一版重训记录、`disp_end_m` 污染修复、三张基线实验表 |

原因：`v7_docs/` 里原本是指向仓库根目录的**符号链接**，根目录那三份才是源文件。我把根目录的"重复文件"当成多余删掉了，实际删的是唯一副本。它们在 git 里处于 staged 未提交状态，所以 `git rm` 后从 HEAD 也恢复不了。

**补救选项：**

1. 我从本会话已建立的上下文**重写一份精简版**（内容准确度有限，长期细节会缺）
2. 你从别处（另一台机器 / 备份 / 聊天记录）找回原文件
3. 就按 `docs/README.md` 里的摘要 + 现在的实验记录，往后重新积累

---

## 维护约定

- 新增文档 → 放进 `v7/docs/`，并在 `v7/docs/README.md` 清单补一行
- 新增脚本 → 按用途放进 `v7/scripts/<类别>/`
- **不要再往仓库根目录放 v7 文件**
- 不要再使用符号链接做"文档汇总"——本次的丢失就是这么造成的，请放真实文件
