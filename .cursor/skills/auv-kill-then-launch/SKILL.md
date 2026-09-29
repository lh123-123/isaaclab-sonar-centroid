---
name: auv-kill-then-launch
description: Kills leftover Isaac processes then starts exactly one train, eval, or play job. Use when the user types /auv-kill-then-launch, 先 kill, kill, 开训, or watchdog.
disable-model-invocation: true
---

# Kill then launch one Isaac

同一时刻只允许一个 Isaac Lab / Isaac Sim。GPU 显存和 kit 文件锁撑不住第二个。

硬约束也在 `.cursor/rules/single-isaac-terminal-process.mdc`。本 skill 是操作步骤。

## Do not

- Start a second `isaaclab.sh` / `train.py` / `play_auv` / eval while one is alive.
- `pkill` in the **same** shell command as `nohup` 启动 — 会把刚拉起的 watchdog 一起杀掉。
- Use skrl for this task. Direct-v1 训练走 SB3 脚本 + watchdog。
- Launch Cloud Agent / extra worktree jobs that would start Sim on this GPU.

## Kill

User said 先 kill / kill:

```bash
./scripts/kill_isaac_lab_processes.sh
pkill -f watchdog_train_auv_tracking || true
sleep 2
pgrep -af 'isaaclab|train.py --task Isaac-AUV|play_auv|watchdog_train_auv' | grep -vE 'pgrep|cursor' || echo clean
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

必须看到 `clean`（或只剩无关进程）再启动。

## Launch (exactly one)

Pick the script the user named. Examples:

**Direct-v1 长训（watchdog，8 env，headless）**

```bash
export TERM=xterm-256color
export TRAIN_SCRIPT=./scripts/<train_script>.sh
export RUN_LOG=/tmp/<run>.log
: > "$RUN_LOG"
: > /tmp/<run>_watchdog.log
nohup ./scripts/watchdog_train_auv_tracking_sb3_sac_v1.sh --num_envs 8 --headless \
  >> /tmp/<run>_watchdog.log 2>&1 &
echo "watchdog_pid=$!"
```

Known train scripts:

| 名 | 脚本 |
|---|---|
| A1 | `train_auv_tracking_sb3_sac_v1_asym_cnn_a1.sh` |
| G0 | `train_auv_tracking_sb3_sac_v1_gated_cnn_g0.sh` |
| G1 | `train_auv_tracking_sb3_sac_v1_gated_cnn.sh` |
| ResNet v1 | `train_auv_tracking_sb3_sac_v1.sh`（不要误用在 compact-CNN 上） |

**20 局评测** → 用 skill `auv-20ep-eval`，不要自己拼 `play.py`。

启动后确认：`pgrep` 里只有这一组 watchdog + `train.py` / `play`；`nvidia-smi` 显存在涨。

## If the user did not say kill

Occupied → tell them what is running and wait. Do not kill a live A1/G0 train to start eval unless they asked.
