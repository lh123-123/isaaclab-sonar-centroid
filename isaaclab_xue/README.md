# Isaac Lab 学习笔记（isaaclab_xue）

本目录存放跟随官方 `scripts/tutorials` 与 `docs/source/tutorials` 整理的学习记录，按教程编号分文件，便于对照源码复习。

| 编号 | 主题 | 笔记文件 |
|------|------|----------|
| 00 | 仿真与应用启动 | [00_sim_仿真与应用启动.md](00_sim_仿真与应用启动.md) |
| 01 | 资产与交互 | [01_assets_资产与交互.md](01_assets_资产与交互.md) |
| 02 | 交互式场景 | [02_scene_交互式场景.md](02_scene_交互式场景.md) |
| 03 | 环境与 RL 入口 | [03_envs_环境与强化学习入口.md](03_envs_环境与强化学习入口.md) |
| 04 | 传感器 | [04_sensors_在机器人上添加传感器.md](04_sensors_在机器人上添加传感器.md) |
| 05 | 控制器（IK / OSC） | [05_controllers_逆运动学与操作空间控制.md](05_controllers_逆运动学与操作空间控制.md) |
| — | **Direct 任务扩展**（`isaaclab_tasks.direct`，不含 `lh_2`） | [isaaclab_tasks_direct_Direct工作流与任务清单.md](isaaclab_tasks_direct_Direct工作流与任务清单.md) |

## 仓库内教程脚本一览

`scripts/tutorials` 下 Python 文件均已纳入上表对应章节（04 含 `add_sensors_on_robot` 与同目录四个 `run_*` 脚本）。

## 延伸阅读

更完整的 RL 注册、Direct/Managed 对比、训练配置等见在线文档 `docs/source/tutorials/03_envs/` 中未单独对应脚本的 RST；需要时可从笔记中的任务包与模块名继续深挖。
