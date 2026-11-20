# ROV 定深修改前备份（2026-10-13）

这是修改前的实机版本备份，**不是新固件**。备份时飞控已上锁，推进器和舵机动力断开，手柄控制停止。只读读取 1029/1029 项飞控参数；未写参数或烧录。

## 内容

- `firmware/ardusub.apj`：树莓派上最近一次 `FLASH_SUCCESS` 对应的 Pixhawk1 固件产物。SHA-256：`01f2eb1e16b7b68ecc35e2b357e4215a95f07028a0cdda072ac07b414db45afd`。这是已留存的烧录文件及记录，不是从飞控闪存直接读出的镜像。
- `firmware/flash_status.json`：该固件的上传成功记录。备份时该产物哈希与记录一致。
- `firmware/source/ArduSub/` 和 `firmware/source/libraries/AP_Motors/`：树莓派上该构建的定制源码。与本地发布包 `prepared_depth_bottom_guard_20261008/manifest.json` 中的 22 项源码哈希全部一致。
- `firmware/vehicle_parameters.json`：飞控本次读回的身份和全部 1029 项参数。SHA-256：`3a6d45239463ee3938492d91be40d73a5de2189ce32ad9c753aa14fb27cbf27c`。
- `firmware/vehicle_parameters.params`：同一批参数的 QGC 可读格式。SHA-256：`a9e195288239d389e3d98da2728ba6fe2ce6e4323a39cc1792691206e31f1fa5`。另存副本于 `C:\Users\15057\Documents\QGroundControl\Parameters\ROV_当前版本备份_20261013_171011.params`。
- `pi_deploy/`：备份时树莓派 `/home/liang/underwater_robot/` 的部署文件。
- `pc_control/`：备份时电脑端控制脚本、测试和启动脚本。

关键实机参数读回：`MOT_DH_MAX=25`、`MOT_BAT_CURR_MAX=20`、`SURFACE_MAX_THR=1`、`THR_DZ=100`、`PILOT_ACCEL_Z=100`、`ROV_EN=1`。树莓派时钟与电脑日期不一致，以本文件夹名的电脑日期作为本次备份标记。

## 回滚边界

若新固件水中表现不佳，先停机上锁、断开推进器和舵机动力、停止手柄控制；然后按飞控安全烧录流程重新上传本目录的 `ardusub.apj`，校验 CRC、飞控身份及参数读回。**不要直接整包导入旧参数**：先比较新旧参数，只恢复确实被修改的项；当前这次代码改动计划保留实机参数原值。若电脑或树莓派部署脚本发生变化，可用 `pc_control/` 和 `pi_deploy/` 逐文件比对恢复。任何重新烧录或水中动作都需要现场确认安全状态。

本备份不等于水下闭环验证，也无法单靠 `FLASH_SUCCESS` 证明飞控闪存中每个字节与备份文件相同；可确认的是最新保留的上传成功记录、产物哈希、飞控版本身份和完整参数读回。
