"""Debug v7-flow-hover step 0: print raw image, texture mask, state before termination."""
from __future__ import annotations
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
import torch, numpy as np
from isaaclab_tasks.direct.isaac_auv_tracking.auv_tracking_v7_flow_hover_gsrov_env import (
    AUVTrackingV7FlowHoverGsrovEnv,
    AUVTrackingV7FlowHoverGsrovEnvCfg,
)

cfg = AUVTrackingV7FlowHoverGsrovEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
env = AUVTrackingV7FlowHoverGsrovEnv(cfg)
n = env.num_envs

print(f"\n[cfg] decimation={cfg.decimation}  render_interval={cfg.sim.render_interval}")
print(f"[cfg] cam_width={cfg.cam_width}  cam_height={cfg.cam_height}")
print(f"[cfg] render_width={cfg.render_width}  render_height={cfg.render_height}")
print(f"[cfg] texture_lost_limit_steps={cfg.texture_lost_limit_steps}")
print(f"[cfg] sigma_flow={cfg.sigma_flow}")
print(f"[cfg] min_texture_px={cfg.min_texture_px}")

obs, info = env.reset()
img = obs["policy"]["image"]
st = obs["policy"]["state"]
print(f"\n[obs] after reset  img.shape={tuple(img.shape)}  st.shape={tuple(st.shape)}")
print(f"[obs] img absmax={float(img.abs().max()):.6f}  rew from env.observation...")
# compute reward manually
flow_rms_px = float(st[0, 8]) * cfg.cam_width
print(f"[obs] flow_rms_px={flow_rms_px:.3f}")
print(f"[obs] depth_err={float(st[0,0]):.4f}  att_err={torch.sqrt(st[0,1]**2+st[0,2]**2).item():.4f}")
print(f"[obs] state values: {st[0].tolist()}")

# Step 0 with zero action
act = torch.zeros(n, 4)
obs2, rew, term, trunc, info2 = env.step(act)
img2 = obs2["policy"]["image"]
st2 = obs2["policy"]["state"]
flow_rms_px2 = float(st2[0, 8]) * cfg.cam_width
print(f"\n[step 0] rew={float(rew.mean()):.4f}  flow_rms_px={flow_rms_px2:.3f}")
print(f"[step 0] term={term.tolist()}  trunc={trunc.tolist()}")
print(f"[step 0] img absmax={float(img2.abs().max()):.6f}")
print(f"[step 0] state: {st2[0].tolist()}")

# Check internal buffers
print(f"\n[internal] _texture_count={env._texture_count[0].item():.0f}")
print(f"[internal] _texture_lost_streak={env._texture_lost_streak[0].item():.0f}")
print(f"[internal] _prev_rgb_ok all={env._prev_rgb_ok[0].item()}")

env.close()
print("\n[done]")
raise SystemExit(0)
