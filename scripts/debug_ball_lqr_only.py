#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Ball-only LQR motion demo (no AUV).

Uses InteractiveScene + CPU PhysX by default to avoid GPU PhysX CUDA 700
crashes seen with kinematic Sphere on this machine.
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Ball-only LQR waypoint tracking demo.")
parser.add_argument("--steps", type=int, default=60, help="Control steps (dt=0.2 s).")
parser.add_argument("--speed", type=float, default=1.0, help="Cruise speed m/s.")
parser.add_argument("--workspace", type=float, default=8.0, help="Half-size of XY workspace.")
parser.add_argument("--radius", type=float, default=0.25, help="Sphere radius m.")
parser.add_argument("--z", type=float, default=1.0, help="Ball height m.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Default to CPU PhysX — GPU pipeline hits CUDA 700 with this minimal sphere scene.
if args_cli.device in (None, "", "cuda", "cuda:0"):
    # Keep torch LQR on CUDA if available later; PhysX device forced below via SimulationCfg.
    pass
# Force sim device to cpu unless user explicitly passed cpu already or wants to override
# with --device cpu. For smoke stability we always use cpu physics here.
args_cli.device = "cpu"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from isaaclab_tasks.direct.isaac_auv_tracking.batch_lqr import BatchKinematicTargetLQR


@configclass
class BallSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.38, 0.12)),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            # kinematic OK on CPU PhysX; GPU PhysX was crashing with CUDA 700.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


def main() -> None:
    control_dt = 0.2
    physics_dt = 1.0 / 60.0
    decimation = max(1, int(round(control_dt / physics_dt)))

    sim_cfg = sim_utils.SimulationCfg(dt=physics_dt, device="cpu")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[12.0, 12.0, 10.0], target=[0.0, 0.0, args_cli.z])

    scene_cfg = BallSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)
    scene_cfg.ball.spawn.radius = args_cli.radius
    scene_cfg.ball.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.z))
    scene = InteractiveScene(scene_cfg)
    ball = scene["ball"]

    sim.reset()
    scene.reset()

    # LQR can stay on CPU with the scene.
    device = ball.device
    lqr = BatchKinematicTargetLQR(
        num_envs=1,
        device=device,
        dt=control_dt,
        cruise_speed=args_cli.speed,
        workspace_half_xy=(args_cli.workspace, args_cli.workspace),
        waypoint_reach_dist=1.0,
    )

    state = torch.zeros(1, 4, device=device)
    state[0, 2] = args_cli.speed
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    centers = torch.zeros(1, 2, device=device)
    lqr.reset(env_ids, state, centers)

    def _write_ball(s: torch.Tensor) -> None:
        pose = torch.zeros(1, 7, device=device)
        pose[0, 0] = s[0, 0] + scene.env_origins[0, 0]
        pose[0, 1] = s[0, 1] + scene.env_origins[0, 1]
        pose[0, 2] = args_cli.z
        pose[0, 3] = 1.0
        vel = torch.zeros(1, 6, device=device)
        vel[0, 0] = s[0, 2]
        vel[0, 1] = s[0, 3]
        ball.write_root_pose_to_sim(pose)
        ball.write_root_velocity_to_sim(vel)

    _write_ball(state)
    scene.write_data_to_sim()
    print(
        f"[ball-lqr] device={device} start=({state[0,0]:.2f},{state[0,1]:.2f}) "
        f"wp=({lqr.waypoint[0,0]:.2f},{lqr.waypoint[0,1]:.2f}) steps={args_cli.steps}",
        flush=True,
    )

    pos0 = state[0, :2].clone()
    max_steps = max(1, args_cli.steps)
    for step in range(max_steps):
        if not simulation_app.is_running():
            break
        state = lqr.step(state)
        _write_ball(state)
        scene.write_data_to_sim()
        for _ in range(decimation):
            sim.step()
            scene.update(physics_dt)

        if step % 10 == 0 or step == max_steps - 1:
            spd = math.hypot(float(state[0, 2]), float(state[0, 3]))
            dist_wp = math.hypot(
                float(lqr.waypoint[0, 0] - state[0, 0]),
                float(lqr.waypoint[0, 1] - state[0, 1]),
            )
            # Read back from sim
            sim_xy = ball.data.root_pos_w[0, :2]
            sync = math.hypot(float(sim_xy[0] - state[0, 0] - scene.env_origins[0, 0]),
                              float(sim_xy[1] - state[0, 1] - scene.env_origins[0, 1]))
            print(
                f"[ball-lqr] step={step:04d} "
                f"pos=({state[0,0]:.2f},{state[0,1]:.2f}) "
                f"wp=({lqr.waypoint[0,0]:.2f},{lqr.waypoint[0,1]:.2f}) "
                f"dist_wp={dist_wp:.2f} speed={spd:.2f} sync={sync:.5f}",
                flush=True,
            )

    moved = torch.norm(state[0, :2] - pos0).item()
    print(f"[ball-lqr] displacement={moved:.3f} m", flush=True)
    if moved < 0.05:
        raise SystemExit(f"FAIL: ball barely moved ({moved:.4f} m)")
    print("[ball-lqr] PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
