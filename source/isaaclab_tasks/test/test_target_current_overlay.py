# CPU-only test for current overlay on kinematic target motion.
import importlib.util
from pathlib import Path

import numpy as np
import torch

_BATCH_LQR_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_tasks"
    / "direct"
    / "isaac_auv_tracking"
    / "batch_lqr.py"
)
_spec = importlib.util.spec_from_file_location("batch_lqr", _BATCH_LQR_PATH)
assert _spec and _spec.loader
_batch_lqr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_batch_lqr)
BatchHoloTargetMotion = _batch_lqr.BatchHoloTargetMotion


def test_current_overlay_shifts_xy_by_v_dt():
    device = torch.device("cpu")
    motion = BatchHoloTargetMotion(
        num_envs=1,
        device=device,
        cruise_speed=0.5,
        soft_start_s=0.0,
        easy_s=0.0,
        workspace_half_xy=(20.0, 20.0),
        workspace_margin=1.0,
    )
    env_ids = torch.tensor([0], dtype=torch.long)
    init = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    centers = torch.zeros(1, 2)
    dt = 0.1
    state0 = init.clone()
    np.random.seed(0)
    torch.manual_seed(0)
    motion.reset(env_ids, init, centers, agent_xy=init[:, :2])
    no_cur = motion.step(state0.clone(), dt=dt)
    np.random.seed(0)
    torch.manual_seed(0)
    motion.reset(env_ids, init, centers, agent_xy=init[:, :2])
    cur = torch.tensor([[1.0, 0.0]])
    with_cur = motion.step(state0.clone(), dt=dt, current_vel_xy=cur)
    delta = with_cur[0, :2] - no_cur[0, :2]
    assert torch.allclose(delta, torch.tensor([dt, 0.0]), atol=1e-5)
