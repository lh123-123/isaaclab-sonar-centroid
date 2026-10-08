"""RAFT-Small dense optical flow for the AUV tracking tasks.

Single shared estimator: torchvision's pretrained ``raft_small`` (3.8 MB weights,
pure PyTorch, no extra dependency). Input is render-resolution RGB ``(N, H, W, 3)``
uint8 straight off the TiledCamera; output is dense flow ``(N, 2, H, W)`` float32
in render pixels.

Input MUST stay uint8: the torchvision ``Raft_Small_Weights`` preprocessing
normalises for uint8 input, and passing float silently inflates the flow by ~10x
(measured: EPE 5-29 px vs 0.3-2.7 px on the v6 scene).
"""

from __future__ import annotations

import torch

_ESTIMATOR = None


def get_raft_estimator(device: str | torch.device):
    """Lazily build (once per process) and return the RAFT-Small model in eval mode."""
    global _ESTIMATOR
    if _ESTIMATOR is None or str(_ESTIMATOR["device"]) != str(device):
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _ESTIMATOR = {"model": model, "pre": weights.transforms(), "device": torch.device(device)}
    return _ESTIMATOR


@torch.no_grad()
def estimate_flow(prev_rgb: torch.Tensor, curr_rgb: torch.Tensor) -> torch.Tensor:
    """Dense flow from ``prev_rgb`` to ``curr_rgb``.

    Both ``(N, H, W, 3)`` uint8 RGB on the same CUDA device. Returns ``(N, 2, H, W)``
    float32 in pixels of the input resolution. H and W must be divisible by 8.
    """
    assert prev_rgb.dtype == torch.uint8 and curr_rgb.dtype == torch.uint8, (
        f"RAFT input must be uint8, got {prev_rgb.dtype}/{curr_rgb.dtype}"
    )
    est = get_raft_estimator(prev_rgb.device)
    b1, b2 = est["pre"](
        prev_rgb.permute(0, 3, 1, 2).contiguous(),
        curr_rgb.permute(0, 3, 1, 2).contiguous(),
    )
    return est["model"](b1, b2)[-1]
