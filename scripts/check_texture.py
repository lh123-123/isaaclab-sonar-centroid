"""Quick check: verify texture detection works at 25 Hz with plain RGB."""
import torch, torch.nn.functional as F
from pathlib import Path

# Simulate a simple textured image (gradients)
h, w = 64, 64
img = torch.zeros(1, h, w, dtype=torch.float32)
# Add diagonal stripes
for i in range(64):
    img[0, :, i] = float((i * 4) % 256)
img = img.unsqueeze(0).clamp(0, 255).to(torch.uint8)  # (1,1,h,w) -> (1,h,w,1)

def _texture_from_rgb(rgb_nhwc):
    gray = rgb_nhwc.float().mean(dim=-1)
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, 1:-1, 1:-1] = gray[:, 1:-1, 2:] - gray[:, 1:-1, :-2]
    gy[:, 1:-1, 1:-1] = gray[:, 2:, 1:-1] - gray[:, :-2, 1:-1]
    mag = torch.sqrt(gx * gx + gy * gy)
    rh, rw = gray.shape[-2], gray.shape[-1]
    h2, w2 = 64, 64
    threshold = 8.0
    hot = (mag > threshold).float().unsqueeze(1)
    if (rh, rw) != (h2, w2):
        mask = F.adaptive_max_pool2d(hot, (h2, w2))
    else:
        mask = hot
    count = mask.flatten(1).sum(dim=-1)
    return mask, count

mask, count = _texture_from_rgb(img.float().unsqueeze(-1))
print(f"Test texture: mask sum = {count.item():.0f} px (should be > 20)")

# Now test with grayscale (no texture)
gray_img = torch.ones(1, h, w, 3) * 128.0
mask2, count2 = _texture_from_rgb(gray_img.to(torch.uint8).float().unsqueeze(-1))
print(f"Test flat: mask sum = {count2.item():.0f} px (should be 0)")
