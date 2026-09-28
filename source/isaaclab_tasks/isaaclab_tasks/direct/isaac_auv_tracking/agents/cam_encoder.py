# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Frame-stack encoder for the camera AUV tracking task (Direct-v4-cam).

The actor's image is a K=4 RGB stack, shape ``(12, 64, 64)``. When the observation
dict also has ``flow_image`` ``(2, 64, 64)``, those two channels are concatenated
on the channel axis before the first convolution, so that layer is ``Conv2d(14, 32, ...)``.
Appearance and the GT motion then share the same filters. The critic never sees this
tensor; it reads the belief state only.

Chosen over ResNet18 on measured grounds: ResNet18 is 11.2 M params / 161 M MACs and its layer4
output on a 64x64 input is only 2x2, so almost all spatial information is pooled away. This CNN
keeps 8x8 at its last conv stage for ~0.3 M params / ~30 M MACs.

Why the last stage stays 8x8 -- and why the head is not global average pooling
------------------------------------------------------------------------------
Under Direct-v4-cam's asymmetric split the actor sees ``observations["image"]`` and, when
present, ``observations["flow_image"]``. The belief state goes to the critic alone, so the
actor has to read the target's bearing -- "is it left or right?" -- out of the pixels and
the flow: that is its input for choosing which way to yaw.

An earlier revision ended the CNN with ``nn.AdaptiveAvgPool2d(1)``, which silently destroys that
bearing. Global average pooling is translation-*invariant*: sliding the target across the frame
permutes which cells of the 8x8 map fire but leaves their mean unchanged. Measured on a synthetic
map, sweeping a blob from column 0 to column 7 moved the pooled output by a relative L2 of
**0.0000** -- columns 0.5 and 6.5 gave bit-identical output, i.e. the encoder was exactly
left-right mirror-symmetric. With the real conv stack, two different bearings gave cosine
similarity 1.000000 to six decimals. No later layer can undo an average over positions, so this
is unrecoverable downstream, and the 8x8 map was being computed and then thrown away.

The head is therefore a **spatial softmax** (soft-argmax), which is translation-*equivariant* --
move the evidence one cell and the predicted coordinate moves with it -- and adds no weights.
``MidCNNExtractor`` emits ``concat(GAP, spatial-softmax coords)`` so the actor gets both halves:
    * GAP (C)         -> "is anything there, and how big" (appearance / apparent range)
    * soft-argmax (2C) -> "where it is" (bearing), an expected (x, y) per channel
On the same synthetic sweep the relative span is 1.43, versus 0.0000 for GAP alone.

This is a deliberate *departure* from HoloOcean v1, which has the identical defect: its
``EncoderResNet`` does ``nn.Sequential(*list(resnet.children())[:-1])``, and ResNet50's
``children()[:-1]`` includes the final ``avgpool``, so v1 also global-average-pools its map.

The actor input is the RGB stack, not a luma collapse. An earlier measurement on 20
frames found almost no chroma (89.1% of pixels had ``R == G == B``), but the target is
now orange-tinted, and the stack is kept in colour so that tint stays in the pixels.
The two flow channels are not a separate head: they are extra inputs to this same
first convolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from isaaclab_tasks.direct.isaac_auv_tracking.sparse_flow import SPARSE_FLOW_DIM, sparse_flow_vector


class SpatialSoftmax(nn.Module):
    """``(B, C, h, w)`` -> ``(B, 2C)``: per-channel expected ``(x, y)`` over the grid, in [-1, 1].

    A soft-argmax over the whole ``h * w`` grid (not per row/column), so each channel reports one
    2-D position. Translation-*equivariant*: shifting the evidence by one cell shifts the expected
    coordinate by the same amount, which is exactly what makes bearing readable.

    ``AdaptiveAvgPool2d`` is the opposite -- translation-*invariant* -- which is right for image
    classification and wrong for localisation; see the module docstring for the measurement.

    Degenerate case to keep in mind: a 1x1 input map makes the softmax trivially 1 and the
    coordinates 0, so the branch contributes nothing and the head collapses back to GAP. The conv
    stack above yields 8x8 for a 64x64 input, so this does not apply here.
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        # persistent=False: deterministic constants, not learned state, so they stay out of the
        # state_dict while still following .to(device).
        self.register_buffer("xs", torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width), persistent=False)
        self.register_buffer("ys", torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1), persistent=False)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feature_map.shape
        prob = feature_map.reshape(b, c, h * w).softmax(dim=-1).reshape(b, c, h, w)
        return torch.cat([(prob * self.xs).sum(dim=(2, 3)), (prob * self.ys).sum(dim=(2, 3))], dim=1)


class MidCNNExtractor(BaseFeaturesExtractor):
    """RGB stack ``(K*3, H, W)`` plus optional ``flow_image`` -> ``features_dim``.

    Accepts either a bare ``Box`` image space or the asymmetric ``Dict`` space
    (``{"image", "flow_image", "state"}``). ``flow_image`` is concatenated with the
    stack before the first convolution. The state is consumed directly by the critic.

    Input contract: SB3's ``preprocess_obs`` divides *any* uint8 image-like Box by 255
    before it reaches a features extractor. ``is_image_space`` gates that on dtype, bounds
    and rank only -- its ``check_channels`` argument defaults to False on this path -- so
    ``(K, H, W)`` is treated as an image even when K is not 1, 3 or 4. The tensor normally
    therefore arrives as float32 in [0, 1], and dividing again would silently squash it to
    [0, 0.004]. The rescale below is guarded accordingly (the same pattern
    ``sonar_encoder.SonarEncoder`` uses) so the input is only divided when it is still in
    [0, 255].

    Keeping the observation uint8 is what makes the replay buffer 1 byte/pixel instead of 4.

    No ImageNet mean/std normalisation: there is no pretraining to match, and the input is a stack
    of frames rather than a natural image. (HoloOcean v1 does apply it, but only because it loads
    ImageNet weights and then overwrites them with ``utils.weight_init`` orthogonal init -- so its
    normalisation matches its BatchNorm statistics but not its conv weights. Skipping it keeps
    this encoder self-consistent; the BatchNorm after the first conv learns the input scale.)
    """

    def __init__(
        self,
        observation_space: spaces.Box | spaces.Dict,
        features_dim: int = 256,
        use_sparse_flow: bool = False,
    ):
        # The actor's space is a Dict under the asymmetric split (``"image"`` + ``"state"``), the
        # bare Box when the critic is not reading a state vector. Only the image branch is encoded
        # here -- the state goes straight into the critic MLP, matching HoloOcean v1, whose Encoder
        # reads ``observations['image']`` alone.
        image_space = (
            observation_space["image"] if isinstance(observation_space, spaces.Dict) else observation_space
        )
        super().__init__(observation_space, features_dim=features_dim)

        n_in = int(image_space.shape[0])  # K grayscale frames, or K*3 for an RGB stack
        # GT flow is two extra channels (dx, dy) concatenated with the frame stack
        # before the first convolution, so appearance and motion share the same filters.
        self._flow_image_channels = 0
        if isinstance(observation_space, spaces.Dict) and "flow_image" in observation_space.spaces:
            self._flow_image_channels = int(observation_space["flow_image"].shape[0])
            n_in += self._flow_image_channels
        self.use_sparse_flow = bool(use_sparse_flow)
        teach_dim = 0
        if isinstance(observation_space, spaces.Dict) and "teach" in observation_space.spaces:
            teach_dim = int(observation_space["teach"].shape[0])
        self._teach_dim = teach_dim
        in_h, in_w = int(image_space.shape[1]), int(image_space.shape[2])
        n_chan = 128

        # NOTE: no pooling or flattening at the end -- the map is kept as 8x8 on purpose, because
        # both heads below need the spatial layout. See "Why the last stage stays 8x8".
        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # 32 -> 16
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # 16 -> 16
            nn.Conv2d(64, n_chan, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_chan),
            nn.ReLU(inplace=True),  # 16 -> 8
            nn.Conv2d(n_chan, n_chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_chan),
            nn.ReLU(inplace=True),  # 8 -> 8
        )

        # Spatial size of the map, so SpatialSoftmax knows its grid. eval() + no_grad keep this
        # one-sample dry run from touching the BatchNorm running statistics.
        was_training = self.training
        self.eval()
        with torch.no_grad():
            map_h, map_w = self.cnn(torch.zeros(1, n_in, in_h, in_w)).shape[-2:]
        self.train(was_training)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.spatial_softmax = SpatialSoftmax(int(map_h), int(map_w))
        # GAP (n_chan) + soft-argmax (2 * n_chan). Flow and teach are off unless v4 asks.
        extra = (SPARSE_FLOW_DIM if self.use_sparse_flow else 0) + teach_dim
        self.trunk = nn.Sequential(
            nn.Linear(3 * n_chan + extra, int(features_dim)),
            nn.LayerNorm(int(features_dim)),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        images = observations["image"] if isinstance(observations, dict) else observations
        # Same guard as sonar_encoder.SonarEncoder: preprocess_obs has normally already turned
        # the uint8 frames into float32 in [0, 1]; only rescale if the values are still raw.
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.float()
            if float(images.max()) > 1.5:
                images = images / 255.0
        if getattr(self, "_flow_image_channels", 0) and isinstance(observations, dict):
            flow = observations["flow_image"].float()
            images = torch.cat([images, flow], dim=1)
        feature_map = self.cnn(images)
        # "what/how big" alongside "where": the first is translation-invariant by design, the
        # second is what makes the bearing readable.
        pooled = self.gap(feature_map).flatten(1)
        coords = self.spatial_softmax(feature_map)
        parts = [pooled, coords]
        # Old v4 checkpoints were pickled before these flags existed.
        if getattr(self, "use_sparse_flow", False):
            # v4 measures flow on the 256 camera image and stores the 3-vector. A bare image
            # stack still runs Lucas-Kanade on whatever resolution it was given.
            if isinstance(observations, dict) and "flow" in observations:
                parts.append(observations["flow"].float())
            else:
                parts.append(sparse_flow_vector(images))
        if getattr(self, "_teach_dim", 0):
            teach = observations["teach"].float()
            if teach.shape[-1] != self._teach_dim:
                raise ValueError(f"teach dim {teach.shape[-1]} != {self._teach_dim}")
            parts.append(teach)
        return self.trunk(torch.cat(parts, dim=1))
