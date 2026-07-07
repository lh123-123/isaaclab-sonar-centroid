# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Teach-time reference loading, extraction, and snapshot capture."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..reference_snapshot import ReferenceSnapshot
from .geometry import beacon_bearing_range, beacon_image_rois, beacon_polar_rois, sonar_pos_w, spawn_yaw

if TYPE_CHECKING:
    from .sonar_pipeline import SonarPerceptionPipeline
    from ..sonar_ibvs_drl_env_cfg import SonarIbvsDrlEnvCfg


class ReferenceManager:
    """Manage taught sparse features (file / capture / geometry modes)."""

    def __init__(self, cfg: SonarIbvsDrlEnvCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.mode = str(getattr(cfg, "reference_mode", "file")).lower()
        self.use_file = self.mode == "file"
        self.perception_frame = str(getattr(cfg, "perception_frame", "cart")).lower()

        ocfg = cfg.oceansim_sonar_cfg
        self._ref_sonar_image = torch.zeros(ocfg.height, ocfg.width, device=device)
        self._file_ref_yaw = torch.zeros(1, device=device)
        self.file_ref_yaw_valid = False

    def _cfg_frame(self) -> str:
        return str(getattr(self.cfg, "perception_frame", "cart")).lower()

    def _snapshot_compatible(self, snapshot: ReferenceSnapshot) -> bool:
        snap_frame = str(getattr(snapshot, "perception_frame", "cart") or "cart").lower()
        return snap_frame == self._cfg_frame() and snapshot.track_image is not None

    def _set_reference_env(
        self,
        pipeline: SonarPerceptionPipeline,
        env_id: int,
        ref_points,
        track_image,
    ) -> None:
        pipeline.feature_tracker.set_reference(env_id, ref_points, ref_cart=track_image)

    def load_from_disk(self, pipeline: SonarPerceptionPipeline) -> None:
        path = str(getattr(self.cfg, "reference_snapshot_path", "logs/station_ref/station_ref.npz"))
        snapshot = ReferenceSnapshot.load(path)
        tensors = snapshot.to_tensors(self.device)
        self._ref_sonar_image[:] = tensors["sonar_image"]
        if snapshot.ref_yaw is not None:
            self._file_ref_yaw[0] = float(snapshot.ref_yaw)
            self.file_ref_yaw_valid = True
        ref_pts = snapshot.ref_points
        if self._snapshot_compatible(snapshot):
            track = snapshot.track_image
            for env_id in range(self.num_envs):
                self._set_reference_env(pipeline, env_id, ref_pts, track)
            frame = self._cfg_frame()
            print(
                f"[SonarIbvsDrl] loaded reference: {path} frame={frame} "
                f"n_points={len(ref_pts)} "
                f"ref_yaw={'n/a' if not self.file_ref_yaw_valid else f'{float(self._file_ref_yaw[0].item()):+.4f}rad'}",
                flush=True,
            )
            return
        print(
            f"[SonarIbvsDrl] WARN: snapshot frame={getattr(snapshot, 'perception_frame', 'cart')} "
            f"!= cfg frame={self._cfg_frame()}; will re-extract features on reset.",
            flush=True,
        )

    def apply_on_reset(
        self,
        env_ids: torch.Tensor,
        pipeline: SonarPerceptionPipeline,
        *,
        robot,
        beacon_pos_w: torch.Tensor,
        sonar_mount_offset_b: torch.Tensor,
        max_range: float,
    ) -> None:
        if self.mode in ("capture", "geometry"):
            self.extract_for_envs(env_ids, pipeline, robot, beacon_pos_w, sonar_mount_offset_b, max_range)
            return
        if self.mode == "file":
            path = str(getattr(self.cfg, "reference_snapshot_path", "logs/station_ref/station_ref.npz"))
            try:
                snapshot = ReferenceSnapshot.load(path)
                if self._snapshot_compatible(snapshot):
                    for env_id in env_ids.detach().cpu().tolist():
                        self._set_reference_env(
                            pipeline,
                            env_id,
                            snapshot.ref_points,
                            snapshot.track_image,
                        )
                    return
                print(
                    f"[SonarIbvsDrl] WARN: {path} not compatible with perception_frame="
                    f"{self._cfg_frame()}; extracting on polar/cart live frame.",
                    flush=True,
                )
                self.extract_for_envs(env_ids, pipeline, robot, beacon_pos_w, sonar_mount_offset_b, max_range)
            except (FileNotFoundError, ValueError) as exc:
                print(f"[SonarIbvsDrl] WARN: {exc}; falling back to geometry extraction", flush=True)
                self.extract_for_envs(env_ids, pipeline, robot, beacon_pos_w, sonar_mount_offset_b, max_range)

    def extract_for_envs(
        self,
        env_ids: torch.Tensor,
        pipeline: SonarPerceptionPipeline,
        robot,
        beacon_pos_w: torch.Tensor,
        sonar_mount_offset_b: torch.Tensor,
        max_range: float,
    ) -> None:
        sonar_w = sonar_pos_w(robot.data.root_pos_w, robot.data.root_quat_w, sonar_mount_offset_b)
        bearing, horiz_range = beacon_bearing_range(beacon_pos_w, sonar_w, robot.data.root_quat_w)
        ocfg = self.cfg.oceansim_sonar_cfg
        if self._cfg_frame() == "polar":
            rois = beacon_polar_rois(
                bearing,
                horiz_range,
                env_ids,
                min_range=float(ocfg.min_range),
                max_range=max_range,
                polar_height=int(pipeline.track_image.shape[1]),
                polar_width=int(pipeline.track_image.shape[2]),
                hori_fov_deg=float(ocfg.hori_fov),
            )
        else:
            rois = beacon_image_rois(
                bearing,
                horiz_range,
                env_ids,
                max_range=max_range,
                cart_image_size=int(self.cfg.cart_image_size),
            )
        ft = pipeline.feature_tracker
        for i, env_id in enumerate(env_ids.detach().cpu().tolist()):
            track_np = pipeline.track_image[env_id].detach().cpu().numpy()
            u8 = ft.to_u8_image(track_np)
            if self._cfg_frame() == "polar":
                pts = ft.extract_features(
                    u8,
                    roi_center=rois[i],
                    roi_radius=float(self.cfg.feature_roi_radius),
                    roi_radius_az=float(getattr(self.cfg, "feature_roi_radius_polar_az", 14.0)),
                    roi_radius_range=float(getattr(self.cfg, "feature_roi_radius_polar_range", 18.0)),
                    row_band_from_peak=float(getattr(self.cfg, "feature_polar_row_band", 20.0)),
                    min_intensity_ratio=float(getattr(self.cfg, "feature_polar_min_intensity_ratio", 0.35)),
                )
            else:
                pts = ft.extract_features(
                    u8,
                    roi_center=rois[i],
                    roi_radius=float(self.cfg.feature_roi_radius),
                )
            ft.set_reference(env_id, pts, ref_cart=track_np)

    def capture_snapshot(
        self,
        pipeline: SonarPerceptionPipeline,
        *,
        robot,
        station_origin_w: torch.Tensor,
        beacon_pos_w: torch.Tensor,
        sonar_mount_offset_b: torch.Tensor,
        max_range: float,
        path: str | None = None,
    ) -> ReferenceSnapshot:
        env_ids = robot._ALL_INDICES[:1]
        self.extract_for_envs(env_ids, pipeline, robot, beacon_pos_w, sonar_mount_offset_b, max_range)
        env_id = 0
        track_np = pipeline.track_image[env_id].detach().cpu().numpy()
        cart_np = pipeline.cart_image[env_id].detach().cpu().numpy()
        valid = pipeline.feature_tracker.valid[env_id]
        ref_pts = pipeline.feature_tracker.ref_points[env_id, valid].detach().cpu().numpy()
        yaw = float(
            spawn_yaw(
                env_ids,
                station_origin_w[:1],
                device=self.device,
                use_file_reference=self.use_file,
                spawn_use_teach_yaw=bool(getattr(self.cfg, "spawn_use_teach_yaw", True)),
                file_ref_yaw_valid=self.file_ref_yaw_valid,
                file_ref_yaw=self._file_ref_yaw,
                spawn_yaw_range_deg=float(getattr(self.cfg, "spawn_yaw_range_deg", 5.0)),
                eval_mode=bool(self.cfg.eval_mode),
                station_origin_w=station_origin_w,
                beacon_pos_w=beacon_pos_w,
            )[0].item()
        )
        frame = self._cfg_frame()
        snapshot = ReferenceSnapshot(
            ref_points=ref_pts,
            sonar_image=pipeline.sonar_image[env_id].detach().cpu().numpy(),
            cart_image=cart_np,
            track_image=track_np,
            perception_frame=frame,
            ref_yaw=yaw,
        )
        out = path or str(getattr(self.cfg, "reference_snapshot_path", "logs/station_ref/station_ref.npz"))
        snapshot.save(out)
        print(
            f"[SonarIbvsDrl] saved reference snapshot -> {out} "
            f"({len(ref_pts)} features, frame={frame})",
            flush=True,
        )
        return snapshot

    def spawn_yaw_for_reset(
        self,
        env_ids: torch.Tensor,
        spawn_pos: torch.Tensor,
        station_origin_w: torch.Tensor,
        beacon_pos_w: torch.Tensor,
    ) -> torch.Tensor:
        return spawn_yaw(
            env_ids,
            spawn_pos,
            device=self.device,
            use_file_reference=self.use_file,
            spawn_use_teach_yaw=bool(getattr(self.cfg, "spawn_use_teach_yaw", True)),
            file_ref_yaw_valid=self.file_ref_yaw_valid,
            file_ref_yaw=self._file_ref_yaw,
            spawn_yaw_range_deg=float(getattr(self.cfg, "spawn_yaw_range_deg", 5.0)),
            eval_mode=bool(self.cfg.eval_mode),
            station_origin_w=station_origin_w,
            beacon_pos_w=beacon_pos_w,
        )
