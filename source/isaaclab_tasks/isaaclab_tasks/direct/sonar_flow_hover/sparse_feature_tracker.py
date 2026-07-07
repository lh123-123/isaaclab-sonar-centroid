# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sparse feature extraction and accuracy-gated patch re-localization for sonar IBVS."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import torch

CoordFrame = Literal["cart", "polar"]


class SparseFeatureTracker:
    """Per-env sparse IBVS feature tracking on Cartesian sonar images."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        max_extract: int = 30,
        max_track: int = 20,
        image_size: int = 128,
        edge_margin: int = 10,
        fb_error_threshold: float = 1.0,
        fast_quality: float = 0.01,
        fast_min_distance: float = 5.0,
        blob_threshold_ratio: float = 0.25,
        blob_min_area: float = 4.0,
        patch_half_size: int = 6,
        relocate_window: int = 18,
        relocate_min_score: float = 0.55,
        ref_min_self_score: float = 0.72,
        ref_min_peak_margin: float = 0.10,
        track_min_peak_margin: float = 0.06,
        tracking_mode: str = "patch",
        coord_frame: CoordFrame = "cart",
        image_width: int | None = None,
        image_height: int | None = None,
        boresight_col_exclusion_half_width: int = 0,
    ):
        self.num_envs = num_envs
        self.device = device
        self.coord_frame: CoordFrame = str(coord_frame).lower()  # type: ignore[assignment]
        if self.coord_frame not in ("cart", "polar"):
            raise ValueError(f"coord_frame must be cart|polar, got {coord_frame!r}")
        self.tracking_mode = str(tracking_mode).lower()
        self.max_extract = int(max_extract)
        self.max_track = int(max_track)
        side = int(image_size)
        self.image_width = int(image_width if image_width is not None else side)
        self.image_height = int(image_height if image_height is not None else side)
        self.edge_margin = int(edge_margin)
        self.fb_error_threshold = float(fb_error_threshold)
        self.fast_quality = float(fast_quality)
        self.fast_min_distance = float(fast_min_distance)
        self.blob_threshold_ratio = float(blob_threshold_ratio)
        self.blob_min_area = float(blob_min_area)
        self.patch_half = int(patch_half_size)
        self.relocate_window = int(relocate_window)
        self.relocate_min_score = float(relocate_min_score)
        self.ref_min_self_score = float(ref_min_self_score)
        self.ref_min_peak_margin = float(ref_min_peak_margin)
        self.track_min_peak_margin = float(track_min_peak_margin)
        self.boresight_col_exclusion_half_width = int(boresight_col_exclusion_half_width)
        self._refresh_norm_scales()

        self.ref_points = torch.zeros(num_envs, max_track, 2, device=device)
        self.prev_points = torch.zeros(num_envs, max_track, 2, device=device)
        self.curr_points = torch.zeros(num_envs, max_track, 2, device=device)
        self.valid = torch.zeros(num_envs, max_track, dtype=torch.bool, device=device)
        self.has_prev_frame = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.prev_cart_u8: list[np.ndarray | None] = [None] * num_envs
        self.ref_cart_u8: list[np.ndarray | None] = [None] * num_envs
        self.ref_patches: list[list[np.ndarray | None]] = [[None] * max_track for _ in range(num_envs)]

    @property
    def image_size(self) -> int:
        """Legacy square size (max side) for cart-mode callers."""
        return max(self.image_width, self.image_height)

    def _refresh_norm_scales(self) -> None:
        w = max(float(self.image_width), 1.0)
        h = max(float(self.image_height), 1.0)
        self.diagonal = float(np.hypot(w, h))
        if self.coord_frame == "polar":
            # index-0 = azimuth column, index-1 = range row
            self._lateral_norm = w
            self._range_norm = h
        else:
            self._lateral_norm = self.diagonal
            self._range_norm = self.diagonal

    def _sync_image_shape(self, height: int, width: int) -> None:
        height = int(height)
        width = int(width)
        if height == self.image_height and width == self.image_width:
            return
        self.image_height = height
        self.image_width = width
        self._refresh_norm_scales()

    def mean_lateral_error_signed(self) -> torch.Tensor:
        """Signed lateral IBVS error: cart ex or polar Δaz (normalized)."""
        err = self.feature_errors()
        valid = self.valid
        scale = max(float(self._lateral_norm), 1.0e-6)
        ex = err[:, :, 0] / scale
        counts = self.valid_count().clamp(min=1).float()
        return (ex * valid.float()).sum(dim=1) / counts

    def mean_range_error_signed(self) -> torch.Tensor:
        """Signed range IBVS error: cart ey or polar Δrange row (normalized)."""
        err = self.feature_errors()
        valid = self.valid
        scale = max(float(self._range_norm), 1.0e-6)
        ey = err[:, :, 1] / scale
        counts = self.valid_count().clamp(min=1).float()
        return (ey * valid.float()).sum(dim=1) / counts

    def mean_lateral_error_abs(self) -> torch.Tensor:
        return self.mean_lateral_error_signed().abs()

    def reset(self, env_ids: torch.Tensor) -> None:
        self.valid[env_ids] = False
        self.has_prev_frame[env_ids] = False
        self.prev_points[env_ids] = 0.0
        self.curr_points[env_ids] = 0.0
        for idx in env_ids.detach().cpu().tolist():
            self.prev_cart_u8[idx] = None
            self.ref_cart_u8[idx] = None
            self.ref_patches[idx] = [None] * self.max_track

    def reset_temporal(self, env_ids: torch.Tensor) -> None:
        """Drop tracking history after a pose jump; keep ref_points and teach patches."""
        self.has_prev_frame[env_ids] = False
        for idx in env_ids.detach().cpu().tolist():
            self.prev_cart_u8[idx] = None
            mask = self.valid[idx]
            if mask.any():
                self.curr_points[idx, mask] = self.ref_points[idx, mask]
                self.prev_points[idx, mask] = self.ref_points[idx, mask]

    def _crop_patch(self, img: np.ndarray, x: float, y: float) -> np.ndarray | None:
        half = self.patch_half
        h, w = img.shape
        cx, cy = int(round(x)), int(round(y))
        y0, y1 = cy - half, cy + half + 1
        x0, x1 = cx - half, cx + half + 1
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            return None
        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        return patch.copy()

    def _peak_margin(self, result: np.ndarray) -> float:
        flat = result.reshape(-1)
        if flat.size < 2:
            return float(flat.max()) if flat.size else 0.0
        best = float(flat.max())
        flat_work = flat.copy()
        flat_work[np.argmax(flat_work)] = -1.0
        second = float(flat_work.max())
        return best - second

    def _match_patch(
        self,
        ref_patch: np.ndarray,
        img_u8: np.ndarray,
        search_x: float,
        search_y: float,
        *,
        min_score: float,
        min_peak_margin: float,
        search_window: int | None = None,
    ) -> tuple[np.ndarray | None, float, float]:
        """Search a local window for ``ref_patch``; reject ambiguous matches."""
        ph, pw = ref_patch.shape
        win = int(search_window if search_window is not None else self.relocate_window)
        h, w = img_u8.shape
        cx, cy = int(round(search_x)), int(round(search_y))
        sy0 = max(0, cy - win)
        sy1 = min(h, cy + win + 1)
        sx0 = max(0, cx - win)
        sx1 = min(w, cx + win + 1)
        search = img_u8[sy0:sy1, sx0:sx1]
        if search.shape[0] < ph or search.shape[1] < pw:
            return None, 0.0, 0.0

        result = cv2.matchTemplate(search, ref_patch, cv2.TM_CCOEFF_NORMED)
        peak_margin = self._peak_margin(result)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < min_score or peak_margin < min_peak_margin:
            return None, float(max_val), peak_margin

        new_x = sx0 + max_loc[0] + (pw - 1) * 0.5
        new_y = sy0 + max_loc[1] + (ph - 1) * 0.5
        m = self.edge_margin
        if not (m <= new_x < w - m and m <= new_y < h - m):
            return None, float(max_val), peak_margin
        return np.array([new_x, new_y], dtype=np.float32), float(max_val), peak_margin

    def _validate_teach_point(self, ref_u8: np.ndarray, x: float, y: float) -> tuple[bool, float]:
        """Keep only patches that are self-consistent and locally unique on the teach frame."""
        patch = self._crop_patch(ref_u8, x, y)
        if patch is None:
            return False, 0.0
        pt, score, margin = self._match_patch(
            patch,
            ref_u8,
            x,
            y,
            min_score=self.ref_min_self_score,
            min_peak_margin=self.ref_min_peak_margin,
        )
        if pt is None:
            return False, score
        drift = float(np.linalg.norm(pt - np.array([x, y], dtype=np.float32)))
        if drift > 1.5:
            return False, score
        return True, score

    def _build_ref_patches(self, env_id: int, ref_u8: np.ndarray, n: int) -> None:
        patches: list[np.ndarray | None] = [None] * self.max_track
        for i in range(n):
            x = float(self.ref_points[env_id, i, 0].item())
            y = float(self.ref_points[env_id, i, 1].item())
            patches[i] = self._crop_patch(ref_u8, x, y)
        self.ref_patches[env_id] = patches

    def set_reference(
        self,
        env_id: int,
        ref_points: np.ndarray,
        ref_cart: np.ndarray | None = None,
    ) -> None:
        """Store taught reference points (up to max_track) and optional teach Cartesian image."""
        self.ref_points[env_id, :] = 0.0
        self.valid[env_id, :] = False
        self.has_prev_frame[env_id] = False
        self.prev_cart_u8[env_id] = None
        self.ref_patches[env_id] = [None] * self.max_track

        if ref_cart is not None:
            self.ref_cart_u8[env_id] = self.to_u8_image(ref_cart)
        else:
            self.ref_cart_u8[env_id] = None

        if ref_cart is None or len(ref_points) == 0:
            return

        ref_u8 = self.ref_cart_u8[env_id]
        scored: list[tuple[float, np.ndarray]] = []
        for pt in ref_points[: self.max_extract]:
            ok, score = self._validate_teach_point(ref_u8, float(pt[0]), float(pt[1]))
            if ok:
                scored.append((score, pt.astype(np.float32)))

        scored.sort(key=lambda item: item[0], reverse=True)
        n = min(len(scored), self.max_track)
        if n == 0:
            return

        for i, (_score, pt) in enumerate(scored[:n]):
            self.ref_points[env_id, i, 0] = float(pt[0])
            self.ref_points[env_id, i, 1] = float(pt[1])
            self.prev_points[env_id, i] = self.ref_points[env_id, i]
            self.curr_points[env_id, i] = self.ref_points[env_id, i]
            self.valid[env_id, i] = True
            self.ref_patches[env_id][i] = self._crop_patch(ref_u8, float(pt[0]), float(pt[1]))

    @staticmethod
    def to_u8_image(cart: np.ndarray) -> np.ndarray:
        """Map Cartesian float image [0,1] to uint8 (fixed scale, not per-frame min-max)."""
        return (np.clip(cart, 0.0, 1.0) * 255.0).astype(np.uint8)

    _to_u8 = to_u8_image  # backward compat

    def _passes_boresight_col(self, x: float, width: int) -> bool:
        """Reject features on the polar boresight column (center white seam)."""
        if self.coord_frame != "polar" or self.boresight_col_exclusion_half_width <= 0:
            return True
        cx = (width - 1) * 0.5
        return abs(float(x) - cx) > float(self.boresight_col_exclusion_half_width)

    def _apply_boresight_exclusion(self, mask: np.ndarray, width: int) -> None:
        if self.coord_frame != "polar" or self.boresight_col_exclusion_half_width <= 0:
            return
        cx = int(round((width - 1) * 0.5))
        hw = int(self.boresight_col_exclusion_half_width)
        c0 = max(0, cx - hw)
        c1 = min(width, cx + hw + 1)
        mask[:, c0:c1] = 0

    def _roi_mask(
        self,
        shape: tuple[int, int],
        roi_center: tuple[float, float] | None,
        roi_radius: float,
        *,
        roi_radius_az: float | None = None,
        roi_radius_range: float | None = None,
    ) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        if roi_center is not None:
            cx, cy = roi_center
            if roi_radius_az is not None or roi_radius_range is not None:
                ax = max(float(roi_radius_az if roi_radius_az is not None else roi_radius), 1.0)
                ay = max(float(roi_radius_range if roi_radius_range is not None else roi_radius), 1.0)
                cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 255, -1)
            else:
                cv2.circle(mask, (int(cx), int(cy)), int(roi_radius), 255, -1)
        else:
            mask[:] = 255
        m = self.edge_margin
        mask[:m, :] = 0
        mask[-m:, :] = 0
        mask[:, :m] = 0
        mask[:, -m:] = 0
        self._apply_boresight_exclusion(mask, w)
        return mask

    @staticmethod
    def _peak_row_in_mask(img_u8: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
        """Return (row, peak_u8) of the brightest pixel inside mask."""
        roi = img_u8.copy()
        roi[mask == 0] = 0
        flat = roi.reshape(-1)
        if flat.size == 0 or float(flat.max()) <= 0.0:
            return 0, 0.0
        idx = int(flat.argmax())
        h, w = img_u8.shape
        peak_row, _ = divmod(idx, w)
        return int(peak_row), float(flat.max())

    def _passes_polar_gates(
        self,
        img_u8: np.ndarray,
        pt: np.ndarray,
        *,
        peak_row: int,
        peak_u8: float,
        row_band: float | None,
        min_intensity_ratio: float | None,
    ) -> bool:
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        h, w = img_u8.shape
        if not (0 <= x < w and 0 <= y < h):
            return False
        if row_band is not None and row_band > 0.0 and abs(y - peak_row) > row_band:
            return False
        if min_intensity_ratio is not None and min_intensity_ratio > 0.0 and peak_u8 > 1.0e-3:
            if float(img_u8[y, x]) < peak_u8 * float(min_intensity_ratio):
                return False
        if not self._passes_boresight_col(float(x), w):
            return False
        return True

    @staticmethod
    def _merge_points(
        points: list[np.ndarray],
        *,
        max_points: int,
        min_distance: float,
    ) -> np.ndarray:
        merged: list[np.ndarray] = []
        for batch in points:
            if batch is None or len(batch) == 0:
                continue
            for pt in batch:
                if len(merged) >= max_points:
                    break
                if not merged:
                    merged.append(pt.astype(np.float32))
                    continue
                dists = np.linalg.norm(np.stack(merged, axis=0) - pt, axis=1)
                if float(dists.min()) >= min_distance:
                    merged.append(pt.astype(np.float32))
            if len(merged) >= max_points:
                break
        if not merged:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack(merged, axis=0).astype(np.float32)

    def _extract_fast_points(self, cart_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
        corners = cv2.goodFeaturesToTrack(
            cart_u8,
            maxCorners=self.max_extract,
            qualityLevel=self.fast_quality,
            minDistance=self.fast_min_distance,
            mask=mask,
        )
        if corners is None:
            return np.zeros((0, 2), dtype=np.float32)
        return corners.reshape(-1, 2).astype(np.float32)

    def _blob_binary(self, cart_u8: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        roi = cv2.bitwise_and(cart_u8, cart_u8, mask=mask)
        vals = roi[mask > 0]
        if vals.size == 0:
            return None
        peak = float(vals.max())
        if peak <= 8.0:
            return None
        thresh = max(int(peak * self.blob_threshold_ratio), 8)
        _, binary = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        if cv2.countNonZero(binary) < self.blob_min_area:
            return None
        return binary

    def _extract_gradient_peaks(
        self,
        cart_u8: np.ndarray,
        mask: np.ndarray,
        n_needed: int,
    ) -> np.ndarray:
        """Local gradient maxima on the sonar blob (more distinctive than contour arcs)."""
        if n_needed <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        binary = self._blob_binary(cart_u8, mask)
        if binary is None:
            return np.zeros((0, 2), dtype=np.float32)

        roi = cv2.bitwise_and(cart_u8, cart_u8, mask=binary)
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag[binary == 0] = 0.0

        flat = mag.reshape(-1)
        order = np.argsort(flat)[::-1]
        picked: list[np.ndarray] = []
        h, w = cart_u8.shape
        min_d = max(3.0, self.fast_min_distance * 0.6)
        for idx in order:
            if len(picked) >= n_needed:
                break
            val = float(flat[idx])
            if val <= 1.0e-3:
                break
            y, x = divmod(int(idx), w)
            pt = np.array([float(x), float(y)], dtype=np.float32)
            if mask[y, x] == 0:
                continue
            if picked:
                dists = np.linalg.norm(np.stack(picked, axis=0) - pt, axis=1)
                if float(dists.min()) < min_d:
                    continue
            picked.append(pt)
        if not picked:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack(picked, axis=0).astype(np.float32)

    def extract_features(
        self,
        cart_u8: np.ndarray,
        roi_center: tuple[float, float] | None = None,
        roi_radius: float = 40.0,
        *,
        roi_radius_az: float | None = None,
        roi_radius_range: float | None = None,
        row_band_from_peak: float | None = None,
        min_intensity_ratio: float | None = None,
    ) -> np.ndarray:
        """Extract accuracy-gated FAST + gradient-peak candidates (quality over count)."""
        mask = self._roi_mask(
            cart_u8.shape,
            roi_center,
            roi_radius,
            roi_radius_az=roi_radius_az,
            roi_radius_range=roi_radius_range,
        )
        peak_row, peak_u8 = self._peak_row_in_mask(cart_u8, mask)
        gate_kwargs = dict(
            peak_row=peak_row,
            peak_u8=peak_u8,
            row_band=row_band_from_peak if self.coord_frame == "polar" else None,
            min_intensity_ratio=min_intensity_ratio if self.coord_frame == "polar" else None,
        )

        fast_pts = self._extract_fast_points(cart_u8, mask)
        fast_filtered = [
            pt for pt in fast_pts if self._passes_polar_gates(cart_u8, pt, **gate_kwargs)
        ]
        candidates = self._merge_points(
            [np.stack(fast_filtered, axis=0) if fast_filtered else np.zeros((0, 2), np.float32)],
            max_points=self.max_extract,
            min_distance=self.fast_min_distance,
        )

        scored: list[tuple[float, np.ndarray]] = []
        for pt in candidates:
            if not self._passes_polar_gates(cart_u8, pt, **gate_kwargs):
                continue
            ok, score = self._validate_teach_point(cart_u8, float(pt[0]), float(pt[1]))
            if ok:
                scored.append((score, pt))

        if len(scored) < self.max_track:
            grad_pts = self._extract_gradient_peaks(cart_u8, mask, self.max_extract)
            existing = [pt for _, pt in scored]
            for pt in grad_pts:
                if not self._passes_polar_gates(cart_u8, pt, **gate_kwargs):
                    continue
                if existing:
                    dists = np.linalg.norm(np.stack(existing, axis=0) - pt, axis=1)
                    if float(dists.min()) < self.fast_min_distance:
                        continue
                ok, score = self._validate_teach_point(cart_u8, float(pt[0]), float(pt[1]))
                if ok:
                    scored.append((score, pt))
                    existing.append(pt)
                if len(scored) >= self.max_track:
                    break

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack([pt for _, pt in scored[: self.max_track]], axis=0).astype(np.float32)

    def _relocate_points(
        self,
        env_id: int,
        curr_u8: np.ndarray,
        active_idx: torch.Tensor,
        *,
        search_from_ref: bool = False,
        search_window: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Patch-match from teach ref location (IBVS); reject ambiguous relocations."""
        n = len(active_idx)
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32), np.array([], dtype=bool)

        ref_pts = self.ref_points[env_id, active_idx].detach().cpu().numpy()
        out_pts = np.zeros((n, 2), dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        has_prev = bool(self.has_prev_frame[env_id].item())
        for j, idx in enumerate(active_idx.tolist()):
            patch = self.ref_patches[env_id][idx]
            if patch is None:
                continue
            ref_x = float(ref_pts[j, 0])
            ref_y = float(ref_pts[j, 1])
            if search_from_ref or not bool(self.valid[env_id, idx].item()):
                search_x, search_y = ref_x, ref_y
            elif has_prev:
                search_x = float(self.curr_points[env_id, idx, 0].item())
                search_y = float(self.curr_points[env_id, idx, 1].item())
            else:
                search_x, search_y = ref_x, ref_y

            match_kwargs = dict(
                min_score=self.relocate_min_score,
                min_peak_margin=self.track_min_peak_margin,
                search_window=search_window,
            )
            pt, _score, _margin = self._match_patch(
                patch,
                curr_u8,
                search_x,
                search_y,
                **match_kwargs,
            )
            if pt is None:
                pt, _score, _margin = self._match_patch(
                    patch,
                    curr_u8,
                    ref_x,
                    ref_y,
                    **match_kwargs,
                )
            if pt is not None and self._passes_boresight_col(float(pt[0]), curr_u8.shape[1]):
                out_pts[j] = pt
                valid[j] = True
        return out_pts, valid

    def _mean_valid_shift(self, env_id: int) -> tuple[float, float]:
        """Mean (curr - ref) over valid taught features; zero if none valid."""
        mask = self._taught_mask()[env_id]
        valid = self.valid[env_id] & mask
        if not bool(valid.any().item()):
            return 0.0, 0.0
        err = self.feature_errors()[env_id][valid]
        return float(err[:, 0].mean().item()), float(err[:, 1].mean().item())

    def _recover_lost_features(self, env_id: int, curr_u8: np.ndarray) -> int:
        """Re-acquire lost taught features via teach patches (wide search + shift hint)."""
        mask = self._taught_mask()[env_id]
        taught = torch.where(mask)[0]
        if len(taught) == 0:
            return 0
        lost = taught[~self.valid[env_id][taught]]
        if len(lost) == 0:
            return 0

        shift_x, shift_y = self._mean_valid_shift(env_id)
        wide = max(self.relocate_window * 3, self.relocate_window + 12)
        recovered = 0
        for idx in lost.tolist():
            patch = self.ref_patches[env_id][idx]
            if patch is None:
                continue
            ref_x = float(self.ref_points[env_id, idx, 0].item())
            ref_y = float(self.ref_points[env_id, idx, 1].item())
            hint_x = ref_x + shift_x
            hint_y = ref_y + shift_y
            curr_x = float(self.curr_points[env_id, idx, 0].item())
            curr_y = float(self.curr_points[env_id, idx, 1].item())
            search_centers = [
                (hint_x, hint_y),
                (curr_x, curr_y),
                (ref_x, ref_y),
            ]
            pt = None
            for sx, sy in search_centers:
                pt, _score, _margin = self._match_patch(
                    patch,
                    curr_u8,
                    sx,
                    sy,
                    min_score=self.relocate_min_score,
                    min_peak_margin=self.track_min_peak_margin,
                    search_window=wide,
                )
                if pt is not None:
                    break
            if pt is not None and self._passes_boresight_col(float(pt[0]), curr_u8.shape[1]):
                self.curr_points[env_id, idx] = torch.tensor(pt, device=self.device, dtype=torch.float32)
                self.valid[env_id, idx] = True
                recovered += 1
        return recovered

    def _lk_track(
        self,
        env_id: int,
        prev_u8: np.ndarray,
        curr_u8: np.ndarray,
        active_idx: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classic pyramidal LK optical flow with forward-backward consistency."""
        n = len(active_idx)
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32), np.array([], dtype=bool)

        prev_pts = (
            self.prev_points[env_id, active_idx].detach().cpu().numpy().reshape(-1, 1, 2).astype(np.float32)
        )
        lk_kwargs = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        curr_pts, status_f, _ = cv2.calcOpticalFlowPyrLK(prev_u8, curr_u8, prev_pts, None, **lk_kwargs)
        if curr_pts is None:
            return np.zeros((n, 2), dtype=np.float32), np.zeros(n, dtype=bool)

        back_pts, status_b, _ = cv2.calcOpticalFlowPyrLK(curr_u8, prev_u8, curr_pts, None, **lk_kwargs)
        if back_pts is None:
            return np.zeros((n, 2), dtype=np.float32), np.zeros(n, dtype=bool)

        status_f = status_f.reshape(-1).astype(bool)
        status_b = status_b.reshape(-1).astype(bool)
        fb_err = np.linalg.norm(prev_pts.reshape(n, 2) - back_pts.reshape(n, 2), axis=1)

        out_pts = curr_pts.reshape(n, 2).astype(np.float32)
        h, w = curr_u8.shape
        m = self.edge_margin
        valid = np.zeros(n, dtype=bool)
        for j in range(n):
            if not (status_f[j] and status_b[j]):
                continue
            if fb_err[j] > self.fb_error_threshold:
                continue
            x, y = float(out_pts[j, 0]), float(out_pts[j, 1])
            if m <= x < w - m and m <= y < h - m and self._passes_boresight_col(x, w):
                valid[j] = True
        return out_pts, valid

    def update_env(self, env_id: int, track_image: torch.Tensor) -> None:
        """Track features on cart or polar grayscale image (H, W)."""
        track_np = track_image.detach().cpu().numpy()
        if track_np.ndim != 2:
            raise ValueError(f"Expected (H, W) track image, got {track_np.shape}")
        self._sync_image_shape(track_np.shape[0], track_np.shape[1])
        curr_u8 = self.to_u8_image(track_np)

        taught = torch.where(self._taught_mask()[env_id])[0]
        n_taught = int(taught.numel())
        n_valid = int(self.valid[env_id].sum().item())

        if n_taught == 0:
            self.prev_cart_u8[env_id] = curr_u8
            self.has_prev_frame[env_id] = True
            return

        if self.ref_cart_u8[env_id] is None:
            self.ref_cart_u8[env_id] = curr_u8.copy()

        if n_valid == 0:
            self._recover_lost_features(env_id, curr_u8)
            self.prev_points[env_id, self.valid[env_id]] = self.curr_points[env_id, self.valid[env_id]]
            self.prev_cart_u8[env_id] = curr_u8
            self.has_prev_frame[env_id] = True
            return

        if not self.has_prev_frame[env_id] or self.prev_cart_u8[env_id] is None:
            self.prev_cart_u8[env_id] = curr_u8.copy()
            self.prev_points[env_id, self.valid[env_id]] = self.curr_points[env_id, self.valid[env_id]]
            self.has_prev_frame[env_id] = True
            return

        active_idx = torch.where(self.valid[env_id])[0]
        if self.tracking_mode == "lk":
            curr_pts, track_valid = self._lk_track(env_id, self.prev_cart_u8[env_id], curr_u8, active_idx)
        else:
            curr_pts, track_valid = self._relocate_points(env_id, curr_u8, active_idx)
        new_valid = self.valid[env_id].clone()
        new_valid[active_idx] = False
        for j, idx in enumerate(active_idx.tolist()):
            if track_valid[j]:
                self.curr_points[env_id, idx] = torch.from_numpy(curr_pts[j]).to(self.device)
                new_valid[idx] = True
        self.valid[env_id] = new_valid

        self._recover_lost_features(env_id, curr_u8)

        self.prev_points[env_id] = self.curr_points[env_id].clone()
        self.prev_cart_u8[env_id] = curr_u8
        self.has_prev_frame[env_id] = True

    def update(self, track_images: torch.Tensor) -> None:
        for env_id in range(track_images.shape[0]):
            self.update_env(env_id, track_images[env_id])

    def feature_errors(self) -> torch.Tensor:
        err = self.curr_points - self.ref_points
        return torch.where(self.valid.unsqueeze(-1), err, torch.zeros_like(err))

    def valid_count(self) -> torch.Tensor:
        return self.valid.sum(dim=1)

    def _taught_mask(self) -> torch.Tensor:
        return torch.linalg.norm(self.ref_points, dim=-1) > 1.0e-3

    def ref_centroid(self) -> torch.Tensor:
        """Mean teach feature position in track-image pixels, shape (num_envs, 2)."""
        mask = self._taught_mask()
        counts = mask.sum(dim=1).clamp(min=1).float().unsqueeze(-1)
        masked = self.ref_points * mask.unsqueeze(-1).float()
        return masked.sum(dim=1) / counts

    def tracked_centroid(self) -> torch.Tensor:
        """Mean position of currently valid tracked features in pixels."""
        mask = self._taught_mask()
        valid = self.valid & mask
        counts = valid.sum(dim=1).clamp(min=1).float().unsqueeze(-1)
        weighted = self.curr_points * valid.unsqueeze(-1).float()
        return weighted.sum(dim=1) / counts

    def centroid_errors(self) -> torch.Tensor:
        """Tracked minus teach centroid in pixels, shape (num_envs, 2)."""
        return self.tracked_centroid() - self.ref_centroid()

    def _image_norm_scale(self) -> torch.Tensor:
        return torch.tensor(
            [max(self.image_width, 1), max(self.image_height, 1)],
            device=self.device,
            dtype=self.curr_points.dtype,
        )

    def ref_centroid_norm(self) -> torch.Tensor:
        """Teach centroid in normalized image coords, shape (num_envs, 2)."""
        return self.ref_centroid() / self._image_norm_scale().unsqueeze(0)

    def tracked_centroid_norm(self) -> torch.Tensor:
        """Tracked centroid in normalized image coords, shape (num_envs, 2)."""
        return self.tracked_centroid() / self._image_norm_scale().unsqueeze(0)

    def centroid_errors_norm(self) -> torch.Tensor:
        """Teach-to-track centroid offset normalized by image width/height."""
        err = self.centroid_errors()
        scale = torch.tensor(
            [max(float(self._lateral_norm), 1.0e-6), max(float(self._range_norm), 1.0e-6)],
            device=self.device,
            dtype=self.curr_points.dtype,
        )
        return err / scale.unsqueeze(0)

    def centroid_lateral_error_signed(self) -> torch.Tensor:
        """Signed lateral centroid error (cart ex or polar Δaz), normalized."""
        return self.centroid_errors_norm()[:, 0]

    def centroid_lateral_error_abs(self) -> torch.Tensor:
        return self.centroid_lateral_error_signed().abs()

    def normalized_pixel_error(self) -> torch.Tensor:
        err = self.feature_errors()
        norms = torch.linalg.norm(
            err / torch.tensor([self._lateral_norm, self._range_norm], device=err.device),
            dim=-1,
        )
        counts = self.valid_count().clamp(min=1).float()
        mean_err = (norms * self.valid.float()).sum(dim=1) / counts
        return mean_err / max(float(self.diagonal), 1.0e-6)

    def observation_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.tensor([self._lateral_norm, self._range_norm], device=self.device, dtype=torch.float32)
        err = self.feature_errors() / scale.unsqueeze(0).unsqueeze(0)
        err = err.reshape(self.num_envs, self.max_track * 2)
        flags = self.valid.float()
        return err, flags
