"""CrowdFlow analyzer backed by the RAFT optical-flow implementation.

This module computes dense optical flow with the official RAFT weights shipped in
``torchvision`` and reproduces the short-term/dynamic metrics from the CrowdFlow
benchmark.  Foreground segmentation still relies on MediaPipe's selfie model to
separate foreground pedestrians from the static background.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import cv2  # type: ignore[import]
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# -------------------------------------------------
# RAFT import (torchvision optical flow)
# -------------------------------------------------
_RAFT_AVAILABLE = False
_RAFT_ERROR = ""
RAFT_Large_Weights = None
raft_large = None

try:
    from torchvision.models.optical_flow import raft_large

    # torchvision naming differs across versions
    try:
        from torchvision.models.optical_flow import RAFT_Large_Weights
    except Exception:
        from torchvision.models.optical_flow import Raft_Large_Weights as RAFT_Large_Weights  # type: ignore

    _RAFT_AVAILABLE = True

except Exception as e:
    _RAFT_ERROR = str(e)
    _RAFT_AVAILABLE = False
# -------------------------------------------------

try:
    import mediapipe as mp
    _HAS_MP_SOLUTIONS = hasattr(mp, "solutions")
except Exception:
    mp = None
    _HAS_MP_SOLUTIONS = False

# try:
#     from mediapipe.solutions import selfie_segmentation as mp_selfie  # type: ignore[import]
# except ImportError as exc:  # pragma: no cover - optional dependency guard
#     raise ImportError("CrowdFlowAnalyzer requires the `mediapipe` package. Install it via `pip install mediapipe`.") from exc


@dataclass(frozen=True)
class CrowdFlowConfig:
    """Configuration used when running the CrowdFlow analyzer."""

    dynamic_threshold: float = 0.75
    max_frames: int = 180
    ta_segments: int = 5


class CrowdFlowAnalyzer:
    """Compute CrowdFlow metrics using RAFT optical flow and MediaPipe segmentation."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        dynamic_threshold: float = 0.75,
        max_frames: int = 180,
        ta_segments: int = 5,
    ) -> None:
        # ---- status flags (for SAFE+STRICT policy) ----
        self.available: bool = False
        self.unavailable_reason: str = ""

        # ---- device ----
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested for CrowdFlow but CUDA is unavailable; defaulting to CPU.")

        # ---- config ----
        self.config = CrowdFlowConfig(
            dynamic_threshold=float(dynamic_threshold),
            max_frames=int(max_frames),
            ta_segments=int(ta_segments),
        )

        # ---- segmentation ----
        self._segmentation_cls = mp_selfie.SelfieSegmentation

        # ---- RAFT availability (import-time capability flag) ----
        if not _RAFT_AVAILABLE or RAFT_Large_Weights is None or raft_large is None:
            self.unavailable_reason = (
                "RAFT unavailable (torchvision mismatch). "
                f"_RAFT_AVAILABLE={_RAFT_AVAILABLE} "
                f"error={globals().get('_RAFT_ERROR', '')}"
            )
            logger.warning("[CrowdFlow] disabled: %s", self.unavailable_reason)
            return

        # ---- RAFT init (runtime) ----
        try:
            # Prefer the weight enum you were using, but be robust if it doesn't exist.
            if hasattr(RAFT_Large_Weights, "C_T_V2"):
                self._raft_weights = RAFT_Large_Weights.C_T_V2
            else:
                # Works across more torchvision versions.
                self._raft_weights = RAFT_Large_Weights.DEFAULT

            self._raft = raft_large(weights=self._raft_weights, progress=False).to(self.device).eval()
            self._raft_transforms = self._raft_weights.transforms()

            self.available = True

        except Exception as exc:
            self.unavailable_reason = f"RAFT init failed: {exc}"
            logger.warning("[CrowdFlow] disabled: %s", self.unavailable_reason)
            self.available = False
            return


    def _extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frames_rgb: List[np.ndarray] = []
        frames_gray: List[np.ndarray] = []

        max_frames = self.config.max_frames
        try:
            while len(frames_rgb) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        finally:
            cap.release()

        if not frames_rgb:
            raise RuntimeError("No frames available for CrowdFlow analysis.")
        return frames_rgb, frames_gray

    def _segment_frames(self, frames_rgb: Sequence[np.ndarray]) -> List[np.ndarray]:
        segmentation = self._segmentation_cls(model_selection=1)
        masks: List[np.ndarray] = []

        try:
            for frame in frames_rgb:
                result = segmentation.process(frame)
                mask = result.segmentation_mask if result.segmentation_mask is not None else np.zeros(frame.shape[:2], dtype=np.float32)
                masks.append(mask)
        finally:
            segmentation.close()

        return masks

    def _compute_flows(self, frames_rgb: Sequence[np.ndarray]) -> List[np.ndarray]:
        flows: List[np.ndarray] = []

        for idx in range(len(frames_rgb) - 1):
            img1 = Image.fromarray(frames_rgb[idx])
            img2 = Image.fromarray(frames_rgb[idx + 1])
            tensor1, tensor2 = self._raft_transforms(img1, img2)
            tensor1 = tensor1.unsqueeze(0).to(self.device)
            tensor2 = tensor2.unsqueeze(0).to(self.device)

            with torch.no_grad():
                prediction = self._raft(tensor1, tensor2)

            if isinstance(prediction, (list, tuple)):
                flow_tensor = prediction[-1]
            else:
                flow_tensor = prediction

            flow_tensor = flow_tensor.to(torch.float32)
            # Resize flow back to the pre-transformed spatial resolution.
            flow_np = flow_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            flows.append(flow_np)

        return flows

    def _compute_tracks(self, frames_gray: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
        first_frame = frames_gray[0]
        prev_pts = cv2.goodFeaturesToTrack(
            first_frame,
            mask=None,
            maxCorners=400,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )
        if prev_pts is None:
            return []

        tracks: List[Dict[str, Any]] = [
            {"positions": [pt.reshape(2)], "frames": [0]}
            for pt in prev_pts
        ]
        active_indices = np.arange(len(prev_pts))
        prev_pts_formatted = prev_pts.copy()
        prev_gray = first_frame

        for frame_idx in range(1, len(frames_gray)):
            gray = frames_gray[frame_idx]
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts_formatted, None)
            if next_pts is None or status is None:
                break

            status_flat = status.reshape(-1)
            valid_next = next_pts[status_flat == 1]
            valid_indices = active_indices[status_flat == 1]

            for idx, track_idx in enumerate(valid_indices):
                tracks[track_idx]["positions"].append(valid_next[idx].reshape(2))
                tracks[track_idx]["frames"].append(frame_idx)

            prev_pts_formatted = valid_next.reshape(-1, 1, 2)
            active_indices = valid_indices
            prev_gray = gray

            if len(prev_pts_formatted) == 0:
                break

        return tracks

    @staticmethod
    def _region_stats(flow: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
        if not np.any(mask):
            return 0.0, 0.0
        vectors = flow[mask]
        magnitudes = np.linalg.norm(vectors, axis=1)
        epe = float(np.mean(magnitudes)) if magnitudes.size else 0.0
        unit_vectors = vectors / (magnitudes[:, None] + 1e-6)
        mean_vector = np.mean(unit_vectors, axis=0)
        mean_norm = np.linalg.norm(mean_vector) + 1e-6
        alignment = np.clip(unit_vectors @ (mean_vector / mean_norm), -1.0, 1.0)
        r2 = float((np.mean(alignment) + 1.0) / 2.0)
        return epe, r2

    def _compute_short_term_metrics(
        self,
        flows: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        dynamic_threshold: float,
    ) -> Dict[str, float]:
        if not flows:
            return {key: 0.0 for key in [
                "of_fg_static_epe_st",
                "of_fg_static_r2_st",
                "of_bg_static_epe_st",
                "of_bg_static_r2_st",
                "of_fg_dynamic_epe_st",
                "of_fg_dynamic_r2_st",
                "of_bg_dynamic_epe_st",
                "of_bg_dynamic_r2_st",
                "of_fg_avg_epe_st",
                "of_fg_avg_r2_st",
                "of_bg_avg_epe_st",
                "of_bg_avg_r2_st",
                "of_avg_epe_st",
                "of_avg_r2_st",
                "of_time_length_st",
            ]}

        fg_static_epe: List[float] = []
        fg_static_r2: List[float] = []
        fg_dynamic_epe: List[float] = []
        fg_dynamic_r2: List[float] = []
        bg_static_epe: List[float] = []
        bg_static_r2: List[float] = []
        bg_dynamic_epe: List[float] = []
        bg_dynamic_r2: List[float] = []
        fg_avg_epe: List[float] = []
        fg_avg_r2: List[float] = []
        bg_avg_epe: List[float] = []
        bg_avg_r2: List[float] = []
        all_avg_epe: List[float] = []
        all_avg_r2: List[float] = []

        for flow, mask in zip(flows, masks):
            mask_resized = cv2.resize(mask, (flow.shape[1], flow.shape[0]), interpolation=cv2.INTER_LINEAR)
            magnitude = np.linalg.norm(flow, axis=2)
            dynamic_mask = magnitude > dynamic_threshold

            fg_mask = mask_resized > 0.5
            bg_mask = ~fg_mask

            fg_dynamic = fg_mask & dynamic_mask
            fg_static = fg_mask & ~dynamic_mask
            bg_dynamic = bg_mask & dynamic_mask
            bg_static = bg_mask & ~dynamic_mask

            epe, r2 = self._region_stats(flow, fg_static)
            fg_static_epe.append(epe)
            fg_static_r2.append(r2)

            epe, r2 = self._region_stats(flow, fg_dynamic)
            fg_dynamic_epe.append(epe)
            fg_dynamic_r2.append(r2)

            epe, r2 = self._region_stats(flow, bg_static)
            bg_static_epe.append(epe)
            bg_static_r2.append(r2)

            epe, r2 = self._region_stats(flow, bg_dynamic)
            bg_dynamic_epe.append(epe)
            bg_dynamic_r2.append(r2)

            epe, r2 = self._region_stats(flow, fg_mask)
            fg_avg_epe.append(epe)
            fg_avg_r2.append(r2)

            epe, r2 = self._region_stats(flow, bg_mask)
            bg_avg_epe.append(epe)
            bg_avg_r2.append(r2)

            epe, r2 = self._region_stats(flow, np.ones_like(fg_mask, dtype=bool))
            all_avg_epe.append(epe)
            all_avg_r2.append(r2)

        return {
            "of_fg_static_epe_st": float(np.mean(fg_static_epe)) if fg_static_epe else 0.0,
            "of_fg_static_r2_st": float(np.mean(fg_static_r2)) if fg_static_r2 else 0.0,
            "of_bg_static_epe_st": float(np.mean(bg_static_epe)) if bg_static_epe else 0.0,
            "of_bg_static_r2_st": float(np.mean(bg_static_r2)) if bg_static_r2 else 0.0,
            "of_fg_dynamic_epe_st": float(np.mean(fg_dynamic_epe)) if fg_dynamic_epe else 0.0,
            "of_fg_dynamic_r2_st": float(np.mean(fg_dynamic_r2)) if fg_dynamic_r2 else 0.0,
            "of_bg_dynamic_epe_st": float(np.mean(bg_dynamic_epe)) if bg_dynamic_epe else 0.0,
            "of_bg_dynamic_r2_st": float(np.mean(bg_dynamic_r2)) if bg_dynamic_r2 else 0.0,
            "of_fg_avg_epe_st": float(np.mean(fg_avg_epe)) if fg_avg_epe else 0.0,
            "of_fg_avg_r2_st": float(np.mean(fg_avg_r2)) if fg_avg_r2 else 0.0,
            "of_bg_avg_epe_st": float(np.mean(bg_avg_epe)) if bg_avg_epe else 0.0,
            "of_bg_avg_r2_st": float(np.mean(bg_avg_r2)) if bg_avg_r2 else 0.0,
            "of_avg_epe_st": float(np.mean(all_avg_epe)) if all_avg_epe else 0.0,
            "of_avg_r2_st": float(np.mean(all_avg_r2)) if all_avg_r2 else 0.0,
            "of_time_length_st": float(len(flows)),
        }

    def _segment_ranges(self, length: int, segments: int) -> List[Tuple[int, int]]:
        if length == 0:
            return []
        segments = min(segments, length)
        indices = np.array_split(np.arange(length), segments)
        ranges = []
        for idx in indices:
            if idx.size == 0:
                continue
            ranges.append((int(idx[0]), int(idx[-1] + 1)))
        return ranges

    def _compute_tracking_metrics(
        self,
        tracks: Sequence[Dict[str, Any]],
        ranges: Sequence[Tuple[int, int]],
        dynamic_threshold: float,
    ) -> Dict[str, float]:
        ta_metrics: Dict[str, float] = {}
        ta_values: List[float] = []
        ta_dyn_values: List[float] = []
        pt_values: List[float] = []
        pt_dyn_values: List[float] = []

        for idx, (start_frame, end_frame) in enumerate(ranges, start=1):
            segment_length = max(end_frame - start_frame, 1)
            total_tracks = 0
            track_lengths: List[float] = []
            dynamic_tracks = 0
            dynamic_displacements: List[float] = []

            for track in tracks:
                frames = np.array(track["frames"])
                mask = (frames >= start_frame) & (frames <= end_frame)
                if np.sum(mask) < 2:
                    continue

                total_tracks += 1
                positions = np.array(track["positions"])[mask]
                displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                avg_speed = float(np.mean(displacements)) if displacements.size else 0.0
                track_lengths.append(avg_speed)
                if avg_speed > dynamic_threshold:
                    dynamic_tracks += 1
                    dynamic_displacements.append(avg_speed)

            if total_tracks == 0:
                ta_value = 0.0
                ta_dyn_value = 0.0
                pt_value = 0.0
                pt_dyn_value = 0.0
            else:
                ta_value = float(min(np.mean(track_lengths) / (dynamic_threshold + 1e-6), 1.0)) if track_lengths else 0.0
                ta_dyn_value = float(1.0 - (dynamic_tracks / total_tracks))
                pt_value = float(np.mean(track_lengths) / segment_length) if track_lengths else 0.0
                pt_dyn_value = (
                    float(np.mean(dynamic_displacements) / segment_length)
                    if dynamic_displacements
                    else 0.0
                )

            ta_metrics[f"of_ta_IM0{idx}"] = ta_value
            ta_metrics[f"of_ta_IM0{idx}_Dyn"] = ta_dyn_value
            ta_metrics[f"of_pt_IM0{idx}"] = pt_value
            ta_metrics[f"of_pt_IM0{idx}_Dyn"] = pt_dyn_value

            ta_values.append(ta_value)
            ta_dyn_values.append(ta_dyn_value)
            pt_values.append(pt_value)
            pt_dyn_values.append(pt_dyn_value)

        ta_metrics["of_ta_average"] = float(np.mean(ta_values)) if ta_values else 0.0
        ta_metrics["of_pt_average"] = float(np.mean(pt_values)) if pt_values else 0.0

        # Ensure all expected keys exist even if segments fewer than five
        for idx in range(len(ranges) + 1, self.config.ta_segments + 1):
            ta_metrics.setdefault(f"of_ta_IM0{idx}", 0.0)
            ta_metrics.setdefault(f"of_ta_IM0{idx}_Dyn", 0.0)
            ta_metrics.setdefault(f"of_pt_IM0{idx}", 0.0)
            ta_metrics.setdefault(f"of_pt_IM0{idx}_Dyn", 0.0)

        return ta_metrics

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        if not getattr(self, "available", False):
            return self._dummy_features(video_path)

        frames_rgb, frames_gray = self._extract_frames(video_path)
        masks = self._segment_frames(frames_rgb[:-1])  # segmentation aligned with flow source frames

        flows = self._compute_flows(frames_rgb)

        short_term = self._compute_short_term_metrics(flows, masks, self.config.dynamic_threshold)
        tracks = self._compute_tracks(frames_gray)
        ranges = self._segment_ranges(len(flows), self.config.ta_segments)
        tracking_metrics = self._compute_tracking_metrics(tracks, ranges, self.config.dynamic_threshold)

        result = {**short_term, **tracking_metrics}
        result.update(
            {
                "of_total_tracks": float(len(tracks)),
                "of_video_path": str(video_path),
            }
        )
        return result

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        features = self.analyze_video(video_path)
        return {
            "CrowdFlow": {
                "description": "CrowdFlow optical flow, trajectories, and tracking analysis",
                "features": features,
            }
        }
    
    def _dummy_features(self, video_path: str) -> Dict[str, Any]:
        return {
            "CrowdFlow_available": False,
            "CrowdFlow_mean_motion": 0.0,
            "CrowdFlow_dynamic_ratio": 0.0,
            "CrowdFlow_unavailable_reason": self.unavailable_reason,
    }



def extract_crowdflow_features(video_path: str, device: str = "cpu") -> Dict[str, Any]:
    analyzer = CrowdFlowAnalyzer(device=device)
    return analyzer.get_feature_dict(video_path)
