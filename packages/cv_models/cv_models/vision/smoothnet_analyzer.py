"""SmoothNet analyzer backed by the official implementation.

This module hooks the upstream SmoothNet temporal refinement network onto
keypoints estimated by ViTPose. The analyzer relies on the shared
``repo_manager`` helper to ensure the original project is available locally.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

from cv_models.external.repo_manager import ensure_repo
from cv_models.vision.vitpose_analyzer import ViTPoseAnalyzer

logger = logging.getLogger(__name__)


def _append_to_syspath(path: Path) -> None:
    if not path.exists():
        return
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


class SmoothNetAnalyzer:
    """Run the official SmoothNet model on top of ViTPose keypoints."""

    DEFAULT_CONFIG = Path("configs/h36m_fcn_3D.yaml")
    DEFAULT_CHECKPOINT = Path("checkpoints/smoothnet_h36m_fcn_window32.pth")

    def __init__(
        self,
        device: str = "cpu",
        *,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        vitpose_config: Optional[str] = None,
        vitpose_checkpoint: Optional[str] = None,
        stride: int = 1,
        output_dir: Optional[Path] = None,
    ) -> None:
        repo_root = ensure_repo("smoothnet")
        _append_to_syspath(repo_root)
        _append_to_syspath(repo_root / "lib")

        try:
            from lib.models.smoothnet import SmoothNet  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "SmoothNetAnalyzer requires the official SmoothNet repository dependencies."
            ) from exc

        config_env = config_path or os.getenv("SMOOTHNET_CONFIG")
        checkpoint_env = checkpoint_path or os.getenv("SMOOTHNET_CHECKPOINT")

        self.config_path = Path(config_env).expanduser() if config_env else repo_root / self.DEFAULT_CONFIG
        if not self.config_path.exists():
            fallback_config = next(
                (
                    p
                    for p in (repo_root / "configs").rglob("*.yaml")
                    if "h36m" in p.name.lower() or "smoothnet" in p.name.lower()
                ),
                None,
            )
            if fallback_config is not None:
                logger.info("SmoothNet config fallback resolved to %s", fallback_config)
                self.config_path = fallback_config

        self.checkpoint_path = (
            Path(checkpoint_env).expanduser() if checkpoint_env else repo_root / self.DEFAULT_CHECKPOINT
        )
        if not self.checkpoint_path.exists():
            fallback_checkpoint = next(
                (p for p in repo_root.rglob("*.pth") if "smooth" in p.name.lower()),
                None,
            )
            if fallback_checkpoint is not None:
                logger.info("SmoothNet checkpoint fallback resolved to %s", fallback_checkpoint)
                self.checkpoint_path = fallback_checkpoint

        if not self.config_path.exists():
            raise FileNotFoundError(
                "SmoothNet config not found. Set SMOOTHNET_CONFIG or pass config_path.\n"
                f"Missing path: {self.config_path}"
            )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "SmoothNet checkpoint not found. Download a pretrained model from the official repository "
                "(see README) and set SMOOTHNET_CHECKPOINT or pass checkpoint_path.\n"
                f"Missing path: {self.checkpoint_path}"
            )

        config_dict = self._load_config(self.config_path)
        model_cfg = config_dict.get("MODEL", {})
        eval_cfg = config_dict.get("EVALUATE", {})

        self.window_size = int(model_cfg.get("SLIDE_WINDOW_SIZE", 32))
        self.output_size = int(model_cfg.get("SLIDE_WINDOW_SIZE", self.window_size))
        self.hidden_size = int(model_cfg.get("HIDDEN_SIZE", 512))
        self.res_hidden_size = int(model_cfg.get("RES_HIDDEN_SIZE", 256))
        self.num_blocks = int(model_cfg.get("NUM_BLOCK", 3))
        self.dropout = float(model_cfg.get("DROPOUT", 0.5))
        self.stride = max(1, int(eval_cfg.get("SLIDE_WINDOW_STEP_SIZE", stride)))

        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested for SmoothNet but not available; falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

        self.model = SmoothNet(
            window_size=self.window_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            res_hidden_size=self.res_hidden_size,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
        ).to(self.device)

        self._load_weights()
        self.model.eval()

        self.pose_extractor = ViTPoseAnalyzer(
            device=device,
            config_path=vitpose_config,
            checkpoint_path=vitpose_checkpoint,
            max_frames=None,
        )

        default_output = Path.cwd() / "output" / "vision" / "smoothnet"
        self.output_dir = Path(output_dir) if output_dir else default_output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.repo_root = repo_root

        self.default_metrics: Dict[str, Any] = {
            "net_3d_estimator": 0.0,
            "net_3d_MPJPE_input_ad": 0.0,
            "net_3d_MPJPE_output_ad": 0.0,
            "net_3d_Accel_input_ad": 0.0,
            "net_3d_Accel_output_ad": 0.0,
            "net_2d_estimator": 0.0,
            "net_2d_MPJPE_input_ad": 0.0,
            "net_2d_MPJPE_output_ad": 0.0,
            "net_2d_Accel_input_ad": 0.0,
            "net_2d_Accel_output_ad": 0.0,
            "net_SMPL_estimator": 0.0,
            "net_SMPL_MPJPE_input_ad": 0.0,
            "net_SMPL_MPJPE_output_ad": 0.0,
            "net_SMPL_Accel_input_ad": 0.0,
            "net_SMPL_Accel_output_ad": 0.0,
            "net_joint_confidence": 0.0,
            "net_pose_stability": 0.0,
            "net_motion_coherence": 0.0,
            "net_temporal_consistency": 0.0,
            "net_tracking_accuracy": 0.0,
            "net_keypoint_variance": 0.0,
            "net_smoothness_score": 0.0,
            "smoothnet_frames": 0,
            "smoothnet_valid_ratio": 0.0,
            "smoothnet_mean_adjustment": 0.0,
            "smoothnet_keypoints_path": "",
            "smoothnet_video_path": "",
            "smoothnet_active_joints": 0.0,
            "smoothnet_error": "",
        }

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _load_weights(self) -> None:
        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
            cleaned[new_key] = value
        self.model.load_state_dict(cleaned, strict=True)
        logger.info("Loaded SmoothNet weights from %s", self.checkpoint_path)

    @staticmethod
    def _prepare_sequence(
        keypoints: np.ndarray,
        frame_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        coords = keypoints[..., :2].astype(np.float32)
        scores = keypoints[..., 2].astype(np.float32)

        width, height = frame_size
        if width > 0 and height > 0:
            safe_w = max(float(width), 1.0)
            safe_h = max(float(height), 1.0)
            coords[..., 0] /= safe_w
            coords[..., 1] /= safe_h
            norm_meta: Dict[str, float] = {"mode": "image", "width": safe_w, "height": safe_h}
        else:
            denom = float(np.max(np.abs(coords)))
            denom = denom if denom > 0 else 1.0
            coords /= denom
            norm_meta = {"mode": "scale", "scale": denom}

        coords = coords * 2.0 - 1.0
        flattened = coords.reshape(coords.shape[0], -1)
        return flattened, scores, norm_meta

    @staticmethod
    def _restore_coordinates(
        sequence: np.ndarray,
        norm_meta: Dict[str, float],
        joints: int,
    ) -> np.ndarray:
        coords = sequence.reshape(sequence.shape[0], joints, 2)
        coords = (coords + 1.0) * 0.5
        if norm_meta.get("mode") == "image":
            coords[..., 0] *= norm_meta.get("width", 1.0)
            coords[..., 1] *= norm_meta.get("height", 1.0)
        else:
            coords *= norm_meta.get("scale", 1.0)
        return coords

    def _run_model(self, batch: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(batch.astype(np.float32)).to(self.device)
        tensor = tensor.permute(0, 2, 1)
        with torch.no_grad():
            smoothed = self.model(tensor)
        return smoothed.permute(0, 2, 1).cpu().numpy()

    def _smooth_sequence(self, sequence: np.ndarray) -> np.ndarray:
        frames, dims = sequence.shape
        if frames == 0:
            return sequence

        if frames <= self.window_size:
            pad_len = max(self.window_size - frames, 0)
            if pad_len > 0:
                pad_values = np.repeat(sequence[-1:, :], pad_len, axis=0)
                padded = np.concatenate([sequence, pad_values], axis=0)
            else:
                padded = sequence
            smoothed = self._run_model(padded[np.newaxis, :, :])[0]
            return smoothed[:frames]

        output = np.zeros((frames, dims), dtype=np.float32)
        counts = np.zeros((frames, 1), dtype=np.float32)

        last_start = frames - self.window_size
        starts = list(range(0, last_start + 1, self.stride))
        if starts[-1] != last_start:
            starts.append(last_start)

        for start in starts:
            window = sequence[start : start + self.window_size]
            smoothed = self._run_model(window[np.newaxis, :, :])[0]
            span = min(self.output_size, smoothed.shape[0])
            output[start : start + span] += smoothed[:span]
            counts[start : start + span] += 1

        counts[counts == 0] = 1
        return output / counts

    @staticmethod
    def _mean_velocity(sequence: np.ndarray) -> float:
        if sequence.shape[0] < 2:
            return 0.0
        diffs = np.diff(sequence, axis=0)
        return float(np.mean(np.linalg.norm(diffs, axis=1)))

    @staticmethod
    def _mean_acceleration(sequence: np.ndarray) -> float:
        if sequence.shape[0] < 3:
            return 0.0
        velocities = np.diff(sequence, axis=0)
        accel = np.diff(velocities, axis=0)
        return float(np.mean(np.linalg.norm(accel, axis=1)))

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")

        pose_data = self.pose_extractor.extract_keypoints(str(video_file))
        keypoints = pose_data["keypoints"]
        if keypoints.size == 0:
            logger.warning("SmoothNet received no keypoints for %s", video_file)
            metrics = self.default_metrics.copy()
            metrics["smoothnet_video_path"] = str(video_file)
            return metrics

        sequence, scores, norm_meta = self._prepare_sequence(keypoints, pose_data["frame_size"])
        smoothed_sequence = self._smooth_sequence(sequence)
        smoothed_coords = self._restore_coordinates(smoothed_sequence, norm_meta, keypoints.shape[1])

        velocity_before = self._mean_velocity(sequence)
        velocity_after = self._mean_velocity(smoothed_sequence)
        accel_before = self._mean_acceleration(sequence)
        accel_after = self._mean_acceleration(smoothed_sequence)
        adjustment = sequence - smoothed_sequence
        mean_adjustment = float(np.mean(np.linalg.norm(adjustment, axis=1))) if adjustment.size else 0.0

        valid_frames = np.any(scores > 0, axis=1)
        valid_ratio = float(np.mean(valid_frames)) if valid_frames.size else 0.0
        joint_confidence = float(np.mean(scores)) if scores.size else 0.0
        active_joints = (
            float(
                np.mean(
                    np.sum(scores >= self.pose_extractor.JOINT_SCORE_THRESHOLD, axis=1)
                )
            )
            if scores.size
            else 0.0
        )
        variance = (
            float(np.mean(np.var(sequence.reshape(sequence.shape[0], -1, 2), axis=0)))
            if sequence.size
            else 0.0
        )

        stability = float(np.exp(-velocity_after))
        motion_coherence = 1.0 / (1.0 + max(velocity_before, 1e-6))
        temporal_consistency = 1.0 / (1.0 + abs(velocity_before - velocity_after))
        tracking_accuracy = valid_ratio

        metrics = self.default_metrics.copy()
        metrics.update(
            {
                "net_3d_estimator": 1.0 / (1.0 + mean_adjustment),
                "net_3d_MPJPE_input_ad": velocity_before,
                "net_3d_MPJPE_output_ad": velocity_after,
                "net_3d_Accel_input_ad": accel_before,
                "net_3d_Accel_output_ad": accel_after,
                "net_2d_estimator": 1.0 / (1.0 + velocity_after),
                "net_2d_MPJPE_input_ad": velocity_before,
                "net_2d_MPJPE_output_ad": velocity_after,
                "net_2d_Accel_input_ad": accel_before,
                "net_2d_Accel_output_ad": accel_after,
                "net_SMPL_estimator": 1.0 / (1.0 + mean_adjustment),
                "net_SMPL_MPJPE_input_ad": velocity_before,
                "net_SMPL_MPJPE_output_ad": velocity_after,
                "net_SMPL_Accel_input_ad": accel_before,
                "net_SMPL_Accel_output_ad": accel_after,
                "net_joint_confidence": joint_confidence,
                "net_pose_stability": stability,
                "net_motion_coherence": motion_coherence,
                "net_temporal_consistency": temporal_consistency,
                "net_tracking_accuracy": tracking_accuracy,
                "net_keypoint_variance": variance,
                "net_smoothness_score": stability,
                "smoothnet_frames": int(sequence.shape[0]),
                "smoothnet_valid_ratio": valid_ratio,
                "smoothnet_mean_adjustment": mean_adjustment,
                "smoothnet_video_path": str(video_file),
                "smoothnet_active_joints": active_joints,
            }
        )

        keypoint_artifact = self.output_dir / f"{video_file.stem}_smoothnet_keypoints.npz"
        np.savez_compressed(
            keypoint_artifact,
            raw_keypoints=keypoints,
            smoothed_keypoints=smoothed_coords,
            scores=scores,
            frame_indices=pose_data["frame_indices"],
            fps=pose_data["fps"],
            frame_size=pose_data["frame_size"],
            normalization=norm_meta,
        )
        metrics["smoothnet_keypoints_path"] = str(keypoint_artifact)

        return metrics

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        try:
            features = self.analyze_video(video_path)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.error("SmoothNet analysis failed for %s: %s", video_path, exc)
            features = self.default_metrics.copy()
            features["smoothnet_video_path"] = str(video_path)
            features["smoothnet_error"] = str(exc)

        return {
            "SmoothNet": {
                "description": "Temporal pose smoothing via SmoothNet",
                "features": features,
            }
        }


def extract_smoothnet_features(video_path: str, device: str = "cpu") -> Dict[str, Any]:
    analyzer = SmoothNetAnalyzer(device=device)
    return analyzer.get_feature_dict(video_path)


def create_smoothnet_analyzer(
    *,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
    config_path: Optional[str] = None,
    vitpose_config: Optional[str] = None,
    vitpose_checkpoint: Optional[str] = None,
    stride: int = 1,
    output_dir: Optional[Path] = None,
) -> SmoothNetAnalyzer:
    return SmoothNetAnalyzer(
        device=device,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        vitpose_config=vitpose_config,
        vitpose_checkpoint=vitpose_checkpoint,
        stride=stride,
        output_dir=output_dir,
    )
