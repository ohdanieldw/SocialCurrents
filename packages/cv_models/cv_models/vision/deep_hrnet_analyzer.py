"""Deep HRNet analyzer powered by the official upstream repository."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from cv_models.external.repo_manager import ensure_repo

logger = logging.getLogger(__name__)


def _append_to_syspath(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


@dataclass(frozen=True)
class HRNetResources:
    repo_root: Path
    config_path: Path
    checkpoint_path: Path


class DeepHRNetAnalyzer:
    """Run pose estimation with Deep HRNet weights and decode confidences."""

    DEFAULT_CONFIG = "experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml"
    DEFAULT_CHECKPOINT = "models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth"

    def __init__(
        self,
        *,
        device: str = "cpu",
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        confidence_threshold: float = 0.2,
        max_frames: int = 180,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested for Deep HRNet but CUDA is unavailable; defaulting to CPU.")

        self.confidence_threshold = float(confidence_threshold)
        self.max_frames = int(max_frames)
        self.resources = self._resolve_resources(config_path, checkpoint_path)

        self.cfg: Any = None
        self.model: Optional[torch.nn.Module] = None
        self.transform: Optional[transforms.Compose] = None
        self.image_size: np.ndarray | None = None
        self.heatmap_size: np.ndarray | None = None
        self.pixel_std = 200.0

        self.body_parts = {
            "Head": [0, 1, 2, 3, 4],
            "Shoulder": [5, 6],
            "Elbow": [7, 8],
            "Wrist": [9, 10],
            "Hip": [11, 12],
            "Knee": [13, 14],
            "Ankle": [15, 16],
        }

        self.default_metrics: Dict[str, Any] = {
            **{f"DHiR_{part}": 0.0 for part in self.body_parts},
            "DHiR_Mean": 0.0,
            "DHiR_Meanat0.1": 0.0,
            "DHiR_AP": 0.0,
            "DHiR_AP_5": 0.0,
            "DHiR_AP_75": 0.0,
            "DHiR_AP_M": 0.0,
            "DHiR_AP_L": 0.0,
            "DHiR_AR": 0.0,
            "DHiR_AR_5": 0.0,
            "DHiR_AR_75": 0.0,
            "DHiR_AR_M": 0.0,
            "DHiR_AR_L": 0.0,
        }

        self._initialize_model()

    def _resolve_resources(self, config_override: Optional[str], checkpoint_override: Optional[str]) -> HRNetResources:
        repo_root = ensure_repo("hrnet_pose")
        _append_to_syspath(repo_root)
        _append_to_syspath(repo_root / "lib")

        config = Path(
            config_override
            or os.getenv("HRNET_CONFIG_PATH")
            or repo_root / self.DEFAULT_CONFIG
        ).expanduser()
        checkpoint = Path(
            checkpoint_override
            or os.getenv("HRNET_CHECKPOINT_PATH")
            or repo_root / self.DEFAULT_CHECKPOINT
        ).expanduser()

        if not config.exists():
            raise FileNotFoundError(
                "Deep HRNet config not found. Provide HRNET_CONFIG_PATH or place the default YAML in the cloned repo.\n"
                f"Missing path: {config}"
            )
        if not checkpoint.exists():
            raise FileNotFoundError(
                "Deep HRNet weights not found. Provide HRNET_CHECKPOINT_PATH or download the official checkpoint.\n"
                f"Missing path: {checkpoint}"
            )

        return HRNetResources(repo_root=repo_root, config_path=config, checkpoint_path=checkpoint)

    def _initialize_model(self) -> None:
        try:
            from lib.config import cfg as base_cfg
            from lib.config import update_config
            from lib.models import pose_hrnet
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "DeepHRNetAnalyzer requires the upstream HRNet repo; ensure dependencies like yacs are installed."
            ) from exc

        args = SimpleNamespace(cfg=str(self.resources.config_path), opts=[])
        cfg = base_cfg.clone()
        cfg.defrost()
        update_config(cfg, args)
        cfg.freeze()

        model = pose_hrnet.get_pose_net(cfg, is_train=False)
        state_dict = torch.load(str(self.resources.checkpoint_path), map_location=self.device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "model" in state_dict:
            state_dict = state_dict["model"]
        cleaned_state = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
        if missing:
            logger.warning("Deep HRNet missing weights: %s", sorted(missing))
        if unexpected:
            logger.warning("Deep HRNet unexpected weights: %s", sorted(unexpected))

        model = model.to(self.device)
        model.eval()

        self.cfg = cfg
        self.model = model
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE, dtype=np.float32)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE, dtype=np.float32)

        mean = cfg.DATASET.MEAN if hasattr(cfg, "DATASET") else [0.485, 0.456, 0.406]
        std = cfg.DATASET.STD if hasattr(cfg, "DATASET") else [0.229, 0.224, 0.225]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        logger.info(
            "Deep HRNet initialized with %s and weights %s",
            self.resources.config_path,
            self.resources.checkpoint_path,
        )

    def _preprocess(self, frame: np.ndarray) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        if self.transform is None or self.image_size is None:
            raise RuntimeError("Deep HRNet transform uninitialized")

        h, w, _ = frame.shape
        center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
        scale = np.array([w / self.pixel_std, h / self.pixel_std], dtype=np.float32)

        from lib.utils.transforms import get_affine_transform

        trans = get_affine_transform(center, scale, 0.0, self.image_size)
        input_img = cv2.warpAffine(frame, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        return tensor, center, scale

    def _infer(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.heatmap_size is None:
            raise RuntimeError("Deep HRNet model uninitialized")

        input_tensor, center, scale = self._preprocess(frame)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        if isinstance(outputs, (list, tuple)):
            outputs = outputs[-1]

        heatmaps = outputs.detach().cpu().numpy()

        from lib.core.inference import get_max_preds
        from lib.utils.transforms import transform_preds

        preds, maxvals = get_max_preds(heatmaps)

        coords = transform_preds(preds[0], center, scale, self.heatmap_size)
        confidences = maxvals[0].reshape(-1)
        return coords, confidences

    def _body_part_metrics(self, confidences: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for part, indices in self.body_parts.items():
            if not indices:
                metrics[f"DHiR_{part}"] = 0.0
                continue
            subset = confidences[np.array(indices)]
            metrics[f"DHiR_{part}"] = float(np.mean(subset)) if subset.size else 0.0
        return metrics

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Deep HRNet model not loaded")

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.max_frames
        indices = np.linspace(0, frame_total - 1, min(frame_total, self.max_frames), dtype=int)

        confidences_per_frame: List[np.ndarray] = []
        detection_mask: List[bool] = []

        try:
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok:
                    continue
                try:
                    _, confidences = self._infer(frame)
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning("Skipping frame %s due to HRNet failure: %s", frame_idx, exc)
                    continue

                confidences_per_frame.append(confidences)
                detection_mask.append(bool(np.any(confidences > self.confidence_threshold)))
        finally:
            cap.release()

        if not confidences_per_frame:
            logger.warning("Deep HRNet produced no detections for %s", video_path)
            result = self.default_metrics.copy()
            result.update(
                {
                    "video_path": str(path),
                    "total_frames": 0,
                    "pose_detected_frames": 0,
                    "detection_rate": 0.0,
                    "avg_keypoints_per_frame": 0.0,
                }
            )
            return result

        stacked = np.stack(confidences_per_frame)
        part_metrics = self._body_part_metrics(np.mean(stacked, axis=0))

        body_part_values = list(part_metrics.values())
        mean_conf = float(np.mean(body_part_values)) if body_part_values else 0.0
        mean_over_01 = float(np.mean([v for v in body_part_values if v > 0.1])) if any(v > 0.1 for v in body_part_values) else 0.0

        all_conf = stacked.reshape(-1)
        ap = float(np.mean(all_conf))
        ap_5 = float(np.mean(all_conf > 0.5))
        ap_75 = float(np.mean(all_conf > 0.75))
        ap_m = float(np.mean(all_conf[(all_conf > 0.3) & (all_conf <= 0.7)])) if np.any((all_conf > 0.3) & (all_conf <= 0.7)) else 0.0
        ap_l = float(np.mean(all_conf[all_conf > 0.7])) if np.any(all_conf > 0.7) else 0.0

        frame_max = stacked.max(axis=1)
        ar = float(np.mean(detection_mask))
        ar_5 = float(np.mean(frame_max > 0.5))
        ar_75 = float(np.mean(frame_max > 0.75))
        medium_mask = (frame_max > 0.3) & (frame_max <= 0.7)
        ar_m = float(np.mean(frame_max[medium_mask])) if np.any(medium_mask) else 0.0
        high_mask = frame_max > 0.7
        ar_l = float(np.mean(frame_max[high_mask])) if np.any(high_mask) else 0.0

        avg_visible = float(np.mean((stacked > self.confidence_threshold).sum(axis=1)))

        result: Dict[str, Any] = {
            **self.default_metrics,
            **part_metrics,
            "DHiR_Mean": mean_conf,
            "DHiR_Meanat0.1": mean_over_01,
            "DHiR_AP": ap,
            "DHiR_AP_5": ap_5,
            "DHiR_AP_75": ap_75,
            "DHiR_AP_M": ap_m,
            "DHiR_AP_L": ap_l,
            "DHiR_AR": ar,
            "DHiR_AR_5": ar_5,
            "DHiR_AR_75": ar_75,
            "DHiR_AR_M": ar_m,
            "DHiR_AR_L": ar_l,
            "video_path": str(path),
            "total_frames": len(confidences_per_frame),
            "pose_detected_frames": int(np.sum(detection_mask)),
            "detection_rate": float(np.mean(detection_mask)),
            "avg_keypoints_per_frame": avg_visible,
        }
        return result

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        features = self.analyze_video(video_path)
        return {
            "Deep HRNet Pose": {
                "description": "Deep High-Resolution Network pose estimation",
                "features": features,
            }
        }


def extract_deep_hrnet_features(video_path: str, device: str = "cpu") -> Dict[str, Any]:
    analyzer = DeepHRNetAnalyzer(device=device)
    return analyzer.get_feature_dict(video_path)
