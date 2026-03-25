"""ViTPose analyzer wired to the official MMPose implementation."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cv_models.external.repo_manager import ensure_repo

logger = logging.getLogger(__name__)


def _append_to_syspath(path: Path) -> None:
    """Insert a repository path into ``sys.path`` if required."""

    if not path.exists():
        return
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


class ViTPoseAnalyzer:
    """Run ViTPose inference through the upstream MMPose pipeline."""

    DEFAULT_CONFIG = Path(
        "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_simple_coco_256x192.py"
    )
    DEFAULT_CHECKPOINT = Path("checkpoints/ViTPose_base_simple_coco_256x192.pth")
    JOINT_SCORE_THRESHOLD = 0.5

    def __init__(
        self,
        device: str = "cpu",
        *,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        max_frames: Optional[int] = 120,
    ) -> None:
        self.device = device
        self.max_frames = max_frames

        repo_root = ensure_repo("vitpose")
        _append_to_syspath(repo_root)
        _append_to_syspath(repo_root / "mmpose")
        _append_to_syspath(repo_root / "mmcv_custom")

        self.repo_root = repo_root
        user_config = config_path or os.getenv("VITPOSE_CONFIG")
        user_checkpoint = checkpoint_path or os.getenv("VITPOSE_CHECKPOINT")

        self.config_path = Path(user_config).expanduser() if user_config else repo_root / self.DEFAULT_CONFIG
        self.checkpoint_path = (
            Path(user_checkpoint).expanduser() if user_checkpoint else repo_root / self.DEFAULT_CHECKPOINT
        )

        self.pose_model: Any = None
        self.dataset_name: Optional[str] = None
        self.dataset_info: Any = None
        self.num_keypoints: int = 0
        self._inference_fn = None
        self._metric_template = {
            "vit_AR": 0.0,
            "vit_AP": 0.0,
            "vit_AU": 1.0,
            "vit_mean": 0.0,
            "vit_frames": 0,
            "vit_active_joints": 0.0,
            "vit_video_path": "",
            "vit_fps": 0.0,
            "vit_frame_size": (0, 0),
        }

        self._last_keypoints: Optional[Tuple[str, Dict[str, Any]]] = None

        self._initialize_model()

    def _initialize_model(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(
                "ViTPose config not found. Set VITPOSE_CONFIG or pass config_path with the upstream config file.\n"
                f"Missing path: {self.config_path}"
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "ViTPose checkpoint not found. Download the official weights (see the ViTPose repo README) "
                "and set VITPOSE_CHECKPOINT or pass checkpoint_path.\n"
                f"Missing path: {self.checkpoint_path}"
            )

        try:
            from mmpose.apis import inference_top_down_pose_model, init_pose_model
            from mmpose.datasets import DatasetInfo
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ViTPoseAnalyzer requires mmpose and compatible mmcv/mmengine builds. "
                "Install them via pip using the versions recommended by the ViTPose repository."
            ) from exc

        logger.info("Loading ViTPose model from %s", self.checkpoint_path)
        self.pose_model = init_pose_model(
            str(self.config_path),
            str(self.checkpoint_path),
            device=self.device,
        )
        self._inference_fn = inference_top_down_pose_model

        cfg = self.pose_model.cfg
        self.dataset_name = cfg.data["test"]["type"]
        dataset_meta = cfg.data["test"].get("dataset_info")
        self.dataset_info = DatasetInfo(dataset_meta) if dataset_meta else None

        if self.dataset_info and getattr(self.dataset_info, "keypoint_name2id", None):
            self.num_keypoints = len(self.dataset_info.keypoint_name2id)
        else:  # fall back to model config when dataset info is missing
            head_cfg = getattr(cfg, "model", {}).get("keypoint_head", {})
            self.num_keypoints = int(head_cfg.get("out_channels", 0)) or 17

        if self.num_keypoints <= 0:
            self.num_keypoints = 17

    def _infer_frame(self, frame: np.ndarray) -> Sequence[Dict[str, Any]]:
        if self.pose_model is None or self._inference_fn is None:
            raise RuntimeError("ViTPose model not initialised")

        height, width = frame.shape[:2]
        person_results = [{"bbox": np.array([0.0, 0.0, float(width - 1), float(height - 1)])}]
        pose_results, _ = self._inference_fn(
            self.pose_model,
            frame,
            person_results,
            bbox_thr=None,
            format="xyxy",
            dataset=self.dataset_name,
            dataset_info=self.dataset_info,
            return_heatmap=False,
            outputs=None,
        )
        return pose_results

    @staticmethod
    def _select_instance(pose_results: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not pose_results:
            return None

        def _score(entry: Dict[str, Any]) -> float:
            if "score" in entry:
                return float(entry["score"])
            if "bbox_score" in entry:
                return float(entry["bbox_score"])
            keypoints = entry.get("keypoints")
            if keypoints is None:
                return 0.0
            return float(np.mean(np.asarray(keypoints)[:, 2]))

        return max(pose_results, key=_score)

    def _extract_instance_keypoints(self, frame: np.ndarray) -> np.ndarray:
        keypoints_buffer = np.zeros((self.num_keypoints, 3), dtype=np.float32)

        try:
            pose_results = self._infer_frame(frame)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("ViTPose inference failed on frame due to error: %s", exc)
            return keypoints_buffer

        instance = self._select_instance(pose_results)
        if instance is None:
            return keypoints_buffer

        keypoints = np.asarray(instance.get("keypoints", []), dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[1] < 3:
            return keypoints_buffer

        usable = min(self.num_keypoints, keypoints.shape[0])
        keypoints_buffer[:usable] = keypoints[:usable, :3]
        return keypoints_buffer

    def _collect_keypoints(self, video_file: Path) -> Dict[str, Any]:
        cache_key = str(video_file)
        if self._last_keypoints and self._last_keypoints[0] == cache_key:
            return self._last_keypoints[1]

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_file}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        keypoints_list: List[np.ndarray] = []
        frame_indices: List[int] = []

        def _update_dims(frame: np.ndarray) -> None:
            nonlocal frame_width, frame_height
            if frame_width == 0 or frame_height == 0:
                frame_height = int(frame.shape[0])
                frame_width = int(frame.shape[1])

        sequential = (
            self.max_frames is None
            or frame_total <= 0
            or (self.max_frames is not None and frame_total <= self.max_frames)
        )

        if sequential:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                _update_dims(frame)
                keypoints_list.append(self._extract_instance_keypoints(frame))
                frame_indices.append(idx)
                idx += 1
        else:
            sample_count = int(self.max_frames) if self.max_frames else frame_total
            sample_count = max(sample_count, 1)
            indices = np.linspace(0, max(frame_total - 1, 0), sample_count, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok and frame is not None:
                    _update_dims(frame)
                    keypoints = self._extract_instance_keypoints(frame)
                else:
                    keypoints = np.zeros((self.num_keypoints, 3), dtype=np.float32)
                keypoints_list.append(keypoints)
                frame_indices.append(int(idx))

        cap.release()

        if keypoints_list:
            keypoints_array = np.stack(keypoints_list, axis=0)
        else:
            keypoints_array = np.zeros((0, self.num_keypoints, 3), dtype=np.float32)

        data = {
            "keypoints": keypoints_array,
            "frame_indices": np.asarray(frame_indices, dtype=np.int32),
            "fps": fps,
            "frame_size": (frame_width, frame_height),
        }

        self._last_keypoints = (cache_key, data)
        return data

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")

        data = self._collect_keypoints(video_file)

        metrics = self._metric_template.copy()
        metrics["vit_video_path"] = str(video_file)
        metrics["vit_frames"] = int(data["keypoints"].shape[0])
        metrics["vit_fps"] = float(data["fps"])
        metrics["vit_frame_size"] = tuple(int(x) for x in data["frame_size"])

        if data["keypoints"].size == 0:
            logger.warning("ViTPose analyzer produced no frames for %s", video_file)
            return metrics

        score_matrix = data["keypoints"][..., 2]
        if score_matrix.size == 0:
            return metrics

        visibility = score_matrix
        vit_ap = float(np.mean(visibility))
        vit_ar = float(np.mean(visibility >= self.JOINT_SCORE_THRESHOLD))
        if visibility.shape[0] > 1:
            temporal_grad = np.diff(visibility, axis=0)
            vit_au = float(np.clip(np.mean(np.abs(temporal_grad)), 0.0, 1.0))
        else:
            vit_au = float(np.clip(np.mean(1.0 - visibility), 0.0, 1.0))

        active_per_frame = np.sum(visibility >= self.JOINT_SCORE_THRESHOLD, axis=1)
        vit_active = float(np.mean(active_per_frame))

        metrics.update(
            {
                "vit_AR": vit_ar,
                "vit_AP": vit_ap,
                "vit_AU": vit_au,
                "vit_mean": float(np.mean([vit_ar, vit_ap, 1.0 - vit_au])),
                "vit_active_joints": vit_active,
            }
        )
        return metrics

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        return {
            "ViTPose": {
                "description": "Vision Transformer based pose estimation",
                "features": self.analyze_video(video_path),
            }
        }

    def extract_keypoints(self, video_path: str) -> Dict[str, Any]:
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")

        return self._collect_keypoints(video_file)


def create_vitpose_analyzer(
    *,
    device: str = "cpu",
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    max_frames: Optional[int] = 120,
) -> ViTPoseAnalyzer:
    return ViTPoseAnalyzer(
        device=device,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        max_frames=max_frames,
    )
