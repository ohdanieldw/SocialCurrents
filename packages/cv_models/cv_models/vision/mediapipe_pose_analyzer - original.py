"""
Google MediaPipe Pose Estimation and Tracking (TASKS API)
Uses: mediapipe.tasks.python.vision.PoseLandmarker (no mediapipe.solutions)

This version is compatible with MediaPipe builds that do NOT ship `mediapipe.solutions`.
"""

import os
import io
import cv2
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:
    mp = None
    mp_python = None
    mp_vision = None

logger = logging.getLogger(__name__)


class MediaPipePoseAnalyzer:
    """
    MediaPipe analyzer for pose estimation and tracking.

    Outputs 33 pose landmarks:
      - Normalized landmarks: x,y,z,visibility,presence
      - World landmarks: x,y,z,visibility,presence (if provided by model)
    """

    def __init__(
        self,
        device: str = "cpu",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ):
        self.device = device  # MediaPipe tasks runs on CPU in this setup
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.initialized = False

        if mp is None or mp_python is None or mp_vision is None:
            raise ImportError(
                "mediapipe.tasks is not available. Please install mediapipe>=0.10.x."
            )

        # Resolve model path
        self.model_path = self._resolve_model_path(model_path)

        # Pose landmarker
        self.landmarker = None

        # 33 pose landmarks
        self.landmark_names = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]

        # Default metrics
        self.default_metrics: Dict[str, Any] = {}

        for i in range(1, 34):
            self.default_metrics[f"GMP_land_x_{i}"] = 0.0
            self.default_metrics[f"GMP_land_y_{i}"] = 0.0
            self.default_metrics[f"GMP_land_z_{i}"] = 0.0
            self.default_metrics[f"GMP_land_visi_{i}"] = 0.0
            self.default_metrics[f"GMP_land_presence_{i}"] = 0.0

        for i in range(1, 34):
            self.default_metrics[f"GMP_world_x_{i}"] = 0.0
            self.default_metrics[f"GMP_world_y_{i}"] = 0.0
            self.default_metrics[f"GMP_world_z_{i}"] = 0.0
            self.default_metrics[f"GMP_world_visi_{i}"] = 0.0
            self.default_metrics[f"GMP_world_presence_{i}"] = 0.0

        self.default_metrics["GMP_SM_pic"] = ""

    @staticmethod
    def _resolve_model_path(explicit_path: Optional[str]) -> str:
        """
        Resolve the Pose Landmarker .task file path.

        Priority:
          1) explicit arg model_path
          2) env var MEDIAPIPE_POSE_TASK_MODEL
          3) common repo asset locations
        """
        if explicit_path:
            p = Path(explicit_path)
            if p.exists():
                return str(p)

        envp = os.environ.get("MEDIAPIPE_POSE_TASK_MODEL")
        if envp:
            p = Path(envp)
            if p.exists():
                return str(p)

        # common locations in this repo
        candidates = [
            Path(__file__).resolve().parents[1] / "assets" / "pose_landmarker_full.task",
            Path(__file__).resolve().parents[1] / "assets" / "pose_landmarker_lite.task",
            Path(__file__).resolve().parents[1] / "assets" / "pose_landmarker_heavy.task",
        ]
        for c in candidates:
            if c.exists():
                return str(c)

        raise FileNotFoundError(
            "Pose Landmarker model (.task) not found.\n"
            "Set env var MEDIAPIPE_POSE_TASK_MODEL to a .task file path, e.g.\n"
            "  $env:MEDIAPIPE_POSE_TASK_MODEL='...\\pose_landmarker_full.task'\n"
        )

    def _initialize_model(self):
        if self.initialized:
            return

        logger.info("Initializing MediaPipe Pose Landmarker (tasks API)...")
        base = mp_python.BaseOptions(model_asset_path=self.model_path)

        options = mp_vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
        )

        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.initialized = True
        logger.info("MediaPipe Pose Landmarker initialized successfully")

    def _process_frame(
        self, frame_bgr: np.ndarray, timestamp_ms: int
    ) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
        if not self.initialized:
            self._initialize_model()

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run landmarker
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        metrics = self.default_metrics.copy()
        annotated_frame = None  # keep None to avoid dependencies on drawing_utils

        # result.pose_landmarks: list[ list[NormalizedLandmark] ]
        if getattr(result, "pose_landmarks", None):
            lm_list = result.pose_landmarks[0]  # first detected pose

            for i, landmark in enumerate(lm_list):
                idx = i + 1
                metrics[f"GMP_land_x_{idx}"] = float(getattr(landmark, "x", 0.0))
                metrics[f"GMP_land_y_{idx}"] = float(getattr(landmark, "y", 0.0))
                metrics[f"GMP_land_z_{idx}"] = float(getattr(landmark, "z", 0.0))
                metrics[f"GMP_land_visi_{idx}"] = float(getattr(landmark, "visibility", 0.0))
                metrics[f"GMP_land_presence_{idx}"] = float(getattr(landmark, "presence", 0.0))

        # result.pose_world_landmarks: list[ list[Landmark] ] (if model provides)
        if getattr(result, "pose_world_landmarks", None):
            wlm_list = result.pose_world_landmarks[0]
            for i, landmark in enumerate(wlm_list):
                idx = i + 1
                metrics[f"GMP_world_x_{idx}"] = float(getattr(landmark, "x", 0.0))
                metrics[f"GMP_world_y_{idx}"] = float(getattr(landmark, "y", 0.0))
                metrics[f"GMP_world_z_{idx}"] = float(getattr(landmark, "z", 0.0))
                metrics[f"GMP_world_visi_{idx}"] = float(getattr(landmark, "visibility", 0.0))
                metrics[f"GMP_world_presence_{idx}"] = float(getattr(landmark, "presence", 0.0))

        return metrics, annotated_frame

    def _encode_image_to_base64(self, image_bgr: np.ndarray) -> str:
        try:
            rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{image_base64}"
        except Exception as e:
            logger.warning(f"Failed to encode image to base64: {e}")
            return ""

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        if not self.initialized:
            self._initialize_model()

        logger.info(f"Analyzing pose landmarks in video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_metrics: List[Dict[str, Any]] = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0  # fallback

        best_frame = None
        max_landmarks_detected = 0

        try:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = int((frame_idx / fps) * 1000.0)

                metrics, annotated = self._process_frame(frame, timestamp_ms)

                landmarks_detected = sum(
                    1 for i in range(1, 34) if metrics.get(f"GMP_land_visi_{i}", 0.0) > 0.5
                )

                if landmarks_detected > max_landmarks_detected:
                    max_landmarks_detected = landmarks_detected
                    best_frame = annotated if annotated is not None else None

                metrics["frame_idx"] = frame_idx
                metrics["timestamp"] = frame_idx / fps

                frame_metrics.append(metrics)
                frame_idx += 1

                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames")

        finally:
            cap.release()

        logger.info(f"Completed pose analysis: {len(frame_metrics)} frames processed")
        return self._aggregate_results(frame_metrics, best_frame, video_path)

    def _aggregate_results(
        self,
        frame_metrics: List[Dict[str, Any]],
        best_frame: Optional[np.ndarray],
        video_path: str,
    ) -> Dict[str, Any]:
        if not frame_metrics:
            result = self.default_metrics.copy()
            result.update(
                {
                    "video_path": str(video_path),
                    "total_frames": 0,
                    "landmarks_detected_frames": 0,
                    "detection_rate": 0.0,
                }
            )
            return result

        aggregated: Dict[str, Any] = {}
        numeric_keys = [
            k for k in frame_metrics[0].keys() if k.startswith("GMP_") and k != "GMP_SM_pic"
        ]

        for key in numeric_keys:
            vals = [float(f.get(key, 0.0)) for f in frame_metrics]
            aggregated[key] = float(np.mean(vals))

        # no annotated frame produced in this tasks-only implementation
        aggregated["GMP_SM_pic"] = "" if best_frame is None else self._encode_image_to_base64(best_frame)

        landmarks_detected_frames = sum(
            1
            for frame in frame_metrics
            if any(frame.get(f"GMP_land_visi_{i}", 0.0) > 0.5 for i in range(1, 34))
        )

        aggregated.update(
            {
                "video_path": str(video_path),
                "total_frames": len(frame_metrics),
                "landmarks_detected_frames": landmarks_detected_frames,
                "detection_rate": landmarks_detected_frames / len(frame_metrics),
                "avg_landmarks_per_frame": float(
                    np.mean(
                        [
                            sum(
                                1
                                for i in range(1, 34)
                                if frame.get(f"GMP_land_visi_{i}", 0.0) > 0.5
                            )
                            for frame in frame_metrics
                        ]
                    )
                ),
            }
        )

        return aggregated

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        try:
            results = self.analyze_video(video_path)
            return {
                "Pose estimation and tracking": {
                    "description": "Google MediaPipe pose landmark detection with 33 landmarks (tasks API)",
                    "features": results,
                }
            }
        except Exception as e:
            logger.error(f"Error in MediaPipe pose analysis: {e}")

            default_result = self.default_metrics.copy()
            default_result.update(
                {
                    "video_path": str(video_path),
                    "total_frames": 0,
                    "landmarks_detected_frames": 0,
                    "detection_rate": 0.0,
                    "avg_landmarks_per_frame": 0.0,
                    "error": str(e),
                }
            )

            return {
                "Pose estimation and tracking": {
                    "description": "Google MediaPipe pose landmark detection with 33 landmarks (tasks API)",
                    "features": default_result,
                }
            }


def extract_mediapipe_pose_features(video_path: str, device: str = "cpu") -> Dict[str, Any]:
    analyzer = MediaPipePoseAnalyzer(device=device)
    return analyzer.get_feature_dict(video_path)
