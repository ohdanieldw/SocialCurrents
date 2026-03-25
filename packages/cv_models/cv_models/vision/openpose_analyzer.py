#!/usr/bin/env python3
"""
OpenPose analyzer for pose estimation and tracking.
Based on CMU's OpenPose: Real-time multi-person keypoint detection library.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

class OpenPoseAnalyzer:
    """
    OpenPose analyzer for real-time multi-person pose estimation and tracking.

    This class invokes the official OpenPose binary, parses JSON outputs,
    and aggregates them into pipeline features.

    SAFE + STRICT POLICY:
    - If OpenPose is unavailable, return dummy features instead of crashing.
    """

    def __init__(
        self,
        device: str = 'cpu',
        confidence_threshold: float = 0.1,
        openpose_bin: Optional[str] = None,
        model_folder: Optional[str] = None,
        extra_flags: Optional[List[str]] = None,
        enable_face: bool = True,
        enable_hand: bool = False,
        render_pose: int = 2,
        keep_json: bool = False,
        output_root: Optional[Path] = None,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.initialized = False

        # ---- NEW: availability flags (aligned with AVHuBERT / CrowdFlow) ----
        self.available = True
        self.unavailable_reason = ""

        # Resolve OpenPose binary
        self.openpose_bin = (
            openpose_bin
            or os.environ.get("OPENPOSE_BIN")
            or shutil.which("openpose")  # <-- NEW fallback
        )

        self.model_folder = model_folder or os.environ.get("OPENPOSE_MODEL_FOLDER")
        env_flags = os.environ.get("OPENPOSE_FLAGS")
        if env_flags and not extra_flags:
            extra_flags = [flag for flag in env_flags.split() if flag]
        self.extra_flags = extra_flags or []
        self.enable_face = enable_face
        self.enable_hand = enable_hand
        self.render_pose = render_pose
        self.keep_json = keep_json
        self.number_people_max = 10
        self.output_root = Path(output_root) if output_root else None

        # Body keypoint names (COCO format)
        self.keypoint_names = [
            'Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
            'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee',
            'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye',
            'LEye', 'REar', 'LEar'
        ]

        # CPU-only mode detection
        self.openpose_gpu_mode = (os.environ.get("OPENPOSE_GPU_MODE") or "").upper()
        self.openpose_use_cuda = (os.environ.get("OPENPOSE_USE_CUDA") or "").upper()
        self.cpu_only_mode = (
            self.openpose_gpu_mode == "CPU_ONLY"
            or self.openpose_use_cuda in {"OFF", "0", "FALSE"}
        )
        if self.cpu_only_mode and isinstance(self.render_pose, int) and self.render_pose > 0:
            logger.info("OpenPose CPU-only mode → disabling render_pose")
            self.render_pose = 0

        # ---- Default metrics (unchanged) ----
        self.default_metrics = {
            'openPose_total_frames': 0,
            'openPose_pose_detected_frames': 0,
            'openPose_detection_rate': 0.0,
            'openPose_avg_keypoints_per_frame': 0.0,
            'openPose_avg_confidence': 0.0,
            'openPose_max_persons_detected': 0,
            'openPose_pose_video_path': "",
            'openPose_pose_gif_path': "",
            'openPose_SM_pic': "",
            'openPose_SM_preview_available': 0.0,
            'openPose_SM_preview_mean_intensity': 0.0,
            'openPose_SM_preview_contrast': 0.0,
        }

        for keypoint in self.keypoint_names:
            self.default_metrics[f'openPose_{keypoint.lower()}_x'] = 0.0
            self.default_metrics[f'openPose_{keypoint.lower()}_y'] = 0.0
            self.default_metrics[f'openPose_{keypoint.lower()}_confidence'] = 0.0

        self.default_metrics.update({
            'openPose_left_arm_angle': 0.0,
            'openPose_right_arm_angle': 0.0,
            'openPose_left_leg_angle': 0.0,
            'openPose_right_leg_angle': 0.0,
            'openPose_torso_angle': 0.0,
            'openPose_shoulder_width': 0.0,
            'openPose_hip_width': 0.0,
            'openPose_body_height': 0.0
        })

        # ---- NEW: pre-check availability ----
        if not self.openpose_bin:
            self.available = False
            self.unavailable_reason = "OpenPose binary not found (OPENPOSE_BIN not set and 'openpose' not in PATH)"
            logger.warning(f"[OpenPose] disabled: {self.unavailable_reason}")

    def _initialize_model(self) -> None:
        """Resolve OpenPose binary and ensure prerequisites are in place."""
        if self.initialized or not self.available:
            return

        logger.info("Initializing OpenPose analyzer")

        candidate_path = Path(self.openpose_bin)
        if candidate_path.exists():
            resolved_bin = candidate_path
        else:
            resolved = shutil.which(self.openpose_bin)
            if not resolved:
                self.available = False
                self.unavailable_reason = f"OpenPose executable not found: {self.openpose_bin}"
                logger.warning(f"[OpenPose] disabled: {self.unavailable_reason}")
                return
            resolved_bin = Path(resolved)

        if not os.access(resolved_bin, os.X_OK):
            self.available = False
            self.unavailable_reason = f"OpenPose binary not executable: {resolved_bin}"
            logger.warning(f"[OpenPose] disabled: {self.unavailable_reason}")
            return

        if self.model_folder:
            model_folder_path = Path(self.model_folder)
            if not model_folder_path.exists():
                self.available = False
                self.unavailable_reason = f"OpenPose model folder missing: {model_folder_path}"
                logger.warning(f"[OpenPose] disabled: {self.unavailable_reason}")
                return
            self.model_folder = str(model_folder_path.resolve())

        self.openpose_bin = str(resolved_bin.resolve())
        self.initialized = True
        logger.info("OpenPose initialized with binary %s", self.openpose_bin)

    # -------------------- EVERYTHING BELOW IS UNCHANGED --------------------
    # (except analyze_video() safe fallback)

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Run OpenPose on a video and aggregate its outputs."""

        if not self.initialized:
            self._initialize_model()

        # ---- NEW: SAFE FALLBACK ----
        if not self.available:
            metrics = self.default_metrics.copy()
            metrics["openPose_unavailable_reason"] = self.unavailable_reason
            logger.warning(f"[OpenPose] returning dummy features: {self.unavailable_reason}")
            return metrics

        logger.info("Analyzing poses via OpenPose: %s", video_path)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return self.default_metrics.copy()

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()

        video_path_obj = Path(video_path)
        _, json_dir, render_video_path, gif_path = self._prepare_output_paths(video_path_obj)

        self._run_openpose_cli(video_path_obj, json_dir, render_video_path)

        frame_metrics_map, best_frame_idx = self._parse_frame_metrics(json_dir)
        if not frame_metrics_map:
            logger.warning("OpenPose produced no keypoints for %s", video_path)

        max_idx = max(frame_metrics_map.keys(), default=-1)
        if total_frames <= 0 or (max_idx >= 0 and max_idx + 1 > total_frames):
            total_frames = max_idx + 1 if max_idx >= 0 else 0

        frame_metrics: List[Dict[str, Any]] = []
        for frame_idx in range(total_frames):
            metrics = frame_metrics_map.get(frame_idx)
            if metrics is None:
                metrics = {
                    'frame_idx': frame_idx,
                    'poses_detected': 0,
                    'keypoints_detected': 0,
                    'avg_confidence': 0.0,
                }
            frame_metrics.append(metrics)

        results = self._aggregate_frame_results(frame_metrics, total_frames)
        results['openPose_pose_video_path'] = str(render_video_path) if render_video_path.exists() else ""

        gif_path_str, preview_stats = self._extract_preview_media(render_video_path, gif_path, best_frame_idx)
        results['openPose_pose_gif_path'] = gif_path_str
        results['openPose_SM_pic'] = ""
        results['openPose_SM_preview_available'] = preview_stats[0]
        results['openPose_SM_preview_mean_intensity'] = preview_stats[1]
        results['openPose_SM_preview_contrast'] = preview_stats[2]

        if not self.keep_json:
            shutil.rmtree(json_dir, ignore_errors=True)

        logger.info("OpenPose analysis completed for %s", video_path)
        return results

    # (rest of file unchanged)
