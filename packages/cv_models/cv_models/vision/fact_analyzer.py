"""FACT (Face Action Coding Toolkit) inspired analyzer.

This implementation prefers the FaCER detector + Py-Feat AU models when the
dependencies are available. It gracefully falls back to the lightweight Haar
cascade + MediaPipe heuristic when FaCER or PyTorch is missing.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from facer import face_detector as _facer_face_detector  # type: ignore
except Exception:  # pragma: no cover
    _facer_face_detector = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .pyfeat_analyzer import PyFeatAnalyzer
except Exception:  # pragma: no cover
    PyFeatAnalyzer = None  # type: ignore

logger = logging.getLogger(__name__)


class _FaCERPipeline:
    """Small wrapper that runs FaCER's detector (and optional aligner)."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        detector_conf: str = "retinaface/mobilenet",
        aligner_conf: str = "farl/ibug300w/448",
    ) -> None:
        if torch is None:
            raise RuntimeError("torch is required for the FaCER FACT pipeline")
        if _facer_face_detector is None:
            raise RuntimeError("facer is not installed")

        self.device = torch.device(device)
        self.detector_name = detector_conf
        self.aligner_name: Optional[str] = None
        self.detector = _facer_face_detector(detector_conf, device=self.device)

        self.aligner = None
        if aligner_conf:
            try:
                from facer import face_aligner  # type: ignore

                self.aligner = face_aligner(aligner_conf, device=self.device)
                self.aligner_name = aligner_conf
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.info(
                    "FaCER aligner '%s' unavailable, continuing without it: %s",
                    aligner_conf,
                    exc,
                )

    def process_frames(self, frames: List[Tuple[int, np.ndarray]]) -> Dict[str, Any]:
        if torch is None:
            raise RuntimeError("torch is required for the FaCER FACT pipeline")

        faces: List[Dict[str, Any]] = []
        collected_points: List[np.ndarray] = []
        collected_alignment: List[Optional[np.ndarray]] = []

        for frame_index, frame in frames:
            if frame is None:
                continue
            tensor = torch.from_numpy(frame)
            if tensor.ndim != 3:
                continue
            tensor = tensor.to(self.device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():  # pragma: no cover - detection side-effect only
                data = self.detector(tensor)
                if self.aligner is not None:
                    try:
                        data = self.aligner(tensor, data)
                    except Exception as exc:  # pragma: no cover
                        logger.debug("FaCER aligner failed on frame %s: %s", frame_index, exc)
                        data.pop("alignment", None)

            num_faces = int(data.get("rects", torch.zeros(0)).shape[0])
            if num_faces == 0:
                continue

            rects = data["rects"].detach().cpu().numpy().astype(np.float32)
            scores_tensor = data.get("scores")
            scores = (
                scores_tensor.detach().cpu().numpy().astype(np.float32)
                if torch is not None and isinstance(scores_tensor, torch.Tensor) and scores_tensor.numel()
                else np.ones((num_faces,), dtype=np.float32)
            )

            points_tensor = data.get("points")
            if torch is not None and isinstance(points_tensor, torch.Tensor) and points_tensor.numel():
                points_np = points_tensor.detach().cpu().numpy().astype(np.float32)
            else:  # pragma: no cover - retinaface always returns points
                points_np = np.zeros((num_faces, 0, 2), dtype=np.float32)

            alignment_tensor = data.get("alignment")
            if torch is not None and isinstance(alignment_tensor, torch.Tensor) and alignment_tensor.numel():
                alignment_np = alignment_tensor.detach().cpu().numpy().astype(np.float32)
            else:
                alignment_np = None

            for i in range(num_faces):
                faces.append(
                    {
                        "frame_index": int(frame_index),
                        "bbox": [float(x) for x in rects[i].tolist()],
                        "score": float(scores[i]),
                        "landmark_index": len(collected_points),
                    }
                )
                collected_points.append(points_np[i])
                collected_alignment.append(alignment_np[i] if alignment_np is not None else None)

        use_alignment = bool(faces) and all(item is not None for item in collected_alignment)

        if not faces:
            landmarks = np.zeros((0, 0, 2), dtype=np.float32)
        elif use_alignment:
            landmarks = np.stack([np.asarray(item, dtype=np.float32) for item in collected_alignment], axis=0)
        else:
            landmarks = np.stack([np.asarray(item, dtype=np.float32) for item in collected_points], axis=0)

        landmark_source = "alignment" if use_alignment and self.aligner_name else "points"
        landmark_dims = int(landmarks.shape[1]) if landmarks.ndim >= 2 else 0

        summary = {
            "faces": faces,
            "landmarks": landmarks,
            "landmark_source": landmark_source,
            "landmark_dims": landmark_dims,
            "frames_processed": len(frames),
            "detector": self.detector_name,
            "aligner": self.aligner_name if use_alignment else None,
        }
        return summary


class FACTAnalyzer:
    """Produce facial action estimates and metadata."""

    MAX_DETECTION_FRAMES = 12
    ENV_FORCE_LEGACY = "CV_FACT_FORCE_LEGACY"

    def __init__(
        self,
        *,
        output_dir: Optional[Path] = None,
        device: str = "cpu",
        pyfeat_device: str = "cpu",
        use_legacy: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir = self.output_dir / "vision" / "fact"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.pyfeat_device = pyfeat_device

        self._pyfeat: Optional[PyFeatAnalyzer] = None
        self._pyfeat_disabled = False

        env_force_legacy = os.getenv(self.ENV_FORCE_LEGACY, "").lower() in {"1", "true", "yes"}
        self._legacy_requested = use_legacy or env_force_legacy

        self._modern_pipeline: Optional[_FaCERPipeline] = None
        if not self._legacy_requested:
            try:
                self._modern_pipeline = _FaCERPipeline(device=device)
            except Exception as exc:  # pragma: no cover - optional dependencies missing
                logger.info("FaCER pipeline unavailable, falling back to legacy FACT: %s", exc)
                self._modern_pipeline = None

    # ------------------------------------------------------------------
    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        if self._modern_pipeline is not None:
            try:
                return self._modern_fact(video_path)
            except Exception as exc:  # pragma: no cover
                logger.warning("FaCER FACT pipeline failed, switching to legacy fallback: %s", exc, exc_info=True)

        return self._legacy_fact(video_path)

    # ------------------------------------------------------------------
    # Modern pipeline helpers
    def _modern_fact(self, video_path: str) -> Dict[str, Any]:
        frames = self._sample_frames(video_path, self.MAX_DETECTION_FRAMES)
        if not frames:
            logger.warning("FACT: unable to sample frames from %s; using legacy fallback", video_path)
            return self._legacy_fact(video_path)

        assert self._modern_pipeline is not None  # for type checkers
        analysis = self._modern_pipeline.process_frames(frames)

        pyfeat_features = self._compute_pyfeat_features(video_path)
        fact_intensity = self._compute_fact_intensity(pyfeat_features)

        metadata = {
            "mode": "modern",
            "video_path": video_path,
            "detector": analysis["detector"],
            "aligner": analysis["aligner"],
            "frames_sampled": analysis["frames_processed"],
            "landmark_source": analysis["landmark_source"],
            "landmark_dims": analysis["landmark_dims"],
            "num_faces": len(analysis["faces"]),
            "pyfeat_summary": {
                "feature_count": len(pyfeat_features),
                "au_feature_count": len([k for k in pyfeat_features if k.startswith("pf_au")]),
                "error": pyfeat_features.get("pf_error"),
            },
            "faces": analysis["faces"],
        }

        metadata_path = self._write_metadata(video_path, metadata)
        landmarks_path = self._write_landmarks(video_path, analysis["landmarks"])

        result: Dict[str, Any] = {
            "FACT_mode": "modern",
            "FACT_metadata_path": metadata_path,
            "FACT_landmarks_path": landmarks_path,
            "FACT_intensity": fact_intensity,
            "FACT_faces": analysis["faces"],
        }
        if pyfeat_features:
            result["FACT_pyfeat_features"] = pyfeat_features
        return result

    def _sample_frames(self, video_path: str, max_frames: int) -> List[Tuple[int, np.ndarray]]:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            return []

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            total_frames = max_frames

        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

        frames: List[Tuple[int, np.ndarray]] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = capture.read()
            if not success or frame is None:
                continue
            frames.append((index, frame))

        capture.release()
        return frames

    def _compute_pyfeat_features(self, video_path: str) -> Dict[str, Any]:
        if PyFeatAnalyzer is None:
            logger.debug("Py-Feat not installed; FACT intensity will default to 0.0")
            return {}

        if self._pyfeat_disabled:
            return {}

        if self._pyfeat is None:
            try:
                self._pyfeat = PyFeatAnalyzer(device=self.pyfeat_device)
            except Exception as exc:  # pragma: no cover
                logger.warning("Py-Feat initialization failed; disabling FACT AU enrichment: %s", exc)
                self._pyfeat_disabled = True
                return {}

        features_packet = self._pyfeat.get_feature_dict(video_path)
        feature_map = features_packet.get("Facial Expression (Py-Feat)", {}).get("features", {})

        cleaned: Dict[str, Any] = {}
        for key, value in feature_map.items():
            if isinstance(value, (float, int, np.floating, np.integer)):
                cleaned[key] = float(value)
            else:
                cleaned[key] = value
        return cleaned

    @staticmethod
    def _compute_fact_intensity(pyfeat_features: Dict[str, Any]) -> float:
        au_values = [float(value) for key, value in pyfeat_features.items() if key.startswith("pf_au") and isinstance(value, (float, int))]
        return float(np.mean(au_values)) if au_values else 0.0

    def _write_metadata(self, video_path: str, metadata: Dict[str, Any]) -> str:
        summary_path = self.output_dir / f"{Path(video_path).stem}_fact_metadata.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        return str(summary_path)

    def _write_landmarks(self, video_path: str, landmarks: np.ndarray) -> str:
        landmark_path = self.output_dir / f"{Path(video_path).stem}_fact_landmarks.npy"
        np.save(landmark_path, landmarks)
        return str(landmark_path)

    # ------------------------------------------------------------------
    # Legacy helpers
    def _legacy_fact(self, video_path: str) -> Dict[str, Any]:
        frame = self._read_first_frame(video_path)

        faces = self._legacy_detect_faces(frame)
        face_stats = [self._legacy_feature_from_face(face, frame) for face in faces]
        landmarks = self._legacy_landmarks(frame)

        face_records: List[Dict[str, Any]] = []
        for index, (face, stats) in enumerate(zip(faces, face_stats)):
            x, y, w, h = [int(v) for v in face]
            face_records.append(
                {
                    "frame_index": 0,
                    "bbox": [x, y, x + w, y + h],
                    "score": 1.0,
                    "landmark_index": index,
                    "skin_mean": stats["skin_mean"],
                    "skin_std": stats["skin_std"],
                }
            )

        pyfeat_features = self._compute_pyfeat_features(video_path)
        fact_intensity = self._compute_fact_intensity(pyfeat_features)

        metadata = {
            "mode": "legacy",
            "video_path": video_path,
            "detector": "opencv_haar",
            "aligner": "mediapipe_face_mesh" if mp is not None else None,
            "frames_sampled": 1,
            "landmark_source": "mediapipe" if mp is not None else "none",
            "landmark_dims": int(landmarks.shape[1]) if landmarks.ndim >= 2 else 0,
            "num_faces": len(face_records),
            "pyfeat_summary": {
                "feature_count": len(pyfeat_features),
                "au_feature_count": len([k for k in pyfeat_features if k.startswith("pf_au")]),
                "error": pyfeat_features.get("pf_error"),
            },
            "faces": face_records,
        }

        metadata_path = self._write_metadata(video_path, metadata)
        landmarks_path = self._write_landmarks(video_path, landmarks)

        result: Dict[str, Any] = {
            "FACT_mode": "legacy",
            "FACT_metadata_path": metadata_path,
            "FACT_landmarks_path": landmarks_path,
            "FACT_intensity": fact_intensity,
            "FACT_faces": face_records,
        }
        if pyfeat_features:
            result["FACT_pyfeat_features"] = pyfeat_features
        return result

    @staticmethod
    def _read_first_frame(video_path: str) -> np.ndarray:
        capture = cv2.VideoCapture(video_path)
        success, frame = capture.read()
        capture.release()
        if not success or frame is None:
            logger.warning("Unable to read video frame for FACT analysis: %s", video_path)
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        return frame

    def _legacy_detect_faces(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(gray, 1.1, 4)
        return faces

    def _legacy_landmarks(self, frame: np.ndarray) -> np.ndarray:
        if mp is None:  # pragma: no cover - optional dependency
            return np.empty((0, 0, 3), dtype=np.float32)
        mp_face_mesh = mp.solutions.face_mesh  # type: ignore[attr-defined]
        with mp_face_mesh.FaceMesh(static_image_mode=True) as mesh:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)
            if not result.multi_face_landmarks:  # pragma: no cover
                return np.empty((0, 0, 3), dtype=np.float32)
            landmarks = []
            for face_landmarks in result.multi_face_landmarks:
                coords = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                landmarks.append(coords)
            return np.asarray(landmarks, dtype=np.float32)

    @staticmethod
    def _legacy_feature_from_face(face: np.ndarray, frame: np.ndarray) -> Dict[str, float]:
        x, y, w, h = face
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:  # pragma: no cover
            return {"skin_mean": 0.0, "skin_std": 0.0}
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, (0, 48, 80), (20, 255, 255))
        skin_pixels = cv2.bitwise_and(roi, roi, mask=skin_mask)
        values = skin_pixels[skin_mask > 0]
        if values.size == 0:
            return {"skin_mean": 0.0, "skin_std": 0.0}
        luminance = np.mean(values, axis=1)
        return {"skin_mean": float(np.mean(luminance)), "skin_std": float(np.std(luminance))}


__all__ = ["FACTAnalyzer"]