"""EmotiEffNet analyzer backed by the official EmotiEffLib implementation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from cv_models.external.repo_manager import ensure_repo

logger = logging.getLogger(__name__)


def _append_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


class EmotiEffNetAnalyzer:
    """Wrap EmotiEffLib recogniser for emotion / valence / arousal inference."""

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "enet_b0_8_va_mtl",
        engine: str = "torch",
        max_faces_per_frame: int = 1,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.engine = engine
        self.max_faces_per_frame = max_faces_per_frame

        repo_root = ensure_repo("emotiefflib")
        _append_path(repo_root)

        try:
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "EmotiEffLib dependencies are missing. Install torch / torchvision or set up the"
                " environment referenced in EmotiEffLib's requirements."
            ) from exc

        self.recognizer = EmotiEffLibRecognizer(engine=engine, model_name=model_name, device=device)
        self.emotion_map = self.recognizer.idx_to_emotion_class
        self.num_emotions = len(self.emotion_map)
        self.is_mtl = getattr(self.recognizer, "is_mtl", False)

        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _detect_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        detections: List[np.ndarray] = []
        faces = self.face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            return detections

        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
        for x, y, w, h in faces[: self.max_faces_per_frame]:
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            detections.append(crop)
        return detections

    def _run_model(self, face_img: np.ndarray) -> Dict[str, Any]:
        labels, scores = self.recognizer.predict_emotions(face_img, logits=False)
        probs = scores[0]

        if self.is_mtl and probs.shape[0] > self.num_emotions:
            emotion_probs = probs[: self.num_emotions]
            extra = probs[self.num_emotions :]
        else:
            emotion_probs = probs
            extra = np.array([])

        return {
            "labels": labels,
            "emotion_probs": emotion_probs,
            "extra": extra,
        }

    def analyze_video(self, video_path: str, *, max_frames: int = 64, keep_per_frame: bool = False) -> Dict[str, Any]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video at {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
        indices = np.linspace(0, total_frames - 1, min(total_frames, max_frames), dtype=int)

        emotion_vectors: List[np.ndarray] = []
        extra_vectors: List[np.ndarray] = []
        per_sample: List[Dict[str, Any]] = []
        face_hits = 0

        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = capture.read()
            if not ok:
                continue

            first_face = True
            for face in self._detect_faces(frame):
                inference = self._run_model(face)
                emotion_vectors.append(inference["emotion_probs"])
                if inference["extra"].size:
                    extra_vectors.append(inference["extra"])
                face_hits += 1

                if keep_per_frame and first_face:
                    fd: Dict[str, Any] = {"frame_idx": int(idx), "timestamp": int(idx) / fps}
                    for ei, label in sorted(self.emotion_map.items()):
                        if ei < inference["emotion_probs"].shape[0]:
                            fd[f"eln_prob_{label.lower()}"] = float(inference["emotion_probs"][ei])
                    if inference["extra"].size >= 2:
                        fd["eln_valence"] = float(inference["extra"][0])
                        fd["eln_arousal"] = float(inference["extra"][1])
                    elif inference["extra"].size == 1:
                        fd["eln_valence"] = float(inference["extra"][0])
                    per_sample.append(fd)
                    first_face = False

        capture.release()

        if not emotion_vectors:
            logger.warning("EmotiEffNet analyzer could not find faces in %s", video_path)
            result: Dict[str, Any] = {"eln_face_detected_ratio": 0.0, "eln_samples": 0}
            if keep_per_frame:
                result["eln_per_sample"] = []
            return result

        emotion_matrix = np.vstack(emotion_vectors)
        mean_emotions = np.mean(emotion_matrix, axis=0)

        features: Dict[str, Any] = {}
        for index, label in sorted(self.emotion_map.items()):
            key = f"eln_prob_{label.lower()}"
            if index < mean_emotions.shape[0]:
                features[key] = float(mean_emotions[index])

        top_index = int(np.argmax(mean_emotions))
        features["eln_top_emotion"] = self.emotion_map.get(top_index, "unknown")
        features["eln_samples"] = len(emotion_vectors)
        features["eln_face_detected_ratio"] = face_hits / len(indices)

        if extra_vectors:
            extra_matrix = np.vstack(extra_vectors)
            if extra_matrix.shape[1] >= 2:
                # EmotiEffLib MTL models output [arousal, valence] or similar; follow doc order (valence, arousal).
                features["eln_valence"] = float(np.mean(extra_matrix[:, 0]))
                features["eln_arousal"] = float(np.mean(extra_matrix[:, 1]))
            elif extra_matrix.shape[1] == 1:
                features["eln_valence"] = float(np.mean(extra_matrix[:, 0]))

        if keep_per_frame:
            features["eln_per_sample"] = per_sample

        return features

    def get_feature_dict(self, video_path: str, keep_per_frame: bool = False) -> Dict[str, Any]:
        return {
            "EmotiEffNet": {
                "description": "EmotiEffLib affect recognition (official model)",
                "features": self.analyze_video(video_path, keep_per_frame=keep_per_frame),
            }
        }


def extract_emotieffnet_features(
    video_path: str,
    *,
    device: str = "cpu",
    model_name: str = "enet_b0_8_va_mtl",
    engine: str = "torch",
) -> Dict[str, Any]:
    analyzer = EmotiEffNetAnalyzer(device=device, model_name=model_name, engine=engine)
    return analyzer.get_feature_dict(video_path)
