"""DAN analyzer wired to the official implementation."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from cv_models.external.repo_manager import ensure_repo

logger = logging.getLogger(__name__)


def _append_to_syspath(path: Path) -> None:
    if path.exists():
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


class DANAnalyzer:
    """Use the official DAN model for facial expression recognition."""

    DEFAULT_CHECKPOINT = "checkpoints/affecnet8_epoch5_acc0.6209.pth"

    def __init__(
        self,
        device: str = "cpu",
        num_classes: int = 8,
        *,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device)
        self.num_classes = num_classes

        repo_root = ensure_repo("dan")
        _append_to_syspath(repo_root)

        self.repo_root = repo_root
        self.checkpoint_path = Path(
            checkpoint_path
            or os.getenv("DAN_CHECKPOINT_PATH")
            or repo_root / self.DEFAULT_CHECKPOINT
        ).expanduser()

        self.model: Optional[torch.nn.Module] = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        if num_classes == 7:
            self.emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        else:
            self.emotion_labels = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt"]

        self.default_metrics: Dict[str, Any] = {
            **{f"dan_{label}": 0.0 for label in self.emotion_labels},
            "dan_emotion_scores": [0.0] * self.num_classes,
            "dan_face_detected_ratio": 0.0,
        }

        self._initialize_model()

    def _validate_weights(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "DAN checkpoint not found. Download the official weight file and set "
                "DAN_CHECKPOINT_PATH or place it in the cloned repo.\n"
                f"Missing path: {self.checkpoint_path}"
            )

    def _initialize_model(self) -> None:
        self._validate_weights()

        from networks.dan import DAN  # type: ignore[import]

        logger.info("Loading DAN model from %s", self.checkpoint_path)
        model = DAN(num_class=self.num_classes, num_head=4, pretrained=False)

        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        self.model = model

    def _crop_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        faces = self.face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda bbox: bbox[2] * bbox[3])
        return frame[y : y + h, x : x + w]

    def _run_model(self, face_img: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DAN model not initialized")

        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _, _ = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        return probs

    def analyze_video(self, video_path: str, max_frames: int = 120) -> Dict[str, Any]:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
        indices = np.linspace(0, frame_total - 1, min(frame_total, max_frames), dtype=int)

        emotion_vectors: List[np.ndarray] = []
        detected_faces = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            face_roi = self._crop_face(frame)
            if face_roi is None:
                continue
            detected_faces += 1
            try:
                emotion_vectors.append(self._run_model(face_roi))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Skipping frame %s due to inference error: %s", idx, exc)

        cap.release()

        if not emotion_vectors:
            logger.warning("DAN analyzer processed no faces in %s", video_path)
            return self.default_metrics.copy()

        stacked = np.vstack(emotion_vectors)
        aggregated = {
            f"dan_{label}": float(np.mean(stacked[:, i])) for i, label in enumerate(self.emotion_labels)
        }
        aggregated["dan_emotion_scores"] = [float(value) for value in np.mean(stacked, axis=0)]
        aggregated["dan_face_detected_ratio"] = detected_faces / len(indices)
        return aggregated

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        return self.analyze_video(video_path)


def extract_dan_features(
    video_path: str,
    *,
    device: str = "cpu",
    num_classes: int = 8,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    analyzer = DANAnalyzer(
        device=device,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
    )
    return analyzer.get_feature_dict(video_path)
