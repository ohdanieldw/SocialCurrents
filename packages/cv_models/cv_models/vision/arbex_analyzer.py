"""ARBEx analyzer backed by the official implementation.

This module wires the original ARBEx repository into the multimodal pipeline.
It loads the published Poster encoder, classification head, anchor bank, and
(optional) self-attention correction modules to provide robust facial
expression recognition metrics on top of per-frame facial crops.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from cv_models.external.repo_manager import ensure_repo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArbexWeights:
    """Container describing the four learnable components of ARBEx."""

    poster: Path
    classifier: Path
    anchors: Path
    self_attn: Optional[Path] = None


def _append_to_syspath(path: Path) -> None:
    if not path.exists():
        return
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _normalized_entropy(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.clamp(probs, eps, 1.0)
    entropy = -(probs * torch.log(probs)).sum(dim=-1)
    max_entropy = float(np.log(probs.shape[-1]))
    return entropy / max_entropy


def _softmax(tensor: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        tensor = tensor / temperature
    return torch.softmax(tensor, dim=-1)


class ARBExAnalyzer:
    """Run the official ARBEx model across sampled video frames."""

    EMOTIONS = (
        "neutral",
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
        "other",
    )

    DEFAULT_POSTER_ENV = "ARBEX_POSTER_WEIGHTS"
    DEFAULT_CLASSIFIER_ENV = "ARBEX_CLASSIFIER_WEIGHTS"
    DEFAULT_ANCHORS_ENV = "ARBEX_ANCHORS_WEIGHTS"
    DEFAULT_SELF_ATTENTION_ENV = "ARBEX_SELF_ATTENTION_WEIGHTS"
    DEFAULT_FACE_SCALE = 1.2

    def __init__(
        self,
        device: str = "cpu",
        *,
        poster_weights: Optional[str] = None,
        classifier_weights: Optional[str] = None,
        anchor_weights: Optional[str] = None,
        self_attention_weights: Optional[str] = None,
        temperature: float = 1.0,
        delta: float = 1.0,
        max_frames: Optional[int] = 256,
        frame_stride: Optional[int] = None,
        face_scale: float = DEFAULT_FACE_SCALE,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested for ARBEx but CUDA is unavailable; using CPU instead.")

        repo_root = ensure_repo("arbex")
        _append_to_syspath(repo_root)
        _append_to_syspath(repo_root / "arbex")

        try:
            from arbex.models.head import ClassificationHead  # type: ignore
            from arbex.models.anchors import Anchors  # type: ignore
            from arbex.models.attn import SelfAttn  # type: ignore
            from arbex.models.poster import get_poster  # type: ignore
        except ImportError as exc:  # pragma: no cover - defensive guard
            raise ImportError(
                "ARBExAnalyzer requires the dependencies from the official repository. "
                "Install them with `pip install -r external/vision/ARBEx/requirements.txt`."
            ) from exc

        self._ClassificationHead = ClassificationHead
        self._Anchors = Anchors
        self._SelfAttn = SelfAttn
        self._get_poster = get_poster

        weights = self._resolve_weights(
            repo_root=repo_root,
            poster=poster_weights,
            classifier=classifier_weights,
            anchors=anchor_weights,
            self_attn=self_attention_weights,
        )
        self.weights = weights

        self.poster = self._load_poster(weights.poster).to(self.device)
        self.classifier = self._ClassificationHead(size_out=len(self.EMOTIONS))
        self.classifier.load_state_dict(torch.load(str(weights.classifier), map_location="cpu"))
        self.classifier.to(self.device)
        self.anchors = self._Anchors(n_classes=len(self.EMOTIONS))
        self.anchors.load_state_dict(torch.load(str(weights.anchors), map_location="cpu"))
        self.anchors.to(self.device)

        if weights.self_attn and weights.self_attn.exists():
            self.self_attn = self._SelfAttn(n_classes=len(self.EMOTIONS))
            self.self_attn.load_state_dict(torch.load(str(weights.self_attn), map_location="cpu"))
            self.self_attn.to(self.device)
        else:
            self.self_attn = None
            if weights.self_attn is not None:
                logger.warning(
                    "Self-attention weights not found at %s; skipping attention correction.",
                    weights.self_attn,
                )

        self.poster.eval()
        self.classifier.eval()
        self.anchors.eval()
        if self.self_attn is not None:
            self.self_attn.eval()

        self.temperature = float(temperature)
        self.delta = float(delta)
        self.max_frames = max_frames
        self.frame_stride = frame_stride
        self.face_scale = float(face_scale)

        self.transforms = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            raise RuntimeError("Failed to load OpenCV face cascade. Ensure OpenCV is installed with data files.")

        default_output = Path.cwd() / "output" / "vision" / "arbex"
        self.output_dir = Path(output_dir) if output_dir else default_output
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.default_metrics: Dict[str, Any] = {
            "arbex_top_emotion": "neutral",
            "arbex_top_probability": 0.0,
            "arbex_mean_confidence": 0.0,
            "arbex_entropy": 1.0,
            "arbex_processed_frames": 0,
            "arbex_total_frames": 0,
            "arbex_face_detection_rate": 0.0,
            "arbex_video_path": "",
            "arbex_probabilities_path": "",
            "arbex_attention_available": bool(self.self_attn),
            "arbex_SM_pic": "",
            "arbex_distribution_plot": "",
        }
        for emotion in self.EMOTIONS:
            self.default_metrics[f"arbex_prob_{emotion}"] = 0.0

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------
    def _resolve_weights(
        self,
        *,
        repo_root: Path,
        poster: Optional[str],
        classifier: Optional[str],
        anchors: Optional[str],
        self_attn: Optional[str],
    ) -> ArbexWeights:
        poster_path = self._resolve_weight_path(
            provided=poster,
            env_var=self.DEFAULT_POSTER_ENV,
            default_hint=repo_root,
            glob_pattern="**/poster*.pth",
            component="poster",
        )
        classifier_path = self._resolve_weight_path(
            provided=classifier,
            env_var=self.DEFAULT_CLASSIFIER_ENV,
            default_hint=repo_root,
            glob_pattern="**/classifier*.pth",
            component="classifier",
        )
        anchor_path = self._resolve_weight_path(
            provided=anchors,
            env_var=self.DEFAULT_ANCHORS_ENV,
            default_hint=repo_root,
            glob_pattern="**/anchors*.pth",
            component="anchors",
        )
        self_attn_path: Optional[Path] = None
        if self_attn or os.getenv(self.DEFAULT_SELF_ATTENTION_ENV):
            self_attn_path = self._resolve_weight_path(
                provided=self_attn,
                env_var=self.DEFAULT_SELF_ATTENTION_ENV,
                default_hint=repo_root,
                glob_pattern="**/attn*.pth",
                component="self-attention",
                required=False,
            )

        return ArbexWeights(
            poster=poster_path,
            classifier=classifier_path,
            anchors=anchor_path,
            self_attn=self_attn_path,
        )

    def _resolve_weight_path(
        self,
        *,
        provided: Optional[str],
        env_var: str,
        default_hint: Path,
        glob_pattern: str,
        component: str,
        required: bool = True,
    ) -> Path:
        candidate = provided or os.getenv(env_var)
        if candidate:
            path = Path(candidate).expanduser()
            if path.exists():
                return path
            raise FileNotFoundError(f"Configured {component} weights not found at {path}.")

        matches = sorted(default_hint.glob(glob_pattern))
        for match in matches:
            if match.is_file():
                return match

        message = (
            f"Unable to locate ARBEx {component} weights. Set the path via the `{env_var}` environment variable "
            "or pass it directly to the analyzer."
        )
        if required:
            raise FileNotFoundError(message)
        logger.warning(message)
        return default_hint / "missing"

    def _load_poster(self, weights: Path) -> torch.nn.Module:
        # Poster requires the upstream MobileFaceNet/IR checkpoints under models/pretrained
        poster = self._get_poster(
            path_landmark="models/pretrained/mobilefacenet.pth",
            path_ir="models/pretrained/ir50.pth",
        )
        poster.load_state_dict(torch.load(str(weights), map_location="cpu"))
        return poster

    # ------------------------------------------------------------------
    # Video processing
    # ------------------------------------------------------------------
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_file}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_indices = self._select_frame_indices(total_frames)
        probabilities: List[np.ndarray] = []
        confidences: List[float] = []
        entropies: List[float] = []
        face_hits = 0
        processed = 0

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            face = self._extract_face(frame)
            if face is None:
                continue

            face_hits += 1
            probs, confidence, entropy = self._infer_face(face)
            probabilities.append(probs)
            confidences.append(confidence)
            entropies.append(entropy)
            processed += 1

        cap.release()

        metrics = self.default_metrics.copy()
        metrics["arbex_video_path"] = str(video_file)
        metrics["arbex_total_frames"] = int(total_frames)
        metrics["arbex_processed_frames"] = int(processed)
        metrics["arbex_face_detection_rate"] = (face_hits / len(frame_indices)) if frame_indices else 0.0

        if not probabilities:
            probabilities_path = self._dump_probabilities(video_file.stem, np.empty((0, len(self.EMOTIONS))))
            metrics["arbex_probabilities_path"] = str(probabilities_path)
            return metrics

        prob_array = np.vstack(probabilities)
        mean_probs = prob_array.mean(axis=0)
        top_index = int(np.argmax(mean_probs))
        metrics["arbex_top_emotion"] = self.EMOTIONS[top_index]
        metrics["arbex_top_probability"] = float(mean_probs[top_index])
        metrics["arbex_mean_confidence"] = float(np.mean(confidences))
        metrics["arbex_entropy"] = float(np.mean(entropies))

        for emotion, value in zip(self.EMOTIONS, mean_probs):
            metrics[f"arbex_prob_{emotion}"] = float(value)

        probabilities_path = self._dump_probabilities(video_file.stem, prob_array)
        metrics["arbex_probabilities_path"] = str(probabilities_path)

        preview = self._render_preview(prob_array)
        if preview is not None:
            metrics["arbex_distribution_plot"] = preview
            metrics["arbex_SM_pic"] = preview

        return metrics

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        features = self.analyze_video(video_path)
        return {
            "ARBEx": {
                "description": "Facial expression recognition via ARBEx",
                "features": features,
            }
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _select_frame_indices(self, total_frames: int) -> Sequence[int]:
        if total_frames <= 0:
            return []

        if self.max_frames is None or total_frames <= self.max_frames:
            stride = self.frame_stride or 1
            return list(range(0, total_frames, stride))

        sample_count = min(self.max_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
        return indices.tolist()

    def _extract_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda bbox: bbox[2] * bbox[3])
        cx, cy = x + w / 2, y + h / 2
        scale = self.face_scale
        size = max(w, h) * scale
        x1 = int(max(cx - size / 2, 0))
        y1 = int(max(cy - size / 2, 0))
        x2 = int(min(cx + size / 2, frame.shape[1]))
        y2 = int(min(cy + size / 2, frame.shape[0]))
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        return face

    def _infer_face(self, face_bgr: np.ndarray) -> Tuple[np.ndarray, float, float]:
        image = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transforms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.poster(tensor)
            if isinstance(embedding, (list, tuple)):
                embedding = embedding[-1]
            logits = self.classifier(embedding)
            probs = _softmax(logits, self.temperature)
            confidence = (1.0 - _normalized_entropy(probs)).cpu().numpy()

            distances = self.anchors(embedding)
            sim = torch.softmax(-distances.view(distances.shape[0], -1) / self.delta, dim=-1)
            sim = sim.view_as(distances).sum(-1)
            conf_sim = 1.0 - _normalized_entropy(sim)

            if self.self_attn is not None:
                attn = self.self_attn(embedding)
                attn = torch.softmax(attn, dim=-1)
                conf_attn = 1.0 - _normalized_entropy(attn)
            else:
                attn = None
                conf_attn = None

            corrected = probs.clone()
            weight = torch.ones_like(probs)
            corrected = corrected + sim * conf_sim.view(-1, 1)
            weight = weight + conf_sim.view(-1, 1)
            if attn is not None and conf_attn is not None:
                corrected = corrected + attn * conf_attn.view(-1, 1)
                weight = weight + conf_attn.view(-1, 1)
            corrected = corrected / torch.clamp(weight, min=1e-6)

        probs_np = corrected.squeeze(0).cpu().numpy()
        confidence_np = float(confidence.squeeze())
        entropy_np = float(_normalized_entropy(torch.from_numpy(probs_np[np.newaxis, :])).item())
        return probs_np, confidence_np, entropy_np

    def _dump_probabilities(self, stem: str, probs: np.ndarray) -> Path:
        path = self.output_dir / f"{stem}_arbex_probs.npz"
        np.savez_compressed(path, probabilities=probs, emotions=self.EMOTIONS)
        return path

    def _render_preview(self, probs: np.ndarray) -> Optional[str]:
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover - optional dependency
            return None

        mean_probs = probs.mean(axis=0) if probs.size else np.zeros(len(self.EMOTIONS))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(self.EMOTIONS, mean_probs, color="tab:orange")
        ax.set_ylim(0.0, max(1.0, float(mean_probs.max()) + 0.05))
        ax.set_ylabel("Probability")
        ax.set_title("ARBEx Mean Probabilities")
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("ascii")
        return encoded


def create_arbex_analyzer(
    *,
    device: str = "cpu",
    poster_weights: Optional[str] = None,
    classifier_weights: Optional[str] = None,
    anchor_weights: Optional[str] = None,
    self_attention_weights: Optional[str] = None,
    temperature: float = 1.0,
    delta: float = 1.0,
    max_frames: Optional[int] = 256,
    frame_stride: Optional[int] = None,
    face_scale: float = ARBExAnalyzer.DEFAULT_FACE_SCALE,
    output_dir: Optional[Path] = None,
) -> ARBExAnalyzer:
    """Convenience factory matching legacy call sites."""

    return ARBExAnalyzer(
        device=device,
        poster_weights=poster_weights,
        classifier_weights=classifier_weights,
        anchor_weights=anchor_weights,
        self_attention_weights=self_attention_weights,
        temperature=temperature,
        delta=delta,
        max_frames=max_frames,
        frame_stride=frame_stride,
        face_scale=face_scale,
        output_dir=output_dir,
    )
