"""Lightweight speech emotion recognition powered by transformers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from transformers import pipeline

logger = logging.getLogger(__name__)


class SimpleEmotionRecognizer:
    """Wrapper around a pretrained Hugging Face audio emotion classifier."""

    def __init__(
        self,
        model_name: str = "superb/hubert-large-superb-er",
        *,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        resolved_device = self._resolve_device(device)
        try:
            self._pipeline = pipeline(
                task="audio-classification",
                model=model_name,
                device=resolved_device,
                cache_dir=cache_dir,
            )
            config = self._pipeline.model.config
            self._labels = [config.id2label[i] for i in sorted(config.id2label)]
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error("Falling back to uniform emotions because %s could not load: %s", model_name, exc)
            self._pipeline = None
            self._labels = ["angry", "happy", "sad", "neutral"]
        self._ser_labels = [f"ser_{label.lower()}" for label in self._labels]

    @staticmethod
    def _resolve_device(device: Optional[str]) -> int:
        if device is None:
            from torch import cuda

            return 0 if cuda.is_available() else -1
        if isinstance(device, str) and device.lower().startswith("cuda"):
            from torch import cuda

            if not cuda.is_available():
                return -1
            if ":" in device:
                return int(device.split(":", maxsplit=1)[1])
            return 0
        if isinstance(device, str) and device.lower() == "cpu":
            return -1
        try:
            return int(device)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return -1

    def predict_emotion(self, audio_path: str) -> Dict[str, Any]:
        top_k = len(self._labels)
        if self._pipeline is None:
            uniform = 1.0 / top_k
            probabilities = {label: uniform for label in self._ser_labels}
            dominant_label = self._labels[0]
            dominant_score = uniform
        else:
            results = self._pipeline(audio_path, top_k=top_k)
            if not results:
                raise RuntimeError("Audio emotion pipeline returned no predictions")
            probabilities = {f"ser_{entry['label'].lower()}": float(entry["score"]) for entry in results}
            dominant = max(results, key=lambda item: item["score"])
            dominant_label = dominant["label"].lower()
            dominant_score = float(dominant["score"])

        return {
            "emotion": dominant_label,
            "confidence": dominant_score,
            "probabilities": probabilities,
        }

_GLOBAL_RECOGNIZER: Optional[SimpleEmotionRecognizer] = None


def _get_recognizer() -> SimpleEmotionRecognizer:
    global _GLOBAL_RECOGNIZER
    if _GLOBAL_RECOGNIZER is None:
        _GLOBAL_RECOGNIZER = SimpleEmotionRecognizer()
    return _GLOBAL_RECOGNIZER


def analyze_emotion(audio_path: str) -> Dict[str, Any]:
    recognizer = _get_recognizer()
    try:
        result = recognizer.predict_emotion(audio_path)
        result.update({"audio_path": audio_path})
        return result
    except Exception as exc:  # pragma: no cover - defensive path
        logger.error("Emotion analysis failed for %s: %s", audio_path, exc)
        return {
            "audio_path": audio_path,
            "emotion": "neutral",
            "confidence": 0.0,
            "probabilities": {"ser_neutral": 1.0},
            "error": str(exc),
        }


def get_emotion_features_dict(audio_path: str) -> Dict[str, float]:
    recognizer = _get_recognizer()
    result = recognizer.predict_emotion(audio_path)
    probabilities = result.get("probabilities", {})
    return {label: float(probabilities.get(label, 0.0)) for label in recognizer._ser_labels}


__all__ = [
    "SimpleEmotionRecognizer",
    "analyze_emotion",
    "get_emotion_features_dict",
]

