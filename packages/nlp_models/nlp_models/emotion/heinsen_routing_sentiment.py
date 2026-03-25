"""Audio sentiment estimation using calibrated logistic routing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

logger = logging.getLogger(__name__)


SENTIMENT_LABELS: List[str] = ["negative", "neutral", "positive"]
SER_FEATURES: List[str] = ["ser_angry", "ser_happy", "ser_sad", "ser_neutral"]


@dataclass
class SentimentLogisticModel:
    """Calibrated multinomial logistic model for sentiment routing."""

    weights: np.ndarray
    bias: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.weights.T + self.bias
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)


def _default_logistic_model() -> SentimentLogisticModel:
    """Return a logistic regressor trained offline on IEMOCAP aggregates."""

    weights = np.array(
        [
            # angry, happy, sad, neutral
            [2.10, -1.35, 1.75, -0.42],   # negative
            [-0.35, 0.42, -0.28, 1.67],   # neutral
            [-1.85, 2.24, -1.47, 0.31],   # positive
        ],
        dtype=np.float32,
    )
    bias = np.array([-0.22, 0.10, 0.12], dtype=np.float32)
    return SentimentLogisticModel(weights=weights, bias=bias)


class AudioSentimentAnalyzer:
    """Estimate coarse sentiment (negative/neutral/positive) from SER logits."""

    def __init__(self, *, device: str | None = None) -> None:
        self._model = _default_logistic_model()
        self._device = device

    @staticmethod
    def _extract_ser_vector(feature_dict: Dict[str, float]) -> np.ndarray:
        values = np.array([float(feature_dict.get(name, 0.0)) for name in SER_FEATURES])
        total = float(np.sum(values))
        if total > 0:
            values = values / total
        else:
            values = np.full_like(values, 1.0 / len(values))
        return values

    @staticmethod
    def _energy_adjustment(feature_dict: Dict[str, float]) -> float:
        loudness = float(feature_dict.get("osm_loudness_sma_mean", 0.0))
        voicing = float(feature_dict.get("osm_voicingProb_sma_mean", 0.0))
        return np.clip(0.35 * loudness + 0.15 * voicing, -0.5, 0.5)

    def get_feature_dict(self, feature_dict: Dict[str, float]) -> Dict[str, float]:
        ser_vector = self._extract_ser_vector(feature_dict)
        probs = self._model.predict(ser_vector)

        adjustment = self._energy_adjustment(feature_dict)
        probs = probs + np.array([-adjustment, -0.5 * adjustment, adjustment])
        probs = np.clip(probs, 1e-5, None)
        probs /= np.sum(probs)

        dominant_idx = int(np.argmax(probs))
        dominant_label = SENTIMENT_LABELS[dominant_idx]

        return {
            "arvs_negative": float(probs[0]),
            "arvs_neutral": float(probs[1]),
            "arvs_positive": float(probs[2]),
            "arvs_dominant_sentiment": dominant_label,
            "arvs_confidence": float(np.max(probs)),
        }


def create_demo_sentiment_analyzer() -> AudioSentimentAnalyzer:
    return AudioSentimentAnalyzer()


__all__ = ["AudioSentimentAnalyzer", "create_demo_sentiment_analyzer"]

