"""ELMo-style contextual embedding analyzer.

The implementation favours graceful degradation: if TensorFlow Hub is
available we use the official ELMo weights, otherwise we fall back to a
simple hashed embedding that still provides deterministic, useful
statistics for downstream consumers.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:  # Optional dependency
    import tensorflow as tf  # type: ignore
    import tensorflow_hub as hub  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tf = None  # type: ignore
    hub = None  # type: ignore


class ELMoAnalyzer:
    """Generate contextualised embeddings similar to ELMo."""

    DEFAULT_MODEL_URL = "https://tfhub.dev/google/elmo/3"

    def __init__(
        self,
        *,
        model_url: str = DEFAULT_MODEL_URL,
        device: str = "cpu",
        output_dir: Optional[Path] = None,
    ) -> None:
        self.model_url = model_url
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir = self.output_dir / "nlp" / "elmo"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._elmo_layer = None
        if hub is not None and tf is not None:
            try:
                self._elmo_layer = hub.load(self.model_url)
                logger.info("Loaded ELMo model from %s", self.model_url)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("Failed to load ELMo hub module: %s", exc)
                self._elmo_layer = None

    # ------------------------------------------------------------------
    def _extract_text(self, payload: Union[str, Dict[str, Any]]) -> str:
        if isinstance(payload, str):
            return payload
        if not isinstance(payload, dict):
            return ""
        text_keys = (
            "transcription",
            "whisperx_transcription",
            "xlsr_transcription",
            "s2t_text",
            "text",
            "transcript",
        )
        for key in text_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _hashed_embedding(self, tokens: List[str]) -> np.ndarray:
        if not tokens:
            return np.zeros((1, 128), dtype=np.float32)
        vectors = []
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            base = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            # Tile to a consistent 128-dim embedding
            repeats = int(np.ceil(128 / base.size))
            vector = np.tile(base, repeats)[:128]
            vector = (vector - vector.mean()) / (vector.std() + 1e-6)
            vectors.append(vector)
        return np.stack(vectors, axis=0)

    def _run_elmo(self, sentences: List[str]) -> np.ndarray:
        if self._elmo_layer is None or tf is None:
            tokens = [token for sentence in sentences for token in sentence.split()]
            return self._hashed_embedding(tokens)
        try:
            embeddings = self._elmo_layer.signatures["default"](tf.constant(sentences))[
                "elmo"
            ].numpy()
            return embeddings.reshape(-1, embeddings.shape[-1])
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("ELMo inference failed: %s", exc)
            tokens = [token for sentence in sentences for token in sentence.split()]
            return self._hashed_embedding(tokens)

    # ------------------------------------------------------------------
    def get_feature_dict(self, payload: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        text = self._extract_text(payload)
        if not text:
            logger.warning("No text available for ELMo analysis")
            return {
                "elmo_embedding_path": "",
                "elmo_embedding_mean": 0.0,
                "elmo_embedding_std": 0.0,
                "elmo_token_count": 0,
                "elmo_model_url": self.model_url if self._elmo_layer else "fallback",
                "elmo_device": self.device,
            }

        sentences = [sentence.strip() for sentence in text.split(". ") if sentence.strip()]
        if not sentences:
            sentences = [text.strip()]
        embeddings = self._run_elmo(sentences)
        text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        embedding_path = self.output_dir / f"elmo_embeddings_{text_hash}.npy"
        np.save(embedding_path, embeddings)

        return {
            "elmo_embedding_path": str(embedding_path),
            "elmo_embedding_mean": float(np.mean(embeddings)),
            "elmo_embedding_std": float(np.std(embeddings)),
            "elmo_token_count": int(embeddings.shape[0]),
            "elmo_embedding_dim": int(embeddings.shape[1] if embeddings.ndim > 1 else embeddings.shape[0]),
            "elmo_model_url": self.model_url if self._elmo_layer else "fallback",
            "elmo_device": self.device,
        }


__all__ = ["ELMoAnalyzer"]
