"""Speech emotion recognition wrapper around the upstream SER repository."""

from __future__ import annotations

import importlib.util
import logging
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from audio_models.external.repo_manager import ensure_repo

logger = logging.getLogger(__name__)


class SpeechEmotionRecognizer:
    """Infer speech emotion probabilities using the official SER implementation.

    The ``x4nth055/emotion-recognition-using-speech`` project ships a
    scikit-learn based pipeline trained on the RAVDESS, TESS and EMO-DB
    datasets.  Rather than mimicking that behaviour locally, this wrapper keeps
    a clone of the original repository (managed through
    :mod:`audio_models.external.repo_manager`) and loads its exported
    ``grid/best_classifiers.pickle`` model directly for inference.
    """

    DEFAULT_FEATURES: Sequence[str] = ("mfcc", "chroma", "mel")
    _REQUIRED_FILES: Sequence[str] = (
        "emotion_recognition.py",
        "utils.py",
        "convert_wavs.py",
        "grid/best_classifiers.pickle",
    )

    def __init__(
        self,
        *,
        repo_path: Optional[Path] = None,
        features: Optional[Sequence[str]] = None,
        estimator_index: Optional[int] = None,
    ) -> None:
        """Create a recogniser backed by the real SER repository.

        Args:
            repo_path: Optional override pointing to a local checkout of the
                upstream repository.  When omitted the repo is cloned (or
                reused) under ``external/audio``.
            features: Feature list understood by the upstream extractor.  The
                defaults mirror the original project configuration
                (MFCC/Chroma/MEL).
            estimator_index: Optional index into the upstream ``grid`` results
                when you prefer a specific estimator.  By default the wrapper
                picks the model with the highest cross-validation score.
        """

        self._repo_path = Path(repo_path) if repo_path else ensure_repo("emotion_recognition")
        self._ensure_required_files()
        self._register_repo_on_path()
        self._ser_utils = self._load_utils_module()

        self._features = tuple(features or self.DEFAULT_FEATURES)
        self._audio_config = self._ser_utils.get_audio_config(list(self._features))
        self._extract_feature = self._ser_utils.extract_feature

        self._model, cv_score = self._load_estimator(estimator_index)
        logger.debug("Loaded SER estimator %s (cv_score=%.4f)", self._model.__class__.__name__, cv_score)

        classes = getattr(self._model, "classes_", None)
        if classes is None:
            raise RuntimeError("Loaded SER estimator does not expose `classes_`." )

        self.emotions: List[str] = [str(label).lower() for label in classes]

    def _register_repo_on_path(self) -> None:
        repo_str = str(self._repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def _ensure_required_files(self) -> None:
        missing: List[Path] = [self._repo_path / rel for rel in self._REQUIRED_FILES if not (self._repo_path / rel).exists()]
        if not missing:
            return

        rel_paths = [path.relative_to(self._repo_path).as_posix() for path in missing]
        try:
            subprocess.run(
                ["git", "checkout", "HEAD", "--", *rel_paths],
                cwd=self._repo_path,
                check=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
            logger.warning("Could not restore SER files %s: %s", ", ".join(rel_paths), exc)

    def _load_utils_module(self):
        spec = importlib.util.spec_from_file_location(
            "ser_utils",
            self._repo_path / "utils.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load utils module from SER repository")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_estimator(self, estimator_index: Optional[int]):
        grid_path = self._repo_path / "grid" / "best_classifiers.pickle"
        if not grid_path.exists():
            raise FileNotFoundError(f"Expected SER estimator at {grid_path}")

        try:
            with grid_path.open("rb") as handle:
                estimators = pickle.load(handle)
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Loading SER estimators requires scikit-learn to be installed. "
                "Re-run `poetry install` inside the audio_models package."
            ) from exc

        usable = [entry for entry in estimators if hasattr(entry[0], "predict_proba")]
        if not usable:
            raise RuntimeError("No classifier with predict_proba available in SER grid results")

        if estimator_index is not None:
            if estimator_index < 0 or estimator_index >= len(usable):
                raise IndexError(f"Estimator index {estimator_index} out of range (0-{len(usable)-1})")
            chosen = usable[estimator_index]
        else:
            chosen = max(usable, key=lambda item: item[2])

        model, _, cv_score = chosen
        return model, cv_score

    def predict(self, audio_path: str) -> Dict[str, float]:
        """Return per-emotion probabilities for the given audio clip."""

        feature_vector = np.asarray(
            self._extract_feature(audio_path, **self._audio_config),
            dtype=np.float32,
        )
        if feature_vector.size == 0:
            logger.warning("SER feature extractor returned an empty vector for %s", audio_path)
            uniform = 1.0 / len(self.emotions)
            return {f"ser_{emotion}": uniform for emotion in self.emotions}
        if feature_vector.ndim == 0:
            feature_vector = feature_vector.reshape(1)

        model_input = feature_vector.reshape(1, -1)
        probabilities = self._model.predict_proba(model_input)[0]

        # Normalise defensively in case numerical drift occurred when unpickling.
        probabilities = np.asarray(probabilities, dtype=np.float64)
        total = probabilities.sum()
        if not np.isclose(total, 1.0):
            probabilities = probabilities / total if total else np.full_like(probabilities, 1.0 / len(probabilities))

        return {
            f"ser_{emotion}": float(prob)
            for emotion, prob in zip(self.emotions, probabilities)
        }

