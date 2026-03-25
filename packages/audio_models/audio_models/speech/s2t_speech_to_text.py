"""Speech-to-text analyzer backed by the official Speech2Text checkpoints.

This implementation runs Facebook's Speech2Text models via Hugging Face
``transformers`` and produces transcripts together with a confidence score and
coarse per-word timing derived from the audio duration.  No heuristic fallback
text is emitted—if the model cannot load or execute the caller receives a
runtime error describing the missing dependency.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
import torch

try:
    from transformers import (
        Speech2TextForConditionalGeneration,
        Speech2TextProcessor,
    )
except Exception as exc:  # pragma: no cover - import guard
    Speech2TextForConditionalGeneration = None  # type: ignore[assignment]
    Speech2TextProcessor = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


def _resolve_device(device_spec: str) -> torch.device:
    if device_spec.startswith("cuda"):
        if not torch.cuda.is_available():  # pragma: no cover - depends on env
            raise RuntimeError("CUDA requested for S2T but no GPU is available")
        return torch.device(device_spec)
    if device_spec != "cpu":
        logger.warning("Unknown device '%s', defaulting to CPU", device_spec)
    return torch.device("cpu")


class S2TSpeechToTextAnalyzer:
    """Produce Speech-to-Text (S2T) features using a pretrained model."""

    DEFAULT_MODEL = "facebook/s2t-small-librispeech-asr"

    def __init__(
        self,
        *,
        device: str = "cpu",
        output_dir: Optional[Path] = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        if Speech2TextProcessor is None or Speech2TextForConditionalGeneration is None:
            raise RuntimeError(
                "transformers is not installed. Run 'poetry install' inside packages/audio_models "
                "or add transformers to your environment."
            ) from _TRANSFORMERS_IMPORT_ERROR

        self.device = _resolve_device(device)
        self.model_name = model_name
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir = self.output_dir / "audio" / "s2t"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading Speech2Text model %s", self.model_name)
        self.processor = Speech2TextProcessor.from_pretrained(self.model_name)
        self.model = Speech2TextForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self._target_sr = int(self.processor.feature_extractor.sampling_rate)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_alignment(transcript: str, duration: float) -> Dict[str, object]:
        words = [word for word in transcript.split() if word]
        if not words:
            return {"segments": [], "duration": round(max(duration, 0.0), 3)}

        duration = max(duration, len(words) * 0.25)  # guard against zero-length audio
        step = duration / len(words)
        segments = []
        for index, word in enumerate(words):
            start = round(index * step, 3)
            end = round((index + 1) * step, 3)
            segments.append({"word": word, "start": start, "end": end})
        return {"segments": segments, "duration": round(duration, 3)}

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        audio, sr = librosa.load(str(audio_path), sr=self._target_sr)
        if audio.size == 0:
            raise RuntimeError(f"Audio file '{audio_path}' did not contain samples")
        if sr != self._target_sr:
            logger.debug("Resampled %s to %d Hz for S2T", audio_path, self._target_sr)
        return audio.astype(np.float32)

    # ------------------------------------------------------------------
    def get_feature_dict(
        self,
        audio_path: str,
        existing_features: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        del existing_features  # Unused: the real model handles transcription

        audio_file = Path(audio_path)
        waveform = self._load_audio(audio_file)
        inputs = self.processor(
            waveform,
            sampling_rate=self._target_sr,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with torch.no_grad():
            generation = self.model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=1024,
            )

        transcript = self.processor.batch_decode(
            generation.sequences, skip_special_tokens=True
        )[0].strip()
        if not transcript:
            raise RuntimeError(
                f"Speech2Text model '{self.model_name}' produced an empty transcription for {audio_file}"
            )

        if generation.sequences_scores is not None:
            score = float(torch.exp(generation.sequences_scores[0]).item())
        else:  # pragma: no cover - depends on transformers version
            score = 0.0

        duration = float(waveform.shape[0]) / float(self._target_sr)
        alignment = self._build_alignment(transcript, duration)

        alignment_path = self.output_dir / f"{audio_file.stem}_s2t_alignment.json"
        with alignment_path.open("w", encoding="utf-8") as handle:
            json.dump(alignment, handle, indent=2)

        confidence = np.full(len(alignment["segments"]), score, dtype=np.float32)
        confidence_path = self.output_dir / f"{audio_file.stem}_s2t_confidence.npy"
        np.save(confidence_path, confidence)

        return {
            "s2t_text": transcript,
            "s2t_score": score,
            "s2t_alignment_path": str(alignment_path),
            "s2t_confidence_path": str(confidence_path),
            "s2t_model_name": self.model_name,
            "s2t_inference_device": str(self.device),
            "s2t_audio_duration": duration,
            "s2t_fallback_used": False,
        }


__all__ = ["S2TSpeechToTextAnalyzer"]
