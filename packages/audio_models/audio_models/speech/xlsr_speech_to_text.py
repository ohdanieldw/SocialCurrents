"""XLSR speech-to-text analyzer powered by wav2vec 2.0 checkpoints.

The implementation loads Facebook's multilingual XLSR models via
``transformers`` and exposes both the textual transcription and the last hidden
layer activations.  No synthetic fallbacks are emitted—missing dependencies
result in descriptive runtime errors so the caller can remedy the
environment.
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
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
except Exception as exc:  # pragma: no cover - import guard
    Wav2Vec2ForCTC = None  # type: ignore[assignment]
    Wav2Vec2Processor = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


def _resolve_device(device_spec: str) -> torch.device:
    if device_spec.startswith("cuda"):
        if not torch.cuda.is_available():  # pragma: no cover - env dependent
            raise RuntimeError("CUDA requested for XLSR but no GPU is available")
        return torch.device(device_spec)
    if device_spec != "cpu":
        logger.warning("Unknown device '%s', defaulting to CPU", device_spec)
    return torch.device("cpu")


class XLSRSpeechToTextAnalyzer:
    """Produce XLSR speech-to-text outputs using wav2vec 2.0."""

    DEFAULT_MODEL = "facebook/wav2vec2-large-xlsr-53"

    def __init__(
        self,
        *,
        device: str = "cpu",
        output_dir: Optional[Path] = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        if Wav2Vec2Processor is None or Wav2Vec2ForCTC is None:
            raise RuntimeError(
                "transformers is not installed. Run 'poetry install' inside packages/audio_models "
                "or ensure transformers is available."
            ) from _TRANSFORMERS_IMPORT_ERROR

        self.device = _resolve_device(device)
        self.model_name = model_name
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir = self.output_dir / "audio" / "xlsr"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading XLSR wav2vec2 model %s", self.model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self._target_sr = int(self.processor.feature_extractor.sampling_rate)

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        waveform, sr = librosa.load(str(audio_path), sr=self._target_sr)
        if waveform.size == 0:
            raise RuntimeError(f"Audio file '{audio_path}' did not contain samples")
        if sr != self._target_sr:
            logger.debug("Resampled %s to %d Hz for XLSR", audio_path, self._target_sr)
        return waveform.astype(np.float32)

    def get_feature_dict(
        self,
        audio_path: str,
        existing_features: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        del existing_features  # Real model handles transcription

        audio_file = Path(audio_path)
        waveform = self._load_audio(audio_file)
        inputs = self.processor(
            waveform,
            sampling_rate=self._target_sr,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = self.processor.batch_decode(predicted_ids)[0].strip()
        if not transcript:
            raise RuntimeError(
                f"wav2vec2 model '{self.model_name}' produced an empty transcription for {audio_file}"
            )

        frame_confidence = torch.softmax(logits, dim=-1).max(dim=-1).values
        confidence = float(frame_confidence.mean().item())

        hidden = outputs.hidden_states[-1].squeeze(0).cpu().float().numpy()
        hidden_path = self.output_dir / f"{audio_file.stem}_xlsr_hidden.npy"
        np.save(hidden_path, hidden)

        duration = float(waveform.shape[0]) / float(self._target_sr)
        frame_stride = duration / max(hidden.shape[0], 1)
        frame_timestamps = [
            {
                "index": int(idx),
                "start": round(idx * frame_stride, 3),
                "end": round((idx + 1) * frame_stride, 3),
            }
            for idx in range(hidden.shape[0])
        ]

        metadata_path = self.output_dir / f"{audio_file.stem}_xlsr_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "audio_path": str(audio_file),
                    "model": self.model_name,
                    "device": str(self.device),
                    "num_frames": int(hidden.shape[0]),
                    "hidden_dim": int(hidden.shape[1]) if hidden.ndim == 2 else None,
                    "duration_seconds": duration,
                    "frame_stride_seconds": frame_stride,
                    "frame_timestamps": frame_timestamps,
                },
                handle,
                indent=2,
            )

        return {
            "xlsr_transcription": transcript,
            "xlsr_confidence": confidence,
            "xlsr_hidden_states_path": str(hidden_path),
            "xlsr_num_hidden_frames": int(hidden.shape[0]),
            "xlsr_inference_device": str(self.device),
            "xlsr_model_name": self.model_name,
            "xlsr_metadata_path": str(metadata_path),
            "xlsr_fallback_used": False,
        }


__all__ = ["XLSRSpeechToTextAnalyzer"]
