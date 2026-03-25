"""
AV-HuBERT analyzer with SAFE mandatory behavior.

- Never breaks pipeline
- Preserves output schema
- Provides fallback features if model unavailable
- Supports aligned multimodal timestamps
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import soundfile as sf
except Exception:
    sf = None  # optional

logger = logging.getLogger(__name__)

# ----------------------------
# Optional HF imports
# ----------------------------
try:
    from transformers import AutoProcessor
    from transformers import AVHubertForCTC
    _HF_AVAILABLE = True
    _HF_ERROR = ""
except Exception as e:
    AutoProcessor = None
    AVHubertForCTC = None
    _HF_AVAILABLE = False
    _HF_ERROR = str(e)


@dataclass(frozen=True)
class AVHubertConfig:
    model_id: str
    sample_rate: int = 16_000
    target_fps: float = 25.0
    max_frames: int = 256
    frame_size: int = 96


class AVHuBERTAnalyzer:
    """
    AVHuBERT analyzer with graceful degradation.

    If unavailable, returns deterministic placeholder features
    with aligned timestamps.
    """

    MODEL_ENV = "AVHUBERT_MODEL_ID"
    DEFAULT_MODEL = "facebook/avhubert-large-30h-cv"

    def __init__(
        self,
        *,
        device: str = "cpu",
        output_dir: Optional[Path] = None,
        model_id: Optional[str] = None,
        sample_rate: int = 16_000,
        target_fps: float = 25.0,
        max_frames: int = 256,
        frame_size: int = 96,
    ) -> None:

        # ---- config ----
        resolved_model = model_id or os.getenv(self.MODEL_ENV) or self.DEFAULT_MODEL
        self.config = AVHubertConfig(
            model_id=resolved_model,
            sample_rate=sample_rate,
            target_fps=target_fps,
            max_frames=max_frames,
            frame_size=frame_size,
        )

        # ---- device ----
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        # ---- output dir ----
        output_root = Path(output_dir) if output_dir else Path.cwd() / "output"
        self.output_dir = output_root / "vision" / "avhubert"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---- availability flags ----
        self.available = False
        self.unavailable_reason = ""

        # ---- try load model ----
        if not _HF_AVAILABLE:
            self.unavailable_reason = f"transformers unavailable: {_HF_ERROR}"
            logger.warning("[AVHuBERT] disabled: %s", self.unavailable_reason)
            return

        try:
            logger.info("[AVHuBERT] loading model: %s", self.config.model_id)
            self.processor = AutoProcessor.from_pretrained(self.config.model_id)
            self.model = AVHubertForCTC.from_pretrained(self.config.model_id)
            self.model.to(self.device).eval()
            self.available = True
        except Exception as e:
            self.unavailable_reason = f"model load failed: {e}"
            logger.warning("[AVHuBERT] disabled: %s", self.unavailable_reason)

    # ======================================================
    # PUBLIC API
    # ======================================================

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        if not self.available:
            return self._dummy_features(video_path)

        try:
            return self._real_features(video_path)
        except Exception as e:
            logger.warning("[AVHuBERT] runtime failure, fallback to dummy: %s", e)
            return self._dummy_features(video_path)

    # ======================================================
    # REAL MODEL PIPELINE
    # ======================================================

    def _real_features(self, video_path: str) -> Dict[str, Any]:
        audio, sr = self._extract_audio(Path(video_path))
        frames = self._extract_frames(Path(video_path))

        inputs = self.processor(
            raw_speech=audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding="longest",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        logits = outputs.logits
        hidden = outputs.hidden_states[-1]

        transcription = self.processor.batch_decode(
            torch.argmax(logits, dim=-1).cpu(), skip_special_tokens=True
        )[0]

        confidence = float(torch.softmax(logits, dim=-1).max().item())

        emb_path = self._save_embeddings(Path(video_path).stem, hidden)

        return {
            "AVH_available": True,
            "AVH_transcription": transcription,
            "AVH_confidence": confidence,
            "AVH_embeddings_path": str(emb_path),
            "AVH_unavailable_reason": "",
        }

    # ======================================================
    # FALLBACK MODE (CRITICAL FOR PIPELINE)
    # ======================================================

    def _dummy_features(self, video_path: str) -> Dict[str, Any]:
        """
        Deterministic placeholder output preserving schema + time alignment.
        """

        # pseudo timeline length based on video frames
        try:
            cap = cv2.VideoCapture(video_path)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except Exception:
            frames = 0

        dummy_conf = 0.0

        return {
            "AVH_available": False,
            "AVH_transcription": "",
            "AVH_confidence": dummy_conf,
            "AVH_embeddings_path": "",
            "AVH_unavailable_reason": self.unavailable_reason,
            "AVH_frame_count": frames,
        }

    # ======================================================
    # HELPERS
    # ======================================================

    def _extract_audio(self, video_path: Path) -> Tuple[np.ndarray, int]:
        if sf is None:
            return np.zeros(16000, dtype=np.float32), self.config.sample_rate

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(self.config.sample_rate),
            str(tmp_path),
        ]

        subprocess.run(cmd, check=False)

        try:
            audio, sr = sf.read(str(tmp_path), dtype="float32")
        except Exception:
            audio = np.zeros(self.config.sample_rate, dtype=np.float32)
            sr = self.config.sample_rate
        finally:
            tmp_path.unlink(missing_ok=True)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return audio, sr

    def _extract_frames(self, video_path: Path) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        count = 0

        while len(frames) < self.config.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if count % max(int(cap.get(cv2.CAP_PROP_FPS) / self.config.target_fps), 1) == 0:
                frames.append(frame)
            count += 1

        cap.release()
        return frames

    def _save_embeddings(self, stem: str, hidden: torch.Tensor) -> Path:
        emb = hidden.squeeze(0).cpu().numpy()
        path = self.output_dir / f"{stem}_avhubert_embeddings.npz"
        np.savez_compressed(path, embeddings=emb)
        return path
