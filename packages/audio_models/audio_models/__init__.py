"""Audio models and utilities for the multimodal pipeline."""

__version__ = "0.1.0"

from .audio import (
    AudioFeatureExtractor,
    LibrosaFeatureExtractor,
    OpenSMILEFeatureExtractor,
    AudioStretchyAnalyzer,
)
from .speech import (
    SpeechEmotionRecognizer,
    SpeechSeparator,
    WhisperXTranscriber,
)
from .utils.audio_extraction import (
    extract_audio_from_video,
    extract_audio_from_videos,
)

__all__ = [
    "AudioFeatureExtractor",
    "LibrosaFeatureExtractor",
    "OpenSMILEFeatureExtractor",
    "AudioStretchyAnalyzer",
    "SpeechEmotionRecognizer",
    "SpeechSeparator",
    "WhisperXTranscriber",
    "extract_audio_from_video",
    "extract_audio_from_videos",
]
