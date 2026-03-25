"""MELD-style conversational emotion features via pretrained transformers."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from transformers import pipeline

logger = logging.getLogger(__name__)


class MELDEmotionAnalyzer:
    """Infer dialogue-level MELD statistics using a text emotion classifier."""

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        *,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.emotion_categories = [
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
            "surprise",
        ]
        resolved_device = self._resolve_device(device)
        try:
            self._classifier = pipeline(
                task="text-classification",
                model=model_name,
                device=resolved_device,
                cache_dir=cache_dir,
                return_all_scores=True,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error("Failed to load MELD emotion classifier %s: %s", model_name, exc)
            self._classifier = None

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

    def analyze_conversation_emotions(
        self, transcript: str, speaker_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze emotions in conversational transcript using MELD-inspired features.
        
        Args:
            transcript: Full conversation transcript
            speaker_info: Optional speaker diarization information
            
        Returns:
            Dictionary containing MELD-style emotion analysis features
        """
        features = {}

        # Parse transcript into utterances and speakers
        utterances = self._parse_transcript(transcript, speaker_info)
        
        # Basic conversation statistics
        features.update(self._calculate_basic_stats(utterances, transcript))
        
        # Emotion detection and counting
        features.update(self._analyze_emotions(utterances))
        
        # Emotion shift analysis
        features.update(self._analyze_emotion_shifts(utterances))
        
        # Speaker-specific analysis
        features.update(self._analyze_speaker_patterns(utterances))
        
        # Dialogue structure analysis
        features.update(self._analyze_dialogue_structure(utterances))
        
        return features
    
    def _parse_transcript(self, transcript: str, speaker_info: Dict = None) -> List[Dict]:
        """
        Parse transcript into structured utterances with speaker information.
        
        Args:
            transcript: Raw transcript text
            speaker_info: Speaker diarization data
            
        Returns:
            List of utterance dictionaries with speaker and emotion info
        """
        utterances = []
        
        # Split transcript into sentences/utterances
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If we have speaker info from WhisperX, use it
        if speaker_info and isinstance(speaker_info, dict):
            # Extract speaker segments from WhisperX output
            speaker_segments = []
            for key, value in speaker_info.items():
                if 'SPEAKER_' in key and 'word_' in key:
                    # Extract speaker and word info
                    parts = key.split('_')
                    if len(parts) >= 3:
                        speaker_id = f"{parts[1]}_{parts[2]}"  # SPEAKER_01
                        speaker_segments.append({
                            'speaker': speaker_id,
                            'text': str(value),
                            'key': key
                        })
            
            # Group by speakers and create utterances
            current_speaker = None
            current_text = ""
            
            for segment in speaker_segments:
                if segment['speaker'] != current_speaker:
                    if current_text.strip():
                        utterances.append({
                            'speaker': current_speaker or 'SPEAKER_01',
                            'text': current_text.strip(),
                            'length': len(current_text.strip()),
                            'word_count': len(current_text.strip().split())
                        })
                    current_speaker = segment['speaker']
                    current_text = segment['text']
                else:
                    current_text += " " + segment['text']
            
            # Add final utterance
            if current_text.strip():
                utterances.append({
                    'speaker': current_speaker or 'SPEAKER_01',
                    'text': current_text.strip(),
                    'length': len(current_text.strip()),
                    'word_count': len(current_text.strip().split())
                })
        
        # Fallback: create utterances from sentences with estimated speakers
        if not utterances:
            for i, sentence in enumerate(sentences):
                # Simple speaker alternation assumption
                speaker_id = f"SPEAKER_{i % 2:02d}"
                utterances.append({
                    'speaker': speaker_id,
                    'text': sentence,
                    'length': len(sentence),
                    'word_count': len(sentence.split())
                })
        
        # Add emotion analysis to each utterance via pretrained classifier
        self._annotate_emotions(utterances)
        
        return utterances

    def _annotate_emotions(self, utterances: List[Dict[str, Any]]) -> None:
        if not utterances:
            return

        texts = [utterance["text"] for utterance in utterances]
        if self._classifier is None:
            model_outputs = [[{"label": "neutral", "score": 1.0}] for _ in texts]
        else:
            try:
                model_outputs = self._classifier(texts, truncation=True)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.error("MELD classifier failed: %s", exc)
                model_outputs = [[{"label": "neutral", "score": 1.0}] for _ in texts]

        for utterance, predictions in zip(utterances, model_outputs):
            distribution = {entry["label"].lower(): float(entry["score"]) for entry in predictions}
            normalized = {emotion: distribution.get(emotion, 0.0) for emotion in self.emotion_categories}
            total = sum(normalized.values())
            if total > 0:
                normalized = {k: v / total for k, v in normalized.items()}
            else:
                normalized = {emotion: 1.0 / len(self.emotion_categories) for emotion in self.emotion_categories}

            utterance["emotions"] = normalized
            utterance["dominant_emotion"] = max(normalized, key=normalized.get)
            utterance["emotion_intensity"] = float(max(normalized.values()))
    
    def _calculate_basic_stats(self, utterances: List[Dict], transcript: str) -> Dict[str, Any]:
        """Calculate basic conversation statistics."""
        features = {}
        
        # Modality (text-based analysis)
        features['MELD_modality'] = 'text'
        
        # Unique words
        all_words = []
        for utterance in utterances:
            words = re.findall(r'\b\w+\b', utterance['text'].lower())
            all_words.extend(words)
        features['MELD_unique_words'] = len(set(all_words))
        
        # Utterance length statistics
        lengths = [utterance['length'] for utterance in utterances]
        features['MELD_avg_utterance_length'] = np.mean(lengths) if lengths else 0.0
        features['MELD_max_utterance_length'] = max(lengths) if lengths else 0
        
        # Number of utterances
        features['MELD_num_utterances'] = len(utterances)
        
        # Estimate utterance duration (simplified)
        # Assume ~150 words per minute speaking rate
        word_counts = [utterance['word_count'] for utterance in utterances]
        durations = [wc / 2.5 for wc in word_counts]  # ~2.5 words per second
        features['MELD_avg_utterance_duration'] = np.mean(durations) if durations else 0.0
        
        return features
    
    def _analyze_emotions(self, utterances: List[Dict]) -> Dict[str, Any]:
        """Analyze emotion distribution in the conversation."""
        features = {}
        
        # Count emotions
        emotion_counts = Counter()
        total_emotions = 0
        
        for utterance in utterances:
            dominant_emotion = utterance['dominant_emotion']
            emotion_counts[dominant_emotion] += 1
            total_emotions += 1
        
        # Set emotion counts for each MELD emotion category
        for emotion in self.emotion_categories:
            features[f'MELD_count_{emotion}'] = emotion_counts.get(emotion, 0)
        
        # Average emotions per dialogue (simplified as per utterance)
        features['MELD_avg_num_emotions_per_dialogue'] = total_emotions / max(len(utterances), 1)
        
        return features
    
    def _analyze_emotion_shifts(self, utterances: List[Dict]) -> Dict[str, Any]:
        """Analyze emotion transitions and shifts in the conversation."""
        features = {}
        
        # Count emotion shifts
        emotion_shifts = 0
        for i in range(1, len(utterances)):
            prev_emotion = utterances[i-1]['dominant_emotion']
            curr_emotion = utterances[i]['dominant_emotion']
            if prev_emotion != curr_emotion:
                emotion_shifts += 1
        
        features['MELD_num_emotion_shift'] = emotion_shifts
        
        return features
    
    def _analyze_speaker_patterns(self, utterances: List[Dict]) -> Dict[str, Any]:
        """Analyze speaker-specific patterns."""
        features = {}
        
    # Count unique speakers
        speakers = set(utterance['speaker'] for utterance in utterances)
        features['MELD_num_speakers'] = len(speakers)
        
        # Count dialogues (simplified as speaker changes)
        dialogue_count = 1
        current_speaker = utterances[0]['speaker'] if utterances else None
        
        for utterance in utterances[1:]:
            if utterance['speaker'] != current_speaker:
                dialogue_count += 1
                current_speaker = utterance['speaker']
        
        features['MELD_num_dialogues'] = dialogue_count
        
        return features
    
    def _analyze_dialogue_structure(self, utterances: List[Dict]) -> Dict[str, Any]:
        """Analyze the structure of the dialogue."""
        features = {}
        
        # This could be extended with more sophisticated dialogue analysis
        # For now, we include basic structural features
        
        # Calculate average emotions per dialogue segment
        if utterances:
            non_neutral = [u for u in utterances if u.get('dominant_emotion') != 'neutral']
            features['MELD_avg_num_emotions_per_dialogue'] = len(non_neutral) / max(features.get('MELD_num_dialogues', 1), 1)
        else:
            features['MELD_avg_num_emotions_per_dialogue'] = 0.0
        
        return features
    
    def get_feature_dict(self, transcript_data: Any) -> Dict[str, Any]:
        """
        Main interface for extracting MELD emotion features.
        
        Args:
            transcript_data: Can be string transcript or dict with transcript and speaker info
            
        Returns:
            Dictionary of MELD emotion features
        """
        try:
            # Handle different input types
            if isinstance(transcript_data, str):
                transcript = transcript_data
                speaker_info = None
            elif isinstance(transcript_data, dict):
                transcript = transcript_data.get('transcription', transcript_data.get('text', ''))
                speaker_info = transcript_data
            else:
                transcript = str(transcript_data)
                speaker_info = None
            
            if not transcript or transcript.strip() == '':
                logger.warning("Empty transcript provided to MELD analyzer")
                return self._get_default_features()
            
            # Analyze emotions in conversation
            features = self.analyze_conversation_emotions(transcript, speaker_info)
            
            # Add metadata
            features.update({
                'MELD_analysis_timestamp': np.datetime64('now').astype(str),
                'MELD_text_length': len(transcript),
                'MELD_model_name': 'MELD_conversation_emotion_analyzer',
                'MELD_version': '1.0.0'
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Error in MELD emotion analysis: {e}")
            return self._get_default_features()
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Return default features when analysis fails."""
        features = {}
        
        # Set default values for all MELD features
        features['MELD_modality'] = 'text'
        features['MELD_unique_words'] = 0
        features['MELD_avg_utterance_length'] = 0.0
        features['MELD_max_utterance_length'] = 0
        features['MELD_avg_num_emotions_per_dialogue'] = 0.0
        features['MELD_num_dialogues'] = 0
        features['MELD_num_utterances'] = 0
        features['MELD_num_speakers'] = 0
        features['MELD_num_emotion_shift'] = 0
        features['MELD_avg_utterance_duration'] = 0.0
        
        # Emotion counts
        for emotion in self.emotion_categories:
            features[f'MELD_count_{emotion}'] = 0
        
        # Metadata
        features.update({
            'MELD_analysis_timestamp': np.datetime64('now').astype(str),
            'MELD_text_length': 0,
            'MELD_model_name': 'MELD_conversation_emotion_analyzer',
            'MELD_version': '1.0.0'
        })
        
        return features


def create_meld_analyzer():
    """Factory function to create MELD emotion analyzer."""
    return MELDEmotionAnalyzer()
