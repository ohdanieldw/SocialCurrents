"""WhisperX for time-accurate speech transcription and diarization."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from audio_models.external.repo_manager import ensure_repo

class WhisperXTranscriber:
    """Transcribe and diarize speech using WhisperX."""
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "float32",
        language: str = "en",
        batch_size: int = 16,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the WhisperX transcriber.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device to run on ("cpu" or "cuda")
            compute_type: Type for computation ("float32", "float16", "int8")
            language: Language code for transcription
            batch_size: Batch size for processing
            hf_token: HuggingFace token for accessing gated models
        """
        self.model_size = model_size
        self.device = device
        # Force CPU for macOS and ensure compatible compute type
        if device == "cpu":
            self.compute_type = "float32"  # Force float32 for CPU
        else:
            self.compute_type = compute_type
        self.language = language
        self.batch_size = batch_size
        # Prefer explicit token, otherwise pull from environment
        if hf_token is not None:
            self.hf_token = hf_token
        else:
            self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        
        # Models will be loaded on-demand
        self.model = None
        self.diarization_model = None
        
    def _load_models(self):
        """Load WhisperX and diarization models."""
        repo_path = ensure_repo("whisperx")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        try:
            import whisperx  # type: ignore
            
            print(f"Loading WhisperX model: {self.model_size} on {self.device} with compute_type: {self.compute_type}")
            
            # Load ASR model
            self.model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type,
                language=self.language
            )
            
            print("Loading diarization model...")
            # Load diarization model with proper device handling
            try:
                from whisperx.diarize import DiarizationPipeline  # type: ignore
                self.diarization_model = DiarizationPipeline(
                    use_auth_token=self.hf_token,
                    device=self.device
                )
            except ImportError:
                # Fallback for older versions of WhisperX
                try:
                    self.diarization_model = whisperx.DiarizationPipeline(
                        use_auth_token=self.hf_token,
                        device=self.device
                    )
                except Exception as e:
                    print(f"Warning: Could not load diarization model: {e}")
                    print("Continuing without diarization...")
                    self.diarization_model = None
            except Exception as e:
                print(f"Warning: Could not load diarization model: {e}")
                print("Continuing without diarization...")
                self.diarization_model = None
                
        except ImportError as exc:
            print(
                "WhisperX is not importable even after cloning. "
                "Run `pip install -e external/audio/whisperx` or install the package in this environment."
            )
            raise
    
    def transcribe(
        self, 
        audio_path: str,
        min_speakers: int = None,
        max_speakers: int = None
    ) -> Dict[str, Any]:
        """
        Transcribe and diarize an audio file.
        
        Args:
            audio_path: Path to the audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            
        Returns:
            Dict[str, Any]: Transcription results with diarization
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Load models if not already loaded
        if self.model is None:
            self._load_models()
            
        try:
            import whisperx
            
            print("Starting transcription...")
            # Transcribe with word-level timestamps
            transcribe_result = self.model.transcribe(
                audio_path,
                batch_size=self.batch_size
            )
            print(f"Transcription completed. Language detected: {transcribe_result.get('language', 'unknown')}")
            
            # Align the transcription
            print("Loading alignment model...")
            align_model, align_metadata = whisperx.load_align_model(
                language_code=transcribe_result["language"],
                device=self.device
            )
            
            print("Aligning transcription...")
            aligned_result = whisperx.align(
                transcribe_result["segments"],
                align_model,
                align_metadata,
                audio_path,
                self.device,
                return_char_alignments=False
            )
            
            # Perform diarization if model is available
            if self.diarization_model is not None:
                print("Performing diarization...")
                try:
                    diarize_segments = self.diarization_model(
                        audio_path,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )
                    
                    # Assign speaker labels to the segments
                    result = whisperx.assign_word_speakers(
                        diarize_segments,
                        aligned_result
                    )
                    print("Diarization completed successfully.")
                except Exception as e:
                    print(f"Diarization failed: {e}")
                    print("Returning transcription without speaker labels.")
                    result = aligned_result
            else:
                print("No diarization model available. Returning transcription without speaker labels.")
                result = aligned_result
            
            return result
            
        except Exception as e:
            print(f"Error in transcription: {e}")
            raise
    
    def _limit_speakers(self, result: Dict[str, Any], max_speakers: int) -> Dict[str, Any]:
        """
        Post-process results to limit the number of speakers.
        
        Args:
            result: WhisperX transcription result
            max_speakers: Maximum number of speakers to keep
            
        Returns:
            Dict[str, Any]: Result with limited speakers
        """
        if max_speakers is None or not isinstance(result, dict):
            return result
            
        segments = result.get("segments", [])
        if not segments:
            return result
            
        # Count unique speakers
        unique_speakers = set()
        for segment in segments:
            speaker = segment.get("speaker")
            if speaker:
                unique_speakers.add(speaker)
        
        if len(unique_speakers) <= max_speakers:
            return result  # Already within limit
            
        print(f"Found {len(unique_speakers)} speakers, limiting to {max_speakers}")
        
        # Create mapping from excess speakers to allowed speakers
        sorted_speakers = sorted(list(unique_speakers))
        speaker_mapping = {}
        
        for i, speaker in enumerate(sorted_speakers):
            if i < max_speakers:
                speaker_mapping[speaker] = speaker
            else:
                # Map excess speakers to existing ones cyclically
                speaker_mapping[speaker] = sorted_speakers[i % max_speakers]
        
        print(f"Speaker mapping: {speaker_mapping}")
        
        # Apply the mapping to all segments
        for segment in segments:
            original_speaker = segment.get("speaker")
            if original_speaker and original_speaker in speaker_mapping:
                segment["speaker"] = speaker_mapping[original_speaker]
                
            # Also update words if present
            words = segment.get("words", [])
            for word in words:
                if isinstance(word, dict) and "speaker" in word:
                    original_word_speaker = word["speaker"]
                    if original_word_speaker in speaker_mapping:
                        word["speaker"] = speaker_mapping[original_word_speaker]
        
        return result

    def get_feature_dict(
        self, 
        audio_path: str,
        min_speakers: int = None,
        max_speakers: int = None
    ) -> Dict[str, Any]:
        """
        Get a dictionary of WhisperX features with highlighting for diarization.
        
        Args:
            audio_path: Path to the audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            
        Returns:
            Dict[str, Any]: Dictionary with WhisperX highlighted features
        """
        try:
            result = self.transcribe(audio_path, min_speakers, max_speakers)
            
            # Post-process to limit speakers if needed
            if max_speakers is not None:
                result = self._limit_speakers(result, max_speakers)
            
            # Process the result to extract the highlighted diarization features
            feature_dict = {}
            
            print(f"WhisperX result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            
            # Handle different result formats
            segments = result.get("segments", []) if isinstance(result, dict) else []
            
            # Extract transcription text
            full_text_parts = []
            
            for segment_idx, segment in enumerate(segments):
                speaker = segment.get("speaker", f"speaker_{segment_idx}")
                segment_text = segment.get("text", "")
                
                # Add segment-level information
                segment_key = f"WhX_segment_{segment_idx+1}"
                feature_dict[segment_key] = {
                    "text": segment_text,
                    "speaker": speaker,
                    "start": segment.get("start"),
                    "end": segment.get("end")
                }
                
                # Process words if available
                words = segment.get("words", [])
                for word_idx, word in enumerate(words):
                    # Handle different word formats
                    if isinstance(word, dict):
                        word_text = word.get("word", word.get("text", ""))
                        highlight = "adjusted" in word or "score" in word
                        
                        # Create a key for this word
                        word_key = f"WhX_{speaker}_word_{segment_idx+1}_{word_idx+1}"
                        feature_dict[word_key] = {
                            "text": word_text,
                            "highlighted": highlight,
                            "start": word.get("start"),
                            "end": word.get("end"),
                            "confidence": word.get("score", word.get("confidence", 1.0))
                        }
                        
                        full_text_parts.append(word_text)
                    else:
                        # Handle case where word is just a string
                        full_text_parts.append(str(word))
                
                # If no words, use segment text
                if not words and segment_text:
                    full_text_parts.append(segment_text)
            
            # Add the full transcription
            if full_text_parts:
                feature_dict["transcription"] = " ".join(full_text_parts).strip()
            else:
                # Fallback: try to get text from segments directly
                feature_dict["transcription"] = " ".join([
                    seg.get("text", "") for seg in segments
                ]).strip()
            
            # Add metadata
            feature_dict["language"] = result.get("language", "unknown") if isinstance(result, dict) else "unknown"
            feature_dict["num_segments"] = len(segments)
            
            # Add speaker-specific word features
            speaker_word_features = self.extract_speaker_word_features(result, max_speakers=3)
            feature_dict.update(speaker_word_features)
            
            print(f"Extracted transcription: {feature_dict.get('transcription', 'No transcription')[:100]}...")
            
            return feature_dict
            
        except Exception as e:
            print(f"Error extracting WhisperX features: {e}")
            raise

    def extract_speaker_word_features(self, result: dict, max_speakers: int = 3) -> Dict[str, Any]:
        """
        Extract speaker-specific word features in the format requested.
        
        Args:
            result: WhisperX transcription result
            max_speakers: Maximum number of speakers to extract
            
        Returns:
            Dict with speaker-specific word features
        """
        try:
            segments = result.get("segments", [])
            word_features = {}
            
            # Group words by speaker
            speaker_words = {}
            
            for segment in segments:
                speaker = segment.get("speaker")
                if not speaker:
                    continue
                    
                words = segment.get("words", [])
                if speaker not in speaker_words:
                    speaker_words[speaker] = []
                    
                for word_info in words:
                    if isinstance(word_info, dict) and "word" in word_info:
                        speaker_words[speaker].append({
                            "word": word_info["word"].strip(),
                            "start": word_info.get("start", 0),
                            "end": word_info.get("end", 0),
                            "score": word_info.get("score", 0.0)
                        })
            
            # Sort speakers consistently and limit to max_speakers
            sorted_speakers = sorted(speaker_words.keys())[:max_speakers]
            
            # Create speaker-word features
            for i, speaker in enumerate(sorted_speakers, 1):
                words = speaker_words[speaker]
                
                # Add individual words as features
                for j, word_info in enumerate(words, 1):
                    word_key = f"WhX_highlight_diarize_speaker{i}_word_{j}"
                    word_features[word_key] = {
                        "word": word_info["word"],
                        "start_time": word_info["start"],
                        "end_time": word_info["end"],
                        "confidence": word_info["score"]
                    }
                
                # Add speaker summary
                speaker_key = f"WhX_speaker{i}_summary"
                word_features[speaker_key] = {
                    "speaker_id": speaker,
                    "total_words": len(words),
                    "total_duration": sum(w["end"] - w["start"] for w in words if w["end"] > w["start"]),
                    "avg_confidence": np.mean([w["score"] for w in words]) if words else 0.0
                }
            
            return word_features
            
        except Exception as e:
            print(f"Error extracting speaker word features: {e}")
            return {}
