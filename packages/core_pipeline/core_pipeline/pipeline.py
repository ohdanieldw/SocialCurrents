# packages/core_pipeline/core_pipeline/pipeline.py
"""
Main pipeline for processing audio files with all available features.
"""
import os
import json
import logging
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union
from datetime import datetime
from tempfile import TemporaryDirectory

from audio_models.utils.audio_extraction import (
    extract_audio_from_video,
    extract_audio_from_videos,
)


class MultimodalPipeline:
    """Main pipeline for processing multimodal data."""

    def __init__(
        self,
        output_dir: Union[str, Path] = None,
        features: List[str] = None,
        device: str = "cpu",
        keep_per_frame: bool = False,
        decimal_places: int = 3,
    ):
        """
        Initialize the multimodal pipeline.

        Args:
            output_dir: Directory to save output files
            features: List of features to extract (if None, extract all)
            device: Device to run models on ("cpu" or "cuda")
            keep_per_frame: Keep per-frame sequence outputs for supported models
        """
        self.device = device
        self.keep_per_frame = bool(keep_per_frame)
        self.decimal_places = int(decimal_places)

        # Set up output directory
        if output_dir is None:
            self.output_dir = Path("output")
        else:
            self.output_dir = Path(output_dir)

        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize feature list
        all_features = [
            "basic_audio",      # Volume and pitch from OpenCV
            "librosa_spectral", # Spectral features from librosa
            "opensmile",        # OpenSMILE Low-Level Descriptors and Functionals
            "audiostretchy",    # AudioStretchy high-quality time-stretching analysis
            "speech_emotion",   # Speech emotion recognition
            "heinsen_sentiment", # Heinsen routing sentiment analysis
            "meld_emotion",     # MELD emotion recognition during social interactions
            "speech_separation", # Speech source separation
            "whisperx_transcription", # WhisperX transcription with diarization
            "xlsr_speech_to_text", # XLSR-style speech-to-text & hidden states
            "s2t_speech_to_text",  # Speech-to-Text (S2T) transcript + alignment
            "deberta_text",     # DeBERTa text analysis with benchmark performance metrics
            "simcse_text",      # SimCSE contrastive learning of sentence embeddings
            "albert_text",      # ALBERT language representation analysis
            "sbert_text",       # Sentence-BERT dense vector representations and reranking
            "use_text",         # Universal Sentence Encoder for text classification and semantic analysis
            "elmo_text",        # ELMo contextual embeddings
            "pare_vision",      # PARE 3D human body estimation and pose analysis
            "vitpose_vision",   # ViTPose Vision Transformer pose estimation
            "psa_vision",       # Polarized Self-Attention pose estimation
            # Additional vision / facial analysis feature flags
            "emotieffnet_vision",   # EmotiEffNet facial emotion & AU
            "mediapipe_pose_vision",# Google MediaPipe 33 landmark pose
            "openpose_vision",      # OpenPose 2D multi-person keypoints
            "pyfeat_vision",        # Py-Feat facial expression / AU / geometry
            "me_graphau_vision",    # ME-GraphAU facial action units (ann_*)
            "dan_vision",           # DAN facial emotion recognition (dan_*)
            "ganimation_vision",    # GANimation AU intensities (GAN_*)
            "arbex_vision",         # ARBEx attentive facial expression analysis (arbex_*)
            "instadm_vision",       # Insta-DM depth & motion (indm_*)
            "crowdflow_vision",     # CrowdFlow optical flow crowd features (of_*)
            "deep_hrnet_vision",    # Deep HRNet pose metrics (DHiR_*)
            "simple_baselines_vision", # Simple Baselines pose metrics (SBH_*)
            "rsn_vision",           # RSN keypoint localization (rsn_*)
            "optical_flow_vision",  # Optical flow sparse/dense (avg_motion_*, etc.)
            "videofinder_vision",   # VideoFinder object/people localization (ViF_*)
            "lanegcn_vision",       # LaneGCN motion forecasting (GCN_*)
            "smoothnet_vision",     # SmoothNet temporal pose smoothing (net_*)
            "avhubert_vision",      # AV-HuBERT audio-visual embeddings (AVH_*)
            "fact_vision",          # FACT facial action coding (FACT_*)
            "video_frames_vision",  # Video frame extraction utility (VFE_*)
            "rife_vision",          # RIFE motion estimation (RIFE_*)
        ]

        # Default behavior: extract all features when none specified
        self.features = features if features is not None else all_features

        # Initialize feature extractors (lazily loaded later)
        self.extractors = {}

    def _get_extractor(self, feature_name: str) -> Any:
        """
        Get or initialize a feature extractor.

        Args:
            feature_name: Name of the feature extractor

        Returns:
            Any: The feature extractor object
        """
        if feature_name not in self.extractors:
            if feature_name == "basic_audio":
                try:
                    from audio_models.audio.basic_features import AudioFeatureExtractor
                    self.extractors[feature_name] = AudioFeatureExtractor()
                except Exception as e:
                    print(f"Warning: basic_audio unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "librosa_spectral":
                try:
                    from audio_models.audio.spectral_features import LibrosaFeatureExtractor
                    self.extractors[feature_name] = LibrosaFeatureExtractor()
                except Exception as e:
                    print(f"Warning: librosa_spectral unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "opensmile":
                try:
                    from audio_models.audio.opensmile_features import OpenSMILEFeatureExtractor
                    self.extractors[feature_name] = OpenSMILEFeatureExtractor()
                except Exception as e:
                    print(f"Warning: opensmile unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "audiostretchy":
                try:
                    from audio_models.audio.audiostretchy_features import AudioStretchyAnalyzer
                    self.extractors[feature_name] = AudioStretchyAnalyzer()
                except Exception as e:
                    print(f"Warning: audiostretchy unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "speech_emotion":
                try:
                    from audio_models.speech.emotion_recognition import SpeechEmotionRecognizer
                    self.extractors[feature_name] = SpeechEmotionRecognizer()
                except Exception as e:
                    print(f"Warning: speech_emotion unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "heinsen_sentiment":
                try:
                    from nlp_models.emotion.heinsen_routing_sentiment import AudioSentimentAnalyzer
                    self.extractors[feature_name] = AudioSentimentAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: heinsen_sentiment unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "meld_emotion":
                try:
                    from nlp_models.emotion.meld_emotion_analyzer import MELDEmotionAnalyzer
                    self.extractors[feature_name] = MELDEmotionAnalyzer()
                except Exception as e:
                    print(f"Warning: meld_emotion unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "speech_separation":
                try:
                    from audio_models.speech.speech_separator import SpeechSeparator
                    self.extractors[feature_name] = SpeechSeparator(device=self.device)
                except Exception as e:
                    print(f"Warning: speech_separation unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "whisperx_transcription":
                # Pass HF token if available for gated diarization models
                try:
                    from audio_models.speech.whisperx_transcriber import WhisperXTranscriber
                    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
                    self.extractors[feature_name] = WhisperXTranscriber(device=self.device, hf_token=hf_token)
                except Exception as e:
                    print(f"Warning: whisperx_transcription unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "xlsr_speech_to_text":
                try:
                    from audio_models.speech.xlsr_speech_to_text import XLSRSpeechToTextAnalyzer
                    self.extractors[feature_name] = XLSRSpeechToTextAnalyzer(device=self.device, output_dir=self.output_dir)
                except Exception as e:
                    print(f"Warning: xlsr_speech_to_text unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "s2t_speech_to_text":
                try:
                    from audio_models.speech.s2t_speech_to_text import S2TSpeechToTextAnalyzer
                    self.extractors[feature_name] = S2TSpeechToTextAnalyzer(device=self.device, output_dir=self.output_dir)
                except Exception as e:
                    print(f"Warning: s2t_speech_to_text unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "deberta_text":
                try:
                    from nlp_models.text.deberta_analyzer import DeBERTaAnalyzer
                    self.extractors[feature_name] = DeBERTaAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: deberta_text unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "simcse_text":
                try:
                    from nlp_models.text.simcse_analyzer import SimCSEAnalyzer
                    self.extractors[feature_name] = SimCSEAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: simcse_text unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "albert_text":
                try:
                    from nlp_models.text.albert_analyzer import ALBERTAnalyzer
                    self.extractors[feature_name] = ALBERTAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: albert_text unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "sbert_text":
                try:
                    from nlp_models.text.sbert_analyzer import SBERTAnalyzer
                    self.extractors[feature_name] = SBERTAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: sbert_text unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "use_text":
                try:
                    from nlp_models.text.use_analyzer import USEAnalyzer
                    self.extractors[feature_name] = USEAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: use_text unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "elmo_text":
                try:
                    from nlp_models.text.elmo_analyzer import ELMoAnalyzer
                    self.extractors[feature_name] = ELMoAnalyzer(device=self.device, output_dir=self.output_dir)
                except Exception as e:
                    print(f"Warning: elmo_text unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "pare_vision":
                try:
                    from cv_models.vision.pare_analyzer import PAREAnalyzer
                    # Optional: provide checkpoint via env var PARE_CHECKPOINT
                    pare_ckpt = os.getenv("PARE_CHECKPOINT")
                    self.extractors[feature_name] = PAREAnalyzer(device=self.device, model_path=pare_ckpt)
                except Exception as e:
                    print(f"Warning: pare_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "vitpose_vision":
                try:
                    from cv_models.vision.vitpose_analyzer import ViTPoseAnalyzer
                    # Optional: provide checkpoint via env var VITPOSE_CHECKPOINT
                    vitpose_ckpt = os.getenv("VITPOSE_CHECKPOINT")
                    vitpose_config = os.getenv("VITPOSE_CONFIG")
                    self.extractors[feature_name] = ViTPoseAnalyzer(
                        device=self.device,
                        checkpoint_path=vitpose_ckpt,
                        config_path=vitpose_config,
                    )
                except Exception as e:
                    print(f"Warning: vitpose_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "psa_vision":
                try:
                    from cv_models.vision.psa_analyzer import PSAAnalyzer
                    self.extractors[feature_name] = PSAAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: psa_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "emotieffnet_vision":
                try:
                    from cv_models.vision.emotieffnet_analyzer import EmotiEffNetAnalyzer
                    self.extractors[feature_name] = EmotiEffNetAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: emotieffnet_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "mediapipe_pose_vision":
                try:
                    from cv_models.vision.mediapipe_pose_analyzer import MediaPipePoseAnalyzer
                    self.extractors[feature_name] = MediaPipePoseAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: mediapipe_pose_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "openpose_vision":
                try:
                    from cv_models.vision.openpose_analyzer import OpenPoseAnalyzer
                    self.extractors[feature_name] = OpenPoseAnalyzer(
                        device=self.device,
                        output_root=self.output_dir,
                    )
                except Exception as e:
                    print(f"Warning: openpose_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "pyfeat_vision":
                try:
                    from cv_models.vision.pyfeat_analyzer import PyFeatAnalyzer
                    self.extractors[feature_name] = PyFeatAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: pyfeat_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "me_graphau_vision":
                try:
                    from cv_models.vision.me_graphau_analyzer import MEGraphAUAnalyzer
                    self.extractors[feature_name] = MEGraphAUAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: me_graphau_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "dan_vision":
                try:
                    from cv_models.vision.dan_analyzer import DANAnalyzer
                    self.extractors[feature_name] = DANAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: dan_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "ganimation_vision":
                try:
                    from cv_models.vision.ganimation_analyzer import GANimationAnalyzer
                    self.extractors[feature_name] = GANimationAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: ganimation_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "arbex_vision":
                try:
                    from cv_models.vision.arbex_analyzer import ARBExAnalyzer
                    self.extractors[feature_name] = ARBExAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: arbex_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "instadm_vision":
                try:
                    from cv_models.vision.instadm_analyzer import InstaDMAnalyzer
                    self.extractors[feature_name] = InstaDMAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: instadm_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "crowdflow_vision":
                try:
                    from cv_models.vision.crowdflow_analyzer import CrowdFlowAnalyzer
                    self.extractors[feature_name] = CrowdFlowAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: crowdflow_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "deep_hrnet_vision":
                try:
                    from cv_models.vision.deep_hrnet_analyzer import DeepHRNetAnalyzer
                    self.extractors[feature_name] = DeepHRNetAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: deep_hrnet_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "simple_baselines_vision":
                try:
                    from cv_models.vision.simple_baselines_analyzer import SimpleBaselinesAnalyzer
                    self.extractors[feature_name] = SimpleBaselinesAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: simple_baselines_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "rsn_vision":
                try:
                    from cv_models.vision.rsn_analyzer import RSNAnalyzer
                    self.extractors[feature_name] = RSNAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: rsn_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "optical_flow_vision":
                try:
                    from cv_models.vision.optical_flow_analyzer import OpticalFlowAnalyzer
                    self.extractors[feature_name] = OpticalFlowAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: optical_flow_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "videofinder_vision":
                try:
                    from cv_models.vision.videofinder_analyzer import VideoFinderAnalyzer
                    self.extractors[feature_name] = VideoFinderAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: videofinder_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "lanegcn_vision":
                try:
                    from cv_models.vision.lanegcn_analyzer import LaneGCNAnalyzer
                    self.extractors[feature_name] = LaneGCNAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: lanegcn_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "smoothnet_vision":
                try:
                    from cv_models.vision.smoothnet_analyzer import SmoothNetAnalyzer
                    self.extractors[feature_name] = SmoothNetAnalyzer(device=self.device)
                except Exception as e:
                    print(f"Warning: smoothnet_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "avhubert_vision":
                try:
                    from cv_models.vision.avhubert_analyzer import AVHuBERTAnalyzer
                    self.extractors[feature_name] = AVHuBERTAnalyzer(device=self.device, output_dir=self.output_dir)
                except Exception as e:
                    print(f"Warning: avhubert_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "fact_vision":
                try:
                    from cv_models.vision.fact_analyzer import FACTAnalyzer
                    self.extractors[feature_name] = FACTAnalyzer(output_dir=self.output_dir)
                except Exception as e:
                    print(f"Warning: fact_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "video_frames_vision":
                try:
                    from cv_models.vision.video_frame_extractor import VideoFrameExtractor
                    self.extractors[feature_name] = VideoFrameExtractor(output_dir=self.output_dir)
                except Exception as e:
                    print(f"Warning: video_frames_vision unavailable: {e}")
                    self.extractors[feature_name] = None
            elif feature_name == "rife_vision":
                try:
                    from cv_models.vision.rife_analyzer import RIFEAnalyzer
                    self.extractors[feature_name] = RIFEAnalyzer(output_dir=self.output_dir)
                except Exception as e:
                    print(f"Warning: rife_vision unavailable: {e}")
                    self.extractors[feature_name] = None

        return self.extractors.get(feature_name)

    def extract_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract all enabled features from an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dict[str, Any]: Dictionary with all extracted features
        """
        features = {}

        # Extract basic audio features
        if "basic_audio" in self.features:
            print(f"Extracting basic audio features from {audio_path}")
            extractor = self._get_extractor("basic_audio")
            if extractor is not None:
                try:
                    basic_features = extractor.extract_all_features(audio_path)
                    features.update(basic_features)
                except Exception as e:
                    logging.warning("basic_audio failed, skipping: %s", e)
            else:
                logging.warning("Skipping basic_audio (extractor unavailable)")

        # Extract librosa spectral features
        if "librosa_spectral" in self.features:
            print(f"Extracting librosa spectral features from {audio_path}")
            extractor = self._get_extractor("librosa_spectral")
            if extractor is not None:
                try:
                    spectral_features = extractor.extract_all_features(audio_path)
                    features.update(spectral_features)
                except Exception as e:
                    logging.warning("librosa_spectral failed, skipping: %s", e)
            else:
                logging.warning("Skipping librosa_spectral (extractor unavailable)")

        # Extract OpenSMILE features
        if "opensmile" in self.features:
            print(f"Extracting OpenSMILE features from {audio_path}")
            extractor = self._get_extractor("opensmile")
            if extractor is not None:
                try:
                    opensmile_features = extractor.get_feature_dict(audio_path)
                    features.update(opensmile_features)
                except Exception as e:
                    logging.warning("opensmile failed, skipping: %s", e)
            else:
                logging.warning("Skipping opensmile (extractor unavailable)")

        # Extract AudioStretchy features
        if "audiostretchy" in self.features:
            print(f"Extracting AudioStretchy time-stretching features from {audio_path}")
            extractor = self._get_extractor("audiostretchy")
            if extractor is not None:
                try:
                    audiostretchy_features = extractor.get_feature_dict(audio_path)
                    features.update(audiostretchy_features)
                except Exception as e:
                    logging.warning("audiostretchy failed, skipping: %s", e)
            else:
                logging.warning("Skipping audiostretchy (extractor unavailable)")

        # Extract speech emotion features
        if "speech_emotion" in self.features:
            print(f"Extracting speech emotion features from {audio_path}")
            extractor = self._get_extractor("speech_emotion")
            if extractor is not None:
                try:
                    emotion_features = extractor.predict(audio_path)
                    features.update(emotion_features)
                except Exception as e:
                    logging.warning("speech_emotion failed, skipping: %s", e)
            else:
                logging.warning("Skipping speech_emotion (extractor unavailable)")

        # Extract Heinsen routing sentiment features
        if "heinsen_sentiment" in self.features:
            print(f"Extracting Heinsen routing sentiment features from {audio_path}")
            extractor = self._get_extractor("heinsen_sentiment")
            if extractor is not None:
                try:
                    sentiment_features = extractor.get_feature_dict(features)
                    features.update(sentiment_features)
                except Exception as e:
                    logging.warning("heinsen_sentiment failed, skipping: %s", e)
            else:
                logging.warning("Skipping heinsen_sentiment (extractor unavailable)")

        # Extract speech separation features
        if "speech_separation" in self.features:
            print(f"Extracting speech separation features from {audio_path}")
            extractor = self._get_extractor("speech_separation")
            if extractor is not None:
                try:
                    separation_features = extractor.get_feature_dict(audio_path, None)
                    features.update(separation_features)
                except Exception as e:
                    logging.warning("speech_separation failed, skipping: %s", e)
            else:
                logging.warning("Skipping speech_separation (extractor unavailable)")

        # Extract WhisperX features
        if "whisperx_transcription" in self.features:
            print(f"Extracting WhisperX transcription features from {audio_path}")
            extractor = self._get_extractor("whisperx_transcription")
            if extractor is not None:
                try:
                    whisperx_features = extractor.get_feature_dict(audio_path, max_speakers=3)
                    features.update(whisperx_features)
                except Exception as e:
                    logging.warning("whisperx_transcription failed, skipping: %s", e)
            else:
                logging.warning("Skipping whisperx_transcription (extractor unavailable)")

        # Extract XLSR speech-to-text features
        if "xlsr_speech_to_text" in self.features:
            print(f"Extracting XLSR speech-to-text features from {audio_path}")
            extractor = self._get_extractor("xlsr_speech_to_text")
            if extractor is not None:
                try:
                    xlsr_features = extractor.get_feature_dict(audio_path, features)
                    features.update(xlsr_features)
                except Exception as e:
                    print(f"Warning: XLSR speech-to-text analysis failed: {e}")
            else:
                print("Warning: Skipping xlsr_speech_to_text (extractor unavailable)")

        # Extract S2T speech-to-text features
        if "s2t_speech_to_text" in self.features:
            print(f"Extracting S2T speech-to-text features from {audio_path}")
            extractor = self._get_extractor("s2t_speech_to_text")
            if extractor is not None:
                try:
                    s2t_features = extractor.get_feature_dict(audio_path, features)
                    features.update(s2t_features)
                except Exception as e:
                    print(f"Warning: S2T speech-to-text analysis failed: {e}")
            else:
                print("Warning: Skipping s2t_speech_to_text (extractor unavailable)")

        # Extract MELD emotion features (after WhisperX to access transcription)
        if "meld_emotion" in self.features:
            print(f"Extracting MELD emotion recognition features from {audio_path}")
            extractor = self._get_extractor("meld_emotion")
            if extractor is not None:
                try:
                    meld_features = extractor.get_feature_dict(features)
                    features.update(meld_features)
                except Exception as e:
                    print(f"Warning: MELD emotion analysis failed: {e}")
            else:
                print("Warning: Skipping meld_emotion (extractor unavailable)")

        # Extract DeBERTa text analysis features
        if "deberta_text" in self.features:
            print(f"Extracting DeBERTa text analysis features from {audio_path}")
            extractor = self._get_extractor("deberta_text")
            if extractor is not None:
                try:
                    deberta_features = extractor.get_feature_dict(features)
                    features.update(deberta_features)
                except Exception as e:
                    print(f"Warning: DeBERTa text analysis failed: {e}")
            else:
                print("Warning: Skipping deberta_text (extractor unavailable)")

        # Extract SimCSE text analysis features
        if "simcse_text" in self.features:
            print(f"Extracting SimCSE text analysis features from {audio_path}")
            extractor = self._get_extractor("simcse_text")
            if extractor is not None:
                try:
                    simcse_features = extractor.get_feature_dict(features)
                    features.update(simcse_features)
                except Exception as e:
                    print(f"Warning: SimCSE text analysis failed: {e}")
            else:
                print("Warning: Skipping simcse_text (extractor unavailable)")

        # Extract ALBERT text analysis features
        if "albert_text" in self.features:
            print(f"Extracting ALBERT text analysis features from {audio_path}")
            extractor = self._get_extractor("albert_text")
            if extractor is not None:
                try:
                    albert_features = extractor.get_feature_dict(features)
                    features.update(albert_features)
                except Exception as e:
                    print(f"Warning: ALBERT text analysis failed: {e}")
            else:
                print("Warning: Skipping albert_text (extractor unavailable)")

        # Extract Sentence-BERT text analysis features
        if "sbert_text" in self.features:
            print(f"Extracting Sentence-BERT text analysis features from {audio_path}")
            extractor = self._get_extractor("sbert_text")
            if extractor is not None:
                try:
                    sbert_features = extractor.get_feature_dict(features)
                    features.update(sbert_features)
                except Exception as e:
                    print(f"Warning: Sentence-BERT text analysis failed: {e}")
            else:
                print("Warning: Skipping sbert_text (extractor unavailable)")

        # Extract Universal Sentence Encoder text analysis features
        if "use_text" in self.features:
            print(f"Extracting Universal Sentence Encoder text analysis features from {audio_path}")
            extractor = self._get_extractor("use_text")
            if extractor is not None:
                try:
                    use_features = extractor.get_feature_dict(features)
                    features.update(use_features)
                except Exception as e:
                    print(f"Warning: USE text analysis failed: {e}")
            else:
                print("Warning: Skipping use_text (extractor unavailable)")

        # Extract ELMo contextual embeddings
        if "elmo_text" in self.features:
            print(f"Extracting ELMo contextual embeddings from {audio_path}")
            extractor = self._get_extractor("elmo_text")
            if extractor is not None:
                try:
                    elmo_features = extractor.get_feature_dict(features)
                    features.update(elmo_features)
                except Exception as e:
                    print(f"Warning: ELMo text analysis failed: {e}")
            else:
                print("Warning: Skipping elmo_text (extractor unavailable)")

        return features

    def process_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Process a single audio file through the pipeline.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dict[str, Any]: Extracted features
        """
        audio_path = Path(audio_path)
        print(f"Processing audio file: {audio_path}")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Extract features
        features = self.extract_features(str(audio_path))

        # Group features by model categories
        grouped_features = self._group_features_by_model(features)

        # Save features as JSON and/or parquet
        base_name = audio_path.stem

        # Create a JSON structure with the file name as the first key
        json_features = {base_name: {}}

        # Add metadata
        try:
            import librosa
            audio_data, sr = librosa.load(str(audio_path), sr=None)
            json_features[base_name]["metadata"] = {
                "filename": audio_path.name,
                "file_size_bytes": os.path.getsize(audio_path),
                "duration_seconds": len(audio_data) / sr,
                "sample_rate": sr,
                "channels": audio_data.shape[1] if len(audio_data.shape) > 1 else 1,
                "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "features_included": self.features
            }
        except Exception as e:
            json_features[base_name]["metadata"] = {
                "filename": audio_path.name,
                "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "features_included": self.features,
                "metadata_error": str(e)
            }

        # Add grouped features
        for group_name, group_data in grouped_features.items():
            json_features[base_name][group_name] = {
                "Feature": group_data["Feature"],
                "Model": group_data["Model"],
                "features": {}
            }

            # Process features for JSON
            for key, value in group_data["features"].items():
                if isinstance(value, np.ndarray):
                    # Convert all arrays to JSON-compatible data with statistics for large arrays
                    if value.size > 1000:
                        # Include statistics for large arrays
                        json_features[base_name][group_name]["features"][key] = {
                            'mean': float(np.mean(value)),
                            'min': float(np.min(value)),
                            'max': float(np.max(value)),
                            'std': float(np.std(value)),
                            'shape': list(value.shape),
                            'dtype': str(value.dtype),
                            'samples': [float(x) if isinstance(x, (np.number, np.float32, np.float64)) else x for x in value[:10].tolist()] if value.size > 10 else [float(x) if isinstance(x, (np.number, np.float32, np.float64)) else x for x in value.tolist()]
                        }
                    else:
                        # For smaller arrays, convert to list and include directly
                        if value.dtype.kind in 'fc':  # float or complex
                            json_features[base_name][group_name]["features"][key] = [float(x) for x in value.tolist()]
                        elif value.dtype.kind in 'iu':  # integer
                            json_features[base_name][group_name]["features"][key] = [int(x) for x in value.tolist()]
                        else:
                            json_features[base_name][group_name]["features"][key] = value.tolist()
                elif isinstance(value, (np.number, np.float32, np.float64, np.int32, np.int64)):
                    # Convert numpy scalar types to native Python types
                    json_features[base_name][group_name]["features"][key] = float(value) if isinstance(value, (np.float32, np.float64)) else int(value)
                elif isinstance(value, (str, int, float, bool, list, dict)):
                    # Other Python native types go directly to JSON
                    json_features[base_name][group_name]["features"][key] = value

        # Prepare output paths for per-file JSON
        feature_dir = self.output_dir / "features"
        feature_file_json = feature_dir / f"{base_name}_features.json"

        # Ensure features directory exists only when actually writing
        os.makedirs(feature_dir, exist_ok=True)

        # Save a single JSON file
        with open(feature_file_json, "w") as f:
            json.dump(json_features, f, indent=2)

        return features

    def process_video_file(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video file by extracting audio and then processing it.

        Args:
            video_path: Path to the video file

        Returns:
            Dict[str, Any]: Extracted features
        """
        video_path = Path(video_path)
        print(f"Processing video file: {video_path}")

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        features = {}

        # Capture video metadata (FPS and frame count) for time-indexed CSV
        try:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(str(video_path))
            if _cap.isOpened():
                _fps = _cap.get(_cv2.CAP_PROP_FPS) or 30.0
                _nf = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
                _cap.release()
                features["video_fps"] = float(_fps) if _fps > 0 else 30.0
                features["video_total_frames"] = _nf if _nf > 0 else 0
        except Exception:
            pass

        # Extract audio from video and compute audio-based features
        with TemporaryDirectory(prefix=f"{video_path.stem}_", dir=str(self.output_dir)) as temp_audio_dir:
            audio_path = Path(
                extract_audio_from_video(
                    video_path,
                    temp_audio_dir,
                    format="wav",
                    sample_rate=16000,
                )
            )

            audio_features = self.extract_features(str(audio_path))
            features.update(audio_features)

        # Extract vision features
        vision_feature_flags = [
            "pare_vision", "vitpose_vision", "psa_vision", "emotieffnet_vision", "mediapipe_pose_vision",
            "openpose_vision", "pyfeat_vision", "me_graphau_vision", "dan_vision", "ganimation_vision",
            "arbex_vision", "instadm_vision", "crowdflow_vision", "deep_hrnet_vision", "simple_baselines_vision",
            "rsn_vision", "optical_flow_vision", "videofinder_vision", "lanegcn_vision", "smoothnet_vision",
            "avhubert_vision", "fact_vision", "video_frames_vision", "rife_vision"
        ]

        for vf in vision_feature_flags:
            if vf not in self.features:
                continue

            print(f"Extracting {vf} features from {video_path}")
            extractor = self._get_extractor(vf)

            if extractor is None:
                print(f"Warning: extractor for {vf} not available (possibly missing dependency)")
                continue

            try:
                # Always collect per-frame data for the three key vision analyzers
                if vf in ("mediapipe_pose_vision", "emotieffnet_vision", "pyfeat_vision") and hasattr(extractor, "get_feature_dict"):
                    raw = extractor.get_feature_dict(str(video_path), keep_per_frame=True)
                else:
                    raw = (
                        extractor.get_feature_dict(str(video_path))
                        if hasattr(extractor, "get_feature_dict")
                        else extractor.analyze_video(str(video_path))
                    )

                # Stash per-frame sequences for the time-indexed CSV
                if vf == "mediapipe_pose_vision":
                    try:
                        for group_val in raw.values():
                            if isinstance(group_val, dict):
                                pf = group_val.get("features", {}).get("per_frame", None)
                                if isinstance(pf, list):
                                    features["mediapipe_pose_vision_per_frame"] = pf
                                    break
                    except Exception:
                        pass
                elif vf == "emotieffnet_vision":
                    try:
                        for group_val in raw.values():
                            if isinstance(group_val, dict):
                                pf = group_val.get("features", {}).get("eln_per_sample", None)
                                if isinstance(pf, list):
                                    features["emotieffnet_vision_per_sample"] = pf
                                    break
                    except Exception:
                        pass
                elif vf == "pyfeat_vision":
                    try:
                        for group_val in raw.values():
                            if isinstance(group_val, dict):
                                pf = group_val.get("features", {}).get("pf_per_frame", None)
                                if isinstance(pf, list):
                                    features["pyfeat_vision_per_frame"] = pf
                                    break
                    except Exception:
                        pass

                flat = self._flatten_feature_output(raw)
                features.update(flat)

            except Exception as e:
                print(f"Error extracting {vf}: {e}")

        return features

    def _flatten_feature_output(self, raw: Any) -> Dict[str, Any]:
        """Normalize analyzer outputs into a flat feature dict.

        Acceptable raw forms:
          1. Flat dict of feature_name->value
          2. Dict with a top-level 'features' key
          3. Nested dict {group: { 'features': {...}}}
        """
        if raw is None:
            return {}
        if isinstance(raw, dict):
            # Case 2
            if 'features' in raw and isinstance(raw['features'], dict):
                return raw['features']
            # Case 3
            collected = {}
            multi_group = True
            for v in raw.values():
                if isinstance(v, dict) and 'features' in v and isinstance(v['features'], dict):
                    collected.update(v['features'])
                else:
                    multi_group = False
                    break
            if multi_group and collected:
                return collected
            # Fallback assume already flat
            return raw
        return {}

    def process_directory(self, directory: Union[str, Path], is_video: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Process all files in a directory.

        Args:
            directory: Path to the directory containing files
            is_video: Whether the files are videos (True) or audio (False)

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping filenames to their features
        """
        directory = Path(directory)
        print(f"Processing directory: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        # Find files
        extensions = [".mp4", ".MP4", ".avi", ".mov", ".MOV", ".mkv"] if is_video else [".wav", ".mp3", ".flac"]
        files = []
        for ext in extensions:
            files.extend(list(directory.glob(f"*{ext}")))

        if not files:
            raise FileNotFoundError(f"No {'video' if is_video else 'audio'} files found in {directory}")

        # Process each file and save outputs to a per-video subfolder
        results = {}
        for file_path in files:
            stem = file_path.stem  # e.g. "dyad002_sub003"
            if "_" in stem:
                dyad_id, subject_id = stem.split("_", 1)
                file_prefix = f"{dyad_id}_{subject_id}"
                video_out_dir = self.output_dir / dyad_id / subject_id
            else:
                file_prefix = stem
                video_out_dir = self.output_dir / stem

            video_out_dir.mkdir(parents=True, exist_ok=True)
            _log_handler = logging.FileHandler(video_out_dir / f"{file_prefix}.log")
            _log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logging.getLogger().addHandler(_log_handler)
            try:
                if is_video:
                    features = self.process_video_file(file_path)
                else:
                    features = self.process_audio_file(file_path)

                results[file_path.name] = features
                self._save_file_outputs(file_prefix, features, video_out_dir)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
            finally:
                logging.getLogger().removeHandler(_log_handler)
                _log_handler.close()

        return results

    def _save_file_outputs(self, file_prefix: str, file_features: Dict[str, Any], out_dir: Path) -> None:
        """Save {prefix}_summary_features.json, _summary_features.csv, and _timeseries_features.csv for one video."""
        import pandas as pd

        # Build JSON
        try:
            grouped_features = self._group_features_by_model(file_features)
            file_json: Dict[str, Any] = {}
            for group_name, group_data in grouped_features.items():
                file_json[group_name] = {
                    "Feature": group_data["Feature"],
                    "Model": group_data["Model"],
                    "features": {},
                }
                for key, value in group_data["features"].items():
                    if isinstance(value, np.ndarray):
                        if value.size > 1000:
                            file_json[group_name]["features"][key] = {
                                "mean": float(np.mean(value)),
                                "min": float(np.min(value)),
                                "max": float(np.max(value)),
                                "std": float(np.std(value)),
                                "shape": list(value.shape),
                                "dtype": str(value.dtype),
                                "samples": [float(x) if isinstance(x, (np.number, np.float32, np.float64)) else x for x in value[:5].tolist()] if value.size > 5 else [float(x) if isinstance(x, (np.number, np.float32, np.float64)) else x for x in value.tolist()],
                            }
                        else:
                            if value.dtype.kind in "fc":
                                file_json[group_name]["features"][key] = [float(x) for x in value.tolist()]
                            elif value.dtype.kind in "iu":
                                file_json[group_name]["features"][key] = [int(x) for x in value.tolist()]
                            else:
                                file_json[group_name]["features"][key] = value.tolist()
                    elif not callable(value):
                        if isinstance(value, (np.number, np.float32, np.float64, np.int32, np.int64)):
                            file_json[group_name]["features"][key] = float(value) if isinstance(value, (np.float32, np.float64)) else int(value)
                        else:
                            file_json[group_name]["features"][key] = value

            json_path = out_dir / f"{file_prefix}_summary_features.json"
            with open(json_path, "w") as f:
                json.dump(file_json, f, indent=2)
            print(f"  JSON saved to {json_path}")
        except Exception as e:
            print(f"Warning: Could not save summary_features.json for {file_prefix}: {e}")
            traceback.print_exc()

        # Flat CSV (one row)
        try:
            row = self._flatten_for_csv(file_features)
            df = pd.DataFrame([row])
            df = df.round(self.decimal_places)
            csv_path = out_dir / f"{file_prefix}_summary_features.csv"
            df.to_csv(csv_path, index=False)
            print(f"  CSV saved to {csv_path}  ({len(df.columns)} columns)")
        except Exception as e:
            print(f"Warning: Could not save summary_features.csv for {file_prefix}: {e}")
            traceback.print_exc()

        # Time-series CSV (one row per frame)
        try:
            part = self._build_timeindexed_csv(file_features)
            if part is not None:
                part = part.round(self.decimal_places)
                ts_path = out_dir / f"{file_prefix}_timeseries_features.csv"
                part.to_csv(ts_path, index=False)
                print(f"  Time-series CSV saved to {ts_path}  ({len(part)} rows × {len(part.columns)} columns)")
        except Exception as e:
            print(f"Warning: Could not save timeseries_features.csv for {file_prefix}: {e}")
            traceback.print_exc()

    # Keys that hold per-frame lists-of-dicts — skip in flat CSV, used only for time-indexed CSV
    _PER_FRAME_KEYS = frozenset({
        "mediapipe_pose_vision_per_frame",
        "emotieffnet_vision_per_sample",
        "pyfeat_vision_per_frame",
    })

    def _flatten_for_csv(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a feature dict into a flat {column: scalar} dict suitable for a CSV row.

        Rules:
        - Scalar numbers/bools/strings → kept as-is.
        - numpy scalar types → cast to Python float or int.
        - 1-D numeric arrays / lists → expanded to {key}_mean, {key}_std, {key}_min, {key}_max.
          If the array has ≤ 20 elements every individual value is also stored as {key}_0 … {key}_N.
        - Multi-dimensional arrays → only summary stats.
        - Dicts (e.g. WhisperX segment objects) → JSON-encoded string.
        - Anything else not serialisable → str().
        """
        row: Dict[str, Any] = {}

        _SKIP_CSV = self._PER_FRAME_KEYS | {"per_frame", "GMP_SM_pic", "video_path"}

        for key, value in features.items():
            if callable(value):
                continue
            if key in _SKIP_CSV:
                continue
            if key == "error" or key.endswith("_error"):
                if value:
                    logging.warning("Extractor error in features[%s]: %s", key, value)
                continue

            # ── numpy scalar ────────────────────────────────────────────────
            if isinstance(value, (np.number, np.bool_)):
                row[key] = value.item()

            # ── numpy array ─────────────────────────────────────────────────
            elif isinstance(value, np.ndarray):
                flat = value.flatten()
                if flat.size == 0:
                    continue
                if np.issubdtype(flat.dtype, np.number):
                    clean = flat[np.isfinite(flat)]
                    if clean.size > 0:
                        row[f"{key}_mean"] = float(np.mean(clean))
                        row[f"{key}_std"]  = float(np.std(clean))
                        row[f"{key}_min"]  = float(np.min(clean))
                        row[f"{key}_max"]  = float(np.max(clean))
                    # For short arrays, also store individual values
                    if flat.size <= 20:
                        for i, v in enumerate(flat.tolist()):
                            row[f"{key}_{i}"] = float(v) if isinstance(v, float) else v
                else:
                    row[key] = str(flat.tolist())

            # ── plain Python list ────────────────────────────────────────────
            elif isinstance(value, list):
                if not value:
                    continue
                # Check if it's a numeric list
                if all(isinstance(v, (int, float)) for v in value):
                    arr = np.array(value, dtype=float)
                    finite = arr[np.isfinite(arr)]
                    if finite.size > 0:
                        row[f"{key}_mean"] = float(np.mean(finite))
                        row[f"{key}_std"]  = float(np.std(finite))
                        row[f"{key}_min"]  = float(np.min(finite))
                        row[f"{key}_max"]  = float(np.max(finite))
                    if len(value) <= 20:
                        for i, v in enumerate(value):
                            row[f"{key}_{i}"] = v
                else:
                    row[key] = json.dumps(value)

            # ── dict (e.g. WhisperX segment) ─────────────────────────────────
            elif isinstance(value, dict):
                row[key] = json.dumps(value)

            # ── plain scalar ─────────────────────────────────────────────────
            elif isinstance(value, (int, float, bool, str)):
                row[key] = value

            else:
                try:
                    row[key] = str(value)
                except Exception:
                    pass

        return row

    def _build_timeindexed_csv(self, features: Dict[str, Any]):
        """Return a per-frame DataFrame (one row per video frame) for *filename*.

        Columns:
          filename, frame_idx, time_seconds,
          <audio feature arrays resampled to video frame rate>,
          <vision per-frame features interpolated to full frame count>,
          <scalar features broadcast to every row>.

        Returns None if video metadata is unavailable.
        """
        import pandas as pd

        fps = float(features.get("video_fps", 0.0))
        n_frames = int(features.get("video_total_frames", 0))
        if fps <= 0 or n_frames <= 0:
            return None

        video_times = np.arange(n_frames) / fps

        df = pd.DataFrame({
            "frame_idx": np.arange(n_frames, dtype=np.int32),
            "time_seconds": video_times,
        })
        new_cols: dict = {}

        # ── Audio time-series arrays ────────────────────────────────────────
        # Standard pipeline: sr=16000, hop_length=512  →  ~31.25 audio frames/sec
        hop_length = int(features.get("hop_length", 512))
        sr = int(features.get("sample_rate", 16000))
        AUDIO_PREFIXES = ("oc_audvol", "oc_audpit", "lbrs_", "osm_")

        for key, val in features.items():
            if not isinstance(val, np.ndarray) or val.ndim != 1 or val.size <= 1:
                continue
            if not any(key == p or key.startswith(p) for p in AUDIO_PREFIXES):
                continue
            n_audio = val.size
            audio_times = np.arange(n_audio) * hop_length / sr
            # Clip to avoid extrapolation beyond available audio
            clipped = np.clip(video_times, audio_times[0], audio_times[-1])
            try:
                new_cols[key] = np.interp(clipped, audio_times, val.astype(float))
            except Exception:
                pass

        # ── MediaPipe per-frame (already at full resolution) ────────────────
        mp_pf = features.get("mediapipe_pose_vision_per_frame")
        if isinstance(mp_pf, list) and mp_pf:
            mp_fidx = np.array([d.get("frame_idx", i) for i, d in enumerate(mp_pf)], dtype=float)
            gmp_keys = [k for k in mp_pf[0] if k.startswith("GMP_")]
            def _sf(v, default=0.0):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default

            for k in gmp_keys:
                vals = np.array([_sf(d.get(k, 0.0)) for d in mp_pf])
                if len(mp_fidx) == n_frames:
                    new_cols[k] = vals
                elif len(mp_fidx) >= 2:
                    new_cols[k] = np.interp(np.arange(n_frames), mp_fidx, vals)

        # helper: interpolate sampled-frame vision data to all n_frames
        def _interp_vision(per_sample, key_prefix):
            if not isinstance(per_sample, list) or not per_sample:
                return
            fidx = np.array([d.get("frame_idx", 0) for d in per_sample], dtype=float)
            vis_keys = [k for k in per_sample[0] if k.startswith(key_prefix)]
            for k in vis_keys:
                vals = np.array([float(d.get(k, np.nan)) for d in per_sample])
                ok = np.isfinite(vals)
                if ok.sum() < 2:
                    continue
                new_cols[k] = np.interp(
                    np.arange(n_frames),
                    fidx[ok],
                    vals[ok],
                    left=vals[ok][0],
                    right=vals[ok][-1],
                )

        # ── EmotiEffNet sampled frames ───────────────────────────────────────
        _interp_vision(features.get("emotieffnet_vision_per_sample"), "eln_")

        # ── Py-Feat sampled frames ───────────────────────────────────────────
        _interp_vision(features.get("pyfeat_vision_per_frame"), "pf_")

        # ── Scalar features (broadcast) ──────────────────────────────────────
        SKIP = self._PER_FRAME_KEYS | {
            "video_fps", "video_total_frames", "hop_length", "sample_rate", "num_frames",
            "per_frame", "GMP_SM_pic", "video_path",
            # metadata/config — summary CSV only
            "DEB_analysis_timestamp", "BERT_analysis_timestamp",
            "DEB_model_name", "DEB_device",
            "BERT_embed_model", "BERT_rerank_model", "BERT_device",
            "total_frames", "landmarks_detected", "detection_rate", "avg_landmarks",
        }
        existing_cols = set(df.columns) | set(new_cols)
        for key, val in features.items():
            if key in SKIP or key in existing_cols or callable(val):
                continue
            if isinstance(val, (np.ndarray, list)):
                continue  # arrays already handled above
            if key.startswith("osm_"):
                continue  # osm_ arrays are already in new_cols; scalar keys are file-level functionals
            if isinstance(val, (np.number, np.bool_)):
                scalar = val.item()
            elif isinstance(val, (int, float, bool, str)):
                scalar = val
            else:
                continue
            if isinstance(scalar, float) and not np.isfinite(scalar):
                continue
            new_cols[key] = scalar

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        df = df.replace(-100.0, np.nan)
        return df

    def _group_features_by_model(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Group features by their model/algorithm categories according to the specification.

        Args:
            features: Dictionary of all extracted features

        Returns:
            Dictionary grouped by "Feature" categories
        """
        # Define feature groupings according to the specification table
        feature_groups = {
            "Audio volume": {
                "exact_matches": ["oc_audvol"],
                "model_name": "OpenCV"
            },
            "Change in audio volume": {
                "exact_matches": ["oc_audvol_diff"],
                "model_name": "OpenCV"
            },
            "Average audio pitch": {
                "exact_matches": ["oc_audpit"],
                "model_name": "OpenCV"
            },
            "Change in audio pitch": {
                "exact_matches": ["oc_audpit_diff"],
                "model_name": "OpenCV"
            },
            "Speech emotion/emotional speech classification": {
                "prefixes": ["ser_"],
                "exact_matches": [],
                "model_name": "Speech Emotion Recognition"
            },
            "Time-Accurate Speech Transcription": {
                "prefixes": ["WhX_"],
                "exact_matches": ["transcription", "language", "num_segments"],
                "model_name": "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio"
            },
            "Spectral Features, Pitch, Rhythm": {
                "prefixes": ["lbrs_"],
                "exact_matches": [],
                "model_name": "Librosa"
            },
            "(1) High-quality time-stretching of WAV/MP3 files without changing their pitch; (2) Time-stretch silence separately": {
                "prefixes": ["AS_"],
                "exact_matches": [],
                "model_name": "AudioStretchy"
            },
            "Speech feature extraction": {
                "prefixes": ["osm_"],
                "exact_matches": ["sample_rate", "hop_length", "num_frames"],
                "model_name": "openSMILE"
            },
            "Speech-to-Text (XLSR)": {
                "prefixes": ["xlsr_"],
                "exact_matches": [],
                "model_name": "XLSR / Wav2Vec 2.0"
            },
            "Speech-to-Text (S2T)": {
                "prefixes": ["s2t_"],
                "exact_matches": [],
                "model_name": "Speech-to-Text (S2T)"
            },
            "Sentiment Analysis": {
                "prefixes": ["arvs_"],
                "exact_matches": [],
                "model_name": "AnAlgorithm for Routing Vectors in Sequences"
            },
            "Emotion Recognition during Social Interactions": {
                "prefixes": ["MELD_"],
                "exact_matches": [],
                "model_name": "MELD (Multimodal Multi-Party Dataset for Emotion Recognition in Conversation)"
            },
            "Disentangled Attention Mechanism & Enhanced Mask Decoder": {
                "prefixes": ["DEB_"],
                "exact_matches": [],
                "model_name": "DEBERTA"
            },
            "Contrastive Learning of Sentence Embeddings": {
                "prefixes": ["CSE_"],
                "exact_matches": [],
                "model_name": "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
            },
            "Language representation": {
                "prefixes": ["alb_"],
                "exact_matches": [],
                "model_name": "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"
            },
            "Dense Vector Representations and Reranking": {
                "prefixes": ["BERT_"],
                "exact_matches": [],
                "model_name": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
            },
            "text classification + semantic similarity + semantic cluster": {
                "prefixes": ["USE_"],
                "exact_matches": [],
                "model_name": "Universal Sentence Encoder"
            },
            "Deep contextualized word representations (ELMo)": {
                "prefixes": ["elmo_"],
                "exact_matches": [],
                "model_name": "ELMo"
            },
            "3D Human Body Estimation and Pose Analysis": {
                "prefixes": ["PARE_"],
                "exact_matches": [],
                "model_name": "PARE (Part Attention Regressor for 3D Human Body Estimation)"
            },
            "Pose estimation": {
                "prefixes": ["vit_"],
                "exact_matches": [],
                "model_name": "ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation"
            },
            # Newly added vision/facial groups
            "Facial Emotion & Action Units (EmotiEffNet)": {
                "prefixes": ["eln_"],
                "exact_matches": [],
                "model_name": "EmotiEffNet"
            },
            "Facial Expression (Py-Feat)": {
                "prefixes": ["pf_"],
                "exact_matches": [],
                "model_name": "Py-Feat"
            },
            "Pose Landmarks (MediaPipe)": {
                "prefixes": ["GMP_"],
                "exact_matches": [],
                "model_name": "Google MediaPipe Pose"
            },
            "Deep HRNet Pose Metrics": {
                "prefixes": ["DHiR_"],
                "exact_matches": [],
                "model_name": "Deep HRNet"
            },
            "Simple Baselines Pose Metrics": {
                "prefixes": ["SBH_"],
                "exact_matches": [],
                "model_name": "Simple Baselines"
            },
            "OpenPose Keypoints": {
                "prefixes": ["openPose_"],
                "exact_matches": [],
                "model_name": "OpenPose"
            },
            "ME-GraphAU Facial Action Units": {
                "prefixes": ["ann_"],
                "exact_matches": [],
                "model_name": "ME-GraphAU"
            },
            "DAN Facial Emotions": {
                "prefixes": ["dan_"],
                "exact_matches": [],
                "model_name": "DAN"
            },
            "GANimation AUs": {
                "prefixes": ["GAN_"],
                "exact_matches": [],
                "model_name": "GANimation"
            },
            "ARBEx Facial Emotions": {
                "prefixes": ["arbex_"],
                "exact_matches": [],
                "model_name": "ARBEx"
            },
            "Insta-DM Depth & Motion": {
                "prefixes": ["indm_"],
                "exact_matches": [],
                "model_name": "Insta-DM"
            },
            "AV-HuBERT Audio-Visual Embeddings": {
                "prefixes": ["AVH_"],
                "exact_matches": [],
                "model_name": "AV-HuBERT"
            },
            "FACT Facial Action Coding": {
                "prefixes": ["FACT_"],
                "exact_matches": [],
                "model_name": "FACT"
            },
            "CrowdFlow Optical Flow": {
                "prefixes": ["of_"],
                "exact_matches": [],
                "model_name": "CrowdFlow"
            },
            "VideoFinder Object/People Localization": {
                "prefixes": ["ViF_"],
                "exact_matches": [],
                "model_name": "VideoFinder"
            },
            "SmoothNet Temporal Pose": {
                "prefixes": ["net_"],
                "exact_matches": [],
                "model_name": "SmoothNet"
            },
            "LaneGCN Motion Forecasting": {
                "prefixes": ["GCN_"],
                "exact_matches": [],
                "model_name": "LaneGCN"
            },
            "RSN Keypoint Localization": {
                "prefixes": ["rsn_"],
                "exact_matches": [],
                "model_name": "RSN"
            },
            "Video Frame Extraction": {
                "prefixes": ["VFE_"],
                "exact_matches": [],
                "model_name": "Frame Sampler"
            },
            "RIFE Motion Estimation": {
                "prefixes": ["RIFE_"],
                "exact_matches": [],
                "model_name": "RIFE"
            },
            "PSA Keypoint & Segmentation": {
                "prefixes": ["psa_"],
                "exact_matches": [],
                "model_name": "Polarized Self-Attention"
            }
        }

        # Initialize grouped features
        grouped_features = {}
        ungrouped_features = {}

        # Group features by category
        for feature_name, feature_value in features.items():
            matched = False

            for group_name, group_config in feature_groups.items():
                # Check exact matches first (more specific)
                if feature_name in group_config.get("exact_matches", []):
                    if group_name not in grouped_features:
                        grouped_features[group_name] = {
                            "Feature": group_name,
                            "Model": group_config["model_name"],
                            "features": {}
                        }
                    grouped_features[group_name]["features"][feature_name] = feature_value
                    matched = True
                    break

                # Check prefix matches
                if any(feature_name.startswith(prefix) for prefix in group_config.get("prefixes", [])):
                    if group_name not in grouped_features:
                        grouped_features[group_name] = {
                            "Feature": group_name,
                            "Model": group_config["model_name"],
                            "features": {}
                        }
                    grouped_features[group_name]["features"][feature_name] = feature_value
                    matched = True
                    break

            # If no match found, add to ungrouped
            if not matched:
                ungrouped_features[feature_name] = feature_value

        # Add ungrouped features if any exist
        if ungrouped_features:
            grouped_features["Other"] = {
                "Feature": "Other",
                "Model": "Various",
                "features": ungrouped_features
            }

        return grouped_features