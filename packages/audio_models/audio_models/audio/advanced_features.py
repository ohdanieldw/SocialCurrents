"""
Advanced audio feature extraction for volume and pitch analysis.
"""

import numpy as np
import librosa
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class AdvancedAudioFeatures:
    """
    Extract advanced audio features including volume and pitch statistics.
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def extract_volume_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract volume-related features from audio.
        
        Args:
            audio: Audio waveform as numpy array
            
        Returns:
            Dict with volume features
        """
        try:
            # Calculate RMS energy (volume proxy)
            rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            
            # Convert to dB
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            
            # Calculate volume statistics
            volume_features = {
                'oc_audvol': float(np.mean(rms_db)),  # Mean volume in dB
                'oc_audvol_std': float(np.std(rms_db)),  # Volume standard deviation
                'oc_audvol_max': float(np.max(rms_db)),  # Maximum volume
                'oc_audvol_min': float(np.min(rms_db)),  # Minimum volume
            }
            
            # Calculate volume differences (frame-to-frame changes)
            if len(rms_db) > 1:
                volume_diff = np.diff(rms_db)
                volume_features['oc_audvol_diff'] = float(np.mean(np.abs(volume_diff)))
                volume_features['oc_audvol_diff_std'] = float(np.std(volume_diff))
            else:
                volume_features['oc_audvol_diff'] = 0.0
                volume_features['oc_audvol_diff_std'] = 0.0
            
            return volume_features
            
        except Exception as e:
            logger.error(f"Error extracting volume features: {e}")
            return {
                'oc_audvol': 0.0,
                'oc_audvol_diff': 0.0,
                'oc_audvol_std': 0.0,
                'oc_audvol_max': 0.0,
                'oc_audvol_min': 0.0,
                'oc_audvol_diff_std': 0.0
            }
    
    def extract_pitch_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract pitch-related features from audio.
        
        Args:
            audio: Audio waveform as numpy array
            
        Returns:
            Dict with pitch features
        """
        try:
            # Extract fundamental frequency (F0) using yin algorithm
            f0 = librosa.yin(audio, 
                           fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                           fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                           sr=self.sr)
            
            # Remove unvoiced frames (where f0 is very low or zero)
            voiced_f0 = f0[f0 > 50]  # Remove frames below 50 Hz
            
            if len(voiced_f0) > 0:
                # Convert to semitones for better perceptual relevance
                f0_semitones = 12 * np.log2(voiced_f0 / 440.0)  # A4 = 440 Hz as reference
                
                pitch_features = {
                    'oc_audpit': float(np.mean(f0_semitones)),  # Mean pitch in semitones
                    'oc_audpit_std': float(np.std(f0_semitones)),  # Pitch standard deviation
                    'oc_audpit_max': float(np.max(f0_semitones)),  # Maximum pitch
                    'oc_audpit_min': float(np.min(f0_semitones)),  # Minimum pitch
                    'oc_audpit_range': float(np.max(f0_semitones) - np.min(f0_semitones)),  # Pitch range
                }
                
                # Calculate pitch differences (frame-to-frame changes)
                if len(f0_semitones) > 1:
                    pitch_diff = np.diff(f0_semitones)
                    pitch_features['oc_audpit_diff'] = float(np.mean(np.abs(pitch_diff)))
                    pitch_features['oc_audpit_diff_std'] = float(np.std(pitch_diff))
                else:
                    pitch_features['oc_audpit_diff'] = 0.0
                    pitch_features['oc_audpit_diff_std'] = 0.0
                    
            else:
                # No voiced segments found
                pitch_features = {
                    'oc_audpit': 0.0,
                    'oc_audpit_diff': 0.0,
                    'oc_audpit_std': 0.0,
                    'oc_audpit_max': 0.0,
                    'oc_audpit_min': 0.0,
                    'oc_audpit_range': 0.0,
                    'oc_audpit_diff_std': 0.0
                }
            
            return pitch_features
            
        except Exception as e:
            logger.error(f"Error extracting pitch features: {e}")
            return {
                'oc_audpit': 0.0,
                'oc_audpit_diff': 0.0,
                'oc_audpit_std': 0.0,
                'oc_audpit_max': 0.0,
                'oc_audpit_min': 0.0,
                'oc_audpit_range': 0.0,
                'oc_audpit_diff_std': 0.0
            }
    
    def extract_all_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract all advanced audio features from an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with all advanced audio features
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Extract features
            volume_features = self.extract_volume_features(audio)
            pitch_features = self.extract_pitch_features(audio)
            
            # Combine all features
            all_features = {**volume_features, **pitch_features}
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            return {
                'oc_audvol': 0.0,
                'oc_audvol_diff': 0.0,
                'oc_audpit': 0.0,
                'oc_audpit_diff': 0.0
            }

def extract_advanced_audio_features(audio_path: str) -> Dict[str, float]:
    """
    Convenience function to extract advanced audio features.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dict with advanced audio features
    """
    extractor = AdvancedAudioFeatures()
    return extractor.extract_all_features(audio_path)

if __name__ == "__main__":
    # Test the feature extractor
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        features = extract_advanced_audio_features(audio_path)
        print(f"Advanced audio features for {audio_path}:")
        for feature, value in features.items():
            print(f"  {feature}: {value:.4f}")
    else:
        print("Usage: python advanced_features.py <audio_file>")
