"""
Basic audio feature extraction module using OpenCV.
Extracts features like audio volume and pitch.
"""
import numpy as np
import librosa

class AudioFeatureExtractor:
    """Extract basic audio features using OpenCV and librosa."""
    
    def __init__(self):
        pass
        
    def extract_volume(self, audio_data: np.ndarray, hop_length: int = 512) -> np.ndarray:
        """
        Extract volume (RMS energy) from audio data.
        
        Args:
            audio_data: Audio time series
            hop_length: Number of samples between frames
            
        Returns:
            np.ndarray: RMS energy for each frame
        """
        # Compute the RMS energy
        rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
        return rms
    
    def extract_volume_diff(self, volume: np.ndarray) -> np.ndarray:
        """
        Calculate frame-by-frame differences in volume.
        
        Args:
            volume: Array of volume values
            
        Returns:
            np.ndarray: Differences between consecutive volume values
        """
        if len(volume) <= 1:
            return np.array([0.0])
        return np.diff(volume, prepend=volume[0])
    
    def extract_pitch(self, audio_data: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
        """
        Extract pitch (fundamental frequency) from audio data.
        
        Args:
            audio_data: Audio time series
            sr: Sample rate
            hop_length: Number of samples between frames
            
        Returns:
            np.ndarray: Pitch for each frame
        """
        # Compute the pitch using librosa's piptrack
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr, hop_length=hop_length)
        
        # Get the pitch for each frame
        pitch = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch.append(pitches[index, i])
            
        return np.array(pitch)
    
    def extract_pitch_diff(self, pitch: np.ndarray) -> np.ndarray:
        """
        Calculate frame-by-frame differences in pitch.
        
        Args:
            pitch: Array of pitch values
            
        Returns:
            np.ndarray: Differences between consecutive pitch values
        """
        if len(pitch) <= 1:
            return np.array([0.0])
        return np.diff(pitch, prepend=pitch[0])
    
    def extract_all_features(self, audio_path: str, hop_length: int = 512) -> dict:
        """
        Extract all basic audio features from an audio file.
        
        Args:
            audio_path: Path to the audio file
            hop_length: Number of samples between frames
            
        Returns:
            dict: Dictionary of extracted features
        """
        # Load audio file
        audio_data, sr = librosa.load(audio_path, sr=None)
        
        # Extract features
        volume = self.extract_volume(audio_data, hop_length)
        volume_diff = self.extract_volume_diff(volume)
        pitch = self.extract_pitch(audio_data, sr, hop_length)
        pitch_diff = self.extract_pitch_diff(pitch)
        
        return {
            "oc_audvol": volume,
            "oc_audvol_diff": volume_diff,
            "oc_audpit": pitch,
            "oc_audpit_diff": pitch_diff,
            "sample_rate": sr,
            "hop_length": hop_length,
            "num_frames": len(volume)
        }
