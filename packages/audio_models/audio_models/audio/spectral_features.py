"""
Advanced audio feature extraction using Librosa.
Extracts spectral features, pitch, rhythm characteristics.
"""
import numpy as np
import librosa

class LibrosaFeatureExtractor:
    """Extract advanced audio features using Librosa."""
    
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        """
        Initialize the Librosa feature extractor.
        
        Args:
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def extract_spectral_centroid(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract spectral centroid from audio data.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            np.ndarray: Spectral centroid for each frame
        """
        return librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
    
    def extract_spectral_bandwidth(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract spectral bandwidth from audio data.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            np.ndarray: Spectral bandwidth for each frame
        """
        return librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
    
    def extract_spectral_flatness(self, y: np.ndarray) -> np.ndarray:
        """
        Extract spectral flatness from audio data.
        
        Args:
            y: Audio time series
            
        Returns:
            np.ndarray: Spectral flatness for each frame
        """
        return librosa.feature.spectral_flatness(
            y=y, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
    def extract_spectral_rolloff(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract spectral rolloff from audio data.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            np.ndarray: Spectral rolloff for each frame
        """
        return librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
    
    def extract_zero_crossing_rate(self, y: np.ndarray) -> np.ndarray:
        """
        Extract zero crossing rate from audio data.
        
        Args:
            y: Audio time series
            
        Returns:
            np.ndarray: Zero crossing rate for each frame
        """
        return librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)[0]
    
    def extract_rmse(self, y: np.ndarray) -> np.ndarray:
        """
        Extract root mean square energy from audio data.
        
        Args:
            y: Audio time series
            
        Returns:
            np.ndarray: RMS energy for each frame
        """
        return librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
    
    def extract_tempo(self, y: np.ndarray, sr: int) -> float:
        """
        Extract tempo from audio data.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            float: Estimated tempo in beats per minute
        """
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        # Use the updated function path to avoid deprecation warning
        try:
            tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=self.hop_length)[0]
        except AttributeError:
            # Fall back to the old method if using an older version of librosa
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=self.hop_length)[0]
        return tempo
    
    def extract_spectral_contrast(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract spectral contrast from audio data.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            np.ndarray: Spectral contrast for each frame (time series)
        """
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        # Return mean across frequency bands for each time frame
        return np.mean(contrast, axis=0)
    
    def extract_spectral_contrast_singlevalue(self, y: np.ndarray, sr: int) -> float:
        """
        Extract mean spectral contrast from audio data (single value).
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            float: Mean spectral contrast across all frames
        """
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return np.mean(contrast)
    
    def extract_all_features(self, audio_path: str) -> dict:
        """
        Extract all Librosa features from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            dict: Dictionary of extracted features
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract time-series features
        spectral_centroid = self.extract_spectral_centroid(y, sr)
        spectral_bandwidth = self.extract_spectral_bandwidth(y, sr)
        spectral_flatness = self.extract_spectral_flatness(y)
        spectral_rolloff = self.extract_spectral_rolloff(y, sr)
        zero_crossing_rate = self.extract_zero_crossing_rate(y)
        rmse = self.extract_rmse(y)
        spectral_contrast = self.extract_spectral_contrast(y, sr)
        
        # Extract single-value features
        tempo = self.extract_tempo(y, sr)
        spectral_contrast_singlevalue = self.extract_spectral_contrast_singlevalue(y, sr)
        
        # Calculate single values for time series features (mean across time)
        spectral_centroid_singlevalue = np.mean(spectral_centroid)
        spectral_bandwidth_singlevalue = np.mean(spectral_bandwidth)
        spectral_flatness_singlevalue = np.mean(spectral_flatness)
        spectral_rolloff_singlevalue = np.mean(spectral_rolloff)
        rmse_singlevalue = np.mean(rmse)
        zero_crossing_rate_singlevalue = np.mean(zero_crossing_rate)
        
        return {
            # Time series features (per frame)
            "lbrs_spectral_centroid": spectral_centroid,
            "lbrs_spectral_bandwidth": spectral_bandwidth,
            "lbrs_spectral_flatness": spectral_flatness,
            "lbrs_spectral_rolloff": spectral_rolloff,
            "lbrs_zero_crossing_rate": zero_crossing_rate,
            "lbrs_rmse": rmse,
            "lbrs_spectral_contrast": spectral_contrast,
            
            # Single value features (aggregated across time)
            "lbrs_tempo": tempo,
            "lbrs_tempo_singlevalue": tempo,  # Tempo is already a single value
            "lbrs_spectral_centroid_singlevalue": spectral_centroid_singlevalue,
            "lbrs_spectral_bandwidth_singlevalue": spectral_bandwidth_singlevalue,
            "lbrs_spectral_flatness_singlevalue": spectral_flatness_singlevalue,
            "lbrs_spectral_rolloff_singlevalue": spectral_rolloff_singlevalue,
            "lbrs_spectral_contrast_singlevalue": spectral_contrast_singlevalue,
            "lbrs_rmse_singlevalue": rmse_singlevalue,
            "lbrs_zero_crossing_rate_singlevalue": zero_crossing_rate_singlevalue,
            
            # Metadata
            "sample_rate": sr,
            "hop_length": self.hop_length,
            "num_frames": len(spectral_centroid),
            "duration_seconds": len(y) / sr
        }
    
    def get_feature_dict(self, audio_path: str) -> dict:
        """
        Extract features and format them for the pipeline.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            dict: Dictionary of extracted features
        """
        return self.extract_all_features(audio_path)
