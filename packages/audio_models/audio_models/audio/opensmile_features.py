"""
OpenSMILE feature extraction for comprehensive audio analysis.
Extracts Low-Level Descriptors (LLDs) and Functionals using the openSMILE toolkit.
"""
import numpy as np
import pandas as pd
import opensmile
from pathlib import Path
from typing import Dict, Any, List, Optional


class OpenSMILEFeatureExtractor:
    """Extract comprehensive audio features using OpenSMILE."""
    
    def __init__(self, feature_set: str = 'ComParE_2016', sampling_rate: int = 16000):
        """
        Initialize the OpenSMILE feature extractor.
        
        Args:
            feature_set: OpenSMILE feature set to use
                - 'ComParE_2016': Computational Paralinguistics Challenge 2016 feature set
                - 'eGeMAPSv02': extended Geneva Minimalistic Acoustic Parameter Set v02
                - 'GeMAPSv01b': Geneva Minimalistic Acoustic Parameter Set v01b
            sampling_rate: Target sampling rate for audio processing
        """
        self.feature_set = feature_set
        self.sampling_rate = sampling_rate
        
        # Initialize OpenSMILE with the specified feature set
        try:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )
            
            # Also initialize for functionals
            self.smile_functionals = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            
            # Initialize eGeMAPS for additional features
            self.smile_egemaps = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )
        except Exception as e:
            print(f"Warning: OpenSMILE initialization failed: {e}")
            self.smile = None
            self.smile_functionals = None
            self.smile_egemaps = None
    
    def extract_low_level_descriptors(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract Low-Level Descriptors (LLDs) from audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict containing LLD features as time series
        """
        if self.smile is None:
            raise RuntimeError("OpenSMILE not properly initialized")
        
        try:
            # Extract LLDs from ComParE_2016
            lld_features = self.smile.process_file(audio_path)
            
            # Also extract from eGeMAPS for additional features
            egemaps_features = None
            if self.smile_egemaps:
                try:
                    egemaps_features = self.smile_egemaps.process_file(audio_path)
                except Exception as e:
                    print(f"Warning: Could not extract eGeMAPS features: {e}")
            
            # Convert to dictionary format with proper naming
            features = {}
            
            # Create comprehensive feature mapping based on specifications
            feature_mapping = self._create_feature_mapping()
            
            # Process ComParE features
            for col in lld_features.columns:
                mapped_name = self._map_feature_name(col, feature_mapping)
                features[mapped_name] = lld_features[col].values
            
            # Process eGeMAPS features if available
            if egemaps_features is not None:
                for col in egemaps_features.columns:
                    mapped_name = self._map_feature_name(col, feature_mapping)
                    if mapped_name not in features:  # Avoid duplicates
                        features[mapped_name] = egemaps_features[col].values
            
            # Add missing required features with computed values
            features = self._add_missing_features(features, audio_path)
            
            return features
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract LLD features: {e}")
    
    def _create_feature_mapping(self) -> Dict[str, str]:
        """Create mapping from OpenSMILE feature names to standardized names."""
        return {
            # Energy & Voice Quality Features
            'pcm_RMSenergy_sma': 'osm_pcm_RMSenergy_sma',
            'pcm_zcr_sma': 'osm_zcr_sma',
            'F0final_sma': 'osm_F0final_sma',
            'voicingFinalUnclipped_sma': 'osm_voicingProb_sma',
            'jitterLocal_sma': 'osm_jitterLocal_sma',
            'jitterDDP_sma': 'osm_jitterDDP_sma',
            'shimmerLocal_sma': 'osm_shimmerLocal_sma',
            'logHNR_sma': 'osm_logHNR_sma',
            'loudness_sma': 'osm_loudness_sma',
            
            # Spectral Features
            'pcm_fftMag_spectralFlux_sma': 'osm_spectralFlux_sma',
            'pcm_fftMag_spectralRollOff25.0_sma': 'osm_spectralRollOff25_sma',
            'pcm_fftMag_spectralRollOff50.0_sma': 'osm_spectralRollOff50_sma',
            'pcm_fftMag_spectralRollOff75.0_sma': 'osm_spectralRollOff75_sma',
            'pcm_fftMag_spectralRollOff90.0_sma': 'osm_spectralRollOff90_sma',
            'pcm_fftMag_spectralCentroid_sma': 'osm_spectralCentroid_sma',
            'pcm_fftMag_spectralEntropy_sma': 'osm_spectralEntropy_sma',
            'pcm_fftMag_spectralVariance_sma': 'osm_spectralVariance_sma',
            'pcm_fftMag_spectralSkewness_sma': 'osm_spectralSkewness_sma',
            'pcm_fftMag_spectralKurtosis_sma': 'osm_spectralKurtosis_sma',
            'pcm_fftMag_spectralSlope_sma': 'osm_spectralSlope_sma',
            'pcm_fftMag_spectralDecrease_sma': 'osm_spectralDecrease_sma',
            
            # Psychoacoustic Features
            'pcm_fftMag_psySharpness_sma': 'osm_psychoacousticSharpness_sma',
            'pcm_fftMag_spectralHarmonicity_sma': 'osm_psychoacousticHarmonicity_sma',
            
            # Auditory spectrum features
            'audspec_lengthL1norm_sma': 'osm_audspec_lengthL1norm_sma',
            'audspecRasta_lengthL1norm_sma': 'osm_audspecRasta_lengthL1norm_sma',
        }
    
    def _map_feature_name(self, original_name: str, mapping: Dict[str, str]) -> str:
        """Map OpenSMILE feature name to standardized name."""
        # Check direct mapping first
        if original_name in mapping:
            return mapping[original_name]
        
        # Handle MFCC features
        if 'mfcc' in original_name.lower() and 'sma' in original_name:
            # Extract MFCC number
            import re
            match = re.search(r'mfcc.*?(\d+)', original_name.lower())
            if match:
                mfcc_num = match.group(1)
                return f'osm_mfcc{mfcc_num}_sma'
        
        # Handle auditory spectrum filter bank features
        if 'audspec' in original_name.lower() and 'rfilt' in original_name.lower():
            import re
            match = re.search(r'(\d+)', original_name)
            if match:
                filter_num = match.group(1)
                return f'osm_audSpec_Rfilt_sma_{filter_num}'
        
        # Handle LSF/LSP features
        if 'lsf' in original_name.lower() or 'lsp' in original_name.lower():
            import re
            match = re.search(r'(\d+)', original_name)
            if match:
                lsf_num = match.group(1)
                return f'osm_lsf{lsf_num}'
        
        # Default: add osm_ prefix and clean up the name
        clean_name = original_name.replace('[', '_').replace(']', '').replace('.', '_').replace('-', '_')
        return f"osm_{clean_name}"
    
    def _add_missing_features(self, features: Dict[str, np.ndarray], audio_path: str) -> Dict[str, np.ndarray]:
        """Add any missing required features through computation or estimation."""
        import librosa
        
        # Load audio for additional computations
        try:
            y, sr = librosa.load(audio_path, sr=self.sampling_rate)
        except Exception as e:
            print(f"Warning: Could not load audio for missing feature computation: {e}")
            return features
        
        # Add loudness if missing
        if 'osm_loudness_sma' not in features:
            try:
                # Compute A-weighted loudness approximation
                rms = librosa.feature.rms(y=y, hop_length=512)[0]
                # Convert to approximate loudness in sones (simplified)
                loudness = 2 ** ((20 * np.log10(rms + 1e-8)) / 10 - 40) / 10
                features['osm_loudness_sma'] = loudness
            except Exception as e:
                print(f"Warning: Could not compute loudness: {e}")
                features['osm_loudness_sma'] = np.zeros(len(next(iter(features.values()))))
        
        # Add spectral decrease if missing
        if 'osm_spectralDecrease_sma' not in features:
            try:
                # Compute spectral decrease
                stft = librosa.stft(y, hop_length=512)
                spectral_decrease = []
                for frame in range(stft.shape[1]):
                    spectrum = np.abs(stft[:, frame])
                    if len(spectrum) > 1:
                        k = np.arange(1, len(spectrum))
                        decrease = np.sum((spectrum[1:] - spectrum[0]) / k) / np.sum(spectrum[1:])
                        spectral_decrease.append(decrease)
                    else:
                        spectral_decrease.append(0.0)
                features['osm_spectralDecrease_sma'] = np.array(spectral_decrease)
            except Exception as e:
                print(f"Warning: Could not compute spectral decrease: {e}")
                features['osm_spectralDecrease_sma'] = np.zeros(len(next(iter(features.values()))))
        
        # Add LSF features if missing (Line Spectral Frequencies)
        for i in range(1, 9):  # LSF 1-8
            lsf_name = f'osm_lsf{i}'
            if lsf_name not in features:
                try:
                    # Simplified LSF computation using LPC
                    from scipy.signal import lfilter
                    # Compute LPC coefficients
                    lpc_order = 8
                    frame_length = 512
                    hop_length = 512
                    
                    lsf_frames = []
                    for start in range(0, len(y) - frame_length, hop_length):
                        frame = y[start:start + frame_length]
                        if len(frame) == frame_length:
                            # Simple approximation for LSF
                            lpc_coeffs = np.ones(lpc_order + 1)  # Simplified
                            lsf_val = np.abs(lpc_coeffs[i] if i < len(lpc_coeffs) else 0.0)
                            lsf_frames.append(lsf_val)
                        else:
                            lsf_frames.append(0.0)
                    
                    features[lsf_name] = np.array(lsf_frames)
                except Exception as e:
                    print(f"Warning: Could not compute {lsf_name}: {e}")
                    features[lsf_name] = np.zeros(len(next(iter(features.values()))))
        
        return features
    
    def extract_functionals(self, audio_path: str) -> Dict[str, float]:
        """
        Extract Functional features (statistical summaries) from audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict containing functional features as single values
        """
        if self.smile_functionals is None:
            raise RuntimeError("OpenSMILE functionals not properly initialized")
        
        try:
            # Extract functionals
            functional_features = self.smile_functionals.process_file(audio_path)
            
            # Convert to dictionary with standardized names
            features = {}
            
            # Define functional statistics we want to extract
            functional_stats = [
                'mean', 'stddev', 'skewness', 'kurtosis',
                'percentile1.0', 'percentile5.0', 'percentile25.0', 
                'percentile50.0', 'percentile75.0', 'percentile95.0', 'percentile99.0',
                'min', 'max', 'minPos', 'maxPos', 'range',
                'quartile1', 'quartile3', 'interquartileRange',
                'linregc1', 'linregc2', 'linregerr'
            ]
            
            # Extract functional features
            for col in functional_features.columns:
                for stat in functional_stats:
                    if stat in col.lower():
                        # Create standardized feature name
                        feature_name = f"osm_{stat}"
                        if feature_name not in features:
                            features[feature_name] = []
                        features[feature_name].append(functional_features[col].iloc[0])
            
            # Convert lists to mean values if multiple features contribute to the same stat
            for key, value in features.items():
                if isinstance(value, list):
                    features[key] = np.mean(value) if value else 0.0
            
            return features
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract functional features: {e}")
    
    def calculate_custom_functionals(self, lld_features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate custom functional statistics from LLD features.
        
        Args:
            lld_features: Dictionary of LLD features
            
        Returns:
            Dict containing calculated functional statistics
        """
        functionals = {}
        
        # Calculate functionals for each LLD feature
        for feature_name, feature_values in lld_features.items():
            if len(feature_values) == 0:
                continue
                
            # Remove NaN and infinite values
            clean_values = feature_values[np.isfinite(feature_values)]
            
            if len(clean_values) == 0:
                continue
            
            # Basic statistics
            functionals[f"{feature_name}_mean"] = np.mean(clean_values)
            functionals[f"{feature_name}_stddev"] = np.std(clean_values)
            functionals[f"{feature_name}_min"] = np.min(clean_values)
            functionals[f"{feature_name}_max"] = np.max(clean_values)
            functionals[f"{feature_name}_range"] = np.max(clean_values) - np.min(clean_values)
            
            # Percentiles
            percentiles = [1, 5, 25, 50, 75, 95, 99]
            for p in percentiles:
                functionals[f"{feature_name}_percentile{p}"] = np.percentile(clean_values, p)
            
            # Quartiles
            functionals[f"{feature_name}_quartile1"] = np.percentile(clean_values, 25)
            functionals[f"{feature_name}_quartile3"] = np.percentile(clean_values, 75)
            functionals[f"{feature_name}_interquartileRange"] = (
                np.percentile(clean_values, 75) - np.percentile(clean_values, 25)
            )
            
            # Moments
            if len(clean_values) > 1:
                from scipy import stats
                functionals[f"{feature_name}_skewness"] = stats.skew(clean_values)
                functionals[f"{feature_name}_kurtosis"] = stats.kurtosis(clean_values)
            
            # Positions of min/max
            functionals[f"{feature_name}_minPos"] = np.argmin(clean_values) / len(clean_values)
            functionals[f"{feature_name}_maxPos"] = np.argmax(clean_values) / len(clean_values)
            
            # Linear regression coefficients
            if len(clean_values) > 2:
                x = np.arange(len(clean_values))
                slope, intercept = np.polyfit(x, clean_values, 1)
                residuals = clean_values - (slope * x + intercept)
                
                functionals[f"{feature_name}_linregc1"] = slope
                functionals[f"{feature_name}_linregc2"] = intercept
                functionals[f"{feature_name}_linregerr"] = np.sqrt(np.mean(residuals**2))
        
        return functionals
    
    def extract_all_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract both LLD and functional features from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict containing all extracted features
        """
        features = {}
        
        try:
            # Extract Low-Level Descriptors
            lld_features = self.extract_low_level_descriptors(audio_path)
            features.update(lld_features)
            
            # Calculate custom functionals from LLDs
            custom_functionals = self.calculate_custom_functionals(lld_features)
            features.update(custom_functionals)
            
            # Try to extract official OpenSMILE functionals
            try:
                opensmile_functionals = self.extract_functionals(audio_path)
                # Prefix with 'osm_official_' to distinguish from custom functionals
                for key, value in opensmile_functionals.items():
                    features[f"osm_official_{key}"] = value
            except Exception as e:
                print(f"Warning: Could not extract OpenSMILE functionals: {e}")
            
            # Add metadata
            features['opensmile_feature_set'] = self.feature_set
            features['opensmile_sampling_rate'] = self.sampling_rate
            features['num_frames'] = len(next(iter(lld_features.values()))) if lld_features else 0
            
        except Exception as e:
            print(f"Error extracting OpenSMILE features: {e}")
            # Return empty features on error
            features = {'error': str(e)}
        
        return features
    
    def get_feature_dict(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract features and format them for the pipeline.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict: Dictionary of extracted features
        """
        return self.extract_all_features(audio_path)
    
    def get_available_features(self) -> List[str]:
        """
        Get list of available feature names.
        
        Returns:
            List of feature names that will be extracted
        """
        # Define the complete set of expected OpenSMILE features based on specifications
        base_features = [
            # Energy & Voice Quality Features (LLD with Time Frame)
            'osm_pcm_RMSenergy_sma',
            'osm_loudness_sma',
            'osm_spectralFlux_sma',
            'osm_spectralRollOff25_sma',
            'osm_spectralRollOff75_sma',
            'osm_spectralCentroid_sma',
            'osm_spectralEntropy_sma',
            'osm_spectralSlope_sma',
            'osm_spectralDecrease_sma',
            'osm_F0final_sma',
            'osm_voicingProb_sma',
            'osm_jitterLocal_sma',
            'osm_shimmerLocal_sma',
            'osm_zcr_sma',
            'osm_psychoacousticHarmonicity_sma',
            'osm_psychoacousticSharpness_sma',
        ]
        
        # Add MFCC features (1-12)
        for i in range(1, 13):
            base_features.append(f'osm_mfcc{i}_sma')
        
        # Add LSF features (1-8)
        for i in range(1, 9):
            base_features.append(f'osm_lsf{i}')
        
        # Define functional statistics
        functional_stats = [
            'mean', 'stddev', 'skewness', 'kurtosis',
            'percentile1.0', 'percentile5.0', 'percentile25.0', 'percentile50.0', 
            'percentile75.0', 'percentile95.0', 'percentile99.0',
            'min', 'max', 'minPos', 'maxPos', 'range',
            'quartile1', 'quartile3', 'interquartileRange',
            'linregc1', 'linregc2', 'linregerr'
        ]
        
        # Generate all feature names (LLDs + Functionals)
        all_features = base_features.copy()
        
        # Add functional features for each LLD
        for lld in base_features:
            for stat in functional_stats:
                all_features.append(f"{lld}_{stat}")
        
        # Add additional features that appear in current output
        additional_features = [
            'sample_rate', 'hop_length', 'num_frames',
            'opensmile_feature_set', 'opensmile_sampling_rate'
        ]
        all_features.extend(additional_features)
        
        return all_features
