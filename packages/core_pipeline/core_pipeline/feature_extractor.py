"""
Wrapper class providing a MultimodalFeatureExtractor interface.
This is a convenience wrapper around the MultimodalPipeline class.
"""
from typing import Dict, List, Any, Union
from pathlib import Path
from core_pipeline.pipeline import MultimodalPipeline


class MultimodalFeatureExtractor:
    """
    A wrapper around MultimodalPipeline providing a simpler feature extraction interface.
    """
    
    def __init__(
        self,
        features: List[str] = None,
        device: str = "cpu",
        output_dir: str = None
    ):
        """
        Initialize the multimodal feature extractor.
        
        Args:
            features: List of features to extract (if None, extract all)
            device: Device to run models on ("cpu" or "cuda")
            output_dir: Directory to save output files
        """
        self.pipeline = MultimodalPipeline(
            output_dir=output_dir,
            features=features,
            device=device
        )
    
    def extract_features(self, data: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract features from various data sources.
        
        Args:
            data: Can be:
                - Path to audio/video file (str or Path)
                - Dictionary with feature data (for DeBERTa text analysis)
                
        Returns:
            Dict[str, Any]: Dictionary with all extracted features
        """
        if isinstance(data, dict):
            # Handle dictionary input (for text analysis or pre-extracted features)
            return self._extract_from_dict(data)
        else:
            # Handle file path input
            file_path = Path(data)
            if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                return self.pipeline.process_video_file(str(file_path))
            else:
                return self.pipeline.process_audio_file(str(file_path))
    
    def _extract_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from a dictionary of data.
        
        Args:
            data: Dictionary containing data for feature extraction
            
        Returns:
            Dict[str, Any]: Dictionary with extracted features
        """
        features = {}
        
        # If DeBERTa text analysis is requested, process text data
        if "deberta_text" in self.pipeline.features:
            extractor = self.pipeline._get_extractor("deberta_text")
            deberta_features = extractor.get_feature_dict(data)
            features.update(deberta_features)
        
        # If SimCSE text analysis is requested, process text data
        if "simcse_text" in self.pipeline.features:
            extractor = self.pipeline._get_extractor("simcse_text")
            simcse_features = extractor.get_feature_dict(data)
            features.update(simcse_features)
        
        # If ALBERT text analysis is requested, process text data
        if "albert_text" in self.pipeline.features:
            extractor = self.pipeline._get_extractor("albert_text")
            albert_features = extractor.get_feature_dict(data)
            features.update(albert_features)
        
        # If Sentence-BERT text analysis is requested, process text data
        if "sbert_text" in self.pipeline.features:
            extractor = self.pipeline._get_extractor("sbert_text")
            sbert_features = extractor.get_feature_dict(data)
            features.update(sbert_features)
        
        # If Universal Sentence Encoder text analysis is requested, process text data
        if "use_text" in self.pipeline.features:
            extractor = self.pipeline._get_extractor("use_text")
            use_features = extractor.get_feature_dict(data)
            features.update(use_features)

        if "elmo_text" in self.pipeline.features:
            extractor = self.pipeline._get_extractor("elmo_text")
            elmo_features = extractor.get_feature_dict(data)
            features.update(elmo_features)
        
        # Can add other feature extractors that work with dictionary input here
        
        return features
