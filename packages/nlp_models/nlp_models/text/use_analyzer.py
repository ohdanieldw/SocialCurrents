"""
Universal Sentence Encoder (USE): Text Classification + Semantic Similarity + Semantic Clustering
Implementation for the multimodal pipeline.

Reference: https://tfhub.dev/google/universal-sentence-encoder/1
"""

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    _TF_AVAILABLE = True
except ImportError:
    tf = None
    hub = None
    _TF_AVAILABLE = False

import numpy as np
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import logging
import time
import re

logger = logging.getLogger(__name__)


class USEAnalyzer:
    """
    Universal Sentence Encoder analyzer for text classification, semantic similarity, and clustering.
    
    This implementation provides:
    1. Fixed-length embedding vectors (512 dimensions) for any input text
    2. Sentence-level embeddings regardless of text length
    3. Support for text classification, semantic similarity, and clustering tasks
    """
    
    def __init__(
        self,
        model_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4",
        device: str = "cpu"
    ):
        """
        Initialize Universal Sentence Encoder analyzer.
        
        Args:
            model_url: TensorFlow Hub URL for Universal Sentence Encoder model
            device: Device to run on ("cpu" or "cuda") - USE runs on TensorFlow
        """
        self.model_url = model_url
        self.device = device
        
        # Initialize model
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Universal Sentence Encoder model."""
        if not _TF_AVAILABLE:
            logger.warning("TensorFlow not available — Universal Sentence Encoder is disabled.")
            self.model = None
            return
        try:
            logger.info(f"Loading Universal Sentence Encoder from: {self.model_url}")

            # Configure TensorFlow to use appropriate device
            if self.device == "cpu":
                tf.config.set_visible_devices([], 'GPU')

            # Load the model from TensorFlow Hub
            self.model = hub.load(self.model_url)

            logger.info("Universal Sentence Encoder model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Universal Sentence Encoder: {e}")
            self.model = None
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for individual embedding generation.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting using periods, exclamation marks, and question marks
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If no sentences found, return the original text
        if not sentences:
            sentences = [text.strip()]
            
        return sentences
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate USE embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings (num_texts, 512)
        """
        if not texts or self.model is None:
            return np.array([])
        
        try:
            # Generate embeddings using Universal Sentence Encoder
            embeddings = self.model(texts)
            
            # Convert to numpy array
            embeddings_np = embeddings.numpy()
            
            logger.debug(f"Generated embeddings shape: {embeddings_np.shape}")
            return embeddings_np
            
        except Exception as e:
            logger.error(f"Error generating USE embeddings: {e}")
            return np.array([])
    
    def _compute_semantic_similarity(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Compute semantic similarity metrics between embeddings.
        
        Args:
            embeddings: Array of embeddings (num_texts, 512)
            
        Returns:
            Dictionary with similarity metrics
        """
        if embeddings.size == 0 or len(embeddings) < 2:
            return {
                "USE_avg_cosine_similarity": 0.0,
                "USE_max_cosine_similarity": 0.0,
                "USE_min_cosine_similarity": 0.0
            }
        
        # Compute cosine similarity matrix
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        if len(upper_triangle) > 0:
            return {
                "USE_avg_cosine_similarity": float(np.mean(upper_triangle)),
                "USE_max_cosine_similarity": float(np.max(upper_triangle)),
                "USE_min_cosine_similarity": float(np.min(upper_triangle))
            }
        else:
            return {
                "USE_avg_cosine_similarity": 0.0,
                "USE_max_cosine_similarity": 0.0,
                "USE_min_cosine_similarity": 0.0
            }
    
    def _compute_clustering_metrics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Compute clustering-related metrics for embeddings.
        
        Args:
            embeddings: Array of embeddings (num_texts, 512)
            
        Returns:
            Dictionary with clustering metrics
        """
        if embeddings.size == 0:
            return {
                "USE_centroid_distance": 0.0,
                "USE_spread_variance": 0.0,
                "USE_avg_pairwise_distance": 0.0
            }
        
        try:
            # Compute centroid
            centroid = np.mean(embeddings, axis=0)
            
            # Distance from each embedding to centroid
            centroid_distances = np.linalg.norm(embeddings - centroid, axis=1)
            
            # Compute spread (variance of distances to centroid)
            spread_variance = float(np.var(centroid_distances))
            
            # Average distance to centroid
            avg_centroid_distance = float(np.mean(centroid_distances))
            
            # Average pairwise distance
            if len(embeddings) > 1:
                pairwise_distances = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        dist = np.linalg.norm(embeddings[i] - embeddings[j])
                        pairwise_distances.append(dist)
                avg_pairwise_distance = float(np.mean(pairwise_distances))
            else:
                avg_pairwise_distance = 0.0
            
            return {
                "USE_centroid_distance": avg_centroid_distance,
                "USE_spread_variance": spread_variance,
                "USE_avg_pairwise_distance": avg_pairwise_distance
            }
            
        except Exception as e:
            logger.error(f"Error computing clustering metrics: {e}")
            return {
                "USE_centroid_distance": 0.0,
                "USE_spread_variance": 0.0,
                "USE_avg_pairwise_distance": 0.0
            }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using Universal Sentence Encoder for embeddings and semantic analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with USE_ prefixed features
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for USE analysis")
            return self._get_default_metrics()
        
        if self.model is None:
            logger.error("Universal Sentence Encoder model not properly initialized")
            return self._get_default_metrics()
        
        try:
            results = {}
            
            # Split text into sentences
            sentences = self._split_into_sentences(text)
            
            # Generate embeddings for each sentence
            embeddings = self._generate_embeddings(sentences)
            
            if embeddings.size > 0:
                # Store individual sentence embeddings (limit to reasonable number)
                max_sentences = 10  # Limit to prevent overly large output
                num_sentences = min(len(sentences), max_sentences)
                
                for i in range(num_sentences):
                    # Store each sentence embedding as a list (512 dimensions)
                    embedding_key = f"USE_embed_sentence{i+1}"
                    results[embedding_key] = embeddings[i].tolist()
                
                # Compute semantic similarity metrics
                similarity_metrics = self._compute_semantic_similarity(embeddings)
                results.update(similarity_metrics)
                
                # Compute clustering metrics
                clustering_metrics = self._compute_clustering_metrics(embeddings)
                results.update(clustering_metrics)
                
                # Overall text embedding (mean of sentence embeddings)
                overall_embedding = np.mean(embeddings, axis=0)
                results["USE_embed_overall"] = overall_embedding.tolist()
                
            else:
                # No embeddings generated
                results["USE_embed_sentence1"] = [0.0] * 512  # Default 512-dim vector
                results.update({
                    "USE_avg_cosine_similarity": 0.0,
                    "USE_max_cosine_similarity": 0.0,
                    "USE_min_cosine_similarity": 0.0,
                    "USE_centroid_distance": 0.0,
                    "USE_spread_variance": 0.0,
                    "USE_avg_pairwise_distance": 0.0
                })
                results["USE_embed_overall"] = [0.0] * 512
            
            # Add metadata
            results.update({
                "USE_analysis_timestamp": time.time(),
                "USE_text_length": len(text),
                "USE_sentence_count": len(sentences),
                "USE_embedding_dimension": 512,
                "USE_model_url": self.model_url,
                "USE_device": self.device
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Universal Sentence Encoder analysis: {e}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """
        Get default metrics when analysis fails.
        
        Returns:
            Dictionary with default values for all metrics
        """
        return {
            "USE_embed_sentence1": [0.0] * 512,
            "USE_embed_overall": [0.0] * 512,
            "USE_avg_cosine_similarity": 0.0,
            "USE_max_cosine_similarity": 0.0,
            "USE_min_cosine_similarity": 0.0,
            "USE_centroid_distance": 0.0,
            "USE_spread_variance": 0.0,
            "USE_avg_pairwise_distance": 0.0,
            "USE_analysis_timestamp": time.time(),
            "USE_text_length": 0,
            "USE_sentence_count": 0,
            "USE_embedding_dimension": 512,
            "USE_model_url": self.model_url,
            "USE_device": self.device
        }
    
    def get_feature_dict(self, text_or_features: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract Universal Sentence Encoder features from text input or existing feature dictionary.
        This method provides compatibility with the multimodal pipeline.
        
        Args:
            text_or_features: Either raw text string or dictionary with extracted features
                             (may contain transcription from WhisperX or other text sources)
            
        Returns:
            Dictionary with USE_ prefixed embedding and semantic analysis features
        """
        # Extract text from various sources
        text = ""
        
        if isinstance(text_or_features, str):
            text = text_or_features
        elif isinstance(text_or_features, dict):
            # Look for text in feature dictionary from various sources
            text_sources = [
                'transcription',          # WhisperX transcription (actual key used)
                'whisperx_transcription', # WhisperX transcription alternative
                'transcript',             # Generic transcript
                'text',                   # Direct text field
                'speech_text',            # Speech-to-text output
                'transcribed_text',       # Alternative transcription field
            ]
            
            for source in text_sources:
                if source in text_or_features and text_or_features[source]:
                    if isinstance(text_or_features[source], str):
                        text = text_or_features[source]
                        break
                    elif isinstance(text_or_features[source], dict):
                        # Handle nested structures
                        if 'text' in text_or_features[source]:
                            text = text_or_features[source]['text']
                            break
                        elif 'transcription' in text_or_features[source]:
                            text = text_or_features[source]['transcription']
                            break
        
        # If no text found, provide default metrics
        if not text or not text.strip():
            logger.warning("No text found for Universal Sentence Encoder analysis")
            return self._get_default_metrics()
        
        # Perform analysis
        return self.analyze_text(text)
    
    def get_available_features(self) -> List[str]:
        """
        Get list of available Universal Sentence Encoder feature names.
        
        Returns:
            List of feature names that will be extracted
        """
        return [
            "USE_embed_sentence1",        # First sentence embedding (512-dim)
            "USE_embed_sentence2",        # Second sentence embedding (if exists)
            "USE_embed_sentence3",        # Third sentence embedding (if exists)
            "USE_embed_sentence4",        # Fourth sentence embedding (if exists)
            "USE_embed_sentence5",        # Fifth sentence embedding (if exists)
            "USE_embed_sentence6",        # Sixth sentence embedding (if exists)
            "USE_embed_sentence7",        # Seventh sentence embedding (if exists)
            "USE_embed_sentence8",        # Eighth sentence embedding (if exists)
            "USE_embed_sentence9",        # Ninth sentence embedding (if exists)
            "USE_embed_sentence10",       # Tenth sentence embedding (if exists)
            "USE_embed_overall",          # Overall text embedding (mean of sentences)
            "USE_avg_cosine_similarity",  # Average cosine similarity between sentences
            "USE_max_cosine_similarity",  # Maximum cosine similarity between sentences
            "USE_min_cosine_similarity",  # Minimum cosine similarity between sentences
            "USE_centroid_distance",      # Average distance to embedding centroid
            "USE_spread_variance",        # Variance in distances to centroid
            "USE_avg_pairwise_distance",  # Average pairwise distance between embeddings
            "USE_analysis_timestamp",     # Analysis timestamp
            "USE_text_length",            # Input text length
            "USE_sentence_count",         # Number of sentences processed
            "USE_embedding_dimension",    # Embedding dimension (512)
            "USE_model_url",              # Model URL identifier
            "USE_device"                  # Computation device
        ]


def test_use_analyzer():
    """Test function for Universal Sentence Encoder analyzer."""
    print("Testing Universal Sentence Encoder Analyzer...")
    
    # Initialize analyzer
    analyzer = USEAnalyzer(device="cpu")
    
    # Test text samples
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing enables computers to understand human text. "
        "Machine learning models can extract meaningful representations from sentences.",
        "This is the first sentence. This is the second sentence. This is the third sentence for testing embeddings.",
        "Universal Sentence Encoder generates fixed-length vectors for any input text. "
        "These embeddings capture semantic meaning effectively. "
        "They can be used for classification, similarity, and clustering tasks."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nAnalyzing text {i}: '{text[:50]}...'")
        features = analyzer.analyze_text(text)
        
        print("Universal Sentence Encoder Features:")
        print(f"  Sentences processed: {features.get('USE_sentence_count', 0)}")
        print(f"  Embedding dimension: {features.get('USE_embedding_dimension', 0)}")
        
        # Show embedding info
        embed_keys = [k for k in features.keys() if k.startswith('USE_embed_sentence')]
        print(f"  Sentence embeddings generated: {len(embed_keys)}")
        
        # Show semantic similarity
        avg_sim = features.get('USE_avg_cosine_similarity', 0)
        print(f"  Average cosine similarity: {avg_sim:.3f}")
        
        # Show clustering metrics
        centroid_dist = features.get('USE_centroid_distance', 0)
        spread_var = features.get('USE_spread_variance', 0)
        print(f"  Centroid distance: {centroid_dist:.3f}")
        print(f"  Spread variance: {spread_var:.3f}")
        
        # Show sample embedding values (first few dimensions)
        if 'USE_embed_sentence1' in features:
            embedding = features['USE_embed_sentence1']
            if isinstance(embedding, list) and len(embedding) > 0:
                sample_values = embedding[:5]  # First 5 dimensions
                print(f"  Sample embedding values: [{', '.join(f'{v:.4f}' for v in sample_values)}, ...]")
    
    print("\nUniversal Sentence Encoder analyzer test completed!")


if __name__ == "__main__":
    test_use_analyzer()
