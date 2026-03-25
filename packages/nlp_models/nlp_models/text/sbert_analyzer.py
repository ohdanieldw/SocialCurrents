"""
Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
Implementation for the multimodal pipeline.

Reference: https://github.com/UKPLab/sentence-transformers
"""

import torch
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
import time
from sklearn.metrics.pairwise import cosine_similarity
import re

logger = logging.getLogger(__name__)


class SBERTAnalyzer:
    """
    Sentence-BERT analyzer for computing dense vector representations and reranking.
    
    This implementation provides:
    1. Dense sentence/paragraph embeddings with correlational matrices
    2. Reranker models for query-passage scoring and ranking
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu"
    ):
        """
        Initialize Sentence-BERT analyzer.
        
        Args:
            embedding_model: SentenceTransformer model for embeddings
            reranker_model: CrossEncoder model for reranking
            device: Device to run on ("cpu" or "cuda")
        """
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.device = device
        
        # Initialize models
        self.embedding_model = None
        self.reranker_model = None
        
        try:
            self._load_models()
        except Exception as e:
            logger.warning(f"Failed to load Sentence-BERT models: {e}")
    
    def _load_models(self):
        """Load Sentence-BERT embedding and reranker models."""
        try:
            logger.info(f"Loading Sentence-BERT models...")
            
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
            
            # Load reranker model
            self.reranker_model = CrossEncoder(self.reranker_model_name, device=self.device)
            
            logger.info("Sentence-BERT models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Sentence-BERT models: {e}")
            self.embedding_model = None
            self.reranker_model = None
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using basic sentence boundary detection.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting using regex
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Input text to split
            
        Returns:
            List of paragraphs
        """
        # Split by double newlines or when sentences are long enough
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If no paragraph breaks, split by sentence count
        if len(paragraphs) == 1 and len(text) > 200:
            sentences = self._split_into_sentences(text)
            # Group sentences into paragraphs of ~3 sentences
            paragraphs = []
            for i in range(0, len(sentences), 3):
                para = ' '.join(sentences[i:i+3])
                if para.strip():
                    paragraphs.append(para)
        
        return paragraphs
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute dense vector representations for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not texts or self.embedding_model is None:
            return np.array([])
        
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            return np.array([])
    
    def _compute_correlational_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute correlational matrix from embeddings.
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            Correlational matrix (cosine similarity matrix)
        """
        if embeddings.size == 0:
            return np.array([])
        
        try:
            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            return similarity_matrix
        except Exception as e:
            logger.error(f"Error computing correlational matrix: {e}")
            return np.array([])
    
    def _compute_reranker_scores(self, query: str, passages: List[str]) -> np.ndarray:
        """
        Compute reranker scores for query-passage pairs.
        
        Args:
            query: Query text
            passages: List of passage texts
            
        Returns:
            Array of reranker scores
        """
        if not passages or self.reranker_model is None:
            return np.array([])
        
        try:
            # Create query-passage pairs
            pairs = [(query, passage) for passage in passages]
            
            # Predict relevance scores
            scores = self.reranker_model.predict(pairs)
            return np.array(scores)
        except Exception as e:
            logger.error(f"Error computing reranker scores: {e}")
            return np.array([])
    
    def _rank_passages(self, query: str, passages: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Rank passages by relevance to query.
        
        Args:
            query: Query text
            passages: List of passage texts
            
        Returns:
            Tuple of (ranked_indices, scores)
        """
        scores = self._compute_reranker_scores(query, passages)
        if scores.size == 0:
            return [], np.array([])
        
        # Get ranked indices (descending order)
        ranked_indices = np.argsort(scores)[::-1].tolist()
        return ranked_indices, scores
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using Sentence-BERT for embeddings and reranking.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with BERT_ prefixed features
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for Sentence-BERT analysis")
            return self._get_default_metrics()
        
        if self.embedding_model is None or self.reranker_model is None:
            logger.error("Sentence-BERT models not properly initialized")
            return self._get_default_metrics()
        
        try:
            results = {}
            
            # Split text into sentences and paragraphs
            sentences = self._split_into_sentences(text)
            paragraphs = self._split_into_paragraphs(text)
            
            # Compute sentence embeddings
            sentence_embeddings = self._compute_embeddings(sentences)
            if sentence_embeddings.size > 0:
                sentence_correlation_matrix = self._compute_correlational_matrix(sentence_embeddings)
                # Store as flattened array for JSON serialization
                results["BERT_tensor_sentences"] = sentence_correlation_matrix.flatten().tolist()
                results["BERT_tensor_sentences_shape"] = list(sentence_correlation_matrix.shape)
            else:
                results["BERT_tensor_sentences"] = []
                results["BERT_tensor_sentences_shape"] = [0, 0]
            
            # Compute paragraph embeddings
            paragraph_embeddings = self._compute_embeddings(paragraphs)
            if paragraph_embeddings.size > 0:
                paragraph_correlation_matrix = self._compute_correlational_matrix(paragraph_embeddings)
                # Store as flattened array for JSON serialization
                results["BERT_tensor_paragraphs"] = paragraph_correlation_matrix.flatten().tolist()
                results["BERT_tensor_paragraphs_shape"] = list(paragraph_correlation_matrix.shape)
            else:
                results["BERT_tensor_paragraphs"] = []
                results["BERT_tensor_paragraphs_shape"] = [0, 0]
            
            # Reranker analysis - use first sentence as query, others as passages
            if len(sentences) > 1:
                query = sentences[0]
                passages = sentences[1:]
                
                # Compute reranker scores
                ranked_indices, scores = self._rank_passages(query, passages)
                
                # Store reranker scores (limit to first 10 for reasonable output size)
                if scores.size > 0:
                    results["BERT_score"] = scores[:min(10, len(scores))].tolist()
                    results["BERT_ranks"] = ranked_indices[:min(10, len(ranked_indices))]
                else:
                    results["BERT_score"] = []
                    results["BERT_ranks"] = []
            else:
                results["BERT_score"] = []
                results["BERT_ranks"] = []
            
            # Add metadata
            results.update({
                "BERT_analysis_timestamp": time.time(),
                "BERT_text_length": len(text),
                "BERT_sentence_count": len(sentences),
                "BERT_paragraph_count": len(paragraphs),
                "BERT_embedding_model": self.embedding_model_name,
                "BERT_reranker_model": self.reranker_model_name,
                "BERT_device": self.device
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Sentence-BERT analysis: {e}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """
        Get default metrics when analysis fails.
        
        Returns:
            Dictionary with empty values for all metrics
        """
        return {
            "BERT_tensor_sentences": [],
            "BERT_tensor_sentences_shape": [0, 0],
            "BERT_tensor_paragraphs": [],
            "BERT_tensor_paragraphs_shape": [0, 0],
            "BERT_score": [],
            "BERT_ranks": [],
            "BERT_analysis_timestamp": time.time(),
            "BERT_text_length": 0,
            "BERT_sentence_count": 0,
            "BERT_paragraph_count": 0,
            "BERT_embedding_model": self.embedding_model_name,
            "BERT_reranker_model": self.reranker_model_name,
            "BERT_device": self.device
        }
    
    def get_feature_dict(self, text_or_features: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract Sentence-BERT features from text input or existing feature dictionary.
        This method provides compatibility with the multimodal pipeline.
        
        Args:
            text_or_features: Either raw text string or dictionary with extracted features
                             (may contain transcription from WhisperX or other text sources)
            
        Returns:
            Dictionary with BERT_ prefixed sentence embedding and reranking features
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
            logger.warning("No text found for Sentence-BERT analysis")
            return self._get_default_metrics()
        
        # Perform analysis
        return self.analyze_text(text)
    
    def get_available_features(self) -> List[str]:
        """
        Get list of available Sentence-BERT feature names.
        
        Returns:
            List of feature names that will be extracted
        """
        return [
            "BERT_tensor_sentences",        # Sentence correlational matrix (flattened)
            "BERT_tensor_sentences_shape",  # Shape of sentence matrix
            "BERT_tensor_paragraphs",       # Paragraph correlational matrix (flattened)
            "BERT_tensor_paragraphs_shape", # Shape of paragraph matrix
            "BERT_score",                   # Reranker scores
            "BERT_ranks",                   # Ranked indices
            "BERT_analysis_timestamp",      # Analysis timestamp
            "BERT_text_length",             # Input text length
            "BERT_sentence_count",          # Number of sentences
            "BERT_paragraph_count",         # Number of paragraphs
            "BERT_embedding_model",         # Embedding model name
            "BERT_reranker_model",          # Reranker model name
            "BERT_device"                   # Computation device
        ]


def test_sbert_analyzer():
    """Test function for Sentence-BERT analyzer."""
    print("Testing Sentence-BERT Analyzer...")
    
    # Initialize analyzer
    analyzer = SBERTAnalyzer(device="cpu")
    
    # Test text samples
    test_texts = [
        "This is the first sentence. This is the second sentence. This is the third sentence for testing.",
        "Natural language processing enables computers to understand human text. "
        "Machine learning models can extract meaningful representations. "
        "Sentence embeddings capture semantic similarity between texts.",
        "Sentence-BERT creates dense vector representations for sentences and paragraphs. "
        "The model uses Siamese networks to learn meaningful embeddings. "
        "Reranker models can score and rank passages based on relevance to a query."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nAnalyzing text {i}: '{text[:50]}...'")
        features = analyzer.analyze_text(text)
        
        print("Sentence-BERT Features:")
        print(f"  Sentences analyzed: {features.get('BERT_sentence_count', 0)}")
        print(f"  Paragraphs analyzed: {features.get('BERT_paragraph_count', 0)}")
        
        # Show matrix shapes
        sent_shape = features.get('BERT_tensor_sentences_shape', [0, 0])
        para_shape = features.get('BERT_tensor_paragraphs_shape', [0, 0])
        print(f"  Sentence correlation matrix: {sent_shape[0]}x{sent_shape[1]}")
        print(f"  Paragraph correlation matrix: {para_shape[0]}x{para_shape[1]}")
        
        # Show reranker scores
        scores = features.get('BERT_score', [])
        if scores:
            print(f"  Reranker scores: {scores[:3]} (showing first 3)")
        else:
            print(f"  Reranker scores: None (need multiple sentences)")
    
    print("\nSentence-BERT analyzer test completed!")


if __name__ == "__main__":
    test_sbert_analyzer()
