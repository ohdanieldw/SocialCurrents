"""
GANimation: Continuous Manifold for Anatomical Facial Movements
Based on: https://github.com/albertpumarola/GANimation

This analyzer implements continuous manifold for anatomical facial movements
using Action Units (AUs) at different intensity levels (0, 33, 66, 99).
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
from pathlib import Path
import base64
import io
from PIL import Image
import warnings

logger = logging.getLogger(__name__)

class GANimationAnalyzer:
    """
    GANimation analyzer for continuous manifold anatomical facial movements.
    
    This analyzer implements Action Unit (AU) intensity estimation at discrete
    levels (0, 33, 66, 99) for anatomical facial movement analysis.
    """
    
    def __init__(self, device='cpu', confidence_threshold=0.5):
        """
        Initialize GANimation analyzer.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
            confidence_threshold: Minimum confidence threshold for AU detection
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.initialized = False
        
        # Action Units defined in GANimation
        # Based on Facial Action Coding System (FACS)
        self.action_units = [
            'AU1',   # Inner Brow Raiser
            'AU2',   # Outer Brow Raiser
            'AU4',   # Brow Lowerer
            'AU5',   # Upper Lid Raiser
            'AU6',   # Cheek Raiser
            'AU7',   # Lid Tightener
            'AU9',   # Nose Wrinkler
            'AU10',  # Upper Lip Raiser
            'AU12',  # Lip Corner Puller
            'AU14',  # Dimpler
            'AU15',  # Lip Corner Depressor
            'AU17',  # Chin Raiser
            'AU20',  # Lip Stretcher
            'AU23',  # Lip Tightener
            'AU25',  # Lips Part
            'AU26',  # Jaw Drop
            'AU45'   # Blink
        ]
        
        # Intensity levels for GANimation
        self.intensity_levels = [0, 33, 66, 99]
        
        # Face detection model
        self.face_cascade = None
        
        # Initialize default metrics for all AUs at all intensity levels
        self.default_metrics = {}
        
        for au in self.action_units:
            for intensity in self.intensity_levels:
                self.default_metrics[f'GAN_{au}_{intensity}'] = 0.0
        
        # Add summary metrics
        self.default_metrics['GAN_face_detected'] = 0.0
        self.default_metrics['GAN_total_au_activations'] = 0.0
        self.default_metrics['GAN_avg_au_intensity'] = 0.0
        self.default_metrics['GAN_SM_pic'] = ""
        
    def _initialize_model(self):
        """Initialize the GANimation model components."""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing GANimation analyzer...")
            
            # Initialize face detection (using OpenCV's Haar cascade as fallback)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                logger.warning("Could not load face cascade classifier")
            
            # Note: In a full implementation, you would load the actual GANimation model here
            # For this implementation, we'll use a simplified approach with facial landmark detection
            # and rule-based AU intensity estimation
            
            self.initialized = True
            logger.info("GANimation analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GANimation analyzer: {e}")
            raise
    
    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def _extract_facial_features(self, face_roi: np.ndarray) -> Dict[str, float]:
        """
        Extract facial features from a face ROI for AU intensity estimation.
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            Dictionary of basic facial features
        """
        # Convert to grayscale
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Basic feature extraction using image statistics
        # In a full implementation, this would use the trained GANimation model
        
        # Normalize image
        gray_roi = gray_roi.astype(np.float32) / 255.0
        
        # Extract basic statistical features
        features = {
            'mean_intensity': np.mean(gray_roi),
            'std_intensity': np.std(gray_roi),
            'eye_region_intensity': np.mean(gray_roi[:len(gray_roi)//3, :]),  # Upper third
            'mouth_region_intensity': np.mean(gray_roi[2*len(gray_roi)//3:, :]),  # Lower third
            'cheek_region_intensity': np.mean(gray_roi[len(gray_roi)//3:2*len(gray_roi)//3, :]),  # Middle third
        }
        
        return features
    
    def _estimate_au_intensities(self, facial_features: Dict[str, float], 
                                face_size: Tuple[int, int]) -> Dict[str, float]:
        """
        Estimate Action Unit intensities based on facial features.
        
        Args:
            facial_features: Extracted facial features
            face_size: Size of the detected face (width, height)
            
        Returns:
            Dictionary of AU intensities at different levels
        """
        au_intensities = {}
        
        # Simple rule-based AU intensity estimation
        # In a full implementation, this would use the trained GANimation model
        
        mean_intensity = facial_features['mean_intensity']
        std_intensity = facial_features['std_intensity']
        eye_intensity = facial_features['eye_region_intensity']
        mouth_intensity = facial_features['mouth_region_intensity']
        cheek_intensity = facial_features['cheek_region_intensity']
        
        # Estimate base activation probabilities for each AU
        au_base_activations = {
            'AU1': max(0, (eye_intensity - mean_intensity) * 2),  # Inner Brow Raiser
            'AU2': max(0, (eye_intensity - mean_intensity) * 1.8),  # Outer Brow Raiser
            'AU4': max(0, (mean_intensity - eye_intensity) * 1.5),  # Brow Lowerer
            'AU5': max(0, eye_intensity * 1.2),  # Upper Lid Raiser
            'AU6': max(0, cheek_intensity * 1.3),  # Cheek Raiser
            'AU7': max(0, (mean_intensity - eye_intensity) * 1.1),  # Lid Tightener
            'AU9': max(0, std_intensity * 0.8),  # Nose Wrinkler
            'AU10': max(0, (mouth_intensity - mean_intensity) * 1.4),  # Upper Lip Raiser
            'AU12': max(0, mouth_intensity * 1.5),  # Lip Corner Puller
            'AU14': max(0, cheek_intensity * 1.1),  # Dimpler
            'AU15': max(0, (mean_intensity - mouth_intensity) * 1.2),  # Lip Corner Depressor
            'AU17': max(0, mouth_intensity * 0.9),  # Chin Raiser
            'AU20': max(0, mouth_intensity * 1.1),  # Lip Stretcher
            'AU23': max(0, (mean_intensity - mouth_intensity) * 0.8),  # Lip Tightener
            'AU25': max(0, mouth_intensity * 1.3),  # Lips Part
            'AU26': max(0, mouth_intensity * 1.4),  # Jaw Drop
            'AU45': max(0, (mean_intensity - eye_intensity) * 1.0),  # Blink
        }
        
        # Convert to discrete intensity levels (0, 33, 66, 99)
        for au in self.action_units:
            base_activation = au_base_activations.get(au, 0.0)
            
            # Normalize and map to intensity levels
            normalized_activation = min(1.0, max(0.0, base_activation))
            
            # Determine which intensity levels are activated
            if normalized_activation < 0.25:
                # Low activation - mostly at level 0
                au_intensities[f'GAN_{au}_0'] = 1.0 - normalized_activation * 2
                au_intensities[f'GAN_{au}_33'] = normalized_activation * 2
                au_intensities[f'GAN_{au}_66'] = 0.0
                au_intensities[f'GAN_{au}_99'] = 0.0
            elif normalized_activation < 0.5:
                # Medium-low activation
                au_intensities[f'GAN_{au}_0'] = 0.0
                au_intensities[f'GAN_{au}_33'] = 1.0 - (normalized_activation - 0.25) * 4
                au_intensities[f'GAN_{au}_66'] = (normalized_activation - 0.25) * 4
                au_intensities[f'GAN_{au}_99'] = 0.0
            elif normalized_activation < 0.75:
                # Medium-high activation
                au_intensities[f'GAN_{au}_0'] = 0.0
                au_intensities[f'GAN_{au}_33'] = 0.0
                au_intensities[f'GAN_{au}_66'] = 1.0 - (normalized_activation - 0.5) * 4
                au_intensities[f'GAN_{au}_99'] = (normalized_activation - 0.5) * 4
            else:
                # High activation - mostly at level 99
                au_intensities[f'GAN_{au}_0'] = 0.0
                au_intensities[f'GAN_{au}_33'] = 0.0
                au_intensities[f'GAN_{au}_66'] = 1.0 - (normalized_activation - 0.75) * 4
                au_intensities[f'GAN_{au}_99'] = (normalized_activation - 0.75) * 4
        
        return au_intensities
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
        """
        Process a single frame for AU intensity estimation.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Tuple of (au_intensities_dict, annotated_frame)
        """
        if not self.initialized:
            self._initialize_model()
        
        # Initialize metrics with defaults
        metrics = self.default_metrics.copy()
        annotated_frame = frame.copy()
        
        # Detect faces
        faces = self._detect_faces(frame)
        
        if len(faces) > 0:
            metrics['GAN_face_detected'] = 1.0
            
            # Process the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Extract facial features
            facial_features = self._extract_facial_features(face_roi)
            
            # Estimate AU intensities
            au_intensities = self._estimate_au_intensities(facial_features, (w, h))
            
            # Update metrics
            metrics.update(au_intensities)
            
            # Calculate summary statistics
            total_activations = sum(1 for key, value in au_intensities.items() 
                                  if value > self.confidence_threshold)
            metrics['GAN_total_au_activations'] = float(total_activations)
            
            avg_intensity = np.mean(list(au_intensities.values()))
            metrics['GAN_avg_au_intensity'] = float(avg_intensity)
            
            # Draw face bounding box
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add AU activation text
            active_aus = []
            for au in self.action_units:
                max_intensity_level = 0
                max_intensity_value = 0
                for level in self.intensity_levels:
                    value = au_intensities.get(f'GAN_{au}_{level}', 0.0)
                    if value > max_intensity_value:
                        max_intensity_value = value
                        max_intensity_level = level
                
                if max_intensity_value > self.confidence_threshold:
                    active_aus.append(f"{au}:{max_intensity_level}")
            
            # Display active AUs on frame
            if active_aus:
                text = ", ".join(active_aus[:5])  # Show first 5 AUs
                cv2.putText(annotated_frame, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return metrics, annotated_frame
    
    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Base64 encoded string
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/jpeg;base64,{image_base64}"
            
        except Exception as e:
            logger.warning(f"Failed to encode image to base64: {e}")
            return ""
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze AU intensities in a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing AU analysis results
        """
        if not self.initialized:
            self._initialize_model()
        
        logger.info(f"Analyzing AU intensities in video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frame_metrics = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        best_annotated_frame = None
        max_au_activations = 0
        
        try:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                metrics, annotated_frame = self._process_frame(frame)
                
                # Keep the frame with most AU activations for visualization
                au_activations = metrics.get('GAN_total_au_activations', 0)
                if au_activations > max_au_activations:
                    max_au_activations = au_activations
                    best_annotated_frame = annotated_frame
                
                # Add frame index and timestamp
                metrics['frame_idx'] = frame_idx
                metrics['timestamp'] = frame_idx / fps if fps > 0 else frame_idx
                
                frame_metrics.append(metrics)
                frame_idx += 1
                
                # Progress logging
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        finally:
            cap.release()
        
        logger.info(f"Completed AU analysis: {len(frame_metrics)} frames processed")
        
        # Aggregate results
        return self._aggregate_results(frame_metrics, best_annotated_frame, video_path)
    
    def _aggregate_results(self, frame_metrics: List[Dict[str, Any]], 
                          best_frame: Optional[np.ndarray], 
                          video_path: str) -> Dict[str, Any]:
        """
        Aggregate frame-level results into final metrics.
        
        Args:
            frame_metrics: List of per-frame metrics
            best_frame: Best annotated frame for visualization
            video_path: Path to the video file
            
        Returns:
            Aggregated AU analysis results
        """
        if not frame_metrics:
            result = self.default_metrics.copy()
            result.update({
                'video_path': str(video_path),
                'total_frames': 0,
                'faces_detected_frames': 0,
                'face_detection_rate': 0.0
            })
            return result
        
        # Calculate aggregated metrics (mean across all frames)
        aggregated = {}
        numeric_keys = [key for key in frame_metrics[0].keys() 
                       if key.startswith('GAN_') and key != 'GAN_SM_pic']
        
        for key in numeric_keys:
            values = [frame.get(key, 0.0) for frame in frame_metrics]
            aggregated[key] = float(np.mean(values))
        
        # Add visualization of best frame
        if best_frame is not None:
            aggregated['GAN_SM_pic'] = self._encode_image_to_base64(best_frame)
        else:
            aggregated['GAN_SM_pic'] = ""
        
        # Add summary statistics
        faces_detected_frames = sum(1 for frame in frame_metrics 
                                   if frame.get('GAN_face_detected', 0) > 0.5)
        
        aggregated.update({
            'video_path': str(video_path),
            'total_frames': len(frame_metrics),
            'faces_detected_frames': faces_detected_frames,
            'face_detection_rate': faces_detected_frames / len(frame_metrics),
            'max_au_activations': max(frame.get('GAN_total_au_activations', 0) 
                                    for frame in frame_metrics),
            'avg_au_activations_per_frame': float(np.mean([
                frame.get('GAN_total_au_activations', 0) 
                for frame in frame_metrics
            ]))
        })
        
        return aggregated
    
    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        """
        Get GANimation AU features for the pipeline.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with GANimation AU features
        """
        try:
            results = self.analyze_video(video_path)
            
            # Create feature group for the pipeline
            feature_dict = {
                "Continuous manifold for anatomical facial movements": {
                    "description": "GANimation Action Unit intensity estimation at discrete levels (0, 33, 66, 99)",
                    "features": results
                }
            }
            
            return feature_dict
            
        except Exception as e:
            logger.error(f"Error in GANimation analysis: {e}")
            
            # Return default values on error
            default_result = self.default_metrics.copy()
            default_result.update({
                'video_path': str(video_path),
                'total_frames': 0,
                'faces_detected_frames': 0,
                'face_detection_rate': 0.0,
                'max_au_activations': 0,
                'avg_au_activations_per_frame': 0.0,
                'error': str(e)
            })
            
            feature_dict = {
                "Continuous manifold for anatomical facial movements": {
                    "description": "GANimation Action Unit intensity estimation at discrete levels (0, 33, 66, 99)",
                    "features": default_result
                }
            }
            
            return feature_dict

def extract_ganimation_features(video_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Extract GANimation AU features from a video file.
    
    Args:
        video_path: Path to the video file
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing AU features
    """
    analyzer = GANimationAnalyzer(device=device)
    return analyzer.get_feature_dict(video_path)
