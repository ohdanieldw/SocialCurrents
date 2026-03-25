"""
VideoFinder: Locate the objects and people
Based on: https://github.com/win4r/VideoFinder-Llama3.2-vision-Ollama

This analyzer implements object and people localization using VideoFinder
with consistency and match evaluation metrics.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import base64
import io
from PIL import Image
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class VideoFinderAnalyzer:
    """
    VideoFinder analyzer for locating objects and people in videos.
    
    This analyzer implements:
    - Object detection and localization
    - People detection and tracking
    - Consistency evaluation across frames
    - Match validation for detected objects/people
    """
    
    def __init__(self, device='cpu', confidence_threshold=0.5):
        """
        Initialize VideoFinder analyzer.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
            confidence_threshold: Minimum confidence threshold for detections
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.initialized = False
        
        # Object detection parameters
        self.detection_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee'
        ]
        
        # People detection model (using OpenCV's HOG descriptor)
        self.hog = None
        
        # Object detector (using OpenCV's DNN module)
        self.net = None
        
        # Initialize default metrics
        self.default_metrics = {}
        
        # Generate metrics for multiple detection instances (up to 10)
        for i in range(1, 11):
            self.default_metrics[f'ViF_consistency_{i}'] = "0/10"
            self.default_metrics[f'ViF_match_{i}'] = "No"
        
        # Add summary metrics
        self.default_metrics['ViF_total_detections'] = 0
        self.default_metrics['ViF_avg_consistency'] = 0.0
        self.default_metrics['ViF_match_rate'] = 0.0
        self.default_metrics['ViF_people_detected'] = 0
        self.default_metrics['ViF_objects_detected'] = 0
        
    def _initialize_model(self):
        """Initialize the VideoFinder model components."""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing VideoFinder analyzer...")
            
            # Initialize HOG descriptor for people detection
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Note: In a full implementation, you would load the actual VideoFinder model here
            # For this implementation, we'll use OpenCV's built-in detectors as a substitute
            # You could also load a YOLO model or other object detection network
            
            self.initialized = True
            logger.info("VideoFinder analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VideoFinder analyzer: {e}")
            raise
    
    def _detect_people(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect people in the frame using HOG descriptor.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of people bounding boxes with confidence (x, y, w, h, confidence)
        """
        # Detect people
        boxes, weights = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(32, 32),
            scale=1.05
        )
        
        # Filter by confidence and add confidence scores
        people_detections = []
        for i, (x, y, w, h) in enumerate(boxes):
            confidence = weights[i] if i < len(weights) else 0.5
            if confidence > self.confidence_threshold:
                people_detections.append((x, y, w, h, confidence))
        
        return people_detections
    
    def _detect_objects(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """
        Detect objects in the frame using simple contour detection.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of object bounding boxes with confidence and class (x, y, w, h, confidence, class)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours and create object detections
        object_detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on contour properties
                aspect_ratio = w / h if h > 0 else 0
                confidence = min(1.0, area / 10000) * (0.5 + 0.5 / (1 + abs(aspect_ratio - 1)))
                
                if confidence > self.confidence_threshold:
                    # Assign a generic object class
                    object_class = "object"
                    object_detections.append((x, y, w, h, confidence, object_class))
        
        return object_detections
    
    def _calculate_consistency(self, detections_history: List[List], frame_window: int = 10) -> float:
        """
        Calculate consistency of detections across frames.
        
        Args:
            detections_history: History of detections across frames
            frame_window: Number of frames to consider for consistency
            
        Returns:
            Consistency score (0-1)
        """
        if len(detections_history) < 2:
            return 0.0
        
        # Calculate consistency based on detection overlap across frames
        recent_frames = detections_history[-frame_window:]
        
        if not recent_frames:
            return 0.0
        
        # Simple consistency: how many frames have similar number of detections
        detection_counts = [len(frame_detections) for frame_detections in recent_frames]
        
        if not detection_counts:
            return 0.0
        
        avg_count = np.mean(detection_counts)
        consistency = 1.0 - (np.std(detection_counts) / (avg_count + 1e-6))
        
        return max(0.0, min(1.0, consistency))
    
    def _evaluate_match(self, detection: Tuple, reference_detections: List[Tuple]) -> bool:
        """
        Evaluate if a detection matches with reference detections.
        
        Args:
            detection: Current detection (x, y, w, h, confidence, ...)
            reference_detections: List of reference detections
            
        Returns:
            Boolean indicating if there's a match
        """
        if not reference_detections:
            return False
        
        x1, y1, w1, h1 = detection[:4]
        
        for ref_detection in reference_detections:
            x2, y2, w2, h2 = ref_detection[:4]
            
            # Calculate IoU (Intersection over Union)
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersection = x_overlap * y_overlap
            
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            iou = intersection / union if union > 0 else 0
            
            # Consider it a match if IoU > 0.3
            if iou > 0.3:
                return True
        
        return False
    
    def _process_frame(self, frame: np.ndarray, frame_idx: int, 
                      detections_history: List[List]) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Process a single frame for object and people detection.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            frame_idx: Current frame index
            detections_history: History of detections from previous frames
            
        Returns:
            Tuple of (detection_metrics, annotated_frame)
        """
        if not self.initialized:
            self._initialize_model()
        
        # Initialize metrics with defaults
        metrics = self.default_metrics.copy()
        annotated_frame = frame.copy()
        
        # Detect people
        people_detections = self._detect_people(frame)
        
        # Detect objects
        object_detections = self._detect_objects(frame)
        
        # Combine all detections
        all_detections = []
        
        # Add people detections
        for detection in people_detections:
            all_detections.append((*detection, "person"))
        
        # Add object detections
        for detection in object_detections:
            all_detections.append(detection)
        
        # Update detections history
        current_detections = [det[:4] for det in all_detections]  # Just bounding boxes
        detections_history.append(current_detections)
        
        # Calculate metrics for each detection
        for i, detection in enumerate(all_detections[:10]):  # Limit to 10 detections
            detection_idx = i + 1
            
            # Calculate consistency
            consistency_score = self._calculate_consistency(detections_history, frame_window=10)
            consistency_out_of_10 = int(consistency_score * 10)
            metrics[f'ViF_consistency_{detection_idx}'] = f"{consistency_out_of_10}/10"
            
            # Evaluate match with previous frame
            match = False
            if len(detections_history) > 1:
                previous_detections = detections_history[-2]
                match = self._evaluate_match(detection, previous_detections)
            
            metrics[f'ViF_match_{detection_idx}'] = "Yes" if match else "No"
            
            # Draw detection on frame
            x, y, w, h = detection[:4]
            confidence = detection[4] if len(detection) > 4 else 0.5
            class_name = detection[5] if len(detection) > 5 else "unknown"
            
            # Choose color based on class
            color = (0, 255, 0) if class_name == "person" else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Update summary metrics
        metrics['ViF_total_detections'] = len(all_detections)
        metrics['ViF_people_detected'] = len(people_detections)
        metrics['ViF_objects_detected'] = len(object_detections)
        
        # Calculate average consistency
        consistency_values = []
        for i in range(1, min(11, len(all_detections) + 1)):
            consistency_str = metrics[f'ViF_consistency_{i}']
            if '/' in consistency_str:
                score = int(consistency_str.split('/')[0])
                consistency_values.append(score / 10.0)
        
        metrics['ViF_avg_consistency'] = np.mean(consistency_values) if consistency_values else 0.0
        
        # Calculate match rate
        match_count = sum(1 for i in range(1, min(11, len(all_detections) + 1)) 
                         if metrics[f'ViF_match_{i}'] == "Yes")
        total_detections = min(len(all_detections), 10)
        metrics['ViF_match_rate'] = match_count / total_detections if total_detections > 0 else 0.0
        
        return metrics, annotated_frame
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze object and people localization in a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing VideoFinder analysis results
        """
        if not self.initialized:
            self._initialize_model()
        
        logger.info(f"Analyzing object and people localization in video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frame_metrics = []
        detections_history = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        best_annotated_frame = None
        max_detections = 0
        
        try:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                metrics, annotated_frame = self._process_frame(frame, frame_idx, detections_history)
                
                # Keep the frame with most detections for visualization
                total_detections = metrics.get('ViF_total_detections', 0)
                if total_detections > max_detections:
                    max_detections = total_detections
                    best_annotated_frame = annotated_frame
                
                # Add frame index and timestamp
                metrics['frame_idx'] = frame_idx
                metrics['timestamp'] = frame_idx / fps if fps > 0 else frame_idx
                
                frame_metrics.append(metrics)
                frame_idx += 1
                
                # Progress logging
                if frame_idx % 50 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        finally:
            cap.release()
        
        logger.info(f"Completed VideoFinder analysis: {len(frame_metrics)} frames processed")
        
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
            Aggregated VideoFinder analysis results
        """
        if not frame_metrics:
            result = self.default_metrics.copy()
            result.update({
                'video_path': str(video_path),
                'total_frames': 0,
                'avg_detections_per_frame': 0.0
            })
            return result
        
        # Calculate aggregated metrics
        aggregated = {}
        
        # Aggregate consistency metrics (take most common value)
        for i in range(1, 11):
            consistency_key = f'ViF_consistency_{i}'
            match_key = f'ViF_match_{i}'
            
            # For consistency, take the most frequent value
            consistency_values = [frame.get(consistency_key, "0/10") for frame in frame_metrics]
            most_common_consistency = max(set(consistency_values), key=consistency_values.count)
            aggregated[consistency_key] = most_common_consistency
            
            # For match, calculate percentage and convert to Yes/No
            match_values = [frame.get(match_key, "No") for frame in frame_metrics]
            yes_count = sum(1 for val in match_values if val == "Yes")
            match_rate = yes_count / len(match_values) if match_values else 0.0
            aggregated[match_key] = "Yes" if match_rate > 0.5 else "No"
        
        # Aggregate numeric metrics (mean across all frames)
        numeric_keys = ['ViF_total_detections', 'ViF_avg_consistency', 'ViF_match_rate', 
                       'ViF_people_detected', 'ViF_objects_detected']
        
        for key in numeric_keys:
            values = [frame.get(key, 0.0) for frame in frame_metrics]
            aggregated[key] = float(np.mean(values))
        
        # Add video metadata
        aggregated.update({
            'video_path': str(video_path),
            'total_frames': len(frame_metrics),
            'avg_detections_per_frame': aggregated.get('ViF_total_detections', 0.0),
            'max_detections_in_frame': max(frame.get('ViF_total_detections', 0) 
                                         for frame in frame_metrics),
        })
        
        return aggregated
    
    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        """
        Get VideoFinder features for the pipeline.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with VideoFinder features
        """
        try:
            results = self.analyze_video(video_path)
            
            # Create feature group for the pipeline
            feature_dict = {
                "Locate the objects and people": {
                    "description": "VideoFinder object and people localization with consistency and match evaluation",
                    "features": results
                }
            }
            
            return feature_dict
            
        except Exception as e:
            logger.error(f"Error in VideoFinder analysis: {e}")
            
            # Return default values on error
            default_result = self.default_metrics.copy()
            default_result.update({
                'video_path': str(video_path),
                'total_frames': 0,
                'avg_detections_per_frame': 0.0,
                'max_detections_in_frame': 0,
                'error': str(e)
            })
            
            feature_dict = {
                "Locate the objects and people": {
                    "description": "VideoFinder object and people localization with consistency and match evaluation",
                    "features": default_result
                }
            }
            
            return feature_dict

def extract_videofinder_features(video_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Extract VideoFinder features from a video file.
    
    Args:
        video_path: Path to the video file
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing VideoFinder features
    """
    analyzer = VideoFinderAnalyzer(device=device)
    return analyzer.get_feature_dict(video_path)
