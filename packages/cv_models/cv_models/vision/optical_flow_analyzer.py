"""
Optical Flow: Movement and Estimation of Motion
Based on: https://github.com/chuanenlin/optical-flow

This analyzer implements optical flow analysis for movement and motion estimation,
providing both sparse and dense flow analysis with visualization capabilities.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
from pathlib import Path
import base64
import io
from PIL import Image
import warnings

logger = logging.getLogger(__name__)

class OpticalFlowAnalyzer:
    """
    Optical Flow analyzer for movement and motion estimation.
    
    This analyzer implements both sparse and dense optical flow analysis
    to track motion patterns and estimate movement in video sequences.
    """
    
    def __init__(self, device='cpu', max_corners=100, quality_level=0.3, min_distance=7):
        """
        Initialize Optical Flow analyzer.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
            max_corners: Maximum number of corners to track for sparse flow
            quality_level: Quality level parameter for corner detection
            min_distance: Minimum distance between corners
        """
        self.device = device
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.initialized = False
        
        # Parameters for optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Parameters for corner detection
        self.feature_params = dict(
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7
        )
        
        # Colors for visualization (deterministic palette)
        indices = np.arange(100, dtype=np.uint8)
        self.colors = np.stack(
            ((indices * 37) % 255, (indices * 67) % 255, (indices * 97) % 255),
            axis=1,
        ).astype(np.uint8)
        
        # Default metrics
        self.default_metrics = {
            'sparse_flow_vis_.png': "",
            'sparse_points.npy': "",
            'dense_flow.npy': "",
            'dense_flow_vis_.png': "",
            'total_frames': 0,
            'motion_detected_frames': 0,
            'motion_detection_rate': 0.0,
            'avg_sparse_points': 0.0,
            'avg_motion_magnitude': 0.0,
            'max_motion_magnitude': 0.0,
            'total_displacement': 0.0,
            'dominant_motion_direction': 0.0,
            'motion_consistency': 0.0
        }
        
    def _initialize_model(self):
        """Initialize the optical flow analyzer components."""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing Optical Flow analyzer...")
            
            # Optical flow uses OpenCV built-in methods, no additional model loading needed
            logger.info("Optical Flow analyzer initialized successfully")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Optical Flow analyzer: {e}")
            raise

    def _detect_corners(self, gray_frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect Shi-Tomasi corners for sparse optical flow tracking."""
        return cv2.goodFeaturesToTrack(gray_frame, mask=None, **self.feature_params)
    
    def _calculate_sparse_flow(self, old_gray: np.ndarray, gray: np.ndarray, 
                              p0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate sparse optical flow using Lucas-Kanade method.
        
        Args:
            old_gray: Previous grayscale frame
            gray: Current grayscale frame
            p0: Previous corner points
            
        Returns:
            Tuple of (new_points, status, error)
        """
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **self.lk_params)
        return p1, st, err
    
    def _calculate_dense_flow(self, old_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """
        Calculate dense optical flow using Farneback method.
        
        Args:
            old_gray: Previous grayscale frame
            gray: Current grayscale frame
            
        Returns:
            Dense flow field
        """
        # Use Farneback method for dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            old_gray, gray, None, 
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        return flow
    
    def _visualize_sparse_flow(self, frame: np.ndarray, p0: np.ndarray, 
                              p1: np.ndarray, status: np.ndarray) -> np.ndarray:
        """
        Visualize sparse optical flow with arrows showing point trajectories.
        
        Args:
            frame: Current frame
            p0: Previous points
            p1: Current points
            status: Status of point tracking
            
        Returns:
            Annotated frame with sparse flow visualization
        """
        vis_frame = frame.copy()
        
        # Select good points
        good_new = p1[status == 1]
        good_old = p0[status == 1]
        
        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            
            # Draw arrow from old to new position
            cv2.arrowedLine(vis_frame, (c, d), (a, b), 
                           self.colors[i % len(self.colors)].tolist(), 2, tipLength=0.3)
            
            # Draw point
            cv2.circle(vis_frame, (a, b), 3, 
                      self.colors[i % len(self.colors)].tolist(), -1)
        
        return vis_frame
    
    def _visualize_dense_flow(self, flow: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Visualize dense optical flow as color-coded flow image.
        
        Args:
            flow: Dense flow field
            frame_shape: Shape of the original frame
            
        Returns:
            Color-coded flow visualization
        """
        h, w = frame_shape[:2]
        
        # Convert flow to polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create HSV image
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255  # Saturation
        
        # Map angle to hue
        hsv[..., 0] = ang * 180 / np.pi / 2
        
        # Map magnitude to value
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert HSV to BGR
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return flow_vis
    
    def _encode_array_to_base64(self, array: np.ndarray) -> str:
        """
        Encode numpy array to base64 string.
        
        Args:
            array: Numpy array to encode
            
        Returns:
            Base64 encoded string
        """
        try:
            # Save array to bytes buffer
            buffer = io.BytesIO()
            np.save(buffer, array)
            buffer.seek(0)
            
            # Encode to base64
            array_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return array_base64
            
        except Exception as e:
            logger.warning(f"Failed to encode array to base64: {e}")
            return ""
    
    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Base64 encoded string
        """
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG', optimize=True)
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.warning(f"Failed to encode image to base64: {e}")
            return ""
    
    def _calculate_motion_metrics(self, flow: np.ndarray, sparse_points: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate motion analysis metrics from optical flow.
        
        Args:
            flow: Dense optical flow field
            sparse_points: Sparse flow points (optional)
            
        Returns:
            Dictionary of motion metrics
        """
        metrics = {}
        
        # Calculate motion magnitude
        if flow is not None and flow.size > 0:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Motion magnitude statistics
            metrics['avg_motion_magnitude'] = float(np.mean(mag))
            metrics['max_motion_magnitude'] = float(np.max(mag))
            
            # Total displacement (sum of all motion vectors)
            metrics['total_displacement'] = float(np.sum(mag))
            
            # Dominant motion direction (average angle weighted by magnitude)
            if np.sum(mag) > 0:
                weighted_angles = ang * mag
                metrics['dominant_motion_direction'] = float(np.sum(weighted_angles) / np.sum(mag))
            else:
                metrics['dominant_motion_direction'] = 0.0
            
            # Motion consistency (standard deviation of motion directions)
            if np.std(ang) > 0:
                metrics['motion_consistency'] = float(1.0 / (1.0 + np.std(ang)))
            else:
                metrics['motion_consistency'] = 1.0
        else:
            metrics.update({
                'avg_motion_magnitude': 0.0,
                'max_motion_magnitude': 0.0,
                'total_displacement': 0.0,
                'dominant_motion_direction': 0.0,
                'motion_consistency': 0.0
            })
        
        # Sparse points statistics
        if sparse_points is not None:
            metrics['avg_sparse_points'] = float(len(sparse_points))
        else:
            metrics['avg_sparse_points'] = 0.0
        
        return metrics
    
    def _process_frame_pair(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_corners: Optional[np.ndarray],
    ) -> Tuple[
        Dict[str, float],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Process a pair of consecutive frames for optical flow analysis.
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            prev_corners: Previous corner points for sparse flow
            
        Returns:
            Tuple containing per-frame metrics and artefacts for best-frame selection
        """
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        metrics: Dict[str, float] = {}
        new_corners: Optional[np.ndarray] = None

        # Sparse optical flow artefacts (kept only for best frame selection)
        sparse_vis: Optional[np.ndarray] = None
        sparse_points_data: Optional[np.ndarray] = None
        
        if prev_corners is not None and len(prev_corners) > 0:
            p0 = prev_corners.astype(np.float32)
            try:
                # Calculate sparse flow
                new_points, status, error = self._calculate_sparse_flow(prev_gray, curr_gray, p0)
            except cv2.error as err:
                logger.warning(f"Sparse optical flow failed: {err}")
                new_points, status = None, None

            if new_points is not None and status is not None:
                # Filter good points
                good_points = new_points[status == 1]
                if len(good_points) > 0:
                    new_corners = good_points.reshape(-1, 1, 2).astype(np.float32)

                    # Visualize sparse flow
                    sparse_vis = self._visualize_sparse_flow(curr_frame, p0, new_points, status)

                    # Store sparse points data
                    sparse_points_data = good_points.astype(np.float32)
        
        # Dense optical flow
        dense_flow = self._calculate_dense_flow(prev_gray, curr_gray)
        dense_vis = self._visualize_dense_flow(dense_flow, curr_frame.shape)
        
        # Calculate motion metrics
        motion_metrics = self._calculate_motion_metrics(dense_flow, sparse_points_data)
        
        metrics.update(motion_metrics)

        return metrics, new_corners, sparse_vis, sparse_points_data, dense_flow, dense_vis
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze optical flow in a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing optical flow analysis results
        """
        if not self.initialized:
            self._initialize_model()
        
        logger.info(f"Analyzing optical flow in video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frame_data = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        prev_frame = None
        prev_corners = None
        motion_detected_frames = 0
        
        # Store best visualizations (frames with most motion)
        best_sparse_vis = ""
        best_dense_vis = ""
        best_sparse_points = ""
        best_dense_flow = ""
        max_motion = 0.0
        
        try:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    # Process frame pair
                    (
                        frame_metrics,
                        new_corners,
                        sparse_vis,
                        sparse_points_arr,
                        dense_flow_arr,
                        dense_vis,
                    ) = self._process_frame_pair(prev_frame, frame, prev_corners)
                    
                    # Check if motion detected
                    motion_magnitude = frame_metrics.get('avg_motion_magnitude', 0.0)
                    if motion_magnitude > 0.1:  # Threshold for motion detection
                        motion_detected_frames += 1
                    
                    # Keep best visualizations
                    if motion_magnitude > max_motion:
                        max_motion = motion_magnitude
                        best_sparse_vis = (
                            self._encode_image_to_base64(sparse_vis)
                            if sparse_vis is not None
                            else ''
                        )
                        best_dense_vis = (
                            self._encode_image_to_base64(dense_vis)
                            if dense_vis is not None
                            else ''
                        )
                        best_sparse_points = (
                            self._encode_array_to_base64(sparse_points_arr)
                            if sparse_points_arr is not None and sparse_points_arr.size > 0
                            else ''
                        )
                        best_dense_flow = (
                            self._encode_array_to_base64(dense_flow_arr)
                            if dense_flow_arr is not None
                            else ''
                        )
                    
                    # Add frame metadata
                    frame_metrics.update({
                        'frame_idx': frame_idx,
                        'timestamp': frame_idx / fps if fps > 0 else frame_idx
                    })
                    
                    frame_data.append(frame_metrics)
                    
                    # Update corners for next iteration
                    prev_corners = new_corners
                else:
                    # First frame - detect initial corners
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    prev_corners = self._detect_corners(gray)
                    if prev_corners is not None:
                        prev_corners = prev_corners.astype(np.float32)
                
                prev_frame = frame
                frame_idx += 1
                
                # Progress logging
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        finally:
            cap.release()
        
        logger.info(f"Completed optical flow analysis: {len(frame_data)} frame pairs processed")
        
        # Aggregate results
        return self._aggregate_results(frame_data, best_sparse_vis, best_dense_vis, 
                                     best_sparse_points, best_dense_flow, 
                                     motion_detected_frames, video_path)
    
    def _aggregate_results(self, frame_data: List[Dict[str, Any]], 
                          best_sparse_vis: str, best_dense_vis: str,
                          best_sparse_points: str, best_dense_flow: str,
                          motion_detected_frames: int, video_path: str) -> Dict[str, Any]:
        """
        Aggregate frame-level results into final metrics.
        
        Args:
            frame_data: List of per-frame flow data
            best_sparse_vis: Best sparse flow visualization
            best_dense_vis: Best dense flow visualization
            best_sparse_points: Best sparse points data
            best_dense_flow: Best dense flow data
            motion_detected_frames: Number of frames with detected motion
            video_path: Path to the video file
            
        Returns:
            Aggregated optical flow analysis results
        """
        if not frame_data:
            result = self.default_metrics.copy()
            result.update({
                'video_path': str(video_path),
                'total_frames': 0,
                'motion_detected_frames': 0,
                'motion_detection_rate': 0.0
            })
            return result
        
        # Calculate aggregated metrics
        numeric_metrics = [
            'avg_motion_magnitude', 'max_motion_magnitude', 'total_displacement',
            'dominant_motion_direction', 'motion_consistency', 'avg_sparse_points'
        ]
        
        aggregated = {}
        for metric in numeric_metrics:
            values = [frame.get(metric, 0.0) for frame in frame_data]
            aggregated[metric] = float(np.mean(values))
        
        # Use best visualizations and data
        aggregated.update({
            'sparse_flow_vis_.png': best_sparse_vis,
            'dense_flow_vis_.png': best_dense_vis,
            'sparse_points.npy': best_sparse_points,
            'dense_flow.npy': best_dense_flow
        })
        
        # Add summary statistics
        total_frames = len(frame_data) + 1  # +1 for the first frame
        aggregated.update({
            'video_path': str(video_path),
            'total_frames': total_frames,
            'motion_detected_frames': motion_detected_frames,
            'motion_detection_rate': motion_detected_frames / len(frame_data) if frame_data else 0.0,
            'overall_max_motion': max(frame.get('max_motion_magnitude', 0.0) for frame in frame_data),
            'cumulative_displacement': sum(frame.get('total_displacement', 0.0) for frame in frame_data)
        })
        
        return aggregated
    
    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        """
        Get optical flow features for the pipeline.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with optical flow features
        """
        try:
            results = self.analyze_video(video_path)
            return results
            
        except Exception as e:
            logger.error(f"Error in optical flow analysis: {e}")
            
            # Return default values on error
            default_result = self.default_metrics.copy()
            default_result.update({
                'video_path': str(video_path),
                'total_frames': 0,
                'motion_detected_frames': 0,
                'motion_detection_rate': 0.0,
                'error': str(e)
            })
            
            return default_result
    
    def visualize_flow(self, video_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Create visualization video with optical flow overlay.
        
        Args:
            video_path: Path to input video
            output_path: Path for output visualization video
            
        Returns:
            Path to output video if successful, None otherwise
        """
        try:
            if output_path is None:
                output_path = str(Path(video_path).with_suffix('_optical_flow.mp4'))
            
            # Process video and create visualization
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
            
            prev_frame = None
            prev_corners = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    (
                        _,
                        new_corners,
                        _,
                        _,
                        _,
                        dense_vis,
                    ) = self._process_frame_pair(prev_frame, frame, prev_corners)
                    
                    # Create side-by-side visualization
                    left_frame = frame.copy()

                    if dense_vis is None:
                        right_frame = np.zeros_like(frame)
                        cv2.putText(
                            right_frame,
                            "Dense Flow",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                        )
                    else:
                        right_frame = dense_vis
                    
                    # Combine frames
                    combined = np.hstack([left_frame, right_frame])
                    out.write(combined)
                    
                    prev_corners = new_corners
                else:
                    # First frame - detect corners
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    prev_corners = self._detect_corners(gray)
                
                prev_frame = frame
            
            cap.release()
            out.release()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create optical flow visualization: {e}")
            return None

def extract_optical_flow_features(video_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Extract optical flow features from a video file.
    
    Args:
        video_path: Path to the video file
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing optical flow features
    """
    analyzer = OpticalFlowAnalyzer(device=device)
    return analyzer.get_feature_dict(video_path)
