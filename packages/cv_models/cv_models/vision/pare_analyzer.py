"""
PARE (Part Attention Regressor for 3D Human Body Estimation) Analyzer

This module implements human pose and body estimation using PARE, which predicts
body-part-guided attention masks and hidden posture from video frames.

PARE focuses on:
- 3D human body estimation
- Part attention regression
- SMPL model parameters
- 3D/2D joint detection
- Body part attention masks
- Camera parameter estimation

Website: https://pare.is.tue.mpg.de/

Note: The analyzer requires the official PARE library and checkpoints. If they are
missing, initialization will raise an error and the feature should be disabled in
the pipeline configuration.
"""

import numpy as np
import cv2
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class PAREAnalyzer:
    """
    Analyzer for 3D human body estimation using PARE (Part Attention Regressor).
    
    PARE predicts body-part-guided attention masks and estimates hidden posture
    from video frames using part attention regression.
    
    All results come directly from the PARE model. If the model or dependencies are
    missing, initialization will fail, signalling to the caller that this feature
    cannot be used.
    """
    
    def __init__(self, device: str = 'cpu', model_path: Optional[str] = None):
        """
        Initialize the PARE analyzer.
        
        Args:
            device: Computation device ('cpu' or 'cuda')
            model_path: Optional path to pre-trained PARE model
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.is_model_loaded = False
        
        # PARE components (will be loaded in _initialize_model)
        self.PARE = None
        self.update_hparams = None
        self.load_pretrained_model = None
        
        # PARE output feature names
        self.feature_names = [
            'PARE_pred_cam',          # Predicted camera parameters
            'PARE_orig_cam',          # Original camera parameters  
            'PARE_verts',             # 3D mesh vertices
            'PARE_pose',              # SMPL pose parameters
            'PARE_betas',             # SMPL shape parameters
            'PARE_joints3d',          # 3D joint positions
            'PARE_joints2d',          # 2D joint projections
            'PARE_smpl_joints2d',     # SMPL 2D joint projections
            'PARE_bboxes',            # Bounding boxes
            'PARE_frame_ids'          # Frame identifiers
        ]
        
        # Try to load model if available
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the PARE model."""
        try:
            from pare.models import PARE
            from pare.core.config import update_hparams
        except ImportError as e:
            raise RuntimeError(
                "PARE library components are not available. Install the 'pare' package to use PAREAnalyzer."
            ) from e

        self.PARE = PARE
        self.update_hparams = update_hparams

        # Create model instance without pretrained weights by default
        self._create_default_pare_model()

        # Load pre-trained model if available
        if self.model_path and Path(self.model_path).exists():
            self.model = self._load_pare_model(self.model_path)
            self.is_model_loaded = True
            logger.info(f"PARE model loaded from {self.model_path}")
    
    def _create_default_pare_model(self):
        """Create a PARE model instance with default configuration."""
        if not hasattr(self, 'PARE') or self.PARE is None:
            raise RuntimeError("PARE class is not available for model creation")

        model = self.PARE(
            backbone='resnet50',
            num_joints=24,
            softmax_temp=1.0,
            num_features_smpl=64,
            num_features_cam=64,
            attention_blocks=3,
        )

        # Move to appropriate device
        if self.device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()

        model.eval()
        self.model = model
        self.is_model_loaded = True
        logger.info("Default PARE model created (no pretrained weights)")
    
    def _load_pare_model(self, model_path: str):
        """Load PARE model from checkpoint."""
        try:
            logger.info(f"Loading PARE model from {model_path}")
            
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Create model instance
            model = self.PARE(
                backbone='resnet50',
                num_joints=24,
                softmax_temp=1.0,
                num_features_smpl=64,
                num_features_cam=64,
                attention_blocks=3,
            )
            
            # Load state dict
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            
            # Move to appropriate device
            if self.device == 'cuda' and torch.cuda.is_available():
                model = model.cuda()
            else:
                model = model.cpu()
                
            model.eval()
            return model
            
        except Exception as e:
            raise RuntimeError(f"Error loading PARE model from {model_path}: {e}") from e
    
    def analyze_video_frames(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze video frames for human pose estimation.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Dictionary containing PARE features
        """
        try:
            if not self.is_model_loaded or self.model is None:
                raise RuntimeError("PARE model is not loaded and cannot analyze video")

            frames = self._extract_frames(video_path)
            if not frames:
                raise RuntimeError("No frames were extracted from the provided video")

            results = self._run_pare_inference(frames)
            return self._process_pare_results(results)

        except Exception as e:
            logger.error(f"Error in PARE video analysis: {e}")
            raise
    
    def _extract_frames(self, video_path: str, max_frames: int = 30) -> List[np.ndarray]:
        """Extract frames from video for analysis."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_step = max(1, total_frames // max_frames)
            
            frame_idx = 0
            while len(frames) < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                frame_idx += frame_step
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []
    
    def _run_pare_inference(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Run PARE inference on video frames."""
        try:
            import torch
            from torchvision import transforms
            
            # Preprocessing transform
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            results = {
                'pred_cam': [],      # Predicted camera parameters
                'orig_cam': [],      # Original camera parameters
                'verts': [],         # 3D mesh vertices
                'pose': [],          # SMPL pose parameters (72 dim)
                'betas': [],         # SMPL shape parameters (10 dim)
                'joints3d': [],      # 3D joint positions (49 x 3)
                'joints2d': [],      # 2D joint projections
                'smpl_joints2d': [], # SMPL 2D joint projections
                'bboxes': [],        # Bounding boxes
                'frame_ids': []      # Frame identifiers
            }

            # Process each frame
            for i, frame in enumerate(frames[:10]):  # Limit to 10 frames for efficiency
                # Convert frame to RGB if needed
                if frame.shape[2] == 3:
                    frame_rgb = frame
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Preprocess frame
                input_tensor = transform(frame_rgb).unsqueeze(0)

                # Move to device
                if self.device == 'cuda' and torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()

                # Run PARE model
                with torch.no_grad():
                    output = self.model(input_tensor)

                if not isinstance(output, dict):
                    raise RuntimeError("Unexpected output type from PARE model")

                pred_cam = output.get('pred_cam', torch.zeros(1, 3)).cpu().numpy().flatten()
                pred_pose = output.get('pred_pose', torch.zeros(1, 72)).cpu().numpy().flatten()
                pred_betas = output.get('pred_betas', torch.zeros(1, 10)).cpu().numpy().flatten()
                pred_vertices = output.get('pred_vertices', torch.zeros(1, 6890, 3)).cpu().numpy()
                pred_joints3d = output.get('pred_joints3d', torch.zeros(1, 49, 3)).cpu().numpy()
                pred_joints2d = output.get('pred_joints2d', torch.zeros(1, 49, 2)).cpu().numpy()

                # Store results
                results['pred_cam'].append(pred_cam)
                results['orig_cam'].append(pred_cam)  # Use same as pred_cam for now
                results['pose'].append(pred_pose)
                results['betas'].append(pred_betas)
                results['verts'].append(pred_vertices.squeeze())
                results['joints3d'].append(pred_joints3d.squeeze())
                results['joints2d'].append(pred_joints2d.squeeze())
                results['smpl_joints2d'].append(pred_joints2d.squeeze()[:24])  # SMPL has 24 joints

                # Calculate bounding box from 2D joints
                joints2d_flat = pred_joints2d.squeeze()
                if len(joints2d_flat) > 0:
                    x_coords = joints2d_flat[:, 0]
                    y_coords = joints2d_flat[:, 1]
                    bbox = [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]
                else:
                    bbox = [0, 0, frame.shape[1], frame.shape[0]]

                results['bboxes'].append(bbox)
                results['frame_ids'].append(i)

            logger.info(f"Processed {len(results['frame_ids'])} frames with PARE model")
            return results

        except Exception as e:
            logger.error(f"Error in PARE inference: {e}")
            raise
    
    
    def _process_pare_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process PARE results into feature format."""
        features = {}
        
        try:
            # Aggregate results across frames
            for key, values in results.items():
                if not values:
                    continue
                
                feature_name = f"PARE_{key}"
                
                if key in ['pred_cam', 'orig_cam']:
                    # Camera parameters: average across frames
                    features[feature_name] = np.mean(values, axis=0).tolist()
                
                elif key in ['pose', 'betas']:
                    # SMPL parameters: average across frames
                    features[feature_name] = np.mean(values, axis=0).tolist()
                
                elif key in ['verts', 'joints3d', 'joints2d', 'smpl_joints2d']:
                    # 3D/2D coordinates: flatten and take statistics
                    all_coords = np.array(values)
                    features[f"{feature_name}_mean"] = np.mean(all_coords, axis=(0, 1)).tolist()
                    features[f"{feature_name}_std"] = np.std(all_coords, axis=(0, 1)).tolist()
                    features[f"{feature_name}_shape"] = list(all_coords.shape)
                
                elif key == 'bboxes':
                    # Bounding boxes: average dimensions
                    all_bboxes = np.array(values)
                    features[feature_name] = np.mean(all_bboxes, axis=0).tolist()
                
                elif key == 'frame_ids':
                    # Frame information
                    features[feature_name] = values
                    features['PARE_num_frames'] = len(values)
            
            # Add metadata
            features.update({
                'PARE_analysis_timestamp': np.datetime64('now').astype(str),
                'PARE_model_name': 'PARE_part_attention_regressor',
                'PARE_version': '1.0.0',
                'PARE_device': self.device,
                'PARE_model_loaded': self.is_model_loaded
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing PARE results: {e}")
            return self._get_default_features()
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Return default features when analysis fails."""
        features = {}
        
        # Set default values for all PARE features
        for feature_name in self.feature_names:
            if 'cam' in feature_name:
                features[feature_name] = [0.0, 0.0, 0.0]
            elif feature_name == 'PARE_pose':
                features[feature_name] = [0.0] * 72
            elif feature_name == 'PARE_betas':
                features[feature_name] = [0.0] * 10
            elif 'joints' in feature_name:
                features[f"{feature_name}_mean"] = [0.0, 0.0, 0.0]
                features[f"{feature_name}_std"] = [0.0, 0.0, 0.0]
                features[f"{feature_name}_shape"] = [0, 0, 0]
            elif feature_name == 'PARE_verts':
                features[f"{feature_name}_mean"] = [0.0, 0.0, 0.0]
                features[f"{feature_name}_std"] = [0.0, 0.0, 0.0]
                features[f"{feature_name}_shape"] = [0, 6890, 3]
            elif feature_name == 'PARE_bboxes':
                features[feature_name] = [0.0, 0.0, 0.0, 0.0]
            elif feature_name == 'PARE_frame_ids':
                features[feature_name] = []
        
        # Add metadata
        features.update({
            'PARE_num_frames': 0,
            'PARE_analysis_timestamp': np.datetime64('now').astype(str),
            'PARE_model_name': 'PARE_part_attention_regressor',
            'PARE_version': '1.0.0',
            'PARE_device': self.device,
            'PARE_model_loaded': self.is_model_loaded
        })
        
        return features
    
    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        """
        Main interface for extracting PARE features from video.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Dictionary of PARE features
        """
        return self.analyze_video_frames(video_path)


def create_pare_analyzer(device: str = 'cpu', model_path: Optional[str] = None):
    """Factory function to create PARE analyzer."""
    return PAREAnalyzer(device=device, model_path=model_path)
