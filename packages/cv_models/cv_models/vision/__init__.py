"""
Vision processing module for the multimodal data pipeline.

SAFE + STRICT IMPORT POLICY

- CORE analyzers: must import successfully (pipeline fails otherwise)
- IMPORTANT analyzers: expected to work but non-fatal if missing
- OPTIONAL analyzers: heavy / experimental; failure is allowed

This prevents cascade import failures while preserving scientific rigor.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

FAILED_ANALYZERS: Dict[str, str] = {}
AVAILABLE_ANALYZERS: Dict[str, bool] = {}


def _import(module: str, symbol: str, required: bool = False) -> Optional[Any]:
    try:
        mod = importlib.import_module(module)
        obj = getattr(mod, symbol)
        globals()[symbol] = obj
        AVAILABLE_ANALYZERS[symbol] = True
        return obj
    except Exception as e:
        FAILED_ANALYZERS[symbol] = str(e)
        AVAILABLE_ANALYZERS[symbol] = False
        if required:
            raise ImportError(f"[CORE ANALYZER FAILED] {symbol}: {e}") from e
        logger.warning(f"[VISION] {symbol} disabled: {e}")
        return None


# ============================================================
# 🟢 CORE ANALYZERS (MUST WORK)
# ============================================================

MediaPipePoseAnalyzer = _import(
    "cv_models.vision.mediapipe_pose_analyzer", "MediaPipePoseAnalyzer", required=True
)
OpenPoseAnalyzer = _import(
    "cv_models.vision.openpose_analyzer", "OpenPoseAnalyzer", required=True
)
ViTPoseAnalyzer = _import(
    "cv_models.vision.vitpose_analyzer", "ViTPoseAnalyzer", required=True
)
create_vitpose_analyzer = _import(
    "cv_models.vision.vitpose_analyzer", "create_vitpose_analyzer", required=True
)
DeepHRNetAnalyzer = _import(
    "cv_models.vision.deep_hrnet_analyzer", "DeepHRNetAnalyzer", required=True
)
VideoFrameExtractor = _import(
    "cv_models.vision.video_frame_extractor", "VideoFrameExtractor", required=True
)
OpticalFlowAnalyzer = _import(
    "cv_models.vision.optical_flow_analyzer", "OpticalFlowAnalyzer", required=True
)

# ============================================================
# 🟡 IMPORTANT ANALYZERS (EXPECTED BUT NON-FATAL)
# ============================================================

PAREAnalyzer = _import("cv_models.vision.pare_analyzer", "PAREAnalyzer")
create_pare_analyzer = _import("cv_models.vision.pare_analyzer", "create_pare_analyzer")

PSAAnalyzer = _import("cv_models.vision.psa_analyzer", "PSAAnalyzer")
create_psa_analyzer = _import("cv_models.vision.psa_analyzer", "create_psa_analyzer")

SmoothNetAnalyzer = _import("cv_models.vision.smoothnet_analyzer", "SmoothNetAnalyzer")
SimpleBaselinesAnalyzer = _import("cv_models.vision.simple_baselines_analyzer", "SimpleBaselinesAnalyzer")
RSNAnalyzer = _import("cv_models.vision.rsn_analyzer", "RSNAnalyzer")
RIFEAnalyzer = _import("cv_models.vision.rife_analyzer", "RIFEAnalyzer")

ARBExAnalyzer = _import("cv_models.vision.arbex_analyzer", "ARBExAnalyzer")
create_arbex_analyzer = _import("cv_models.vision.arbex_analyzer", "create_arbex_analyzer")

# ============================================================
# 🔴 OPTIONAL / EXPERIMENTAL ANALYZERS
# ============================================================

AVHuBERTAnalyzer = _import("cv_models.vision.avhubert_analyzer", "AVHuBERTAnalyzer")
CrowdFlowAnalyzer = _import("cv_models.vision.crowdflow_analyzer", "CrowdFlowAnalyzer")

EmotiEffNetAnalyzer = _import("cv_models.vision.emotieffnet_analyzer", "EmotiEffNetAnalyzer")
PyFeatAnalyzer = _import("cv_models.vision.pyfeat_analyzer", "PyFeatAnalyzer")
MEGraphAUAnalyzer = _import("cv_models.vision.me_graphau_analyzer", "MEGraphAUAnalyzer")
DANAnalyzer = _import("cv_models.vision.dan_analyzer", "DANAnalyzer")
GANimationAnalyzer = _import("cv_models.vision.ganimation_analyzer", "GANimationAnalyzer")

ARBExAnalyzer = _import(
    "cv_models.vision.arbex_analyzer",
    "ARBExAnalyzer",
    required=False,  # IMPORTANT but not CORE
)
create_arbex_analyzer = _import(
    "cv_models.vision.arbex_analyzer",
    "create_arbex_analyzer",
    required=False,
)

InstaDMAnalyzer = _import("cv_models.vision.instadm_analyzer", "InstaDMAnalyzer")
VideoFinderAnalyzer = _import("cv_models.vision.videofinder_analyzer", "VideoFinderAnalyzer")
LaneGCNAnalyzer = _import("cv_models.vision.lanegcn_analyzer", "LaneGCNAnalyzer")
FACTAnalyzer = _import("cv_models.vision.fact_analyzer", "FACTAnalyzer")

# ============================================================
# Public API
# ============================================================

__all__ = list(AVAILABLE_ANALYZERS.keys())


def get_failed_analyzers() -> Dict[str, str]:
    return dict(FAILED_ANALYZERS)

def get_available_analyzers() -> Dict[str, bool]:
    return dict(AVAILABLE_ANALYZERS)


def vision_status() -> Dict[str, Dict[str, str]]:
    """
    Debug helper to inspect analyzer availability.
    Returns:
        {
            "available": {...},
            "failed": {...}
        }
    """
    return {
        "available": dict(AVAILABLE_ANALYZERS),
        "failed": dict(FAILED_ANALYZERS),
    }

__all__ += ["get_failed_analyzers", "get_available_analyzers", "vision_status"]

