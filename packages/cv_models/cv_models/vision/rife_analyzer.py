"""RIFE (Real-Time Intermediate Flow Estimation) inspired optical flow analyzer."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

logger = logging.getLogger(__name__)


class RIFEAnalyzer:
    """Estimate frame-to-frame motion magnitude and persist flow maps."""

    def __init__(self, *, output_dir: Optional[Path] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir = self.output_dir / "vision" / "rife"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _read_frames(self, video_path: str, max_pairs: int = 10) -> np.ndarray:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            logger.warning("Could not open video for RIFE analysis: %s", video_path)
            return np.zeros((0, 2), dtype=np.float32)

        frames = []
        prev_gray = None
        count = 0
        while count < max_pairs:
            success, frame = capture.read()
            if not success or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mag_mean = float(np.mean(mag))
                ang_mean = float(np.mean(ang))
                frames.append(np.array([mag_mean, ang_mean], dtype=np.float32))
                count += 1
            prev_gray = gray
        capture.release()
        return np.asarray(frames, dtype=np.float32)

    def get_feature_dict(self, video_path: str) -> Dict[str, any]:
        flow_summary = self._read_frames(video_path)
        flow_path = self.output_dir / f"{Path(video_path).stem}_rife_flow.npy"
        np.save(flow_path, flow_summary)

        metadata_path = self.output_dir / f"{Path(video_path).stem}_rife_metadata.json"
        metadata = {
            "video_path": video_path,
            "flow_pairs": int(flow_summary.shape[0]),
            "flow_dim": list(flow_summary.shape[1:]),
            "torch_available": bool(torch),
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        total_magnitude = float(np.sum(flow_summary[:, 0])) if flow_summary.size else 0.0
        average_speed = total_magnitude / float(max(flow_summary.shape[0], 1)) if flow_summary.size else 0.0

        return {
            "RIFE_flow_path": str(flow_path),
            "RIFE_metadata_path": str(metadata_path),
            "RIFE_total_motion": total_magnitude,
            "RIFE_average_speed": average_speed,
        }


__all__ = ["RIFEAnalyzer"]
