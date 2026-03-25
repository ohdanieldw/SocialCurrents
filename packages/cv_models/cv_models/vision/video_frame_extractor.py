"""Key frame extraction utilities for videos."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import cv2

logger = logging.getLogger(__name__)


AnalysisCallable = Callable[[Sequence[Path]], Dict[str, Any]]


class VideoFrameExtractor:
    """Extract key frames, optionally producing a PDF and LLM analysis."""

    def __init__(
        self,
        *,
        output_dir: Optional[Path] = None,
        max_frames: int = 16,
        difference_threshold: float = 0.32,
        min_scene_gap: float = 0.75,
        create_pdf: bool = False,
        analysis_fn: Optional[AnalysisCallable] = None,
        analysis_filename: str = "frame_analysis.json",
    ) -> None:
        self.output_root = Path(output_dir) if output_dir else Path("output")
        self.output_root = self.output_root / "vision" / "frames"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.max_frames = max(max_frames, 1)
        self.difference_threshold = difference_threshold
        self.min_scene_gap = max(min_scene_gap, 0.0)
        self.create_pdf = create_pdf
        self.analysis_fn = analysis_fn
        self.analysis_filename = analysis_filename

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        video_file = Path(video_path)
        capture = cv2.VideoCapture(str(video_file))
        if not capture.isOpened():
            logger.warning("Could not open video for frame extraction: %s", video_file)
            return {
                "VFE_frame_paths": [],
                "VFE_frame_count": 0,
                "VFE_video_path": str(video_file),
                "VFE_pdf_path": None,
                "VFE_analysis_path": None,
            }

        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_paths = self._select_key_frames(capture, video_file, fps)
        capture.release()

        pdf_path = self._maybe_create_pdf(video_file, frame_paths)
        analysis_path = self._maybe_run_analysis(video_file, frame_paths)

        return {
            "VFE_frame_paths": [str(path) for path in frame_paths],
            "VFE_frame_count": len(frame_paths),
            "VFE_video_path": str(video_file),
            "VFE_pdf_path": str(pdf_path) if pdf_path else None,
            "VFE_analysis_path": str(analysis_path) if analysis_path else None,
        }

    def _select_key_frames(
        self,
        capture: cv2.VideoCapture,
        video_file: Path,
        fps: float,
    ) -> Sequence[Path]:
        last_hist = None
        last_timestamp = float("-inf")
        frame_index = 0
        selected: list[Path] = []
        video_dir = self.output_root / video_file.stem
        video_dir.mkdir(parents=True, exist_ok=True)

        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break

            timestamp = frame_index / fps if fps > 0 else frame_index
            hist = self._compute_histogram(frame)
            is_first_frame = not selected
            # Use histogram distance to detect scene changes and enforce spacing in seconds.
            scene_changed = (
                last_hist is None
                or cv2.compareHist(last_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                >= self.difference_threshold
            )
            gap_ok = (timestamp - last_timestamp) >= self.min_scene_gap

            if (scene_changed and gap_ok) or is_first_frame:
                frame_path = video_dir / f"frame_{timestamp:.2f}.jpg"
                cv2.imwrite(str(frame_path), frame)
                selected.append(frame_path)
                last_hist = hist
                last_timestamp = timestamp
                if len(selected) >= self.max_frames:
                    break

            frame_index += 1

        if not selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = capture.read()
            if success and frame is not None:
                fallback_path = video_dir / "frame_0.00.jpg"
                cv2.imwrite(str(fallback_path), frame)
                selected = [fallback_path]

        return selected

    @staticmethod
    def _compute_histogram(frame: Any) -> Any:
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _maybe_create_pdf(self, video_file: Path, frame_paths: Sequence[Path]) -> Optional[Path]:
        if not self.create_pdf or not frame_paths:
            return None

        try:
            from PIL import Image
        except ImportError:  # pragma: no cover - optional dependency
            logger.warning(
                "PDF export requested but Pillow is not installed. Skipping PDF creation.")
            return None

        pdf_path = (self.output_root / video_file.stem) / "extracted_frames.pdf"
        images = []
        for path in frame_paths:
            try:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
            except Exception:  # pragma: no cover - protect against corrupt frames
                logger.exception("Failed to include frame in PDF: %s", path)

        if not images:
            return None

        first, *rest = images
        try:
            first.save(pdf_path, save_all=True, append_images=rest)
        finally:
            for image in images:
                image.close()

        return pdf_path

    def _maybe_run_analysis(
        self, video_file: Path, frame_paths: Sequence[Path]
    ) -> Optional[Path]:
        if not self.analysis_fn or not frame_paths:
            return None

        try:
            analysis = self.analysis_fn(frame_paths)
        except Exception:  # pragma: no cover - delegated integration
            logger.exception("Frame analysis callable failed for %s", video_file)
            return None

        if not analysis:
            return None

        analysis_path = (self.output_root / video_file.stem) / self.analysis_filename
        try:
            analysis_path.write_text(json.dumps(analysis, indent=2))
        except Exception:  # pragma: no cover - filesystem issues
            logger.exception("Could not write analysis JSON: %s", analysis_path)
            return None

        return analysis_path


__all__ = ["VideoFrameExtractor"]
