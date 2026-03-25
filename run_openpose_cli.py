#!/usr/bin/env python3
"""Utility to run the OpenPose binary on a collection of videos without feature extraction."""

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List

from cv_models.vision.openpose_analyzer import OpenPoseAnalyzer

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".mpg",
    ".mpeg",
    ".wmv",
}


def find_videos(path: Path, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path
        return

    if recursive:
        iterator = path.rglob("*")
    else:
        iterator = path.iterdir()

    for candidate in iterator:
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
            yield candidate


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OpenPose binary on videos without extracting features.")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data",
        help="Directory containing input videos or a single video file (default: data)",
    )
    parser.add_argument(
        "--openpose-bin",
        dest="openpose_bin",
        default=os.environ.get("OPENPOSE_BIN"),
        help="Path to the OpenPose executable. Defaults to OPENPOSE_BIN environment variable.",
    )
    parser.add_argument(
        "--model-folder",
        dest="model_folder",
        default=os.environ.get("OPENPOSE_MODEL_FOLDER"),
        help="Path to OpenPose model folder. Defaults to OPENPOSE_MODEL_FOLDER environment variable if set.",
    )
    parser.add_argument(
        "--extra-flag",
        dest="extra_flags",
        action="append",
        default=[],
        help="Additional flag to pass to the OpenPose binary. May be provided multiple times.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("OPENPOSE_DEVICE", "cpu"),
        help="Device indicator for downstream tooling (default: cpu)",
    )
    parser.add_argument(
        "--disable-face",
        dest="enable_face",
        action="store_false",
        help="Disable face keypoint detection.",
    )
    parser.add_argument(
        "--enable-hand",
        dest="enable_hand",
        action="store_true",
        help="Enable hand keypoint detection.",
    )
    parser.add_argument(
        "--render-pose",
        dest="render_pose",
        type=int,
        default=2,
        help="Render mode passed to OpenPose --render_pose flag (default: 2).",
    )
    parser.add_argument(
        "--keep-json",
        dest="keep_json",
        action="store_true",
        help="Preserve the JSON keypoint outputs instead of deleting them after processing.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search the input directory recursively for videos.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Root directory for OpenPose outputs (default: alongside input data).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args()


def run_openpose_on_videos(videos: Iterable[Path], analyzer: OpenPoseAnalyzer) -> None:
    videos = list(videos)
    if not videos:
        logging.warning("No video files found to process.")
        return

    logging.info("Found %d video(s) for OpenPose processing", len(videos))

    for video_path in videos:
        logging.info("Processing video: %s", video_path)
        try:
            analyzer.analyze_video(str(video_path))
            if analyzer.output_root:
                output_dir = analyzer.output_root / "openpose_output"
            else:
                output_dir = video_path.parent / "openpose_output"
            logging.info(
                "OpenPose outputs for %s available under %s",
                video_path.name,
                output_dir,
            )
        except Exception as exc:
            logging.error("Failed to run OpenPose for %s: %s", video_path, exc, exc_info=logging.getLogger().isEnabledFor(logging.DEBUG))


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    data_path = Path(args.data_dir).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {data_path}")

    videos = find_videos(data_path, args.recursive)

    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)

    analyzer = OpenPoseAnalyzer(
        device=args.device,
        openpose_bin=args.openpose_bin,
        model_folder=args.model_folder,
        extra_flags=args.extra_flags,
        enable_face=args.enable_face,
        enable_hand=args.enable_hand,
        render_pose=args.render_pose,
        keep_json=args.keep_json,
        output_root=output_root,
    )

    run_openpose_on_videos(videos, analyzer)


if __name__ == "__main__":
    main()
