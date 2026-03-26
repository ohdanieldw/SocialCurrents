#!/usr/bin/env python3
"""
Main entry point for the multimodal data pipeline.
It can be run directly with Poetry without activating a shell:
    poetry run python run_simple.py [options]
"""
import os
import sys
import argparse
import logging
import traceback
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict as _DictType, List as _ListType


FEATURE_CATALOG = [
    {
        "name": "basic_audio",
        "category": "Audio Feature",
        "description": "Audio volume, pitch, and frame deltas (oc_audvol*, oc_audpit*)."
    },
    {
        "name": "librosa_spectral",
        "category": "Audio Feature",
        "description": "Librosa spectral, rhythm, and tempo descriptors (lbrs_*)."
    },
    {
        "name": "opensmile",
        "category": "Audio Feature",
        "description": "openSMILE low-level descriptors and functionals (osm_*)."
    },
    {
        "name": "audiostretchy",
        "category": "Audio Feature",
        "description": "AudioStretchy time-stretch statistics (AS_*)."
    },
    {
        "name": "speech_emotion",
        "category": "Emotional Recognition",
        "description": "Speech emotion recognition class probabilities (ser_*)."
    },
    {
        "name": "heinsen_sentiment",
        "category": "Sentiment Analysis",
        "description": "Heinsen routing sentiment capsules (arvs_*)."
    },
    {
        "name": "meld_emotion",
        "category": "Emotional Recognition",
        "description": "MELD dialogue-level emotion analytics (MELD_*)."
    },
    {
        "name": "speech_separation",
        "category": "Audio Separation",
        "description": "SepFormer speech separation waveforms and paths."
    },
    {
        "name": "whisperx_transcription",
        "category": "Audio Transcript",
        "description": "WhisperX diarized transcription outputs (WhX_*)."
    },
    {
        "name": "xlsr_speech_to_text",
        "category": "Audio Transcript",
        "description": "XLSR (wav2vec2) multilingual speech-to-text with hidden states."
    },
    {
        "name": "s2t_speech_to_text",
        "category": "Audio Transcript",
        "description": "Facebook Speech2Text transcription with alignment and confidence."
    },
    {
        "name": "deberta_text",
        "category": "Text Analysis",
        "description": "DeBERTa benchmark summaries (DEB_*)."
    },
    {
        "name": "simcse_text",
        "category": "Text Analysis",
        "description": "SimCSE sentence similarity metrics (CSE_*)."
    },
    {
        "name": "albert_text",
        "category": "Text Analysis",
        "description": "ALBERT benchmark results (alb_*)."
    },
    {
        "name": "sbert_text",
        "category": "Text Analysis",
        "description": "Sentence-BERT embeddings and reranker scores (BERT_*)."
    },
    {
        "name": "use_text",
        "category": "Text Analysis",
        "description": "Universal Sentence Encoder embeddings (USE_*)."
    },
    {
        "name": "elmo_text",
        "category": "Text Analysis",
        "description": "ELMo deep contextualized word embeddings."
    },
    {
        "name": "pare_vision",
        "category": "Body Pose",
        "description": "PARE 3D human body estimation metrics (PARE_*)."
    },
    {
        "name": "vitpose_vision",
        "category": "Body Pose",
        "description": "ViTPose human pose KPIs (vit_*)."
    },
    {
        "name": "psa_vision",
        "category": "Body Pose",
        "description": "Polarized Self-Attention keypoint benchmarks (psa_*)."
    },
    {
        "name": "emotieffnet_vision",
        "category": "Facial Expression",
        "description": "EmotiEffNet valence/arousal and AU scores (eln_*)."
    },
    {
        "name": "mediapipe_pose_vision",
        "category": "Body Pose",
        "description": "MediaPipe 33-keypoint landmarks and world coordinates (GMP_*)."
    },
    {
        "name": "openpose_vision",
        "category": "Body Pose",
        "description": "OpenPose keypoints and media artifacts (openPose_*)."
    },
    {
        "name": "pyfeat_vision",
        "category": "Facial Expression",
        "description": "Py-Feat action units, emotions, and geometry (pf_*)."
    },
    {
        "name": "me_graphau_vision",
        "category": "Facial Expression",
        "description": "ME-GraphAU facial action unit relations (ann_*)."
    },
    {
        "name": "dan_vision",
        "category": "Facial Expression",
        "description": "DAN facial emotion classification probabilities (dan_*)."
    },
    {
        "name": "ganimation_vision",
        "category": "Facial Expression",
        "description": "GANimation synthesized AU intensities (GAN_*)."
    },
    {
        "name": "arbex_vision",
        "category": "Facial Expression",
        "description": "ARBEx reliable facial emotion extraction (arbex_*)."
    },
    {
        "name": "instadm_vision",
        "category": "Video Understanding",
        "description": "Insta-DM dense depth and motion analytics (indm_*)."
    },
    {
        "name": "crowdflow_vision",
        "category": "Video Understanding",
        "description": "CrowdFlow optical flow crowd statistics (of_*)."
    },
    {
        "name": "deep_hrnet_vision",
        "category": "Body Pose",
        "description": "Deep HRNet pose evaluation metrics (DHiR_*)."
    },
    {
        "name": "simple_baselines_vision",
        "category": "Body Pose",
        "description": "Simple Baselines pose estimation metrics (SBH_*)."
    },
    {
        "name": "rsn_vision",
        "category": "Body Pose",
        "description": "Residual Steps Network keypoint statistics (rsn_*)."
    },
    {
        "name": "optical_flow_vision",
        "category": "Video Understanding",
        "description": "Optical flow motion descriptors and magnitudes."
    },
    {
        "name": "videofinder_vision",
        "category": "Video Understanding",
        "description": "VideoFinder object/people localization metrics (ViF_*)."
    },
    {
        "name": "lanegcn_vision",
        "category": "Video Understanding",
        "description": "LaneGCN motion forecasting metrics (GCN_*)."
    },
    {
        "name": "smoothnet_vision",
        "category": "Body Pose",
        "description": "SmoothNet temporally smoothed pose series (net_*)."
    },
    {
        "name": "avhubert_vision",
        "category": "Audio-Visual",
        "description": "AV-HuBERT audio-visual speech embeddings."
    },
    {
        "name": "fact_vision",
        "category": "Video Understanding",
        "description": "FACT frame-action cross-attention temporal features."
    },
    {
        "name": "video_frames_vision",
        "category": "Video Understanding",
        "description": "Key frame extraction with optional LLM-powered frame analysis."
    },
    {
        "name": "rife_vision",
        "category": "Video Understanding",
        "description": "RIFE optical flow magnitude and angle per frame pair."
    },
]

ALL_FEATURES: _ListType[str] = [entry["name"] for entry in FEATURE_CATALOG]
FEATURE_DESCRIPTIONS: _DictType[str, str] = {
    entry["name"]: entry["description"] for entry in FEATURE_CATALOG
}

# Extractors excluded by --skip-slow (CPU-prohibitive; only run when explicitly requested via -f)
SLOW_FEATURES: _ListType[str] = ["s2t_speech_to_text", "xlsr_speech_to_text", "elmo_text"]

# Ensure in-repo packages are importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGES_DIR = PROJECT_ROOT / "packages"
if PACKAGES_DIR.exists():
    sys.path.insert(0, str(PACKAGES_DIR))


def _print_feature_catalog() -> None:
    """Print the feature catalog grouped by category."""
    print("Features defined by the pipeline (grouped by requirements category):")
    current_category = None
    for entry in FEATURE_CATALOG:
        category = entry["category"] or "Other"
        if category != current_category:
            print(f"\n{category}:")
            current_category = category
        print(f"  - {entry['name']}: {entry['description']}")
    print("\nTip: pass --features name1,name2 to limit extraction, or omit to run everything.")


def _check_python_dependencies() -> bool:
    """Verify required Python modules are importable."""
    import importlib

    required_packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "librosa": "librosa",
        "opencv-python": "cv2",
        "torch": "torch",
        "torchaudio": "torchaudio",
        "ffmpeg-python": "ffmpeg",
        "soundfile": "soundfile",
        "pyarrow": "pyarrow",
        "transformers": "transformers",
    }

    missing = []
    print("Checking required dependencies:")
    for pkg_name, import_name in required_packages.items():
        try:
            importlib.import_module(import_name)
            print(f"  ✓ {pkg_name}")
        except ImportError:
            missing.append(pkg_name)
            print(f"  ✗ {pkg_name} - MISSING")

    if missing:
        print("\nSome dependencies are missing.")
        print("Tip: run 'poetry install' in the repo root, or use the run_all scripts which install automatically.")
        print("You can also install manually with: poetry add " + " ".join(missing))
        return False

    print("\nAll dependencies are installed correctly!")
    if shutil.which("ffmpeg") is None:
        print("Warning: ffmpeg binary not found on PATH. Install ffmpeg to enable media processing.")
    return True


def main() -> None:
    """Entry point for running the multimodal pipeline from the CLI."""
    parser = argparse.ArgumentParser(description="Multimodal Data Pipeline")
    parser.add_argument("-d", "--data-dir", default="data", help="Directory with video/audio files")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory (default: ./output/YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "-f",
        "--features",
        help="Comma-separated features to extract (default: all features defined in the catalog)",
    )
    parser.add_argument("--list-features", action="store_true", help="List available features and exit")
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help=f"Skip CPU-prohibitive extractors ({', '.join(SLOW_FEATURES)}); has no effect when -f is used",
    )
    parser.add_argument("--is-audio", action="store_true", help="Process files as audio instead of video")
    parser.add_argument("--log-file", help="Optional path for a run-level log file (per-subject logs are written automatically to each subject's output folder)")
    parser.add_argument(
        "--decimal-places",
        type=int,
        default=3,
        help="Round float values in CSV output to this many decimal places (default: 3)",
    )
    parser.add_argument(
        "--check-dependencies",
        action="store_true",
        help="Check if all required dependencies are installed",
    )
    parser.add_argument("--version", action="version", version="SocialCurrents 0.1.0")
    args = parser.parse_args()

    if args.list_features:
        _print_feature_catalog()
        sys.exit(0)

    if args.check_dependencies:
        success = _check_python_dependencies()
        sys.exit(0 if success else 1)

    try:
        from core_pipeline import MultimodalPipeline  # type: ignore[import]
    except ImportError:
        print("Error: Could not import MultimodalPipeline.")
        print("Ensure dependencies are installed (e.g., 'poetry install'), or run via run_all scripts which auto-install and execute the pipeline.")
        sys.exit(1)

    features = None
    if args.features:
        features = [item.strip() for item in args.features.split(",") if item.strip()]
        if not features:
            features = ALL_FEATURES
    else:
        features = ALL_FEATURES
        if args.skip_slow:
            features = [f for f in features if f not in SLOW_FEATURES]
            logging.info("--skip-slow: excluding %s", ", ".join(SLOW_FEATURES))

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output") / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    _handlers: list = [logging.StreamHandler()]
    if args.log_file:
        _handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=_handlers,
    )

    if shutil.which("ffmpeg") is None:
        logging.error("ffmpeg binary not found on PATH. Install ffmpeg before running the pipeline.")
        sys.exit(1)

    try:
        pipeline = MultimodalPipeline(output_dir=output_dir, features=features, device="cpu", decimal_places=args.decimal_places)

        try:
            import torch  # type: ignore[import]

            if torch.cuda.is_available():
                logging.info("CUDA is available. GPU will be used when supported by downstream models.")
        except (ImportError, AttributeError):
            logging.info("torch not fully available or CUDA check failed; continuing with CPU.")

        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logging.error("Data directory %s does not exist.", data_dir)
            sys.exit(1)

        logging.info("Processing directory: %s", data_dir)
        results = pipeline.process_directory(data_dir, is_video=(not args.is_audio))

        logging.info("Successfully processed %d files.", len(results))
        for filename in results:
            logging.info("  - %s", filename)

        logging.info("Results saved to: %s", output_dir)
        logging.info("Per-subject outputs: %s/{dyadID}/{subjectID}/{dyadID}_{subjectID}_summary_features.csv|json, _timeseries_features.csv, .log", output_dir)

        print(f"\nResults saved to: {output_dir}")
        print(f"Per-subject outputs: {output_dir}/{{dyadID}}/{{subjectID}}/{{dyadID}}_{{subjectID}}_summary_features.csv|json, _timeseries_features.csv, .log")

    except Exception as exc:  # pragma: no cover - defensive logging
        logging.error("Error processing files: %s", exc)
        logging.error(traceback.format_exc())
        print(f"Error processing files: {exc}")
        print("Check the log file for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
