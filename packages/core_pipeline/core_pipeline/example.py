# packages/core_pipeline/core_pipeline/example.py
"""
CLI script for running the multimodal data pipeline and exporting
frame-by-frame CSV outputs for supported analyzers.

Supports:
- MediaPipe per-frame CSV export
- basic_audio per-frame CSV export
"""

import argparse
import csv
import json
from pathlib import Path

from .pipeline import MultimodalPipeline
from .utils.file_utils import ensure_dir


def _json_default(o):
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()

    if isinstance(o, np.floating):
        return float(o)

    if isinstance(o, np.integer):
        return int(o)

    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", errors="replace")

    return str(o)


def _write_mediapipe_perframe_csv(results: dict, csv_path: Path) -> None:
    """
    Write MediaPipe per-frame rows to CSV if present.
    Force frame_index to be the first column when possible.
    """
    mp_key = "mediapipe_pose_vision_per_frame"
    per_frame = results.get(mp_key)

    if not isinstance(per_frame, list) or not per_frame:
        print("No MediaPipe per-frame data found to export.")
        return

    # Collect all keys
    seen = set()
    other_fields = []

    has_frame_idx = False
    has_timestamp = False

    for row in per_frame:
        if not isinstance(row, dict):
            continue

        if "frame_idx" in row:
            has_frame_idx = True
        if "timestamp" in row:
            has_timestamp = True

        for key in row.keys():
            if key in ("frame_idx", "timestamp"):
                continue
            if key not in seen:
                seen.add(key)
                other_fields.append(key)

    # Rename frame_idx -> frame_index in exported CSV
    fieldnames = ["frame_index"] + other_fields
    if has_timestamp:
        fieldnames.append("timestamp")

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in per_frame:
            if not isinstance(row, dict):
                continue

            out_row = {}

            if has_frame_idx:
                out_row["frame_index"] = row.get("frame_idx", "")
            else:
                out_row["frame_index"] = ""

            for key in other_fields:
                out_row[key] = row.get(key, "")

            if has_timestamp:
                out_row["timestamp"] = row.get("timestamp", "")

            writer.writerow(out_row)

    print(f"Saved MediaPipe per-frame CSV to {csv_path}")


def _write_basic_audio_perframe_csv(results: dict, csv_path: Path) -> None:
    """
    Write basic audio frame-by-frame arrays to CSV if present.

    Expected keys from AudioFeatureExtractor:
        - oc_audvol
        - oc_audvol_diff
        - oc_audpit
        - oc_audpit_diff
        - sample_rate
        - hop_length
        - num_frames

    Output columns:
        frame_index,time_sec,oc_audvol,oc_audvol_diff,oc_audpit,oc_audpit_diff
    """
    vol = results.get("oc_audvol")
    vol_diff = results.get("oc_audvol_diff")
    pit = results.get("oc_audpit")
    pit_diff = results.get("oc_audpit_diff")

    if vol is None and vol_diff is None and pit is None and pit_diff is None:
        print("No basic audio per-frame data found to export.")
        return

    sample_rate = results.get("sample_rate")
    hop_length = results.get("hop_length")

    def _to_list(x):
        if x is None:
            return None
        try:
            return x.tolist() if hasattr(x, "tolist") else list(x)
        except Exception:
            return None

    vol = _to_list(vol)
    vol_diff = _to_list(vol_diff)
    pit = _to_list(pit)
    pit_diff = _to_list(pit_diff)

    lengths = [len(x) for x in [vol, vol_diff, pit, pit_diff] if x is not None]
    if not lengths:
        print("Audio feature arrays were found but could not be converted to rows.")
        return

    n = max(lengths)

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_index",
        "time_sec",
        "oc_audvol",
        "oc_audvol_diff",
        "oc_audpit",
        "oc_audpit_diff",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(n):
            row = {"frame_index": i}

            if sample_rate and hop_length:
                row["time_sec"] = i * float(hop_length) / float(sample_rate)
            else:
                row["time_sec"] = ""

            row["oc_audvol"] = vol[i] if vol is not None and i < len(vol) else ""
            row["oc_audvol_diff"] = vol_diff[i] if vol_diff is not None and i < len(vol_diff) else ""
            row["oc_audpit"] = pit[i] if pit is not None and i < len(pit) else ""
            row["oc_audpit_diff"] = pit_diff[i] if pit_diff is not None and i < len(pit_diff) else ""

            writer.writerow(row)

    print(f"Saved basic audio per-frame CSV to {csv_path}")


def _export_single_video_outputs(
    results: dict,
    output_dir: Path,
    write_json: bool,
    write_mediapipe_csv: bool,
    write_audio_csv: bool,
    json_name: str = "pipeline_features.json",
    mediapipe_csv_name: str = "mediapipe_per_frame.csv",
    audio_csv_name: str = "basic_audio_per_frame.csv",
) -> None:
    """
    Export outputs for a single processed item.
    """
    if write_mediapipe_csv:
        _write_mediapipe_perframe_csv(results, output_dir / mediapipe_csv_name)

    if write_audio_csv:
        _write_basic_audio_perframe_csv(results, output_dir / "audio" / audio_csv_name)

    if write_json:
        out_path = output_dir / json_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=_json_default)
        print(f"Saved JSON to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the multimodal data pipeline")

    parser.add_argument("--video", help="Path to a video file")
    parser.add_argument("--audio", help="Path to an audio file")
    parser.add_argument("--data-dir", help="Directory containing video files")
    parser.add_argument("--output-dir", default="output/example", help="Output directory")

    parser.add_argument(
        "--features",
        default="basic_audio,librosa_spectral",
        help="Comma-separated list of features to extract",
    )

    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save pipeline_features.json",
    )

    parser.add_argument(
        "--keep-per-frame",
        action="store_true",
        help="Keep per-frame outputs for sequence models (e.g. MediaPipe)",
    )

    parser.add_argument(
        "--mediapipe-perframe-csv",
        action="store_true",
        help="Write MediaPipe per-frame CSV when available",
    )

    parser.add_argument(
        "--audio-perframe-csv",
        action="store_true",
        help="Write basic audio per-frame CSV when available",
    )

    args = parser.parse_args()

    features = [x.strip() for x in args.features.split(",") if x.strip()]
    output_dir = Path(ensure_dir(args.output_dir))

    pipeline = MultimodalPipeline(
        output_dir=output_dir,
        features=features,
        device="cpu",
        keep_per_frame=bool(args.keep_per_frame),
    )

    if args.video:
        results = pipeline.process_video_file(args.video)

        _export_single_video_outputs(
            results=results,
            output_dir=output_dir,
            write_json=bool(args.save_json),
            write_mediapipe_csv=bool(args.keep_per_frame and args.mediapipe_perframe_csv),
            write_audio_csv=bool(args.audio_perframe_csv),
        )

    elif args.audio:
        print(f"Processing audio file: {args.audio}")
        results = pipeline.process_audio_file(args.audio)

        if args.audio_perframe_csv:
            _write_basic_audio_perframe_csv(
                results,
                output_dir / "audio" / "basic_audio_per_frame.csv",
            )

        if args.save_json:
            out_path = output_dir / "pipeline_features.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=_json_default)
            print(f"Saved JSON to {out_path}")

        print(f"Results saved to {output_dir}")

    elif args.data_dir:
        print(f"Processing all videos in directory: {args.data_dir}")
        results = pipeline.process_directory(args.data_dir, is_video=True)

        # Export per-file CSVs
        for filename, file_results in results.items():
            stem = Path(filename).stem

            if args.keep_per_frame and args.mediapipe_perframe_csv:
                _write_mediapipe_perframe_csv(
                    file_results,
                    output_dir / f"{stem}_mediapipe_per_frame.csv",
                )

            if args.audio_perframe_csv:
                _write_basic_audio_perframe_csv(
                    file_results,
                    output_dir / "audio" / f"{stem}_basic_audio_per_frame.csv",
                )

        if args.save_json:
            out_path = output_dir / "pipeline_features.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=_json_default)
            print(f"Saved JSON to {out_path}")

        print(f"Processed {len(results)} files. Results saved to {output_dir}")

    else:
        default_data_dir = Path(__file__).parent.parent / "data"
        if default_data_dir.exists() and any(default_data_dir.iterdir()):
            print(f"Processing all videos in default directory: {default_data_dir}")
            results = pipeline.process_directory(default_data_dir, is_video=True)
            print(f"Processed {len(results)} files. Results saved to {output_dir}")
        else:
            print("No input provided and default data directory is empty or doesn't exist.")
            print("Please specify --video, --audio, or --data-dir.")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())