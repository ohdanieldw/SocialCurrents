# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Pipeline

```bash
# First-time setup (creates conda env "pipeline-env", installs all deps)
bash setup_macos.sh

# Run on a directory of videos
bash run_macos.sh -d data/ -o output/

# Run specific features only (faster)
bash run_macos.sh -d data/ -f basic_audio,librosa_spectral,mediapipe_pose_vision

# Run directly after activating env
conda activate pipeline-env
python run_pipeline.py -d data/ -o output/
python run_pipeline.py --list-features
python run_pipeline.py --check-dependencies
```

The `run_macos.sh` wrapper handles `conda run` activation; it calls `run_pipeline.py` with all passed arguments.

## Known Issues

### VFR (Variable Frame Rate) videos
Videos with variable frame rates cause MediaPipe to fail with:
```
Input timestamp must be monotonically increasing
```
Pre-process with ffmpeg to convert to constant frame rate before running the pipeline:
```bash
ffmpeg -i input.MP4 -vsync cfr -r 25 output.MP4
```

## Environment

- **Conda env:** `pipeline-env`, Python 3.11
- **PyTorch:** CPU-only build (from `https://download.pytorch.org/whl/cpu`)
- **Local packages:** installed editable via `pip install -e packages/<name>`; do NOT use `poetry install` on macOS (Poetry is only used for the root `pyproject.toml` metadata and lock file management, not for the actual environment)
- To reinstall a single local package after editing: `conda run -n pipeline-env pip install -e packages/<name>`

## Architecture

### Package layout

```
run_pipeline.py           # CLI entry point; defines FEATURE_CATALOG
packages/
  core_pipeline/          # MultimodalPipeline orchestrator
  audio_models/           # Audio, speech, and ASR extractors
  cv_models/              # Vision, pose, and facial analysis extractors
  nlp_models/             # Text/NLP extractors (text comes from transcription)
external/
  audio/                  # Git-cloned upstream repos (cloned on demand)
  vision/
```

### Execution flow

1. **`run_pipeline.py`** parses CLI args → calls `MultimodalPipeline(features=[...]).process_directory(data_dir)`
2. **`MultimodalPipeline.process_video_file()`** (`packages/core_pipeline/core_pipeline/pipeline.py`):
   - Captures video FPS and total frame count
   - Extracts 16 kHz mono WAV to a temp dir via ffmpeg (`audio_models/utils/audio_extraction.py`)
   - Calls `extract_features(audio_path)` for all audio/speech/NLP extractors
   - Calls each vision extractor (from `cv_models/vision/`) in order
   - MediaPipe, Py-Feat, and EmotiEffNet are always called with `keep_per_frame=True` to capture per-frame data
3. **`process_directory()`** calls `_save_file_outputs()` for each video immediately after processing it, writing to `self.output_dir / file_path.stem /`:
   - `features_timeseries.csv` — one row per video frame (built by `_build_timeindexed_csv()`)
   - `features.csv` — one row per file (arrays summarised to `_mean/_std/_min/_max`)
   - `features.json` — nested JSON grouped by model
   - `pipeline.log` is written at the top-level output directory (covers the whole batch run)

### Feature extractor conventions

All extractors expose a `get_feature_dict(path) -> dict` method that returns either:
- A flat `{key: value}` dict, or
- A nested `{"Group name": {"description": ..., "features": {...}}}` dict

`_flatten_feature_output()` in `pipeline.py` normalises both forms to a flat dict before merging into `features`.

Extractors are **lazily initialised** in `_get_extractor(feature_name)` — missing deps cause the extractor to be set to `None` and that feature silently skipped (prints a `Warning:` line).

### Per-frame data contract

Three vision models return per-frame sequences for the time-series CSV:

| Model | Key in features dict | Frames sampled |
|---|---|---|
| MediaPipe (`mediapipe_pose_vision`) | `mediapipe_pose_vision_per_frame` | Every frame |
| Py-Feat (`pyfeat_vision`) | `pyfeat_vision_per_frame` | Up to 120 evenly sampled |
| EmotiEffNet (`emotieffnet_vision`) | `emotieffnet_vision_per_sample` | Up to 64 evenly sampled |

Each entry is a list of dicts with `frame_idx`, `timestamp`, and the model's feature keys. `_build_timeindexed_csv()` interpolates sampled-frame data to all video frames via `np.interp`.

Audio arrays (`oc_*`, `lbrs_*`, `osm_*`) are resampled from ~31 Hz (hop_length=512, sr=16000) to video frame rate using `np.interp`. These keys are identified by prefix match against `AUDIO_PREFIXES = ("oc_audvol", "oc_audpit", "lbrs_", "osm_")`.

`_PER_FRAME_KEYS` (class-level frozenset on `MultimodalPipeline`) lists the three per-frame dict keys; they are skipped in the flat per-file CSV to avoid serialising huge list-of-dict columns.

### External repo cloning

Many vision/audio analyzers call `ensure_repo(repo_key)` at init time (`cv_models/external/repo_manager.py`, `audio_models/external/repo_manager.py`). This clones the upstream repo into `external/audio/<name>` or `external/vision/<name>` on first use and adds it to `sys.path`. Override a clone location with the corresponding env var (e.g. `VITPOSE_REPO`, `WHISPERX_REPO`).

### Adding a new feature extractor

1. Create `packages/<package>/…/<name>_analyzer.py` with a `get_feature_dict(path) -> dict` method
2. Add a lazy-init branch in `_get_extractor()` in `pipeline.py`
3. Add the feature name to `vision_feature_flags` (if vision) or the `extract_features()` chain (if audio/NLP)
4. Add an entry to `FEATURE_CATALOG` in `run_pipeline.py`
5. Update `MANUAL.md` with temporality (time-varying vs static scalar) and output keys

## Key Files

| File | Purpose |
|---|---|
| `run_pipeline.py` | CLI entry point; `FEATURE_CATALOG` defines all 35 extractors |
| `packages/core_pipeline/core_pipeline/pipeline.py` | `MultimodalPipeline` — orchestration, output writing, CSV building |
| `packages/cv_models/cv_models/vision/mediapipe_pose_analyzer.py` | Full per-frame pose, `keep_per_frame=True` returns `per_frame` list |
| `packages/cv_models/cv_models/vision/pyfeat_analyzer.py` | Facial AUs/emotions, per-frame collection in `_detect_features()` |
| `packages/cv_models/cv_models/vision/emotieffnet_analyzer.py` | Valence/arousal/emotions, per-sample collection in `analyze_video()` |
| `packages/audio_models/audio_models/utils/audio_extraction.py` | ffmpeg wrapper — extracts mono 16 kHz WAV from video |
| `packages/cv_models/cv_models/external/repo_manager.py` | On-demand git clone for upstream vision model repos |
| `MANUAL.md` | End-user documentation (output format, feature reference with temporality notes) |
