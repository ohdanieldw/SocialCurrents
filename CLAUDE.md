# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure Convention

SocialCurrents is installed once (e.g. `~/tools/socialcurrents/`). Each study is a separate directory:

```
~/studies/my_study/
  data/
    videos/             # input video files
    ratings/            # continuous trait ratings by dimension
    neural/             # fNIRS, EEG, fMRI
  output/               # all pipeline output (follows neuroimaging convention below)
```

All commands run from the SocialCurrents directory, pointing `-i`, `-t`, and `-o` at the study paths.

## Running the Pipeline

```bash
# First-time setup (creates conda env "pipeline-env", installs all deps)
bash setup_macos.sh

# Run on a directory of videos
bash run_macos.sh -i ~/studies/my_study/data/videos/ -o ~/studies/my_study/output/

# Run specific features only (faster)
bash run_macos.sh -i ~/studies/my_study/data/videos/ -o ~/studies/my_study/output/ \
  -f basic_audio,librosa_spectral,mediapipe_pose_vision,pyfeat_vision

# Run directly after activating env
conda activate pipeline-env
python analysis/extract.py -i ~/studies/my_study/data/videos/ -o ~/studies/my_study/output/
python analysis/extract.py --list-features
python analysis/extract.py --check-dependencies
```

The `run_macos.sh` wrapper handles `conda run` activation; it calls `analysis/extract.py` with all passed arguments.

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

Note: MediaPipe's PoseLandmarker is now reset between videos in batch processing (via `_reset_landmarker()`), so the timestamp error no longer occurs when processing multiple constant-frame-rate videos sequentially.

### WhisperX on Apple Silicon
WhisperX transcription segfaults on Apple Silicon Macs when the conda environment runs x86 Python under Rosetta. ctranslate2 requires native ARM or proper x86 hardware. Transcription features (whisperx, sbert_text, deberta_text) must be run on an x86 Linux machine or GPU server until the conda environment is recreated as ARM-native.

### WhisperX setup
WhisperX speaker diarization requires:
- `pyannote.audio` installed
- A HuggingFace token: run `huggingface-cli login`
- Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1 and https://huggingface.co/pyannote/segmentation-3.0

### CPU-only limitations
These extractors are too slow without GPU and should generally be skipped on CPU-only machines (use `--skip-slow` or exclude via `-f`):
`facer_vision`, `optical_flow_vision`, `avhubert_vision`, `rife_vision`, `fact_vision`, `video_frames_vision`

Note: `pyfeat_vision` now runs efficiently on CPU thanks to every-Nth-frame sampling (`--pyfeat-sample-rate`, default 5). Each frame is processed individually via `detect_image()` (~4s/frame).

## Environment

- **Conda env:** `pipeline-env`, Python 3.11
- **PyTorch:** CPU-only build (from `https://download.pytorch.org/whl/cpu`)
- **Local packages:** installed editable via `pip install -e packages/<name>`; do NOT use `poetry install` on macOS (Poetry is only used for the root `pyproject.toml` metadata and lock file management, not for the actual environment)
- To reinstall a single local package after editing: `conda run -n pipeline-env pip install -e packages/<name>`

## Output Directory Convention

All output follows a neuroimaging-style directory structure. Folder names use `_from_` for directional prediction (DV_from_IV, matching APA convention) and `_by_` for non-directional grouping comparisons.

```
output/
  dyad{NNN}/
    sub{NNN}/
      describe/         # descriptive analysis (describe.py)
      extract/          # pipeline feature extraction outputs
      correlate/        # correlation analysis (correlate.py)
      segments/         # HMM/segmentation (segment.py)
      map_states/       # state-outcome linking (map_states.py)
    sub{NNN}_sub{NNN}/
      synchrony/              # dyad-level synchrony (synchronize.py)
        grouped/              # one subfolder per reduction method (7 methods)
        pca/
        ica/
        fa/
        cca/                  # + cca_loadings.csv
        grouped-pca/
        cluster/              # + cluster_assignments.csv
      trust_from_synch/       # ratings predicted from synchrony
        grouped/              # same 7 methods as synchrony/
        pca/
        ...
      synch_from_features/    # synchrony predicted from features
        grouped/
        pca/
        ...
      synch_by_states/        # synchrony compared by behavioral state
        grouped/
        pca/
        ...
  group/                     # group-level results (cross-subject/dyad)
    group_correlate/          # group_correlate.py
    group_synchrony/          # group_synchrony.py
    group_outcome_from_synch/ # group_outcome_from_synch.py
    group_synch_from_features/ # group_synch_from_features.py
    group_synch_by_states/    # group_synch_by_states.py
```

All scripts must read from and write to this structure. Use `--output-dir` to point to the appropriate `output/dyad{NNN}/sub{NNN}/{verb}/` path. Never create flat top-level directories like `output/describe/`.

### Orientation normalization

Analysis scripts (`correlate.py`, `synchronize.py`, `describe.py`, `segment.py`) accept `--subjects path/to/subjects.csv` with columns: `dyad, subject, seat_position, facing_direction`. When provided, `load_features()` negates x-axis pose columns (`GMP_world_x_*`, `GMP_land_x_*`, `pf_facerectx`) for left-facing participants so all subjects share a canonical right-facing orientation. This is a load-time correction; raw CSVs are unchanged. Movement energy, audio, and facial AU features are direction-invariant and unaffected. The `resolve_facing()` utility in `analysis/utils.py` handles subject ID extraction and lookup. Group-level scripts accept `--subjects` as an informational flag and note it in their summary reports.

### Synchrony output files

`synchronize.py` produces these files in the synchrony output directory:

| File | Format | Description |
|---|---|---|
| `synchrony_timeseries.csv` | Wide (one row per timepoint) | Columns: `time_seconds`, `{metric}_{dimension}` (e.g. `pearson_r_movement_energy`). Merges windowed pearson, crosscorr, concordance, and Granger results. Interpolated to `--output-resolution` grid (default: same as `--time-resolution`, 0.5s). Compatible with `correlate.py` as both `-f` and `-t` input. |
| `windowed_synchrony_long.csv` | Long (window_time x dimension) | Original long-format windowed results. One row per (window, dimension) pair. |
| `windowed_granger.csv` | Long (window_time x dimension) | Windowed Granger causality F-statistics and leader. |
| `wavelet_timeseries.csv` | Wide (one row per timepoint) | Wavelet coherence and phase dynamics: `time_seconds`, `mean_phase_diff`, `mean_coherence`, `leader`. |
| `synchrony_summary.csv` | One row per dimension | Overall summary statistics across all methods. |

The `--output-resolution` flag controls interpolation of the wide-format file. Set to `0` to skip interpolation and output at window-step resolution. The interpolation uses `np.interp` (linear) between window centers.

## Architecture

### Package layout

```
analysis/
  extract.py           # CLI entry point; defines FEATURE_CATALOG
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

1. **`analysis/extract.py`** parses CLI args → calls `MultimodalPipeline(features=[...]).process_directory(input_path)` or `.process_files([input_path])`
2. **`MultimodalPipeline.process_video_file()`** (`packages/core_pipeline/core_pipeline/pipeline.py`):
   - Captures video FPS and total frame count
   - Extracts 16 kHz mono WAV to a temp dir via ffmpeg (`audio_models/utils/audio_extraction.py`)
   - Calls `extract_features(audio_path)` for all audio/speech/NLP extractors
   - Calls each vision extractor (from `cv_models/vision/`) in order
   - MediaPipe, Py-Feat, and EmotiEffNet are always called with `keep_per_frame=True` to capture per-frame data
3. **`process_directory()`** calls `_save_file_outputs()` for each video immediately after processing it. The output path is determined by parsing the filename: `{dyadID}_{subjectID}.ext` → `self.output_dir / dyadID / subjectID`; filenames without an underscore fall back to `self.output_dir / stem`. A `file_prefix` (`{dyadID}_{subjectID}`) is prepended to every output filename. A per-subject `logging.FileHandler` is added before processing each file and removed after.
   - `{prefix}_timeseries_features.csv` = one row per video frame (built by `_build_timeindexed_csv()`); no `filename` column
   - `{prefix}_summary_features.csv` = one row per file (arrays summarised to `_mean/_std/_min/_max`); no `filename` column
   - `{prefix}_summary_features.json` = nested JSON grouped by model
   - `{prefix}.log` = processing log for this subject (e.g., `dyad002_sub003.log`)

### Feature extractor conventions

All extractors expose a `get_feature_dict(path) -> dict` method that returns either:
- A flat `{key: value}` dict, or
- A nested `{"Group name": {"description": ..., "features": {...}}}` dict

`_flatten_feature_output()` in `pipeline.py` normalises both forms to a flat dict before merging into `features`.

Extractors are **lazily initialised** in `_get_extractor(feature_name)`. Missing deps cause the extractor to be set to `None` and that feature silently skipped (prints a `Warning:` line).

### Per-frame data contract

Three vision models return per-frame sequences for the time-series CSV:

| Model | Key in features dict | Frames sampled |
|---|---|---|
| MediaPipe (`mediapipe_pose_vision`) | `mediapipe_pose_vision_per_frame` | Every frame |
| Py-Feat (`pyfeat_vision`) | `pyfeat_vision_per_frame` | Every Nth frame (default N=5, configurable via `--pyfeat-sample-rate`) |
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
4. Add an entry to `FEATURE_CATALOG` in `analysis/extract.py`
5. Update `MANUAL.md` with temporality (time-varying vs static scalar) and output keys

## Key Files

| File | Purpose |
|---|---|
| `analysis/extract.py` | CLI entry point; `FEATURE_CATALOG` defines all 35 extractors |
| `packages/core_pipeline/core_pipeline/pipeline.py` | `MultimodalPipeline`: orchestration, output writing, CSV building |
| `packages/cv_models/cv_models/vision/mediapipe_pose_analyzer.py` | Full per-frame pose, `keep_per_frame=True` returns `per_frame` list |
| `packages/cv_models/cv_models/vision/pyfeat_analyzer.py` | Facial AUs/emotions, per-frame collection in `_detect_features()` |
| `packages/cv_models/cv_models/vision/emotieffnet_analyzer.py` | Valence/arousal/emotions, per-sample collection in `analyze_video()` |
| `packages/audio_models/audio_models/utils/audio_extraction.py` | ffmpeg wrapper: extracts mono 16 kHz WAV from video |
| `packages/cv_models/cv_models/external/repo_manager.py` | On-demand git clone for upstream vision model repos |
| `analysis/correlate.py` | Lagged cross-correlation between pipeline features and target signals (ratings, EEG, fNIRS) |
| `analysis/synchronize.py` | 10 interpersonal synchrony methods with 7 reduction options |
| `analysis/utils.py` | Shared analysis utilities: loading, binning, reduction, correlation, plotting |
| `analysis/group_correlate.py` | Group-level: which features predict outcome ratings across subjects |
| `analysis/group_synchrony.py` | Group-level: synchrony reliability, leader-follower, method comparison |
| `analysis/group_outcome_from_synch.py` | Group-level: does synchrony predict outcome ratings (`--label`) |
| `analysis/group_synch_from_features.py` | Group-level: which features drive synchrony, overlap with outcome |
| `analysis/group_synch_by_states.py` | Group-level: do behavioral states differ in synchrony (eta-squared) |
