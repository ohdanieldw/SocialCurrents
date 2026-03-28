# Changelog

All notable changes to SocialCurrents will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.2] — 2026-03-29

### Added
- `analysis/cross_corr.py` — standalone lagged cross-correlation analysis script
  - Aligns pipeline timeseries features with continuous rating data (e.g., trustworthiness)
  - PCA (top 10 components) + 4 conceptual summary dimensions (movement energy, vocal energy, spectral complexity, openSMILE summary)
  - Lagged cross-correlation with configurable lag range and time bin size
  - Benjamini-Hochberg FDR correction via statsmodels (graceful fallback if not installed)
  - Self-documenting output: `--rater`, `--target`, `--rating-col` args; CSV header comments with full run parameters
  - PCA legend labels show top 3 contributing features per component
- `--pyfeat-sample-rate` flag (default: 5) — process every Nth frame for Py-Feat instead of all frames
- `--pyfeat-batch-timeout` flag (default: 30s) — kill and skip a Py-Feat batch that hangs
- `--overwrite` flag for `analysis/cross_corr.py`

### Fixed
- **MediaPipe batch processing bug:** PoseLandmarker now resets between videos, fixing "Input timestamp must be monotonically increasing" error on the second video in a batch
- **Py-Feat hanging on full videos:** replaced `signal.alarm` timeout (ineffective against C/PyTorch code) with `multiprocessing.Pool` subprocess timeout that can forcibly terminate stuck batches

### Changed
- Py-Feat frame sampling changed from 120 evenly-spaced frames to every-Nth-frame (default every 5th frame = 5 Hz at 25 fps); interpolation to full frame rate is unchanged
- Py-Feat detection now runs in subprocess per batch for timeout safety
- `pyfeat_vision` added to the default quick-test feature set in documentation examples

## [0.1.1] — 2026-03-27

### Added
- `--decimal-places` flag to control CSV float rounding (default: 3)
- `--skip-slow` flag to exclude CPU-prohibitive extractors
- `-i` / `--input` flag (replaces deprecated `-d` / `--data-dir`)
- Single-file input support (`-i path/to/video.mp4`)
- Per-subject log files (`{prefix}.log`) inside each subject folder

### Changed
- Output directory structure: `output/{dyadID}/{subjectID}/` with prefixed filenames
- Filename column removed from both CSVs (identity encoded in folder/filename)
- Metadata/config columns moved from timeseries to summary CSV only
- OpenSMILE -100 sentinels replaced with NaN in timeseries
- All extractors now catch runtime failures gracefully (log + skip)

### Fixed
- `import logging` missing in pipeline.py
- DataFrame fragmentation PerformanceWarning
- ctranslate2 `use_auth_token` constructor error in WhisperX
- speech_separation crashing entire pipeline on runtime error

## [0.1.0] — 2026-03-26

Initial public release.

### Added
- 41 feature extractors across audio, speech/ASR, NLP, body pose, facial expression, and video modalities
- Three output formats per recording session:
  - `{prefix}_timeseries_features.csv` — one row per video frame
  - `{prefix}_summary_features.csv` — per-recording summary statistics
  - `{prefix}_summary_features.json` — full nested JSON with raw arrays
- Per-frame time-series for MediaPipe (all frames), Py-Feat (120 sampled), and EmotiEffNet (64 sampled); audio arrays resampled to video frame rate
- macOS setup script (`setup_macos.sh`) and run wrapper (`run_macos.sh`) with conda environment management
- `--features` flag to run a subset of extractors
- `--list-features` to print the full catalog with descriptions
- `--version` flag (`SocialCurrents 0.1.0`)
- On-demand git cloning of upstream model repos via `ensure_repo()`
