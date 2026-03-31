# Changelog

All notable changes to SocialCurrents will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.1] - 2026-03-29

### Added
- `analysis/correlate.py`: standalone lagged cross-correlation analysis script
  - Aligns pipeline timeseries features with continuous rating data (e.g., trustworthiness)
  - PCA (top 10 components) + 4 conceptual summary dimensions (movement energy, vocal energy, spectral complexity, openSMILE summary)
  - Lagged cross-correlation with configurable lag range and time bin size
  - Benjamini-Hochberg FDR correction via statsmodels (graceful fallback if not installed)
  - Self-documenting output: `--rater`, `--target`, `--rating-col` args; CSV header comments with full run parameters
  - PCA legend labels show top 3 contributing features per component
- `--overwrite` flag for both the main pipeline and `analysis/correlate.py`
- `--decimal-places` flag to control CSV float rounding (default: 3)
- `--skip-slow` flag to exclude CPU-prohibitive extractors
- `-i` / `--input` flag (replaces deprecated `-d` / `--data-dir`)
- Single-file input support (`-i path/to/video.mp4`)
- Per-subject log files (`{prefix}.log`) inside each subject folder
- Py-Feat configuration flags:
  - `--pyfeat-sample-rate` (default: 5) - process every Nth frame
  - `--pyfeat-face-model` (default: `mtcnn`) - face detector selection
  - `--pyfeat-au-model` (default: `svm`) - action unit model selection
- Multi-face selection: when multiple faces detected, keeps the face closest to the horizontal center of the frame (suits lab setups with centrally-seated participants)
- Plain-English command reference and Py-Feat configuration guide in README

### Changed
- Output directory structure: `output/{dyadID}/{subjectID}/` with prefixed filenames
- Filename column removed from both CSVs (identity encoded in folder/filename)
- Metadata/config columns moved from timeseries to summary CSV only
- OpenSMILE -100 sentinels replaced with NaN in timeseries
- All extractors now catch runtime failures gracefully (log + skip)
- Py-Feat frame sampling changed from 120 evenly-spaced frames to every-Nth-frame (default every 5th = 5 Hz at 25 fps); interpolation to full frame rate is unchanged
- Py-Feat default models changed to faster combination: `mtcnn` + `mobilefacenet` + `svm` + `resmasknet` + `img2pose` (previous defaults hung on many videos)
- Py-Feat detection runs in subprocess per batch for timeout safety
- `pyfeat_vision` added to the default quick-test feature set in documentation examples
- Suppressed noisy `timm` FutureWarning and `sklearn` UserWarning in console output

### Fixed
- **MediaPipe batch processing bug:** PoseLandmarker now resets between videos, fixing "Input timestamp must be monotonically increasing" error on the second video in a batch
- **Py-Feat all-zeros output:** `face_detection_threshold` and scipy compat patch were missing from the subprocess where detection actually runs; per-image fallback for py-feat versions without `detect_images()` (plural)
- **Py-Feat hanging:** replaced `signal.alarm` timeout (ineffective against C/PyTorch code) with `multiprocessing.Pool` subprocess timeout that can forcibly terminate stuck batches
- `import logging` missing in pipeline.py
- DataFrame fragmentation PerformanceWarning
- ctranslate2 `use_auth_token` constructor error in WhisperX
- speech_separation crashing entire pipeline on runtime error

## [0.1.0] - 2026-03-26

Initial public release.

### Added
- 41 feature extractors across audio, speech/ASR, NLP, body pose, facial expression, and video modalities
- Three output formats per recording session:
  - `{prefix}_timeseries_features.csv` = one row per video frame
  - `{prefix}_summary_features.csv` = per-recording summary statistics
  - `{prefix}_summary_features.json` = full nested JSON with raw arrays
- Per-frame time-series for MediaPipe (all frames), Py-Feat (120 sampled), and EmotiEffNet (64 sampled); audio arrays resampled to video frame rate
- macOS setup script (`setup_macos.sh`) and run wrapper (`run_macos.sh`) with conda environment management
- `--features` flag to run a subset of extractors
- `--list-features` to print the full catalog with descriptions
- `--version` flag (`SocialCurrents 0.1.0`)
- On-demand git cloning of upstream model repos via `ensure_repo()`
