# Changelog

All notable changes to SocialCurrents will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
