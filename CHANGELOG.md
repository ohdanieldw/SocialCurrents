# Changelog

All notable changes to SocialCurrents will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.0] - 2026-04-02

### Added

#### Group-Level Analysis Tools

Five scripts that aggregate subject/dyad-level results into publication-ready group statistics. All accept `--subjects` for orientation metadata and `--covariates` for moderator analyses.

- **`group_correlate.py`**: Cross-subject feature-outcome analysis
  - Grouped dimension t-tests (Fisher-z transformed peak r, FDR corrected, Cohen's d)
  - Feature vote counting through PCA/ICA/FA loadings: traces back to raw features to find which behavioral cues consistently predict the outcome across subjects, even when subjects load on different components
  - Method comparison (box plot of mean |peak_r| per reduction method)
  - `--label` for any outcome dimension (default: Trustworthiness)
  - Outputs: forest plots, feature vote bar charts, lag distribution, method comparison

- **`group_synchrony.py`**: Cross-dyad synchrony patterns
  - One-sample tests (t-test + Wilcoxon) on whether synchrony metrics are reliably > 0
  - Leader-follower consistency (binomial test for A-leads vs B-leads across dyads)
  - RQA summary (descriptive stats for recurrence rate, determinism, laminarity, entropy)
  - Reduction method comparison (Friedman test)
  - Outputs: synchrony heatmaps, leader-follower bar charts, cross-dyad dot plots

- **`group_outcome_from_synch.py`**: Does synchrony predict ratings?
  - Per synchrony metric: one-sample t-test on Fisher-z(peak_r), FDR corrected
  - Cross-method comparison: which synchrony computation method best predicts the outcome
  - Lag direction analysis: binomial + t-test on whether synchrony change precedes or follows rating change
  - `--label` for any rating dimension (trustworthiness, warmth, competence, etc.)
  - Outputs: forest plots, method comparison bar chart, lag direction histogram

- **`group_synch_from_features.py`**: Which features drive synchrony?
  - Feature vote counting on multi-mode `synch_from_features/` results
  - **Overlap analysis**: cross-references with `group_correlate.py` to find features that predict BOTH the outcome AND synchrony -- the headline finding
  - Outputs: feature vote bar chart, overlap bar plot (outcome-only / both / synchrony-only)

- **`group_synch_by_states.py`**: Do behavioral states differ in synchrony?
  - Eta-squared approach for non-aligned states (states mean different things across dyads)
  - Per-dyad effect sizes, max |Cohen's d| from pairwise comparisons
  - Proportion of dyads with at least one significant pairwise comparison
  - Method comparison across reduction methods
  - Outputs: method comparison box plot, per-dyad bar plots

#### Orientation Normalization

- `--subjects` flag on `correlate.py`, `synchronize.py`, `describe.py`, `segment.py`
- Flips x-axis pose columns (`GMP_world_x_*`, `GMP_land_x_*`, `pf_facerectx`) for left-facing participants; right-facing is canonical orientation
- Load-time correction only -- raw `timeseries_features.csv` files unchanged
- Movement energy, audio features, facial AU/emotion features are direction-invariant (unaffected)
- `subjects.csv` holds seating info, demographics, and questionnaire scales in one file

#### Covariate and Moderator Analysis

- `--covariates` flag on all group-level scripts (comma-separated column names from subjects.csv)
- Spearman correlations between peak effect sizes and participant-level variables
- Separate analyses for rater and target roles where applicable
- Auto-filters: only numeric columns with <= 20% missing values
- Exploratory with small-N caveat noted in all reports

#### Synchrony Enhancements (synchronize.py)

- 3 new feature reduction methods: CCA (canonical correlation maximizing cross-person coupling), grouped-PCA (PCA within each of 4 modality groups), cluster (hierarchical clustering then PCA per cluster)
- 7 reduction methods total as subfolders: grouped, pca, fa, ica, cca, grouped-pca, cluster
- `--reduce-features all` runs all methods and produces separate output per method

#### Analysis Infrastructure

- `analysis/utils.py` expanded with group-level helpers:
  - `fisher_z()` / `inverse_fisher_z()` for correlation meta-analysis
  - `extract_peak_correlation()`: peak |r|, lag, and p per dimension
  - `discover_outputs()`: recursive file discovery
  - `parse_csv_header()`: read `# key: value` metadata from CSVs
  - `feature_to_modality()`: prefix-based modality classification (MODALITY_MAP)
  - `normalize_orientation()` / `resolve_facing()` / `load_subjects_df()`: orientation support
  - `get_covariate_columns()`: validate and filter covariate columns
- Folder naming convention: `_from_` for directional prediction (DV_from_IV, APA style), `_by_` for grouping comparisons
- `_batch/` directory for cross-subject/dyad group-level results
- `write_csv_with_header()` used throughout for reproducible metadata in all output CSVs
- Reference benchmarks (Hirschberg et al., Bernieri & Rosenthal, Dale & Spivey, Cohen) printed in terminal and saved to `summary_report.txt`

#### Individual-Level Analysis Tools

- **`describe.py`**: Descriptive statistics, stationarity tests (ADF), PCA diagnostics (scree plot, loadings heatmap), feature grouping by modality, batch mode for cross-subject comparison
- **`segment.py`**: HMM/changepoint/kmeans-window segmentation, auto state selection via BIC/AIC with elbow detection (kneedle), energy-sorted states (quietest to most animated), minimum duration smoothing (merge short visits), transition matrices
- **`map_states.py`**: State-outcome mapping with per-state means and 95% CIs, pairwise Mann-Whitney U tests (FDR corrected, Cohen's d), within-segment trend analysis (slopes), 3-panel figure
- **`correlate.py`** expanded: multi mode for EEG/fNIRS with CCA/ROI-average target reduction, `--label` flag, `--select-features` glob filtering, `--reduce-features all` to run all methods

### Changed

- `--questionnaire` flag removed from all group-level scripts; consolidated into `--subjects` + `--covariates`
- `group_trust_from_synch.py` renamed to `group_outcome_from_synch.py` with `--label` flag for outcome-agnostic usage
- Output filenames in group scripts use slugified label (e.g., `group_trustworthiness_from_features_grouped.csv`)

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
