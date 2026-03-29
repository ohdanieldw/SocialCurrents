# SocialCurrents

![Version](https://img.shields.io/badge/version-0.1.1-blue)

Extract 400+ behavioral features from video recordings and analyze social dynamics — interpersonal synchrony, conversational states, and impression formation — with no manual annotation required.

SocialCurrents is a multimodal feature extraction and analysis toolkit for social and behavioral research. Given video recordings of conversations, it extracts time-stamped behavioral features covering body movement, facial expression, speech, and language. Its analysis tools then relate these features to dynamic trait ratings, multi-channel neural recordings (fNIRS, EEG, fMRI), or any external timeseries — using lagged cross-correlation, HMM segmentation, and 10 interpersonal synchrony methods from the psychophysiology literature.

## Table of contents

- [What it measures](#what-it-measures)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Command reference](#command-reference)
- [Output format](#output-format)
- [Feature extractors](#feature-extractors)
- [Feature reference](#feature-reference)
- [Facial expression extraction (Py-Feat)](#facial-expression-extraction-py-feat)
- [Analysis toolkit](#analysis-toolkit)
- [Optional & heavy features](#optional--heavy-features)
- [Environment variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Funding](#funding)
- [Citation](#citation)

## What it measures

| Modality | What you get |
|---|---|
| **Body movement** | 33 pose landmarks per frame (MediaPipe), joint velocities, movement magnitude |
| **Facial expression** | Action units (AU1–AU28), valence/arousal, discrete emotion probabilities (Py-Feat, EmotiEffNet) |
| **Speech** | Pitch, volume, spectral features (librosa, openSMILE); speech emotion recognition |
| **Language** | Transcript (WhisperX with speaker diarization), sentiment, semantic similarity, NLI benchmarks |

## Requirements

- macOS (Intel or Apple Silicon)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda recommended)
- [Homebrew](https://brew.sh) (for ffmpeg)
- ~15 GB disk space for models and dependencies
- A free [HuggingFace](https://huggingface.co) account and token for speaker diarization (WhisperX)

**Note on Apple Silicon:** WhisperX transcription (and dependent NLP extractors) currently requires native x86 or GPU hardware. On Apple Silicon Macs running x86 conda environments under Rosetta, the transcription step will crash. Audio and vision extractors work normally. Full pipeline functionality including transcription is available on Linux servers or GPU-equipped machines.

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/ohdanieldw/socialcurrents.git
cd socialcurrents

# 2. Run the one-time setup (creates a conda env, installs all dependencies)
bash setup_macos.sh
```

Setup takes 5-15 minutes on first run. It creates a conda environment called `pipeline-env` and installs all Python packages automatically.

For WhisperX speaker diarization, set your HuggingFace token (get one free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)):

```bash
export HF_TOKEN="hf_your_token_here"
```

Add this line to `~/.zshrc` or `~/.bash_profile` so it persists across sessions.

## Quick start

```bash
git clone https://github.com/ohdanieldw/socialcurrents.git
cd socialcurrents
bash setup_macos.sh
bash run_macos.sh -i data/dyad002_sub003.MP4 -o output/
```

## Usage

```bash
# Run on a folder of video files
bash run_macos.sh -i data/ -o output/

# Skip CPU-prohibitive extractors (recommended on CPU-only machines)
bash run_macos.sh -i data/ -o output/ --skip-slow

# Run only specific feature extractors (faster)
bash run_macos.sh -i data/ -o output/ -f basic_audio,mediapipe_pose_vision,pyfeat_vision

# Process a single video file
bash run_macos.sh -i data/dyad002_sub003.MP4 -o output/

# Reprocess files even if output already exists
bash run_macos.sh -i data/ -o output/ --overwrite

# Fix variable-frame-rate videos automatically (re-encodes to 25 fps)
bash run_macos.sh -i data/ -o output/ --normalize-fps

# Round CSV floats to 2 decimal places (default: 3)
bash run_macos.sh -i data/ -o output/ --decimal-places 2

# See all available extractors
bash run_macos.sh --list-features
```

**Variable frame rate videos** may cause errors. Use `--normalize-fps` to automatically re-encode to constant 25 fps before processing, or manually pre-process with `ffmpeg -i input.MP4 -vsync cfr -r 25 output.MP4`.

If an extractor fails (e.g., missing dependency, incompatible model), the pipeline logs a warning and skips that extractor rather than crashing. All other extractors continue running and output files are still generated.

### Common workflows

**"I want movement and facial data only"**
```bash
bash run_macos.sh -i data/ -f mediapipe_pose_vision,pyfeat_vision,emotieffnet_vision
```

**"I want speech and language features"**
```bash
bash run_macos.sh -i data/ -f basic_audio,librosa_spectral,whisperx_transcription,deberta_text,sbert_text
```

**"I want everything"**
```bash
bash run_macos.sh -i data/ -o output/
```

### Input filename convention

Name your video files using the pattern `{dyadID}_{subjectID}.extension`, e.g.:

```
dyad002_sub003.MP4
dyad002_sub007.MP4
dyad015_sub042.MP4
```

The pipeline splits on the first underscore to extract the dyad and subject IDs. If a filename does not contain an underscore, a single folder named after the file stem is created as a fallback.

### Supported input formats

| Type | Extensions |
|---|---|
| Video | `.mp4`, `.MP4`, `.avi`, `.mov`, `.MOV`, `.mkv` |
| Audio | `.wav`, `.mp3`, `.flac` (use `--is-audio`) |

## Command reference

All the flags you can pass to `run_macos.sh` (or `python run_pipeline.py`), explained in plain English:

| Flag | What it does |
|---|---|
| `-i data/` | **Which videos to process.** Point this at a folder of video files, or a single file. Default: `data/` |
| `-o output/` | **Where to save results.** Each subject gets their own subfolder inside this directory. Default: a timestamped folder under `output/` |
| `-f feat1,feat2,...` | **Pick specific analyses.** Instead of running all 35+ extractors, list only the ones you need (comma-separated, no spaces). Much faster when you only need a few. |
| `--skip-slow` | **Skip the slowest extractors.** Three extractors are too slow to run on a laptop without a GPU. This flag leaves them out. Has no effect when `-f` is used. |
| `--overwrite` | **Redo files that already have output.** By default, the pipeline skips videos that already have results in the output folder. Use this flag to reprocess them. |
| `--normalize-fps` | **Fix choppy or variable-speed videos.** Some cameras record at inconsistent frame rates, which causes errors. This flag re-encodes videos to a steady 25 frames per second before processing. |
| `--decimal-places 3` | **How many decimal places in the CSV.** Default is 3 (e.g., 0.142). Set to 2 for slightly smaller files, or 6 for maximum precision. |
| `--is-audio` | **Process audio files instead of video.** Use this when your input folder contains `.wav`, `.mp3`, or `.flac` files rather than video. |
| `--log-file run.log` | **Save a copy of the processing log.** In addition to the per-subject logs saved automatically, this saves a single log covering the entire run. |
| `--pyfeat-sample-rate 5` | **How often to measure facial expressions.** See [Facial expression extraction](#facial-expression-extraction-py-feat) below. |
| `--pyfeat-batch-timeout 30` | **Give up on stuck facial analysis.** See [Facial expression extraction](#facial-expression-extraction-py-feat) below. |
| `--pyfeat-face-model mtcnn` | **Which face detector to use.** See [Facial expression extraction](#facial-expression-extraction-py-feat) below. |
| `--pyfeat-au-model svm` | **Which action unit model to use.** `svm` (default, fast) or `xgb` (may hang on some systems). See [Facial expression extraction](#facial-expression-extraction-py-feat) below. |

## Output format

Each subject gets its own subfolder nested under their dyad:

```
output/
  dyad002/
    sub003/
      dyad002_sub003_timeseries_features.csv   <- one row per video frame
      dyad002_sub003_summary_features.csv      <- one row per recording
      dyad002_sub003_summary_features.json     <- nested JSON with model metadata
      dyad002_sub003.log                       <- processing log for this subject
    sub007/
      ...
```

### Time-series CSV (`{prefix}_timeseries_features.csv`)

The primary analysis output. Each row represents one video frame.

| Column | Description |
|---|---|
| `frame_idx` | 0-based frame index |
| `time_seconds` | Timestamp in seconds (`frame_idx / fps`) |
| `oc_audvol`, `oc_audpit`, ... | Audio arrays linearly interpolated from audio frame rate to video frame rate |
| `lbrs_*`, `osm_*` | Librosa / openSMILE LLD arrays, same interpolation |
| `GMP_land_*`, `GMP_world_*` | MediaPipe pose landmarks -- one value per video frame (full resolution) |
| `pf_au*`, `pf_anger`, ... | Py-Feat AUs / emotions -- sampled every Nth frame, interpolated to fill all rows |
| `eln_prob_*`, `eln_valence`, `eln_arousal` | EmotiEffNet -- sampled from up to 64 frames, interpolated |
| All other scalar features | Broadcast (same value repeated across all rows for that file) |

**Temporal resolution notes:**
- Audio arrays (librosa, openSMILE LLDs, basic audio) are captured at ~31 samples/sec and linearly interpolated to video frame rate.
- MediaPipe processes every video frame; each row has exact landmark values for that frame.
- Py-Feat processes every Nth frame (default N=5); values between samples are linearly interpolated.
- EmotiEffNet processes up to 64 evenly sampled frames; values between samples are linearly interpolated.
- NLP features (transcription-based) and AudioStretchy configuration parameters are scalars that do not vary over time.

```python
import pandas as pd

ts = pd.read_csv("output/dyad002/sub003/dyad002_sub003_timeseries_features.csv")

# Pose visibility for left wrist landmark over time
ts.plot(x="time_seconds", y="GMP_land_visi_16")

# Happiness and arousal as functions of time
ts[["time_seconds", "pf_happiness", "eln_arousal"]].set_index("time_seconds").plot()

# Audio volume trajectory
ts.plot(x="time_seconds", y="oc_audvol")
```

### Summary CSV (`{prefix}_summary_features.csv`)

One row per input file. Array-valued features are summarised into statistics columns.

| Feature type | Example | CSV columns produced |
|---|---|---|
| Scalar | `GMP_land_visi_26` = `0.94` | `GMP_land_visi_26` = `0.94` |
| Long array (>20 elements) | `oc_audvol` = `[0.01, ...]` | `oc_audvol_mean`, `_std`, `_min`, `_max` |
| Short array (<=20) | `lbrs_spectral_contrast` = 7 values | `..._mean/_std/_min/_max` + `..._0` ... `..._6` |
| String / transcript | `transcription` = `"hello world"` | `transcription` = `"hello world"` |

### JSON (`{prefix}_summary_features.json`)

Nested structure grouped by model. Large arrays (>1000 elements) are stored as statistics objects with `mean`, `min`, `max`, `std`, `shape`, `dtype`, and `samples` fields.

## Feature extractors

| Name | Category | Output prefix |
|---|---|---|
| `basic_audio` | Audio | `oc_` |
| `librosa_spectral` | Audio | `lbrs_` |
| `opensmile` | Audio | `osm_` |
| `audiostretchy` | Audio | `AS_` |
| `speech_emotion` | Speech | `ser_` |
| `whisperx_transcription` | Speech/ASR | `WhX_` |
| `xlsr_speech_to_text` | Speech/ASR | -- |
| `s2t_speech_to_text` | Speech/ASR | -- |
| `deberta_text` | NLP | `DEB_` |
| `simcse_text` | NLP | `CSE_` |
| `albert_text` | NLP | `alb_` |
| `sbert_text` | NLP | `BERT_` |
| `use_text` | NLP | `USE_` |
| `elmo_text` | NLP | -- |
| `mediapipe_pose_vision` | Pose | `GMP_` |
| `vitpose_vision` | Pose | `vit_` |
| `pyfeat_vision` | Facial | `pf_` |
| `emotieffnet_vision` | Facial | `eln_` |
| `dan_vision` | Facial | `dan_` |
| `ganimation_vision` | Facial | `GAN_` |
| `arbex_vision` | Facial | `arbex_` |
| `crowdflow_vision` | Video | `of_` |
| `instadm_vision` | Video | `indm_` |
| `optical_flow_vision` | Video | -- |
| `videofinder_vision` | Video | `ViF_` |
| `lanegcn_vision` | Video | `GCN_` |
| `openpose_vision` | Pose | `openPose_` |
| `pare_vision` | Pose | `PARE_` |
| `psa_vision` | Pose | `psa_` |
| `deep_hrnet_vision` | Pose | `DHiR_` |
| `simple_baselines_vision` | Pose | `SBH_` |
| `rsn_vision` | Pose | `rsn_` |
| `smoothnet_vision` | Pose | `net_` |
| `me_graphau_vision` | Facial | `ann_` |
| `heinsen_sentiment` | NLP | `arvs_` |
| `meld_emotion` | NLP | `MELD_` |
| `avhubert_vision` | Audio-Visual | -- |
| `fact_vision` | Video | -- |
| `video_frames_vision` | Video | -- |
| `rife_vision` | Video | -- |

Run `bash run_macos.sh --list-features` for descriptions of each extractor.

## Feature reference

Detailed output keys for every extractor. Temporality indicates whether a feature changes per frame (time-varying) or is a single value for the whole recording (scalar).

### Audio features

#### `basic_audio` -- Volume & Pitch (`oc_`)

Time-varying arrays at ~31 samples/sec, resampled to video frame rate.

| Key | Description |
|---|---|
| `oc_audvol` | RMS energy (volume) per audio frame |
| `oc_audvol_diff` | Frame-to-frame volume change |
| `oc_audpit` | Pitch (fundamental frequency) per audio frame |
| `oc_audpit_diff` | Frame-to-frame pitch change |

#### `librosa_spectral` -- Spectral & Rhythm (`lbrs_`)

Time-varying arrays resampled to video frame rate. `lbrs_tempo` and `lbrs_*_singlevalue` are scalars.

| Key | Description |
|---|---|
| `lbrs_spectral_centroid` | Spectral centroid per audio frame |
| `lbrs_spectral_bandwidth` | Spectral bandwidth per audio frame |
| `lbrs_spectral_flatness` | Spectral flatness per audio frame |
| `lbrs_spectral_rolloff` | Spectral roll-off per audio frame |
| `lbrs_zero_crossing_rate` | Zero-crossing rate per audio frame |
| `lbrs_rmse` | RMS energy per audio frame |
| `lbrs_spectral_contrast` | Spectral contrast per audio frame |
| `lbrs_tempo` | Estimated tempo in BPM (scalar) |

#### `opensmile` -- Low-Level Descriptors (~1,512 features, `osm_`)

LLD keys (`osm_*_sma`) are time-varying; functional keys (`osm_*_mean`, `osm_*_stddev`, etc.) are scalars.

Key time-varying outputs: `osm_pcm_RMSenergy_sma`, `osm_loudness_sma`, `osm_F0final_sma`, `osm_voicingProb_sma`, `osm_jitterLocal_sma`, `osm_shimmerLocal_sma`, `osm_logHNR_sma`, `osm_mfcc1_sma`...`osm_mfcc12_sma`, `osm_spectralCentroid_sma`, `osm_spectralFlux_sma`, `osm_spectralRollOff25_sma`...`osm_spectralRollOff90_sma`, `osm_lsf1`...`osm_lsf8`.

#### `audiostretchy` -- Time-Stretching Analysis (`AS_`)

All static scalars: `AS_ratio`, `AS_gap_ratio`, `AS_lower_freq`, `AS_upper_freq`, `AS_buffer_ms`, `AS_threshold_gap_db`, `AS_sample_rate`, `AS_input_duration_sec`, `AS_output_duration_sec`.

### Speech features

#### `speech_emotion` -- Speech Emotion Recognition (`ser_`)

Static scalars (probabilities summing to 1.0): `ser_neutral`, `ser_calm`, `ser_happy`, `ser_sad`, `ser_angry`, `ser_fear`, `ser_disgust`, `ser_ps`, `ser_boredom`.

#### `whisperx_transcription` -- Transcription & Diarization (`WhX_`)

Requires `HF_TOKEN` for speaker diarization. Static (full-recording transcript broadcast to all rows).

| Key | Description |
|---|---|
| `transcription` | Full transcript text |
| `language` | Detected language code |
| `num_segments` | Number of speech segments |
| `WhX_segment_1` ... `WhX_segment_N` | Per-segment: text, speaker, start, end |
| `WhX_speaker1_summary` | Per-speaker: total_words, total_duration, avg_confidence |

### NLP / text features

> Text features require a transcription. Run `whisperx_transcription` first, or include it in the same `-f` list. All NLP features are static scalars broadcast to every row.

#### `deberta_text` (`DEB_`)
NLI benchmark scores: `DEB_SQuAD_1.1_F1`, `DEB_MNLI-m_Acc`, `DEB_SST-2_Acc`, `DEB_QNLI_Acc`, `DEB_CoLA_MCC`, `DEB_RTE_Acc`, `DEB_MRPC_F1`, `DEB_QQP_F1`, `DEB_STS-B_P`.

#### `simcse_text` (`CSE_`)
Sentence embedding benchmarks: `CSE_STS12`...`CSE_STS16`, `CSE_STSBenchmark`, `CSE_SICKRelatedness`, `CSE_Avg`.

#### `sbert_text` (`BERT_`)
Sentence embeddings: `BERT_tensor_sentences`, `BERT_score`, `BERT_ranks`.

#### `albert_text` (`alb_`)
GLUE benchmarks: `alb_mnli`, `alb_qnli`, `alb_qqp`, `alb_rte`, `alb_sst`, `alb_mrpc`, `alb_cola`, `alb_sts`.

#### `heinsen_sentiment` (`arvs_`)
Sentiment: `arvs_negative`, `arvs_neutral`, `arvs_positive`, `arvs_dominant_sentiment`, `arvs_confidence`.

#### `meld_emotion` (`MELD_`)
Dialogue-level emotion: `MELD_num_utterances`, `MELD_num_speakers`, `MELD_count_anger/disgust/fear/joy/neutral/sadness/surprise`, `MELD_num_emotion_shift`.

### Vision features

> Vision features are extracted from the video directly (not the audio track). They are skipped when `--is-audio` is used.

#### `mediapipe_pose_vision` -- 33 Pose Landmarks (`GMP_`)

Time-varying -- processes every video frame. 33 body landmarks, each with 10 attributes (330+ features):

| Attribute group | Keys | Description |
|---|---|---|
| Normalized coords | `GMP_land_x_1`...`33`, `GMP_land_y_1`...`33`, `GMP_land_z_1`...`33` | Image coordinates [0,1] |
| Visibility / presence | `GMP_land_visi_1`...`33`, `GMP_land_presence_1`...`33` | Detection confidence |
| World coords | `GMP_world_x_1`...`33`, `GMP_world_y_1`...`33`, `GMP_world_z_1`...`33` | Metric coordinates (meters) |
| World vis / presence | `GMP_world_visi_1`...`33`, `GMP_world_presence_1`...`33` | World-space confidence |

Landmark mapping (1-indexed): 1=Nose, 12=Left shoulder, 13=Right shoulder, 14=Left elbow, 15=Right elbow, 16=Left wrist, 17=Right wrist, 24=Left hip, 25=Right hip.

#### `pyfeat_vision` -- Facial Expression Analysis (`pf_`)

Time-varying -- samples every Nth frame (default N=5), interpolated to all rows. 37 features:

| Group | Keys | Description |
|---|---|---|
| Action Units (20) | `pf_au01`...`pf_au43` | FACS AU intensities 0-1 |
| Emotions (7) | `pf_anger`, `pf_disgust`, `pf_fear`, `pf_happiness`, `pf_sadness`, `pf_surprise`, `pf_neutral` | Emotion probabilities |
| Face geometry | `pf_facerectx`, `pf_facerecty`, `pf_facerectwidth`, `pf_facerectheight`, `pf_facescore` | Bounding box + confidence |
| Head pose | `pf_pitch`, `pf_roll`, `pf_yaw` | Head orientation (degrees) |

See [Facial expression extraction (Py-Feat)](#facial-expression-extraction-py-feat) for configuration options.

#### `emotieffnet_vision` -- EmotiEffNet (`eln_`)

Time-varying (up to 64 sampled frames, interpolated): `eln_arousal`, `eln_valence`, `eln_prob_{emotion}`. Static: `eln_top_emotion`, `eln_face_detected_ratio`, `eln_samples`.

#### Other vision extractors

| Extractor | Prefix | What it measures |
|---|---|---|
| `dan_vision` | `dan_` | Facial emotion probabilities (8 classes) |
| `ganimation_vision` | `GAN_` | AU intensities at 4 levels (68+ features) |
| `arbex_vision` | `arbex_` | Facial expression with reliability balancing |
| `me_graphau_vision` | `ann_` | Graph-based AU relations (12 AUs) |
| `openpose_vision` | `openPose_` | 18 body keypoints + joint angles |
| `vitpose_vision` | `vit_` | 17 COCO keypoints via Vision Transformer |
| `pare_vision` | `PARE_` | 3D body shape/pose parameters |
| `crowdflow_vision` | `of_` | Dense optical flow statistics |
| `optical_flow_vision` | -- | Sparse/dense optical flow (OpenCV) |
| `instadm_vision` | `indm_` | Depth map statistics and motion descriptors |

## Facial expression extraction (Py-Feat)

Py-Feat (`pyfeat_vision`) measures facial action units, emotions, and head pose from video. It works in two steps: first it finds the face in each frame, then it measures the expressions. The settings below let you control how this process works.

### Face detector (`--pyfeat-face-model`)

Think of the face detector as the part of the system that first finds where the face is in each frame before measuring expressions.

| Option | When to use it |
|---|---|
| **`mtcnn`** (default) | Recommended for most lab recordings. Works reliably when participants are seated and facing the camera. Slightly slower to start up but detects faces accurately. |
| `retinaface` | An alternative detector. Can be faster in some cases but is known to hang (freeze) on certain video types. Only try this if `mtcnn` fails on your videos. |
| `img2pose` | A third option that estimates head pose simultaneously. Experimental; use only if the other two don't work for your data. |

```bash
# Use the default (mtcnn) -- no flag needed
bash run_macos.sh -i data/ -f pyfeat_vision

# Explicitly choose a different detector
bash run_macos.sh -i data/ -f pyfeat_vision --pyfeat-face-model retinaface
```

### Sample rate (`--pyfeat-sample-rate`)

Instead of analyzing every single frame of video (which would be very slow), the pipeline analyzes every Nth frame and fills in the gaps automatically using interpolation. The default is 5, meaning it looks at one out of every 5 frames.

For a typical 25 fps video, this means **5 facial measurements per second** -- more than enough to capture meaningful changes in expression during a conversation.

| Sample rate | Frames analyzed per second (at 25 fps) | 10-min video: frames analyzed | Speed |
|---|---|---|---|
| `1` | 25 (every frame) | 15,000 | Slowest |
| **`5` (default)** | **5** | **3,000** | Recommended |
| `10` | 2.5 | 1,500 | Faster, still good |
| `25` | 1 | 600 | Fastest, coarser |

```bash
# Analyze more frames (slower but more detailed)
bash run_macos.sh -i data/ -f pyfeat_vision --pyfeat-sample-rate 2

# Analyze fewer frames (faster, good for long videos)
bash run_macos.sh -i data/ -f pyfeat_vision --pyfeat-sample-rate 10
```

### Action unit model (`--pyfeat-au-model`)

The AU model is the algorithm that measures how much each facial muscle group is activated. Two options are available:

| Option | When to use it |
|---|---|
| **`svm`** (default) | Fast and reliable. Completes in about 4 seconds per frame on a laptop CPU. Recommended for all standard use. |
| `xgb` | An alternative model. May produce slightly different AU intensity values but is known to hang on some systems. Only use if you have a specific reason. |

### Multiple faces

When more than one face appears in a frame (e.g., in a group recording or when someone walks behind the participant), the pipeline automatically selects the face closest to the horizontal center of the frame. This works well for standard lab setups where the target participant is seated centrally in their camera view. No configuration is needed.

### Batch timeout (`--pyfeat-batch-timeout`)

If facial analysis gets stuck on a difficult frame (e.g., heavy occlusion, unusual lighting), this setting tells the pipeline to give up after 30 seconds and move on to the next batch of frames rather than freezing indefinitely. Results collected before the stuck batch are kept.

You generally don't need to change this. Lower it if you want the pipeline to skip problematic frames faster; raise it if your computer is slow and needs more time per batch.

```bash
# Shorter timeout (skip stuck frames faster)
bash run_macos.sh -i data/ -f pyfeat_vision --pyfeat-batch-timeout 15
```

## Analysis toolkit

SocialCurrents includes four analysis tools that take the extracted features and produce publication-ready results. Each tool is a standalone script with flexible options for different research designs.

### Workflow

```
1. Extract features          python run_pipeline.py -i data/ -o output/
       |
2. Describe your data        python analysis/describe.py -f output/
       |
3. Relate to outcomes        python analysis/correlate.py -f ... -t ratings.csv
       |
4. Discover states           python analysis/segment.py -f ...
       |
5. Measure coordination      python analysis/synchronize.py --person-a ... --person-b ...
```

### `describe.py` -- Understand Your Data

Generates a comprehensive descriptive summary of your extracted features before you run any statistical analyses. Reports per-feature statistics (mean, SD, skewness, kurtosis, percent zero/NaN), groups features by modality, tests stationarity (ADF test), and produces diagnostic plots: timeseries over time, PCA scree plot, loadings heatmap, feature distributions, and inter-dimension correlations. In **batch mode**, pass a directory to process all subjects at once and get a cross-subject comparison with box plots.

```bash
# Single subject
python analysis/describe.py \
  -f output/dyad005/sub010/dyad005_sub010_timeseries_features.csv \
  -o results/sub010_descriptives/

# Batch -- all subjects in a directory
python analysis/describe.py -f output/test_full5/ -o results/descriptives_all/
```

### `correlate.py` -- Relate Features to Outcomes

Computes lagged cross-correlation between extracted behavioral features and external target signals. **Single mode** correlates features with a dynamic rating timeseries (e.g., continuous trustworthiness judgments from a slider task), a physiological signal, or any continuous measure -- with adjustable lag range to capture reaction time in ratings. **Multi mode** correlates features with multi-channel neural data (EEG, fNIRS, multi-voxel patterns), supporting PCA, Factor Analysis, ICA, CCA, and ROI-based reduction of target channels. All modes include Benjamini-Hochberg FDR correction and flexible dimensionality reduction (`pca`, `fa`, `ica`, `grouped`, `every`, or `all` to run all methods and compare).

```bash
# Single: behavioral features vs. trustworthiness rating
python analysis/correlate.py --mode single \
  -f output/dyad005/sub010/dyad005_sub010_timeseries_features.csv \
  -t data/Trait/Trustworthiness/sub009rating.csv \
  --reduce-features pca --n-components 5 --label Trustworthiness \
  -o results/sub010/

# Multi: behavioral features vs. fNIRS with ROI averaging
python analysis/correlate.py --mode multi \
  -f output/dyad005/sub010/dyad005_sub010_timeseries_features.csv \
  -t data/fNIRS/dyad005_sub009_fnirs.csv \
  --reduce-features pca --reduce-target roi-average \
  --roi-config data/fNIRS/roi_config.json \
  -o results/sub009_neural/
```

### `segment.py` -- Discover Conversational States

Segments a conversation into distinct behavioral states using Hidden Markov Models (HMM), changepoint detection, or windowed k-means clustering. When set to `auto`, the number of states is selected via BIC/AIC model comparison. States are automatically sorted by overall energy level (State 1 = quietest, State N = most animated) so they are comparable across subjects. Output includes state profiles showing what each state "looks like" across feature dimensions, transition probability matrices, per-visit duration statistics, and a color-coded state timeline. Directly links to impression rating analysis -- e.g., "which conversational states predict changes in perceived trustworthiness?"

```bash
python analysis/segment.py \
  -f output/dyad005/sub010/dyad005_sub010_timeseries_features.csv \
  --method hmm --n-states auto --max-states 8 \
  --reduce-features pca --n-components 5 \
  -o results/sub010_segments/
```

### `synchronize.py` -- Measure Interpersonal Coordination

Implements 10 synchrony methods spanning the full literature on interpersonal coordination, from basic windowed correlation to nonlinear dynamics and directional causality models.

| Category | Methods |
|---|---|
| **Time-domain** | Rolling Pearson, windowed cross-correlation, Lin's concordance |
| **Nonlinear dynamics** | Cross-recurrence quantification (RQA), detrended cross-correlation (DCCA) |
| **Frequency-domain** | Spectral coherence, wavelet coherence |
| **Directionality** | Granger causality, transfer entropy, coupled oscillator models |

All methods support lagged analysis and permutation-based surrogate testing for statistical validity. Directional methods report leader-follower dynamics and how leadership shifts over the conversation. Select any subset of methods via `--methods` or run all at once.

```bash
python analysis/synchronize.py \
  --person-a output/dyad005/sub009/dyad005_sub009_timeseries_features.csv \
  --person-b output/dyad005/sub010/dyad005_sub010_timeseries_features.csv \
  --methods pearson,crosscorr,rqa,granger,coherence \
  --reduce-features grouped --window-size 30 \
  -o results/dyad005_synchrony/
```

### Why this toolkit

Most multimodal pipelines stop at feature extraction. SocialCurrents goes further -- from raw video to publication-ready analyses of how behavioral cues predict social perception, how conversation partners coordinate their behavior, and what latent states structure a social interaction. The analysis tools work with any timeseries data, not just SocialCurrents output -- researchers can use `correlate.py` with EEG, fNIRS, or any multi-channel neural recording, and `synchronize.py` with any pair of behavioral or physiological timeseries.

## Optional & heavy features

Some features require extra setup or are disabled by default:

| Feature | Requirement | How to enable |
|---|---|---|
| `openpose_vision` | OpenPose C++ library compiled from source | Build OpenPose, set `OPENPOSE_PYTHON_PATH` env var |
| `use_text` | TensorFlow | `pip install -e packages/nlp_models[tensorflow-stack]` |
| `whisperx_transcription` diarization | HuggingFace token | `export HF_TOKEN=hf_...` |
| `videofinder_vision` | Ollama running locally | `ollama serve && ollama pull llava` |

## Environment variables

| Variable | Used by | Description |
|---|---|---|
| `HF_TOKEN` | WhisperX diarization | HuggingFace access token |
| `PARE_CHECKPOINT` | `pare_vision` | Path to custom PARE model checkpoint |
| `VITPOSE_CHECKPOINT` | `vitpose_vision` | Path to custom ViTPose checkpoint |
| `VITPOSE_CONFIG` | `vitpose_vision` | Path to custom ViTPose config file |

## Troubleshooting

### `ffmpeg not found`
```bash
brew install ffmpeg
```

### `No video/audio files found in <dir>`
Check that your files have supported extensions. Video: `.mp4`, `.avi`, `.mov`, `.mkv`. Audio: `.wav`, `.mp3`, `.flac`.

### `ModuleNotFoundError: No module named 'core_pipeline'`
The local packages are not installed. Re-run setup:
```bash
bash setup_macos.sh
```

### `Warning: mediapipe_pose_vision unavailable`
MediaPipe model file may not have downloaded. It is fetched automatically on first use; ensure you have internet access.

### Models download on first run
Many models (Whisper, DeBERTa, MediaPipe, etc.) are downloaded on first use. This can take several minutes. Subsequent runs use cached versions at `~/.cache/huggingface/hub/`.

### Running out of memory
Use `-f` to select only the features you need:
```bash
bash run_macos.sh -i data/ -f basic_audio,librosa_spectral,mediapipe_pose_vision,pyfeat_vision
```
Vision models are the heaviest consumers. Audio-only runs are much lighter.

## Funding

Development of SocialCurrents was supported by a Seed Fund Grant from the Centre for Social Sciences and Humanities (CSSH), National University of Singapore, awarded to Daniel DongWon Oh.

## Citation

If you use SocialCurrents in published research, please cite:

> Oh, D. D. (2026). *SocialCurrents: A multimodal feature extraction pipeline for social and behavioral research* (Version 0.1.1) [Software]. GitHub. https://github.com/ohdanieldw/socialcurrents

## Acknowledgments

Initial pipeline scaffolding by Kenneth Dao; testing and debugging by Shuo Duan. All subsequent development, integration, and testing by Daniel DongWon Oh.

## License

MIT -- see [LICENSE](LICENSE).
Copyright 2026 Daniel DongWon Oh
