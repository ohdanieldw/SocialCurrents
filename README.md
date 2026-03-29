# SocialCurrents

![Version](https://img.shields.io/badge/version-0.1.1-blue)

Extract 400+ behavioral features from video recordings automatically.

SocialCurrents is a multimodal feature extraction pipeline for social and behavioral research. Given a folder of video recordings (e.g., dyadic interactions, interviews, group conversations), it produces time-stamped, analysis-ready feature files covering body movement, facial expression, speech, and language — with no manual annotation required.

## What it measures

| Modality | What you get |
|---|---|
| **Body movement** | 33 pose landmarks per frame (MediaPipe), joint velocities, movement magnitude |
| **Facial expression** | Action units (AU1–AU28), valence/arousal, discrete emotion probabilities (Py-Feat, EmotiEffNet) |
| **Speech** | Pitch, volume, spectral features (librosa, openSMILE); speech emotion recognition |
| **Language** | Transcript (WhisperX with speaker diarization), sentiment, semantic similarity, NLI benchmarks |

## Input filename convention

Name your video files using the pattern `{dyadID}_{subjectID}.extension`, e.g.:

```
dyad002_sub003.MP4
dyad002_sub007.MP4
dyad015_sub042.MP4
```

The pipeline splits on the first underscore to extract the dyad and subject IDs. If a filename does not contain an underscore, a single folder named after the file stem is created as a fallback.

## Output files

Each subject gets its own subfolder nested under their dyad, e.g.:

```
output/
  dyad002/
    sub003/
      dyad002_sub003_timeseries_features.csv   ← one row per video frame
      dyad002_sub003_summary_features.csv      ← one row per recording (array features → summary stats)
      dyad002_sub003_summary_features.json     ← nested JSON with raw arrays and model metadata
      dyad002_sub003.log                       ← processing log for this subject
    sub007/
      dyad002_sub007_timeseries_features.csv
      ...
```

The time-series CSV is the primary file for most behavioral analyses (e.g., time-lagged cross-correlations, windowed synchrony, event-locked averages).

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

Setup takes 5–15 minutes on first run. It creates a conda environment called `pipeline-env` and installs all Python packages automatically.

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

`--skip-slow` excludes three extractors that are prohibitively slow on CPU: `s2t_speech_to_text`, `xlsr_speech_to_text`, and `elmo_text`. When `-f` is used instead, `--skip-slow` has no effect — all explicitly requested extractors run regardless.

**Variable frame rate videos** may cause errors. Use `--normalize-fps` to automatically re-encode to constant 25 fps before processing, or manually pre-process with `ffmpeg -i input.MP4 -vsync cfr -r 25 output.MP4`. See [MANUAL.md](MANUAL.md#known-issues) for details.

If an extractor fails (e.g., missing dependency, incompatible model), the pipeline logs a warning and skips that extractor rather than crashing. All other extractors continue running and output files are still generated.

For WhisperX speaker diarization, export your HuggingFace token first:

```bash
export HF_TOKEN=hf_your_token_here
bash run_macos.sh -i data/ -o output/
```

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

### Command reference

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

## Feature extractors

| Name | Category | Output prefix |
|---|---|---|
| `basic_audio` | Audio | `oc_` |
| `librosa_spectral` | Audio | `lbrs_` |
| `opensmile` | Audio | `osm_` |
| `audiostretchy` | Audio | `AS_` |
| `speech_emotion` | Speech | `ser_` |
| `whisperx_transcription` | Speech/ASR | `WhX_` |
| `xlsr_speech_to_text` | Speech/ASR | — |
| `s2t_speech_to_text` | Speech/ASR | — |
| `deberta_text` | NLP | `DEB_` |
| `simcse_text` | NLP | `CSE_` |
| `albert_text` | NLP | `alb_` |
| `sbert_text` | NLP | `BERT_` |
| `use_text` | NLP | `USE_` |
| `elmo_text` | NLP | — |
| `mediapipe_pose_vision` | Pose | `GMP_` |
| `vitpose_vision` | Pose | `vit_` |
| `pyfeat_vision` | Facial | `pf_` |
| `emotieffnet_vision` | Facial | `eln_` |
| `dan_vision` | Facial | `dan_` |
| `ganimation_vision` | Facial | `GAN_` |
| `arbex_vision` | Facial | `arbex_` |
| `crowdflow_vision` | Video | `of_` |
| `instadm_vision` | Video | `indm_` |
| `optical_flow_vision` | Video | — |
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
| `avhubert_vision` | Audio-Visual | — |
| `fact_vision` | Video | — |
| `video_frames_vision` | Video | — |
| `rife_vision` | Video | — |

Run `bash run_macos.sh --list-features` for descriptions of each extractor.

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
# Use the default (mtcnn) — no flag needed
bash run_macos.sh -i data/ -f pyfeat_vision

# Explicitly choose a different detector
bash run_macos.sh -i data/ -f pyfeat_vision --pyfeat-face-model retinaface
```

### Sample rate (`--pyfeat-sample-rate`)

Instead of analyzing every single frame of video (which would be very slow), the pipeline analyzes every Nth frame and fills in the gaps automatically using interpolation. The default is 5, meaning it looks at one out of every 5 frames.

For a typical 25 fps video, this means **5 facial measurements per second** — more than enough to capture meaningful changes in expression during a conversation.

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

## Analysis tools

### Lagged cross-correlation (`analysis/cross_corr.py`)

Correlate pipeline features with continuous behavioral ratings (e.g., trustworthiness, warmth):

```bash
conda activate pipeline-env
python analysis/cross_corr.py \
  -f output/dyad001/sub001/dyad001_sub001_timeseries_features.csv \
  -r ratings/sub001_trustworthiness.csv \
  --rater sub001 --target sub007 --rating-col Value \
  -o analysis_output/dyad001/
```

Outputs a cross-correlation CSV (with FDR-corrected p-values) and a plot showing 10 PCA components + 4 conceptual dimensions (movement energy, vocal energy, spectral complexity, openSMILE summary) at lags from -5s to +15s.

## Documentation

See [MANUAL.md](MANUAL.md) for the full feature reference and output format details.

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Funding

Development of SocialCurrents was supported by a Seed Fund Grant from the Centre for Social Sciences and Humanities (CSSH), National University of Singapore, awarded to Daniel DongWon Oh.

## Citation

If you use SocialCurrents in published research, please cite:

> Oh, D. D. (2026). *SocialCurrents: A multimodal feature extraction pipeline for social and behavioral research* (Version 0.1.1) [Software]. GitHub. https://github.com/ohdanieldw/socialcurrents

## Acknowledgments

Initial pipeline scaffolding by Kenneth Dao; testing and debugging by Shuo Duan. All subsequent development, integration, and testing by Daniel DongWon Oh.

## License

MIT — see [LICENSE](LICENSE).
Copyright © 2026 Daniel DongWon Oh
