# SocialCurrents

![Version](https://img.shields.io/badge/version-0.1.0-blue)

SocialCurrents is a multimodal feature extraction pipeline for social and behavioral research. Given a folder of video recordings (e.g., dyadic interactions, interviews, group conversations), it produces time-stamped, analysis-ready feature files covering body movement, facial expression, speech, and language — with no manual annotation required.

## Documentation

See [MANUAL.md](MANUAL.md) for the full feature reference and output format details.

See [CHANGELOG.md](CHANGELOG.md) for version history.

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

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/ohdanieldw/socialcurrents.git
cd socialcurrents

# 2. Run the one-time setup (creates a conda env, installs all dependencies)
bash setup_macos.sh
```

Setup takes 5–15 minutes on first run. It creates a conda environment called `pipeline-env` and installs all Python packages automatically.

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

## Citation

If you use SocialCurrents in published research, please cite:

> Oh, D. D. (2026). *SocialCurrents: A multimodal feature extraction pipeline for social and behavioral research* (Version 0.1.0) [Software]. GitHub. https://github.com/ohdanieldw/socialcurrents

## Acknowledgments

Initial pipeline scaffolding by Kenneth Dao; testing and debugging by Shuo Duan. All subsequent development, integration, and testing by Daniel DongWon Oh.

## License

MIT — see [LICENSE](LICENSE).
Copyright © 2026 Daniel DongWon Oh
