# SocialCurrents

SocialCurrents is a multimodal feature extraction pipeline for social and behavioral research. Given a folder of video recordings (e.g., dyadic interactions, interviews, group conversations), it produces time-stamped, analysis-ready feature files covering body movement, facial expression, speech, and language — with no manual annotation required.

## What it measures

| Modality | What you get |
|---|---|
| **Body movement** | 33 pose landmarks per frame (MediaPipe), joint velocities, movement magnitude |
| **Facial expression** | Action units (AU1–AU28), valence/arousal, discrete emotion probabilities (Py-Feat, EmotiEffNet) |
| **Speech** | Pitch, volume, spectral features (librosa, openSMILE); speech emotion recognition |
| **Language** | Transcript (WhisperX with speaker diarization), sentiment, semantic similarity, NLI benchmarks |

## Output files

For each recording session, the pipeline writes three files to your output directory:

| File | What it is |
|---|---|
| `pipeline_features_timeseries.csv` | One row per video frame — pose, facial, and audio features varying over time |
| `pipeline_features.csv` | One row per recording — per-feature summary statistics (mean, SD, min, max) |
| `pipeline_features.json` | Full nested JSON with raw arrays and model metadata |

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
bash run_macos.sh -d data/ -o output/

# Run only specific feature extractors (faster)
bash run_macos.sh -d data/ -o output/ -f basic_audio,mediapipe_pose_vision,pyfeat_vision

# See all 35 available extractors
bash run_macos.sh --list-features
```

For WhisperX speaker diarization, export your HuggingFace token first:

```bash
export HF_TOKEN=hf_your_token_here
bash run_macos.sh -d data/ -o output/
```

### Common workflows

**"I want movement and facial data only"**
```bash
bash run_macos.sh -d data/ -f mediapipe_pose_vision,pyfeat_vision,emotieffnet_vision
```

**"I want speech and language features"**
```bash
bash run_macos.sh -d data/ -f basic_audio,librosa_spectral,whisperx_transcription,deberta_text,sbert_text
```

**"I want everything"**
```bash
bash run_macos.sh -d data/ -o output/
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
| `deberta_text` | NLP | `DEB_` |
| `simcse_text` | NLP | `CSE_` |
| `albert_text` | NLP | `alb_` |
| `sbert_text` | NLP | `BERT_` |
| `use_text` | NLP | `USE_` |
| `mediapipe_pose_vision` | Pose | `GMP_` |
| `vitpose_vision` | Pose | `vit_` |
| `pyfeat_vision` | Facial | `pf_` |
| `emotieffnet_vision` | Facial | `eln_` |
| `dan_vision` | Facial | `dan_` |
| `ganimation_vision` | Facial | `GAN_` |
| `arbex_vision` | Facial | `arbex_` |
| `crowdflow_vision` | Video | `of_` |
| `instadm_vision` | Video | `indm_` |
| *(+ 15 more)* | | |

Run `bash run_macos.sh --list-features` for the full list with descriptions.

## Documentation

See [MANUAL.md](MANUAL.md) for the full feature reference, output format details, and per-feature temporality notes (which features are time-varying vs. summary scalars).

## Citation

If you use SocialCurrents in published research, please cite:

> Oh, D. D. (2026). *SocialCurrents: A multimodal feature extraction pipeline for social and behavioral research* [Software]. GitHub. https://github.com/ohdanieldw/socialcurrents

## License

MIT — see [LICENSE](LICENSE).
Copyright © 2026 Daniel DongWon Oh
