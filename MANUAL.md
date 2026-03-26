# Multimodal Feature Extraction Pipeline — User Manual

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation (macOS)](#installation-macos)
4. [Quick Start](#quick-start)
5. [CLI Reference](#cli-reference)
6. [Selecting Features](#selecting-features)
7. [Output Format](#output-format)
8. [Feature Reference](#feature-reference)
   - [Audio Features](#audio-features)
   - [Speech Features](#speech-features)
   - [NLP / Text Features](#nlp--text-features)
   - [Vision Features](#vision-features)
9. [Optional & Heavy Features](#optional--heavy-features)
10. [Environment Variables](#environment-variables)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This pipeline extracts ~2,660+ features from video or audio recordings across four modalities:

| Modality | Examples |
|---|---|
| **Audio** | Volume, pitch, spectral descriptors, openSMILE LLDs |
| **Speech** | Transcription (WhisperX), emotion recognition, speaker diarization |
| **NLP / Text** | DeBERTa, SimCSE, ALBERT, Sentence-BERT, Universal Sentence Encoder |
| **Vision** | Pose (MediaPipe, ViTPose, OpenPose), facial AUs (Py-Feat, EmotiEffNet, GANimation), optical flow |

Inputs can be **video files** (`.mp4`, `.avi`, `.mov`, `.mkv`) or **audio files** (`.wav`, `.mp3`, `.flac`).

Each video gets its own subfolder under the output directory (e.g., `output/run/sub007/`). Outputs per video:
- **`features_timeseries.csv`** — one row per video frame; time-varying features (audio waveform statistics, pose landmarks, facial expressions) change row-by-row while scalar features are broadcast to every row
- **`features.csv`** — one row per input file; array-valued features are summarised as `_mean/_std/_min/_max` columns
- **`features.json`** — nested JSON grouped by model, with full arrays and metadata
- **`pipeline.log`** — execution log for the whole batch run, written at the top level of the output directory

---

## Requirements

| Requirement | Notes |
|---|---|
| **macOS** (Intel or Apple Silicon) | Tested on macOS 10.15+ |
| **Conda** (Anaconda or Miniconda) | [Install Miniconda](https://docs.conda.io/en/latest/miniconda.html) |
| **Homebrew** | For installing `ffmpeg` — [brew.sh](https://brew.sh) |
| **ffmpeg** | Installed automatically by `setup_macos.sh` if Homebrew is available |
| **Internet** | Required during setup to download models (~several GB on first run) |
| **Disk space** | ~10–20 GB for models and dependencies |

---

## Installation (macOS)

### Step 1 — Run the setup script

```bash
cd /path/to/pipeline
bash setup_macos.sh
```

This script will:
- Install `ffmpeg` via Homebrew (if not already present)
- Create a conda environment named **`pipeline-env`** with Python 3.11
- Install all Python dependencies via `pip`
- Install the four local packages (`core_pipeline`, `audio_models`, `cv_models`, `nlp_models`) in editable mode

The first run takes 5–15 minutes depending on your internet speed.

### Step 2 — (Optional) Set HuggingFace token

Speaker diarization (WhisperX) requires access to gated HuggingFace models. Export your token before running:

```bash
export HF_TOKEN="hf_your_token_here"
```

Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (free account required).
You can add this line to `~/.zshrc` or `~/.bash_profile` so it persists.

### Custom environment name

```bash
bash setup_macos.sh --name my-custom-env
```

If you use a custom name, edit the `ENV_NAME` variable at the top of `run_macos.sh` to match.

---

## Quick Start

```bash
# Run on a directory of videos (creates output/ subfolder automatically)
bash run_macos.sh -d data/

# Specify an output directory
bash run_macos.sh -d data/ -o results/session1/

# Process audio files instead of video
bash run_macos.sh -d data/ --is-audio

# Run only a subset of features (faster)
bash run_macos.sh -d data/ -f basic_audio,librosa_spectral,mediapipe_pose_vision

# List all available features
bash run_macos.sh --list-features

# Verify dependencies are installed
bash run_macos.sh --check-dependencies
```

Alternatively, activate the environment manually and run Python directly:

```bash
conda activate pipeline-env
python run_pipeline.py -d data/ -o output/
```

---

## CLI Reference

```
usage: python run_pipeline.py [-h] [-d DATA_DIR] [-o OUTPUT_DIR] [-f FEATURES]
                              [--list-features] [--is-audio] [--log-file LOG_FILE]
                              [--check-dependencies]
```

| Flag | Default | Description |
|---|---|---|
| `-d`, `--data-dir` | `data/` | Directory containing input video or audio files |
| `-o`, `--output-dir` | `output/YYYYMMDD_HHMMSS/` | Directory where results are saved |
| `-f`, `--features` | *(all features)* | Comma-separated list of feature names to extract |
| `--list-features` | — | Print all available feature names grouped by category, then exit |
| `--is-audio` | `False` | Treat input files as audio (`.wav`/`.mp3`/`.flac`) instead of video |
| `--log-file` | `<output_dir>/pipeline.log` | Path for the log file |
| `--check-dependencies` | — | Check whether required Python packages are installed, then exit |

### Supported input formats

| Type | Extensions |
|---|---|
| Video | `.mp4`, `.MP4`, `.avi`, `.mov`, `.MOV`, `.mkv` |
| Audio | `.wav`, `.mp3`, `.flac` |

---

## Selecting Features

Pass a comma-separated list to `-f` / `--features`:

```bash
# Audio only
python run_pipeline.py -d data/ -f basic_audio,librosa_spectral,opensmile

# Pose and face only
python run_pipeline.py -d data/ -f mediapipe_pose_vision,pyfeat_vision,emotieffnet_vision

# Quick text analysis on top of transcription
python run_pipeline.py -d data/ -f whisperx_transcription,deberta_text,sbert_text
```

Run `--list-features` to see all names.

---

## Output Format

After processing, the output directory contains:

```
output/
  pipeline.log          ← execution log (covers the whole batch run)
  sub007/               ← one subfolder per video (named after the file stem)
    features_timeseries.csv  ← time-indexed CSV (one row per video frame)
    features.csv             ← summary CSV (one row per file, arrays → stats columns)
    features.json            ← nested JSON with model groupings and full arrays
  sub003/
    features_timeseries.csv
    ...
```

---

### Time-series CSV (`features_timeseries.csv`)

The primary analysis output. Each row represents one video frame. Columns:

| Column | Description |
|---|---|
| `filename` | Source video filename |
| `frame_idx` | 0-based frame index |
| `time_seconds` | Timestamp in seconds (`frame_idx / fps`) |
| `oc_audvol`, `oc_audpit`, … | Audio arrays linearly interpolated from audio frame rate to video frame rate |
| `lbrs_*`, `osm_*` | Librosa / openSMILE LLD arrays, same interpolation |
| `GMP_land_*`, `GMP_world_*` | MediaPipe pose landmarks — one value per video frame (full resolution) |
| `pf_au*`, `pf_anger`, … | Py-Feat AUs / emotions — sampled from up to 120 frames, interpolated to fill all rows |
| `eln_prob_*`, `eln_valence`, `eln_arousal` | EmotiEffNet — sampled from up to 64 frames, interpolated |
| All other scalar features | Broadcast (same value repeated across all rows for that file) |

**Temporal resolution notes:**
- Audio arrays (librosa, openSMILE LLDs, basic audio) are captured at ~31 samples/sec (hop_length=512, sr=16000) and linearly interpolated to video frame rate.
- MediaPipe processes every video frame; each row has the exact landmark values for that frame.
- Py-Feat processes up to 120 evenly sampled frames; values between samples are linearly interpolated.
- EmotiEffNet processes up to 64 evenly sampled frames; values between samples are linearly interpolated.
- NLP features (transcription-based), AudioStretchy configuration parameters, and WhisperX summary statistics are scalars that do not vary over time.

```python
import pandas as pd

ts = pd.read_csv("output/sub007/features_timeseries.csv")

# Pose visibility for left wrist landmark over time
ts.plot(x="time_seconds", y="GMP_land_visi_16")

# Happiness and arousal as functions of time
ts[["time_seconds", "pf_happiness", "eln_arousal"]].set_index("time_seconds").plot()

# Audio volume trajectory
ts.plot(x="time_seconds", y="oc_audvol")
```

---

### Summary CSV (`features.csv`)

One row per input file. Array-valued features are summarised into statistics columns. Useful for file-level comparisons.

```python
import pandas as pd
df = pd.read_csv("output/sub007/features.csv", index_col="filename")
df["oc_audvol_mean"]      # mean volume across the recording
df["pf_happiness"]        # mean Py-Feat happiness (already a scalar in this file)
df["GMP_land_visi_26"]    # mean visibility of landmark 26 across all frames
```

**How arrays are stored in the summary CSV:**

| Feature type | Example input | CSV columns produced |
|---|---|---|
| Scalar | `GMP_land_visi_26` = `0.94` | `GMP_land_visi_26` = `0.94` |
| Long array (>20 elements) | `oc_audvol` = `[0.01, …]` (N values) | `oc_audvol_mean`, `_std`, `_min`, `_max` |
| Short array (≤20 elements) | `lbrs_spectral_contrast` = 7 values | `…_mean/_std/_min/_max` + `…_0` … `…_6` |
| String / transcript | `transcription` = `"hello world"` | `transcription` = `"hello world"` |
| Dict | `WhX_segment_1` = `{…}` | `WhX_segment_1` = JSON string |

---

### JSON (`features.json`)

Nested structure grouped by model. Large arrays (>1000 elements) are stored as statistics objects:

```json
{
  "video_filename": {
    "Audio volume": {
      "Feature": "Audio volume",
      "Model": "OpenCV",
      "features": {
        "oc_audvol": {
          "mean": 0.014, "min": 0.001, "max": 0.089, "std": 0.008,
          "shape": [4321], "dtype": "float32",
          "samples": [0.012, 0.013, ...]
        }
      }
    },
    "Pose Landmarks (MediaPipe)": {
      "features": {
        "GMP_land_x_1": 0.512,
        "GMP_land_visi_26": 0.94
      }
    }
  }
}
```

---

## Feature Reference

### Audio Features

#### `basic_audio` — Volume & Pitch
**Model:** librosa
**Prefix:** `oc_`
**Temporality:** time-varying — arrays at ~31 samples/sec, resampled per video frame in the time-series CSV

| Output key | Type | Description |
|---|---|---|
| `oc_audvol` | array | RMS energy (volume) per audio frame |
| `oc_audvol_diff` | array | Frame-to-frame volume change |
| `oc_audpit` | array | Pitch (fundamental frequency) per audio frame |
| `oc_audpit_diff` | array | Frame-to-frame pitch change |

---

#### `librosa_spectral` — Spectral, Pitch, Rhythm
**Model:** Librosa
**Prefix:** `lbrs_`
**Temporality:** time-varying arrays at ~31 samples/sec resampled per video frame; `lbrs_tempo` and `lbrs_*_singlevalue` keys are static scalars

| Output key | Type | Description |
|---|---|---|
| `lbrs_spectral_centroid` | array | Spectral centroid per audio frame |
| `lbrs_spectral_bandwidth` | array | Spectral bandwidth per audio frame |
| `lbrs_spectral_flatness` | array | Spectral flatness per audio frame |
| `lbrs_spectral_rolloff` | array | Spectral roll-off per audio frame |
| `lbrs_zero_crossing_rate` | array | Zero-crossing rate per audio frame |
| `lbrs_rmse` | array | RMS energy per audio frame |
| `lbrs_spectral_contrast` | array | Spectral contrast per audio frame |
| `lbrs_tempo` | scalar | Estimated tempo in BPM |
| `lbrs_*_singlevalue` | scalar | Mean of the corresponding time-series feature |

---

#### `opensmile` — Low-Level Descriptors & Functionals
**Model:** openSMILE ComParE_2016 + eGeMAPSv02
**Prefix:** `osm_`
**Total features:** ~1,512
**Temporality:** LLD keys (`osm_*_sma`) are time-varying arrays resampled per video frame; functional keys (`osm_*_mean`, `osm_*_stddev`, etc.) are static scalars

Key LLD outputs (time-varying, one value per audio frame):

| Output key | Description |
|---|---|
| `osm_pcm_RMSenergy_sma` | RMS energy |
| `osm_loudness_sma` | Perceptual loudness |
| `osm_F0final_sma` | Fundamental frequency (F0) |
| `osm_voicingProb_sma` | Voicing probability |
| `osm_jitterLocal_sma` | Pitch jitter |
| `osm_shimmerLocal_sma` | Amplitude shimmer |
| `osm_logHNR_sma` | Log harmonics-to-noise ratio |
| `osm_mfcc1_sma` … `osm_mfcc12_sma` | Mel-frequency cepstral coefficients 1–12 |
| `osm_spectralCentroid_sma` | Spectral centroid |
| `osm_spectralFlux_sma` | Spectral flux |
| `osm_spectralRollOff25_sma` … `osm_spectralRollOff90_sma` | Spectral roll-off at 25/50/75/90% |
| `osm_lsf1` … `osm_lsf8` | Line spectral frequencies 1–8 |

Functional statistics appended to each LLD (e.g. `osm_pcm_RMSenergy_sma_mean`):
`mean`, `stddev`, `skewness`, `kurtosis`, `min`, `max`, `range`, `percentile1`–`percentile99`, `quartile1`, `quartile3`, `interquartileRange`, `linregc1`, `linregc2`, `linregerr`

---

#### `audiostretchy` — Time-Stretching Analysis
**Model:** AudioStretchy
**Prefix:** `AS_`
**Temporality:** all static scalars (configuration and summary statistics)

| Output key | Description |
|---|---|
| `AS_ratio` | Time-stretch ratio (1.0 = unchanged) |
| `AS_gap_ratio` | Silence-stretch ratio |
| `AS_lower_freq` / `AS_upper_freq` | Analysis frequency bounds (Hz) |
| `AS_buffer_ms` | Buffer size (ms) |
| `AS_threshold_gap_db` | Silence detection threshold (dB) |
| `AS_fast_detection` | Fast detection mode flag |
| `AS_sample_rate` | Audio sample rate |
| `AS_input_nframes` / `AS_output_nframes` | Frame counts before/after stretch |
| `AS_input_duration_sec` / `AS_output_duration_sec` | Duration before/after stretch |

---

### Speech Features

#### `speech_emotion` — Speech Emotion Recognition
**Model:** SER (scikit-learn, trained on RAVDESS/TESS/EMO-DB)
**Prefix:** `ser_`
**Temporality:** static scalars (averaged over full recording)

| Output key | Description |
|---|---|
| `ser_neutral` | Probability: neutral |
| `ser_calm` | Probability: calm |
| `ser_happy` | Probability: happy |
| `ser_sad` | Probability: sad |
| `ser_angry` | Probability: angry |
| `ser_fear` | Probability: fear |
| `ser_disgust` | Probability: disgust |
| `ser_ps` | Probability: pleasant surprise |
| `ser_boredom` | Probability: boredom |

Probabilities sum to 1.0.

---

#### `whisperx_transcription` — Time-Accurate Transcription
**Model:** WhisperX (Whisper + forced alignment + speaker diarization)
**Prefix:** `WhX_`
**Requires:** `HF_TOKEN` env var for diarization
**Temporality:** static (full-recording transcript and per-segment/word dicts broadcast to all rows in the time-series CSV)

| Output key | Description |
|---|---|
| `transcription` | Full transcript text |
| `language` | Detected language code |
| `num_segments` | Number of speech segments |
| `WhX_segment_1` … `WhX_segment_N` | Per-segment dict: `text`, `speaker`, `start`, `end` |
| `WhX_speaker1_word_1` … | Per-word dict: `text`, `start`, `end`, `confidence` |
| `WhX_highlight_diarize_speaker1_word_1` … | Speaker-attributed words with timing |
| `WhX_speaker1_summary` | Per-speaker: `total_words`, `total_duration`, `avg_confidence` |

---

#### `speech_separation` — Source Separation
**Model:** SepFormer (SpeechBrain)
**Output:** Paths to separated speaker waveforms saved under `output/audio/separated/`

---

#### `xlsr_speech_to_text` — XLSR / Wav2Vec 2.0 Transcription
**Model:** Wav2Vec 2.0 / XLSR
**Prefix:** `xlsr_`

| Output key | Description |
|---|---|
| `xlsr_transcription` | Transcript text |
| `xlsr_hidden_states_path` | Path to saved hidden state tensors |
| `xlsr_num_hidden_frames` | Number of hidden-state frames |
| `xlsr_model_name` | Model identifier used |
| `xlsr_inference_device` | Device used (`cpu`/`cuda`) |
| `xlsr_fallback_used` | `True` if deterministic fallback was used |

---

#### `s2t_speech_to_text` — Speech-to-Text (S2T)
**Model:** Whisper-small via HuggingFace Transformers
**Prefix:** `s2t_`

| Output key | Description |
|---|---|
| `s2t_text` | Transcript text |
| `s2t_score` | Confidence score |
| `s2t_alignment_path` | Path to per-token alignment JSON |
| `s2t_confidence_path` | Path to per-token confidence tensor |
| `s2t_model_name` | Model identifier used |
| `s2t_fallback_used` | `True` if deterministic fallback was used |

---

### NLP / Text Features

> Text features require a transcription to be available. Run `whisperx_transcription` (or `xlsr_speech_to_text` / `s2t_speech_to_text`) first, or include it in the same `-f` list.
>
> **Temporality:** all NLP features are **static scalars** — they are computed once over the full transcript and broadcast to every row in the time-series CSV.

---

#### `heinsen_sentiment` — Sentiment Analysis
**Model:** Heinsen routing (logistic regressor over SER features)
**Prefix:** `arvs_`

| Output key | Description |
|---|---|
| `arvs_negative` | Probability: negative sentiment |
| `arvs_neutral` | Probability: neutral sentiment |
| `arvs_positive` | Probability: positive sentiment |
| `arvs_dominant_sentiment` | Dominant label (`negative`/`neutral`/`positive`) |
| `arvs_confidence` | Confidence of dominant prediction |

---

#### `meld_emotion` — Dialogue-Level Emotion (MELD)
**Model:** MELD dataset analytics
**Prefix:** `MELD_`

| Output key | Description |
|---|---|
| `MELD_num_utterances` | Number of utterances detected |
| `MELD_num_speakers` | Number of distinct speakers |
| `MELD_unique_words` | Vocabulary size |
| `MELD_avg_utterance_length` | Mean utterance word count |
| `MELD_count_anger/disgust/fear/joy/neutral/sadness/surprise` | Emotion occurrence counts |
| `MELD_avg_num_emotions_per_dialogue` | Average emotions per dialogue |
| `MELD_num_emotion_shift` | Number of emotion transitions |

---

#### `deberta_text` — DeBERTa Benchmark Features
**Model:** DeBERTa-v3 (microsoft/deberta-v3-base)
**Prefix:** `DEB_`

| Output key | Description |
|---|---|
| `DEB_SQuAD_1.1_F1` / `DEB_SQuAD_1.1_EM` | SQuAD 1.1 F1 and Exact Match |
| `DEB_SQuAD_2.0_F1` / `DEB_SQuAD_2.0_EM` | SQuAD 2.0 F1 and Exact Match |
| `DEB_MNLI-m_Acc` / `DEB_MNLI-mm_Acc` | MNLI matched/mismatched accuracy |
| `DEB_SST-2_Acc` | SST-2 sentiment accuracy |
| `DEB_QNLI_Acc` | QNLI accuracy |
| `DEB_CoLA_MCC` | CoLA Matthews correlation |
| `DEB_RTE_Acc` | RTE accuracy |
| `DEB_MRPC_Acc` / `DEB_MRPC_F1` | MRPC accuracy and F1 |
| `DEB_QQP_Acc` / `DEB_QQP_F1` | QQP accuracy and F1 |
| `DEB_STS-B_P` / `DEB_STS-B_S` | STS-B Pearson/Spearman correlation |

---

#### `simcse_text` — SimCSE Sentence Embeddings
**Model:** SimCSE (princeton-nlp/sup-simcse-bert-base-uncased)
**Prefix:** `CSE_`

| Output key | Description |
|---|---|
| `CSE_STS12` … `CSE_STS16` | Spearman correlation on STS12–STS16 benchmarks |
| `CSE_STSBenchmark` | STS Benchmark Spearman correlation |
| `CSE_SICKRelatedness` | SICK-Relatedness Spearman correlation |
| `CSE_Avg` | Average across all 7 benchmarks |

---

#### `albert_text` — ALBERT Language Representation
**Model:** ALBERT (albert-base-v2)
**Prefix:** `alb_`

| Output key | Description |
|---|---|
| `alb_mnli` | MNLI accuracy |
| `alb_qnli` | QNLI accuracy |
| `alb_qqp` | QQP F1 |
| `alb_rte` | RTE accuracy |
| `alb_sst` | SST-2 accuracy |
| `alb_mrpc` | MRPC F1 |
| `alb_cola` | CoLA MCC |
| `alb_sts` | STS-B Spearman |
| `alb_squad11dev` / `alb_squad20dev` / `alb_squad20test` | SQuAD 1.1/2.0 F1 |
| `alb_racetestmiddlehigh` | RACE accuracy |

---

#### `sbert_text` — Sentence-BERT
**Model:** sentence-transformers/all-MiniLM-L6-v2
**Prefix:** `BERT_`

| Output key | Description |
|---|---|
| `BERT_tensor_sentences` | Sentence embedding matrix |
| `BERT_tensor_sentences_shape` | Shape of embedding matrix |
| `BERT_tensor_paragraphs` | Paragraph-level embedding matrix |
| `BERT_score` | Cross-encoder similarity score |
| `BERT_ranks` | Reranking order |

---

#### `use_text` — Universal Sentence Encoder
**Model:** Google USE v4 (TensorFlow Hub) — *requires TensorFlow*
**Prefix:** `USE_`

| Output key | Description |
|---|---|
| `USE_embed_sentence1` … | 512-d embedding per sentence |
| `USE_embed_overall` | 512-d aggregate embedding |
| `USE_avg_cosine_similarity` | Mean pairwise cosine similarity |
| `USE_max_cosine_similarity` | Maximum pairwise cosine similarity |
| `USE_centroid_distance` | Mean distance from centroid |
| `USE_spread_variance` | Variance of embedding spread |

---

### Vision Features

> Vision features are extracted from the **video** directly (not the audio track). They are skipped when `--is-audio` is used.
>
> **Temporality:** features from MediaPipe, Py-Feat, and EmotiEffNet are **time-varying** and populate distinct values per row in the time-series CSV. Other vision models produce **static scalars** (averaged over frames) that are broadcast to all rows.

---

#### `mediapipe_pose_vision` — Pose Landmarks (MediaPipe)
**Model:** Google MediaPipe PoseLandmarker
**Prefix:** `GMP_`
**Total features:** 330+
**Temporality:** time-varying — processes every video frame; each row in the time-series CSV has the exact landmark values for that frame. The summary CSV shows per-file means.

33 body landmarks, each with 10 attributes:

| Attribute group | Keys | Description |
|---|---|---|
| Normalized coords | `GMP_land_x_1`…`33`, `GMP_land_y_1`…`33`, `GMP_land_z_1`…`33` | Normalized [0,1] image coordinates |
| Visibility / presence | `GMP_land_visi_1`…`33`, `GMP_land_presence_1`…`33` | Detection confidence per frame |
| World coords | `GMP_world_x_1`…`33`, `GMP_world_y_1`…`33`, `GMP_world_z_1`…`33` | Metric world coordinates (meters) |
| World vis / presence | `GMP_world_visi_1`…`33`, `GMP_world_presence_1`…`33` | World-space confidence |
| Visualization | `GMP_SM_pic` | Base64-encoded annotated frame (static, best-detected frame) |

Landmark index → body part mapping (0-indexed):
0=Nose, 11=Left shoulder, 12=Right shoulder, 13=Left elbow, 14=Right elbow, 15=Left wrist, 16=Right wrist, 23=Left hip, 24=Right hip, 25=Left knee, 26=Right knee, 27=Left ankle, 28=Right ankle, etc.

---

#### `pyfeat_vision` — Facial Expression Analysis (Py-Feat)
**Model:** Py-Feat
**Prefix:** `pf_`
**Total features:** 37
**Temporality:** time-varying — samples up to 120 frames evenly from the video; values are linearly interpolated to fill every row in the time-series CSV. The summary CSV shows per-file means.

| Group | Keys | Description |
|---|---|---|
| Action Units (20) | `pf_au01`…`pf_au43` | FACS AU intensities 0–1, varying per frame |
| Emotions (7) | `pf_anger`, `pf_disgust`, `pf_fear`, `pf_happiness`, `pf_sadness`, `pf_surprise`, `pf_neutral` | Emotion probabilities, varying per frame |
| Face geometry | `pf_facerectx`, `pf_facerecty`, `pf_facerectwidth`, `pf_facerectheight`, `pf_facescore` | Bounding box + detection confidence, varying per frame |
| Head pose | `pf_pitch`, `pf_roll`, `pf_yaw` | Head orientation angles (degrees), varying per frame |
| 3D position | `pf_x`, `pf_y`, `pf_z` | Estimated face position, varying per frame |

---

#### `emotieffnet_vision` — EmotiEffNet
**Model:** EmotiEffNet (sb-ai-lab)
**Prefix:** `eln_`
**Temporality:** time-varying — samples up to 64 frames evenly from the video; emotion/valence/arousal values are linearly interpolated to fill every row in the time-series CSV. `eln_top_emotion`, `eln_samples`, and `eln_face_detected_ratio` are static scalars.

| Output key | Type | Description |
|---|---|---|
| `eln_arousal` | time-varying | Continuous arousal (per sampled frame, interpolated) |
| `eln_valence` | time-varying | Continuous valence (per sampled frame, interpolated) |
| `eln_prob_{emotion}` | time-varying | Emotion class probability per sampled frame (e.g. `eln_prob_happiness`) |
| `eln_top_emotion` | static | Most frequent predicted emotion label across the recording |
| `eln_face_detected_ratio` | static | Fraction of sampled frames where a face was detected |
| `eln_samples` | static | Number of sampled frames with face detections |

---

#### `dan_vision` — DAN Facial Emotion
**Model:** DAN (Distillation-based Attentive Networks)
**Prefix:** `dan_`

| Output key | Description |
|---|---|
| `dan_neutral`, `dan_happiness`, `dan_sadness`, `dan_surprise`, `dan_fear`, `dan_disgust`, `dan_anger`, `dan_contempt` | Emotion probabilities |

---

#### `ganimation_vision` — GANimation Action Units
**Model:** GANimation
**Prefix:** `GAN_`
**Total features:** 68+ (17 AUs × 4 intensity levels)

`GAN_AU{n}_{level}` where `n` ∈ {1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45} and `level` ∈ {0, 33, 66, 99}

Additional:

| Output key | Description |
|---|---|
| `GAN_face_detected` | Whether a face was detected |
| `GAN_total_au_activations` | Count of activated AUs |
| `GAN_avg_au_intensity` | Mean intensity across all AUs |
| `GAN_SM_pic` | Base64-encoded annotated visualization |

---

#### `arbex_vision` — ARBEx Facial Expression
**Model:** ARBEx (Attentive Feature Extraction with Reliability Balancing)
**Prefix:** `arbex_`

| Output key | Description |
|---|---|
| `arbex_primary` | Initial emotion label |
| `arbex_final` | Final emotion label (after reliability balancing) |
| `arbex_primary_{emotion}` | Probability per emotion (primary stage) |
| `arbex_final_{emotion}` | Probability per emotion (final stage) |
| `arbex_confidence_primary` / `arbex_confidence_final` | Confidence scores |
| `arbex_reliability_score` | Feature reliability measure |

Emotions: `neutral`, `anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`, `others`

---

#### `me_graphau_vision` — ME-GraphAU Action Units
**Model:** ME-GraphAU
**Prefix:** `ann_`

Outputs graph-based AU relation features for 12 action units.

---

#### `openpose_vision` — OpenPose Keypoints
**Model:** CMU OpenPose
**Prefix:** `openPose_`
**Total features:** 50+
**Note:** Requires OpenPose C++ library to be compiled (optional — see below)

18 keypoints: nose, neck, r/l shoulder, r/l elbow, r/l wrist, r/l hip, r/l knee, r/l ankle, r/l eye, r/l ear.
Each keypoint: `openPose_{name}_x`, `openPose_{name}_y`, `openPose_{name}_confidence`

Additional:

| Output key | Description |
|---|---|
| `openPose_left_arm_angle`, `openPose_right_arm_angle` | Joint angle (degrees) |
| `openPose_left_leg_angle`, `openPose_right_leg_angle` | Joint angle (degrees) |
| `openPose_torso_angle` | Torso tilt angle |
| `openPose_shoulder_width`, `openPose_hip_width`, `openPose_body_height` | Body measurements (px) |
| `openPose_detection_rate` | Fraction of frames with a detected pose |
| `openPose_pose_video_path`, `openPose_pose_gif_path` | Paths to annotated output media |

---

#### `pare_vision` — PARE 3D Body Estimation
**Model:** PARE (Part Attention Regressor)
**Prefix:** `PARE_`

Outputs 3D body shape/pose parameters and per-part attention weights.

---

#### `vitpose_vision` — ViTPose
**Model:** ViTPose (Vision Transformer)
**Prefix:** `vit_`

17 COCO keypoints with confidence scores and AP/AR metrics.

---

#### `crowdflow_vision` — CrowdFlow Optical Flow
**Model:** CrowdFlow
**Prefix:** `of_`

Dense optical flow statistics: mean magnitude, max magnitude, flow direction histogram.

---

#### `optical_flow_vision` — Sparse/Dense Optical Flow
**Model:** OpenCV (Lucas-Kanade / Farnebäck)

Average motion magnitude, directional components, frame-to-frame flow statistics.

---

#### `instadm_vision` — Insta-DM Depth & Motion
**Model:** Insta-DM
**Prefix:** `indm_`

Depth map statistics and motion field descriptors.

---

#### `videofinder_vision` — VideoFinder
**Model:** Ollama-backed object/people localization
**Prefix:** `ViF_`
**Requires:** Ollama running locally (`ollama serve`)

---

#### `smoothnet_vision`, `lanegcn_vision`, `rsn_vision`, `deep_hrnet_vision`, `simple_baselines_vision`, `psa_vision`

Additional pose estimation and tracking models. See `checklist.md` for full output column listings.

---

#### `avhubert_vision` — AV-HuBERT Audio-Visual Embeddings
**Model:** AV-HuBERT
**Prefix:** `AVH_`

Joint audio-visual embeddings (requires video with synchronized audio).

---

#### `video_frames_vision` — Video Frame Extraction
**Model:** Frame sampler
**Prefix:** `VFE_`

Saves uniformly sampled frames to disk and records their paths and timestamps.

---

## Optional & Heavy Features

Some features require extra setup or are disabled by default:

| Feature | Requirement | How to enable |
|---|---|---|
| `openpose_vision` | OpenPose C++ library compiled from source | Build OpenPose, set `OPENPOSE_PYTHON_PATH` env var |
| `use_text` | TensorFlow (`pip install tensorflow tensorflow-hub`) | Install TF extras: `pip install -e packages/nlp_models[tensorflow-stack]` |
| `whisperx_transcription` diarization | HuggingFace token for `pyannote/speaker-diarization` | `export HF_TOKEN=hf_...` |
| `videofinder_vision` | Ollama running locally with a vision model | `ollama serve && ollama pull llava` |
| `speech_separation` | SpeechBrain (installed by setup script) | No extra action needed |

---

## Environment Variables

| Variable | Used by | Description |
|---|---|---|
| `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` | WhisperX diarization | HuggingFace access token |
| `PARE_CHECKPOINT` | `pare_vision` | Path to custom PARE model checkpoint |
| `VITPOSE_CHECKPOINT` | `vitpose_vision` | Path to custom ViTPose checkpoint |
| `VITPOSE_CONFIG` | `vitpose_vision` | Path to custom ViTPose config file |
| `OPENPOSE_TARGET` | `run_all.sh` | Data directory for the OpenPose stage |

---

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
Or manually:
```bash
conda activate pipeline-env
pip install -e packages/core_pipeline -e packages/audio_models -e packages/cv_models -e packages/nlp_models
```

### `Warning: whisperx_transcription unavailable`
WhisperX or one of its dependencies failed to install. Check the setup log. You can still run all other features.

### `Warning: mediapipe_pose_vision unavailable`
MediaPipe model file (`.task`) may not have downloaded. It is fetched automatically on first use; ensure you have internet access.

### Models download on first run
Many models (Whisper, DeBERTa, MediaPipe, etc.) are downloaded from HuggingFace or the internet on first use. This can take several minutes. Subsequent runs use the cached versions.
Default cache location: `~/.cache/huggingface/hub/`

### Running out of memory
Use `-f` to select only the features you need:
```bash
bash run_macos.sh -d data/ -f basic_audio,librosa_spectral,mediapipe_pose_vision
```
Vision models are the heaviest consumers. Audio-only runs are much lighter.

### `conda: command not found` inside a script
Source conda's shell hook before running:
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pipeline-env
python run_pipeline.py -d data/
```
