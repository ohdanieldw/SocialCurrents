# Audio Models package

This sub-project isolates dependencies required by the audio and speech
feature extractors. Install it with Poetry to keep heavy libraries such as
`librosa`, `opensmile`, `torchaudio`, and `whisperx` out of the main
application environment until you explicitly opt in.

```bash
poetry install
poetry run python -m audio_models.scripts.setup_audio_repos
```

The second command clones the original research repositories referenced in
``requirements.csv`` (for example the Speech Emotion Recognition project from
`x4nth055/emotion-recognition-using-speech` and Facebook's `fairseq`).  The
wrappers inside ``audio_models.speech`` now rely on this local checkout rather
than on simplified reimplementations.

Runtime implementations now live inside the `audio_models` package itself
(`audio_models.audio`, `audio_models.speech`, and `audio_models.utils`). This
sub-project remains an isolated dependency sandbox so you can iterate on audio
code without polluting other modality stacks.

## Dependency highlights

- `torch>=2.6` and `torchaudio>=2.6` patch the CVE-2025-32434 vulnerability hit
	when loading Hubert-based checkpoints. Re-run `poetry install` after pulling
	these changes so the upgraded wheels are built.
- `scipy` is constrained to `>=1.11,<1.13` to match the updated PyTorch builds
	and to restore compatibility with `scipy.stats.binom_test` used by legacy
	modules.
- `scikit-learn` powers the speech emotion recogniser by loading the official
	`grid/best_classifiers.pickle` estimator exported from the upstream project.
- `transformers` + `sentencepiece` load the original Speech2Text and
	wav2vec2/XLSR checkpoints so speech transcription features run the genuine
	models instead of simplified surrogates.

Both the Speech2Text (`facebook/s2t-small-librispeech-asr`) and XLSR
(`facebook/wav2vec2-large-xlsr-53`) analyzers now require an internet
connection the first time they are executed so the pretrained weights can be
downloaded from Hugging Face.  Subsequent runs use the cached checkpoints.

## External repositories

The table below summarises the upstream projects currently pulled in by
``setup_audio_repos``.  Set the listed environment variable if you prefer to
point the pipeline at an existing local checkout instead of cloning again.

| Key | Repository | Environment override |
| --- | --- | --- |
| `emotion_recognition` | https://github.com/x4nth055/emotion-recognition-using-speech | `EMOTION_RECOGNITION_REPO` |
| `fairseq` | https://github.com/facebookresearch/fairseq | `FAIRSEQ_REPO` |
| `whisperx` | https://github.com/m-bain/whisperX | `WHISPERX_REPO` |
| `speechbrain` | https://github.com/speechbrain/speechbrain | `SPEECHBRAIN_REPO` |

When the variables are unset the repositories are cloned under
``<project_root>/external/audio/<repo-name>``.  The checkouts are kept outside
of the Python package itself so they remain editable and do not bloat the wheel
distribution when the project is packaged.
