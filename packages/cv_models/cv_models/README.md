# Computer Vision Models package

Install this sub-project to provision the heavyweight computer vision stack
required by pose estimation, facial analysis, and optical flow components.
Keeping it in its own Poetry environment prevents clashes with audio or NLP
packages that prefer different versions of PyTorch or OpenCV.

```bash
poetry install
```

All vision analyzers have been relocated into `cv_models.vision`, so this
package now contains both the dependency sandbox and the runtime modules.

## Updated binary wheels

- Vision features now depend on `torch>=2.6` and `torchvision>=0.21` to align
  with the audio stack and to satisfy upstream security mitigations.
- `scipy` is capped at `<1.13` because several analyzers still rely on
  soon-to-be-deprecated APIs such as `scipy.stats.bayes_mvs`.
- After syncing, run `poetry install` here (and optionally `poetry lock`) so
  the new versions are compiled before you launch the pipeline.
- FACT now pulls in [`FacePerceiver/facer`](https://github.com/FacePerceiver/facer)
  for RetinaFace detection and FaRL alignment. The analyzer automatically
  combines FaCER crops with Py-Feat's AU head; export `CV_FACT_FORCE_LEGACY=1`
  if you need to fall back to the old Haar/MediaPipe heuristic. Installing
  `mmsegmentation` is optional but enables the dense 68-point alignment model.

## Upstream vision analyzers

- SmoothNet now runs the official repo from `external/vision/SmoothNet` and
  smooths ViTPose keypoints automatically. Provide `SMOOTHNET_CHECKPOINT` to
  override the default `*.pth` in that tree.
- ARBEx is backed by the upstream POSTER + classification stack housed in
  `external/vision/ARBEx`. Place the published checkpoints under that
  directory or set the env vars `ARBEX_POSTER_WEIGHTS`,
  `ARBEX_CLASSIFIER_WEIGHTS`, `ARBEX_ANCHORS_WEIGHTS`, and optionally
  `ARBEX_SELF_ATTENTION_WEIGHTS` when using custom paths.
  The analyzer saves frame-level probability arrays to
  `output/vision/arbex` for downstream inspection.
- AV-HuBERT loads the official `facebook/avhubert-large-30h-cv` checkpoint via
  `transformers`.  Ensure `ffmpeg`, `soundfile`, and `librosa` are installed and
  optionally set `AVHUBERT_MODEL_ID` to override the default Hugging Face ID.
- CrowdFlow replaces the synthetic Farnebäck heuristic with the RAFT model
  bundled in `torchvision`. Install `torchvision>=0.15` with the RAFT weights and
  keep `mediapipe` available for foreground masks. GPU inference is supported
  when `torch.cuda.is_available()`.
- Deep HRNet now clones `deep-high-resolution-net.pytorch`, loading the official
  HRNet-W32 config and checkpoints. Set `HRNET_CONFIG_PATH` or
  `HRNET_CHECKPOINT_PATH` to override the defaults and install the repo's Python
  requirements (`yacs`, `opencv-python`, etc.) before running pose inference.
