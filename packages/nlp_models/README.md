# NLP Models package

This Poetry project isolates the transformer and TensorFlow stacks that power
text, sentiment, and dialogue analysis features. Use it whenever you need to
work on `nlp_models.text` or `nlp_models.emotion` modules without pulling those
heavy requirements into unrelated environments.

```bash
poetry install
```

Dependencies can diverge safely from the ones used in `audio_models` or
`cv_models`, letting you upgrade HuggingFace tooling without breaking vision
pipelines.

## Version guardrails

- PyTorch-dependent analyzers require `torch>=2.6` (shared with the other
	modality packages). Installing this project will also pull the aligned
	`scipy>=1.11,<1.13` build.
- Transformers pipelines now depend on `tf-keras>=2.15` so they can run even if
	Keras 3 is present in the base environment. Poetry will install it alongside
	the existing optional TensorFlow extras.
