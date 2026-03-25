"""SimCSE analyzer that reports CSE_* metrics using real embeddings."""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

__all__ = ["SimCSEAnalyzer"]


class SimCSEAnalyzer:
    """Generate STS-style metrics by running SimCSE embeddings over text variants."""

    BENCHMARK_ORDER = (
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STSBenchmark",
        "SICKRelatedness",
    )

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "princeton-nlp/sup-simcse-bert-base-uncased",
        max_length: int = 256,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.max_length = max_length

        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None

        try:
            self._load_model()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("SimCSE model (%s) load failed: %s", model_name, exc)

    def _load_model(self) -> None:
        logger.info("Loading SimCSE backbone %s", self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def _encode(self, sentences: Iterable[str]) -> Dict[str, torch.Tensor]:
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("SimCSE model is not initialized")

        unique_sentences = list(dict.fromkeys(sentence for sentence in sentences if sentence.strip()))
        embeddings: Dict[str, torch.Tensor] = {}
        for start in range(0, len(unique_sentences), 12):
            batch = unique_sentences[start : start + 12]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0]
                normalized = F.normalize(cls_embeddings, p=2, dim=1)
            for sentence, vector in zip(batch, normalized):
                embeddings[sentence] = vector.cpu()
        return embeddings

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text.strip()) if segment.strip()]
        if not sentences and text.strip():
            sentences = [text.strip()]
        return sentences

    @staticmethod
    def _word_rotate(sentence: str) -> str:
        words = sentence.split()
        if len(words) < 3:
            return sentence
        return " ".join(words[1:] + words[:1])

    @staticmethod
    def _lexical_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _negate_sentence(sentence: str) -> str:
        if " not " in sentence.lower():
            return sentence.replace(" not ", " ", 1)
        if "n't" in sentence.lower():
            return sentence.replace("n't", "", 1)
        return sentence + " not"

    def _construct_pairs(self, text: str) -> Dict[str, List[Tuple[str, str, float]]]:
        sentences = self._split_sentences(text)
        if not sentences:
            return defaultdict(list)

        dataset_pairs: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)

        for sentence in sentences:
            lower = sentence.lower()
            dataset_pairs["STS12"].append((sentence, lower, self._lexical_ratio(sentence, lower)))

            rotated = self._word_rotate(sentence)
            dataset_pairs["STS13"].append((sentence, rotated, self._lexical_ratio(sentence, rotated)))

            simplified = re.sub(r"\bthe\b", "a", sentence, flags=re.IGNORECASE)
            dataset_pairs["STS14"].append((sentence, simplified, self._lexical_ratio(sentence, simplified)))

            expanded = sentence + " in this context."
            dataset_pairs["STS15"].append((sentence, expanded, self._lexical_ratio(sentence, expanded)))

            negated = self._negate_sentence(sentence)
            dataset_pairs["STS16"].append((sentence, negated, self._lexical_ratio(sentence, negated)))

        if len(sentences) == 1:
            repeated = sentences[0] + " " + sentences[0]
            dataset_pairs["STSBenchmark"].append(
                (sentences[0], repeated, self._lexical_ratio(sentences[0], repeated))
            )
            inverted = sentences[0][::-1]
            dataset_pairs["SICKRelatedness"].append(
                (sentences[0], inverted, self._lexical_ratio(sentences[0], inverted))
            )
        else:
            for idx in range(len(sentences) - 1):
                first, second = sentences[idx], sentences[idx + 1]
                dataset_pairs["STSBenchmark"].append(
                    (first, second, self._lexical_ratio(first, second))
                )
                question = f"What happens when {second.lower()}"
                dataset_pairs["SICKRelatedness"].append(
                    (first, question, self._lexical_ratio(first, question))
                )

        return dataset_pairs

    @staticmethod
    def _correlation(predictions: List[float], targets: List[float]) -> float:
        if not predictions:
            return 0.0
        if len(predictions) == 1 or len(set(targets)) < 2:
            return float(predictions[0])
        pearson = pearsonr(predictions, targets)[0]
        spearman = spearmanr(predictions, targets)[0]
        values = [value for value in (pearson, spearman) if not np.isnan(value)]
        if not values:
            return float(np.mean(predictions))
        return float(np.mean(values))

    def _score_dataset(self, pairs: List[Tuple[str, str, float]]) -> float:
        if not pairs:
            return 0.0
        sentences = [sentence for pair in pairs for sentence in pair[:2]]
        embeddings = self._encode(sentences)
        predictions: List[float] = []
        targets: List[float] = []
        for sentence_a, sentence_b, target in pairs:
            if sentence_a not in embeddings or sentence_b not in embeddings:
                continue
            vector_a = embeddings[sentence_a]
            vector_b = embeddings[sentence_b]
            similarity = F.cosine_similarity(vector_a.unsqueeze(0), vector_b.unsqueeze(0)).item()
            predictions.append(similarity)
            targets.append(target)
        return self._correlation(predictions, targets)

    def _get_default_metrics(self) -> Dict[str, float]:
        return {
            "CSE_STS12": 0.0,
            "CSE_STS13": 0.0,
            "CSE_STS14": 0.0,
            "CSE_STS15": 0.0,
            "CSE_STS16": 0.0,
            "CSE_STSBenchmark": 0.0,
            "CSE_SICKRelatedness": 0.0,
            "CSE_Avg": 0.0,
            "CSE_analysis_timestamp": time.time(),
            "CSE_text_length": 0,
            "CSE_word_count": 0,
            "CSE_model_name": self.model_name,
            "CSE_device": self.device,
        }

    def analyze_text(self, text: str) -> Dict[str, float]:
        if not text or not text.strip():
            logger.warning("Empty text provided for SimCSE analysis")
            return self._get_default_metrics()
        if self._tokenizer is None or self._model is None:
            logger.error("SimCSE backbone not initialized")
            return self._get_default_metrics()

        dataset_pairs = self._construct_pairs(text)
        scores: Dict[str, float] = {}
        collected: List[float] = []
        for dataset_name in self.BENCHMARK_ORDER:
            value = self._score_dataset(dataset_pairs.get(dataset_name, []))
            scores[f"CSE_{dataset_name}"] = value
            collected.append(value)

        scores["CSE_Avg"] = float(np.mean(collected)) if collected else 0.0
        scores["CSE_analysis_timestamp"] = time.time()
        scores["CSE_text_length"] = len(text)
        scores["CSE_word_count"] = len(text.split())
        scores["CSE_model_name"] = self.model_name
        scores["CSE_device"] = self.device
        return scores

    def get_feature_dict(self, text_or_features: Union[str, Dict[str, Any]]) -> Dict[str, float]:
        text = ""
        if isinstance(text_or_features, str):
            text = text_or_features
        elif isinstance(text_or_features, dict):
            for key in (
                "transcription",
                "whisperx_transcription",
                "transcript",
                "text",
                "speech_text",
                "transcribed_text",
            ):
                value = text_or_features.get(key)
                if isinstance(value, str) and value.strip():
                    text = value
                    break
                if isinstance(value, dict):
                    nested = value.get("text") or value.get("transcription")
                    if isinstance(nested, str) and nested.strip():
                        text = nested
                        break
        if not text or not text.strip():
            logger.warning("No text found for SimCSE analysis")
            return self._get_default_metrics()
        return self.analyze_text(text)

    @staticmethod
    def get_available_features() -> List[str]:
        return [
            "CSE_STS12",
            "CSE_STS13",
            "CSE_STS14",
            "CSE_STS15",
            "CSE_STS16",
            "CSE_STSBenchmark",
            "CSE_SICKRelatedness",
            "CSE_Avg",
            "CSE_analysis_timestamp",
            "CSE_text_length",
            "CSE_word_count",
            "CSE_model_name",
            "CSE_device",
        ]

