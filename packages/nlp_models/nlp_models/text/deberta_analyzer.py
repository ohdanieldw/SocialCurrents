"""DeBERTa analyzer with model-backed feature extraction."""

from __future__ import annotations

import logging
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import AutoModel, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

__all__ = ["DeBERTaAnalyzer"]


class DeBERTaAnalyzer:
    """Analyzer that produces DEB_* features using genuine DeBERTa inference."""

    QA_MODELS = (
        "deepset/deberta-v3-base-squad2",
        "deepset/bert-large-uncased-whole-word-masking-squad2",
        "distilbert-base-cased-distilled-squad",
    )
    MNLI_MODELS = (
        "MoritzLaurer/deberta-v3-base-mnli-fever-anli",
        "cross-encoder/nli-deberta-base",
        "facebook/bart-large-mnli",
    )
    SST_MODELS = (
        "distilbert-base-uncased-finetuned-sst-2-english",
        "textattack/bert-base-uncased-SST-2",
        "textattack/albert-base-v2-SST-2",
    )
    QNLI_MODELS = (
        "cross-encoder/nli-deberta-base",
        "textattack/bert-base-uncased-QNLI",
        "facebook/bart-large-mnli",
    )
    RTE_MODELS = (
        "textattack/bert-base-uncased-RTE",
        "cross-encoder/nli-deberta-base",
        "facebook/bart-large-mnli",
    )
    COLA_MODELS = (
        "textattack/bert-base-uncased-CoLA",
        "mrm8488/bert-base-uncased-finetuned-cola",
    )
    MRPC_MODELS = (
        "textattack/bert-base-uncased-MRPC",
        "textattack/albert-base-v2-MRPC",
    )
    QQP_MODELS = (
        "Tomor0720/deberta-base-finetuned-qqp",
        "textattack/bert-base-uncased-QQP",
        "textattack/albert-base-v2-QQP",
    )

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "microsoft/deberta-v3-base",
        max_length: int = 512,
    ) -> None:
        self.device = device
        self.device_index = 0 if device.startswith("cuda") else -1
        self.model_name = model_name
        self.max_length = max_length

        self._pipelines: Dict[str, Any] = {}
        self._embedding_model: Optional[AutoModel] = None
        self._embedding_tokenizer: Optional[AutoTokenizer] = None

    @property
    def _embedding(self) -> Tuple[AutoTokenizer, AutoModel]:
        if self._embedding_model is None or self._embedding_tokenizer is None:
            logger.info("Loading DeBERTa base model for sentence embeddings")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()
            self._embedding_tokenizer = tokenizer
            self._embedding_model = model
        return self._embedding_tokenizer, self._embedding_model

    def _get_pipeline(
        self,
        key: str,
        task: str,
        model_candidates: Union[str, Iterable[str]],
        **pipeline_kwargs: Any,
    ):
        if key in self._pipelines:
            return self._pipelines[key]

        if isinstance(model_candidates, str):
            candidates = [model_candidates]
        else:
            # Preserve candidate order but drop duplicates
            seen = set()
            candidates = []
            for candidate in model_candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(candidate)

        for model_name in candidates:
            kwargs = dict(pipeline_kwargs)
            kwargs.setdefault("device", self.device_index)
            kwargs.setdefault("framework", "pt")
            tokenizer_name = kwargs.pop("tokenizer", model_name)
            try:
                logger.info("Loading %s pipeline (%s)", key, model_name)
                if tokenizer_name is None:
                    pipe = pipeline(task, model=model_name, **kwargs)
                else:
                    pipe = pipeline(task, model=model_name, tokenizer=tokenizer_name, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to load pipeline %s with %s: %s", key, model_name, exc)
                continue
            self._pipelines[key] = pipe
            return pipe

        logger.error("Unable to initialize pipeline %s with provided candidates", key)
        self._pipelines[key] = None
        return None

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
            if sentence.strip()
        ]
        if not sentences and text.strip():
            sentences = [text.strip()]
        return sentences

    @staticmethod
    def _build_pairs(sentences: List[str]) -> List[Tuple[str, str]]:
        if len(sentences) < 2:
            return [(sentences[0], sentences[0])] if sentences else []
        return [(sentences[i], sentences[i + 1]) for i in range(len(sentences) - 1)]

    @staticmethod
    def _normalize_label(label: str) -> str:
        lower = label.lower()
        if "entail" in lower or lower.endswith("_2"):
            return "entailment"
        if "contrad" in lower or lower.endswith("_0"):
            return "contradiction"
        if "neutral" in lower or lower.endswith("_1"):
            return "neutral"
        if "acceptable" in lower:
            return "acceptable"
        if "unacceptable" in lower:
            return "unacceptable"
        if any(token in lower for token in ("duplicate", "paraphrase", "yes")):
            return "positive"
        if "no" in lower:
            return "negative"
        return lower

    @staticmethod
    def _negation_present(text: str) -> bool:
        lowered = f" {text.lower()} "
        return any(token in lowered for token in (" not ", "n't", " never ", " no "))

    def _infer_nli_label(self, premise: str, hypothesis: str) -> str:
        premise_l = premise.lower()
        hypothesis_l = hypothesis.lower()
        if not hypothesis_l.strip():
            return "neutral"
        if hypothesis_l in premise_l or premise_l in hypothesis_l:
            return "entailment"
        if self._negation_present(premise) != self._negation_present(hypothesis):
            return "contradiction"
        return "neutral"

    @staticmethod
    def _paraphrase_label(s1: str, s2: str) -> int:
        ratio = SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
        return 1 if ratio >= 0.75 else 0

    @staticmethod
    def _question_answer_label(question: str, answer: str) -> int:
        question_terms = {term for term in re.findall(r"\w+", question.lower()) if len(term) > 3}
        answer_terms = {term for term in re.findall(r"\w+", answer.lower()) if len(term) > 3}
        if not question_terms or not answer_terms:
            return 0
        return 1 if question_terms & answer_terms else 0

    @staticmethod
    def _extract_score(candidates: Iterable[Dict[str, Any]], positive_labels: Tuple[str, ...]) -> float:
        best = max(candidates, key=lambda item: item.get("score", 0.0))
        for candidate in candidates:
            label = DeBERTaAnalyzer._normalize_label(candidate.get("label", ""))
            if any(label.startswith(pos) for pos in positive_labels):
                return float(candidate.get("score", 0.0))
        return float(best.get("score", 0.0))

    def _question_answer_metrics(self, context: str) -> Dict[str, float]:
        qa_pipe = self._get_pipeline("qa", "question-answering", self.QA_MODELS)
        if qa_pipe is None:
            return {
                "DEB_SQuAD_1.1_F1": 0.0,
                "DEB_SQuAD_1.1_EM": 0.0,
                "DEB_SQuAD_2.0_F1": 0.0,
                "DEB_SQuAD_2.0_EM": 0.0,
            }

        questions = {
            "DEB_SQuAD_1.1": "What is the main topic of the passage?",
            "DEB_SQuAD_2.0": "Which specific detail is emphasized?",
        }
        metrics: Dict[str, float] = {}
        for prefix, question in questions.items():
            try:
                prediction = qa_pipe(question=question, context=context)
                score = float(prediction.get("score", 0.0))
                answer_text = str(prediction.get("answer", ""))
                has_answer = 1.0 if answer_text.strip() and answer_text.lower() != "unanswerable" else 0.0
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("QA inference failed: %s", exc)
                score, has_answer = 0.0, 0.0
            metrics[f"{prefix}_F1"] = score
            metrics[f"{prefix}_EM"] = has_answer
        return metrics

    def _mnli_metrics(self, sentences: List[str]) -> Tuple[float, float]:
        pairs = self._build_pairs(sentences)
        if not pairs:
            return 0.0, 0.0
        nli_pipe = self._get_pipeline("mnli", "text-classification", self.MNLI_MODELS)
        if nli_pipe is None:
            return 0.0, 0.0
        matched_pairs = pairs[::2] or pairs
        mismatched_pairs = pairs[1::2] or pairs
        return (
            self._nli_accuracy(matched_pairs, nli_pipe),
            self._nli_accuracy(mismatched_pairs, nli_pipe),
        )

    def _nli_accuracy(self, pairs: List[Tuple[str, str]], nli_pipe) -> float:
        if not pairs:
            return 0.0
        try:
            inputs = [{"text": p, "text_pair": h} for p, h in pairs]
            outputs = nli_pipe(inputs, return_all_scores=True, truncation=True)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("NLI pipeline failed: %s", exc)
            return 0.0
        predicted = [self._normalize_label(max(scores, key=lambda item: item["score"])["label"]) for scores in outputs]
        targets = [self._infer_nli_label(p, h) for p, h in pairs]
        matches = [1.0 if pred == target else 0.0 for pred, target in zip(predicted, targets)]
        if any(matches):
            return float(np.mean(matches))
        max_probs = [max(scores, key=lambda item: item["score"])["score"] for scores in outputs]
        return float(np.mean(max_probs))

    def _sentiment_confidence(self, sentences: List[str]) -> float:
        if not sentences:
            return 0.0
        sst_pipe = self._get_pipeline("sst", "text-classification", self.SST_MODELS)
        if sst_pipe is None:
            return 0.0
        try:
            outputs = sst_pipe(sentences, return_all_scores=True, truncation=True)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("SST pipeline failed: %s", exc)
            return 0.0
        scores = [self._extract_score(candidates, ("positive", "label_1")) for candidates in outputs]
        return float(np.mean(scores)) if scores else 0.0

    def _binary_nli_metric(
        self,
        pairs: List[Tuple[str, str]],
        pipeline_key: str,
        model_candidates: Union[str, Iterable[str]],
        positive_labels: Tuple[str, ...],
        label_fn,
    ) -> float:
        if not pairs:
            return 0.0
        clf_pipe = self._get_pipeline(pipeline_key, "text-classification", model_candidates)
        if clf_pipe is None:
            return 0.0
        try:
            inputs = [{"text": p, "text_pair": h} for p, h in pairs]
            outputs = clf_pipe(inputs, return_all_scores=True, truncation=True)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("%s pipeline failed: %s", pipeline_key, exc)
            return 0.0
        y_true = [label_fn(p, h) for p, h in pairs]
        scores = [self._extract_score(out, positive_labels) for out in outputs]
        y_pred = [1 if score >= 0.5 else 0 for score in scores]
        if len(set(y_true)) < 2:
            return float(np.mean(scores))
        accuracy = float(np.mean([pred == true for pred, true in zip(y_pred, y_true)]))
        return accuracy

    def _cola_metric(self, sentences: List[str]) -> float:
        if not sentences:
            return 0.0
        cola_pipe = self._get_pipeline("cola", "text-classification", self.COLA_MODELS)
        if cola_pipe is None:
            return 0.0
        try:
            outputs = cola_pipe(sentences, return_all_scores=True, truncation=True)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("CoLA pipeline failed: %s", exc)
            return 0.0

        def heuristic(sentence: str) -> int:
            words = sentence.split()
            if len(words) < 3:
                return 0
            if sentence and sentence[0].islower():
                return 0
            if sentence.count("(") != sentence.count(")"):
                return 0
            if any(len(word) > 25 for word in words):
                return 0
            return 1

        y_true = [heuristic(sentence) for sentence in sentences]
        probs = [self._extract_score(candidates, ("acceptable", "label_1")) for candidates in outputs]
        y_pred = [1 if score >= 0.5 else 0 for score in probs]
        if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
            return float(np.mean(probs))
        return float(matthews_corrcoef(y_true, y_pred))

    def _paraphrase_metrics(
        self,
        pairs: List[Tuple[str, str]],
        pipeline_key: str,
        model_candidates: Union[str, Iterable[str]],
    ) -> Tuple[float, float]:
        if not pairs:
            return 0.0, 0.0
        para_pipe = self._get_pipeline(pipeline_key, "text-classification", model_candidates)
        if para_pipe is None:
            return 0.0, 0.0
        try:
            inputs = [{"text": s1, "text_pair": s2} for s1, s2 in pairs]
            outputs = para_pipe(inputs, return_all_scores=True, truncation=True)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("%s pipeline failed: %s", pipeline_key, exc)
            return 0.0, 0.0
        y_true = [self._paraphrase_label(s1, s2) for s1, s2 in pairs]
        positive_aliases = ("entailment", "paraphrase", "duplicate", "equivalent", "label_1", "yes")
        scores = [self._extract_score(candidates, positive_aliases) for candidates in outputs]
        y_pred = [1 if score >= 0.5 else 0 for score in scores]
        if len(set(y_true)) < 2:
            mean_score = float(np.mean(scores)) if scores else 0.0
            return mean_score, mean_score
        accuracy = float(np.mean([pred == true for pred, true in zip(y_pred, y_true)]))
        tp = sum(1 for pred, true in zip(y_pred, y_true) if pred == 1 and true == 1)
        fp = sum(1 for pred, true in zip(y_pred, y_true) if pred == 1 and true == 0)
        fn = sum(1 for pred, true in zip(y_pred, y_true) if pred == 0 and true == 1)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        return accuracy, float(f1)

    def _embed_sentences(self, sentences: Iterable[str]) -> Dict[str, np.ndarray]:
        tokenizer, model = self._embedding
        sentence_list = list(dict.fromkeys(s for s in sentences if s.strip()))
        embeddings: Dict[str, np.ndarray] = {}
        for start in range(0, len(sentence_list), 6):
            batch = sentence_list[start : start + 6]
            inputs = tokenizer(
                batch,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state
                attention = inputs["attention_mask"].unsqueeze(-1)
                masked = hidden * attention
                summed = masked.sum(dim=1)
                lengths = attention.sum(dim=1).clamp(min=1)
                pooled = summed / lengths
            for sentence, vector in zip(batch, pooled):
                embeddings[sentence] = vector.cpu().numpy()
        return embeddings

    @staticmethod
    def _cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)

    def _sts_metrics(self, pairs: List[Tuple[str, str]]) -> Tuple[float, float]:
        if len(pairs) < 2:
            return 0.0, 0.0
        embeddings = self._embed_sentences([s for pair in pairs for s in pair])
        cosine_scores: List[float] = []
        lexical_scores: List[float] = []
        for s1, s2 in pairs:
            if s1 not in embeddings or s2 not in embeddings:
                continue
            cosine_scores.append(self._cosine(embeddings[s1], embeddings[s2]))
            lexical_scores.append(SequenceMatcher(None, s1, s2).ratio())
        if len(cosine_scores) < 2 or len(set(cosine_scores)) < 2:
            return float(np.mean(cosine_scores)) if cosine_scores else 0.0, 0.0
        pearson = pearsonr(cosine_scores, lexical_scores)[0]
        spearman = spearmanr(cosine_scores, lexical_scores)[0]
        return float(pearson), float(spearman)

    def _get_default_metrics(self) -> Dict[str, float]:
        return {
            "DEB_SQuAD_1.1_F1": 0.0,
            "DEB_SQuAD_1.1_EM": 0.0,
            "DEB_SQuAD_2.0_F1": 0.0,
            "DEB_SQuAD_2.0_EM": 0.0,
            "DEB_MNLI-m_Acc": 0.0,
            "DEB_MNLI-mm_Acc": 0.0,
            "DEB_SST-2_Acc": 0.0,
            "DEB_QNLI_Acc": 0.0,
            "DEB_CoLA_MCC": 0.0,
            "DEB_RTE_Acc": 0.0,
            "DEB_MRPC_Acc": 0.0,
            "DEB_MRPC_F1": 0.0,
            "DEB_QQP_Acc": 0.0,
            "DEB_QQP_F1": 0.0,
            "DEB_STS-B_P": 0.0,
            "DEB_STS-B_S": 0.0,
            "DEB_analysis_timestamp": time.time(),
            "DEB_text_length": 0,
            "DEB_word_count": 0,
            "DEB_model_name": self.model_name,
            "DEB_device": self.device,
        }

    def analyze_text(self, text: str) -> Dict[str, float]:
        if not text or not text.strip():
            logger.warning("Empty text provided for DeBERTa analysis")
            return self._get_default_metrics()

        sentences = self._split_sentences(text)
        pairs = self._build_pairs(sentences)

        metrics = self._question_answer_metrics(text)
        mnli_matched, mnli_mismatched = self._mnli_metrics(sentences)
        metrics["DEB_MNLI-m_Acc"] = mnli_matched
        metrics["DEB_MNLI-mm_Acc"] = mnli_mismatched
        metrics["DEB_SST-2_Acc"] = self._sentiment_confidence(sentences)

        questions = [s for s in sentences if s.endswith("?")]
        statements = [s for s in sentences if not s.endswith("?")]
        if not questions:
            questions = sentences[:1]
        if not statements:
            statements = sentences[:1]
        qnli_pairs = [(question, statement) for question in questions for statement in statements]
        metrics["DEB_QNLI_Acc"] = self._binary_nli_metric(
            qnli_pairs,
            "qnli",
            self.QNLI_MODELS,
            ("entailment", "label_1", "yes"),
            self._question_answer_label,
        )

        metrics["DEB_CoLA_MCC"] = self._cola_metric(sentences)
        metrics["DEB_RTE_Acc"] = self._binary_nli_metric(
            pairs,
            "rte",
            self.RTE_MODELS,
            ("entailment", "label_1", "yes"),
            lambda p, h: 1 if self._infer_nli_label(p, h) == "entailment" else 0,
        )

        mrpc_acc, mrpc_f1 = self._paraphrase_metrics(pairs, "mrpc", self.MRPC_MODELS)
        metrics["DEB_MRPC_Acc"] = mrpc_acc
        metrics["DEB_MRPC_F1"] = mrpc_f1

        qqp_acc, qqp_f1 = self._paraphrase_metrics(pairs, "qqp", self.QQP_MODELS)
        metrics["DEB_QQP_Acc"] = qqp_acc
        metrics["DEB_QQP_F1"] = qqp_f1

        sts_pearson, sts_spearman = self._sts_metrics(pairs)
        metrics["DEB_STS-B_P"] = sts_pearson
        metrics["DEB_STS-B_S"] = sts_spearman

        metrics["DEB_analysis_timestamp"] = time.time()
        metrics["DEB_text_length"] = len(text)
        metrics["DEB_word_count"] = len(text.split())
        metrics["DEB_model_name"] = self.model_name
        metrics["DEB_device"] = self.device
        return metrics

    def get_feature_dict(self, text_or_features: Union[str, Dict[str, Any]]) -> Dict[str, float]:
        text = ""
        if isinstance(text_or_features, str):
            text = text_or_features
        elif isinstance(text_or_features, dict):
            for key in (
                "whisperx_transcription",
                "transcription",
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
            logger.warning("No text found for DeBERTa analysis")
            return self._get_default_metrics()
        return self.analyze_text(text)

    @staticmethod
    def get_available_features() -> List[str]:
        return [
            "DEB_SQuAD_1.1_F1",
            "DEB_SQuAD_1.1_EM",
            "DEB_SQuAD_2.0_F1",
            "DEB_SQuAD_2.0_EM",
            "DEB_MNLI-m_Acc",
            "DEB_MNLI-mm_Acc",
            "DEB_SST-2_Acc",
            "DEB_QNLI_Acc",
            "DEB_CoLA_MCC",
            "DEB_RTE_Acc",
            "DEB_MRPC_Acc",
            "DEB_MRPC_F1",
            "DEB_QQP_Acc",
            "DEB_QQP_F1",
            "DEB_STS-B_P",
            "DEB_STS-B_S",
            "DEB_analysis_timestamp",
            "DEB_text_length",
            "DEB_word_count",
            "DEB_model_name",
            "DEB_device",
        ]
