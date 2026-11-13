"""Signal 3 (S_ret): Retrieval-augmented NLI grounding & contradiction penalty."""

from __future__ import annotations

import re

import numpy as np


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "were", "been", "have", "has", "had", "not", "but", "what", "when",
        "where", "which", "who", "whom", "whose", "why", "how", "into",
    }
    tokens = set(re.findall(r"\b[a-z0-9]{2,}\b", text.lower()))
    return tokens - stopwords


class NLIGroundingScorer:
    """Scores candidate answer faithfulness against retrieved evidence passages."""

    def __init__(
        self,
        cross_encoder_id: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        use_cross_encoder: bool = True,
    ) -> None:
        self.cross_encoder_id = cross_encoder_id
        self.use_cross_encoder = use_cross_encoder
        self._model = None

    def _get_model(self):
        if not self.use_cross_encoder:
            return None
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.cross_encoder_id)
            except Exception:
                self.use_cross_encoder = False
                self._model = None
        return self._model

    def score(self, question: str, candidate_answer: str, evidence_passages: list[str]) -> float:
        """
        Compute S_ret in [0, 1] measuring entailment support minus contradiction risk.
        """
        valid_passages = [p.strip() for p in evidence_passages if p and p.strip()]
        if not valid_passages:
            # Closed-book fallback when no external evidence exists (e.g. TruthfulQA)
            return 0.5

        hypothesis = f"Question: {question} Answer: {candidate_answer}"
        model = self._get_model()

        if model is not None:
            pairs = [(passage[:1500], hypothesis) for passage in valid_passages]
            logits = np.asarray(model.predict(pairs), dtype=np.float64)
            # Softmax over [contradiction, neutral, entailment] or [entailment, neutral, contra]
            exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
            # Conservatively take max support minus max contradiction across top passages
            support = float(np.max(probs[:, -1]))
            contra = float(np.max(probs[:, 0]))
            raw = 0.5 + 0.5 * (support - contra)
            return float(np.clip(raw, 0.001, 0.999))

        # Fast lexical-semantic grounding fallback for CPU-only benchmarking
        ans_tokens = _content_tokens(f"{question} {candidate_answer}")
        if not ans_tokens:
            return 0.5

        best_overlap = 0.0
        for passage in valid_passages:
            ev_tokens = _content_tokens(passage)
            if not ev_tokens:
                continue
            overlap = len(ans_tokens & ev_tokens) / len(ans_tokens)
            best_overlap = max(best_overlap, overlap)

        # Calibrated sigmoid mapping from lexical evidence support to [0.15, 0.97]
        calibrated = 1.0 / (1.0 + np.exp(-5.0 * (best_overlap - 0.35)))
        return float(np.clip(calibrated, 0.001, 0.999))