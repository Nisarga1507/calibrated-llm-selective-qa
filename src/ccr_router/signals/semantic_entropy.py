"""Signal 2 (S_sem): Bidirectional semantic clustering & normalized semantic entropy."""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import numpy as np


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _token_jaccard(a: str, b: str) -> float:
    ta = set(_normalize_text(a).split())
    tb = set(_normalize_text(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class SemanticEntropyEstimator:
    """Clusters K temperature-sampled answers into meaning equivalence classes."""

    def __init__(
        self,
        entailment_fn: Callable[[str, str], bool] | None = None,
        jaccard_threshold: float = 0.65,
    ) -> None:
        self.entailment_fn = entailment_fn
        self.jaccard_threshold = jaccard_threshold

    def _are_equivalent(self, ans_a: str, ans_b: str) -> bool:
        na, nb = _normalize_text(ans_a), _normalize_text(ans_b)
        if na == nb:
            return True
        if self.entailment_fn is not None:
            return self.entailment_fn(ans_a, ans_b) and self.entailment_fn(ans_b, ans_a)
        return _token_jaccard(ans_a, ans_b) >= self.jaccard_threshold

    def score(self, sampled_answers: list[str]) -> tuple[float, int]:
        """
        Return (S_sem confidence in [0, 1], num_semantic_clusters).

        High semantic agreement across K samples -> low semantic entropy -> high S_sem.
        """
        m = len(sampled_answers)
        if m == 0:
            return 0.5, 0
        if m == 1:
            return 0.85, 1

        clusters: list[list[str]] = []
        for ans in sampled_answers:
            placed = False
            for cluster in clusters:
                if self._are_equivalent(ans, cluster[0]):
                    cluster.append(ans)
                    placed = True
                    break
            if not placed:
                clusters.append([ans])

        probs = np.array([len(c) / m for c in clusters], dtype=np.float64)
        sem_entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        max_entropy = math.log(m)
        norm_entropy = sem_entropy / max_entropy if max_entropy > 0 else 0.0

        s_sem = float(np.clip(1.0 - norm_entropy, 0.001, 0.999))
        return s_sem, len(clusters)