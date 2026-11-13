"""Selective QA metrics: Selective Risk, Coverage, AURC, AUROC, and ECE."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_aurc(scores: np.ndarray, is_correct: np.ndarray) -> float:
    """Area Under the Risk-Coverage Curve (lower is better)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(is_correct, dtype=bool)
    order = np.argsort(-s)
    y_sorted = y[order]

    n = len(y_sorted)
    if n == 0:
        return 1.0

    cum_errors = np.cumsum(~y_sorted)
    ranks = np.arange(1, n + 1)
    risks = cum_errors / ranks
    coverages = ranks / n
    return float(np.trapezoid(risks, coverages))


def compute_ece(probs: np.ndarray, is_correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(is_correct, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if np.any(mask):
            ece += np.mean(mask) * abs(np.mean(p[mask]) - np.mean(y[mask]))
    return float(ece)


def evaluate_selective_operating_point(
    unified_scores: np.ndarray,
    is_correct: np.ndarray,
    tau_accept: float,
) -> dict[str, float]:
    """Compute held-out test metrics at calibrated threshold tau_accept."""
    s = np.asarray(unified_scores, dtype=np.float64)
    y = np.asarray(is_correct, dtype=bool)
    accepted = s >= tau_accept

    coverage = float(np.mean(accepted))
    if np.any(accepted):
        selective_risk = float(np.mean(~y[accepted]))
        selective_accuracy = float(np.mean(y[accepted]))
    else:
        selective_risk = 0.0
        selective_accuracy = 1.0

    auroc = float(roc_auc_score(y.astype(int), s)) if len(np.unique(y)) > 1 else 0.5
    aurc = compute_aurc(s, y)
    ece = compute_ece(s, y)

    return {
        "coverage": round(coverage, 4),
        "selective_risk": round(selective_risk, 4),
        "selective_accuracy": round(selective_accuracy, 4),
        "auroc": round(auroc, 4),
        "aurc": round(aurc, 4),
        "ece": round(ece, 4),
    }