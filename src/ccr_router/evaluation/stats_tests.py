"""Paired bootstrap confidence intervals and McNemar's test for selective QA."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from ccr_router.evaluation.selective_metrics import compute_aurc


def paired_bootstrap_aurc_auroc(
    scores_base: np.ndarray,
    scores_full: np.ndarray,
    is_correct: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Compute paired bootstrap 95% CIs and one-sided p-values for:
      - delta_aurc = AURC(base) - AURC(full)  (> 0 means full has lower risk-coverage area)
      - delta_auroc = AUROC(full) - AUROC(base) (> 0 means full discriminates better)
    """
    sb = np.asarray(scores_base, dtype=np.float64)
    sf = np.asarray(scores_full, dtype=np.float64)
    y = np.asarray(is_correct, dtype=bool)
    n = len(y)

    obs_delta_aurc = compute_aurc(sb, y) - compute_aurc(sf, y)
    obs_delta_auroc = (
        float(roc_auc_score(y.astype(int), sf) - roc_auc_score(y.astype(int), sb))
        if len(np.unique(y)) > 1
        else 0.0
    )

    rng = np.random.default_rng(seed)
    boot_daurc = np.empty(n_boot, dtype=np.float64)
    boot_dauroc = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        boot_daurc[b] = compute_aurc(sb[idx], yb) - compute_aurc(sf[idx], yb)
        if len(np.unique(yb)) > 1:
            boot_dauroc[b] = roc_auc_score(yb.astype(int), sf[idx]) - roc_auc_score(
                yb.astype(int), sb[idx]
            )
        else:
            boot_dauroc[b] = 0.0

    p_aurc = float(np.mean(boot_daurc <= 0.0))
    p_auroc = float(np.mean(boot_dauroc <= 0.0))

    return {
        "delta_aurc": round(obs_delta_aurc, 4),
        "delta_aurc_ci_lo": round(float(np.percentile(boot_daurc, 2.5)), 4),
        "delta_aurc_ci_hi": round(float(np.percentile(boot_daurc, 97.5)), 4),
        "p_value_aurc": round(max(p_aurc, 1.0 / (n_boot + 1)), 4),
        "delta_auroc": round(obs_delta_auroc, 4),
        "delta_auroc_ci_lo": round(float(np.percentile(boot_dauroc, 2.5)), 4),
        "delta_auroc_ci_hi": round(float(np.percentile(boot_dauroc, 97.5)), 4),
        "p_value_auroc": round(max(p_auroc, 1.0 / (n_boot + 1)), 4),
    }


def mcnemar_selective_test(
    accepted_base: np.ndarray,
    accepted_full: np.ndarray,
    is_correct: np.ndarray,
) -> dict[str, float | int]:
    """
    Exact McNemar test comparing selective utility (accepted & correct, or abstained & wrong).
    """
    ab = np.asarray(accepted_base, dtype=bool)
    af = np.asarray(accepted_full, dtype=bool)
    y = np.asarray(is_correct, dtype=bool)

    # Ideal selective action: accept when correct, abstain when incorrect
    good_base = ab == y
    good_full = af == y

    # Discordant pairs: n01 = full right & base wrong; n10 = base right & full wrong
    n01 = int(np.sum((~good_base) & good_full))
    n10 = int(np.sum(good_base & (~good_full)))
    total_disc = n01 + n10

    p_val = (
        float(binomtest(n01, n=total_disc, p=0.5, alternative="two-sided").pvalue)
        if total_disc > 0
        else 1.0
    )

    return {
        "n01_full_wins": n01,
        "n10_base_wins": n10,
        "mcnemar_p_value": round(p_val, 4),
    }