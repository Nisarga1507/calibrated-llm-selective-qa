"""UK Enterprise GBP Operational Cost & Cascade Savings Simulator."""

from __future__ import annotations

import numpy as np


def simulate_uk_enterprise_economics(
    unified_scores: np.ndarray,
    stage2_triggered: np.ndarray,
    is_correct: np.ndarray,
    tau_accept: float,
    stage1_cost_gbp: float = 0.0015,
    stage2_cost_gbp: float = 0.0045,
    human_review_cost_gbp: float = 18.50,
    hallucination_penalty_gbp: float = 500.00,
) -> dict[str, float]:
    """Compare total expected GBP cost per 1,000 queries across routing policies."""
    n = len(is_correct)
    if n == 0:
        return {}

    y = np.asarray(is_correct, dtype=bool)
    s2 = np.asarray(stage2_triggered, dtype=bool)
    accepted = np.asarray(unified_scores, dtype=np.float64) >= tau_accept

    # 1. Always-Accept baseline (runs full Stage-2 on every query, 0% human escalation)
    always_accept_compute = n * (stage1_cost_gbp + stage2_cost_gbp)
    always_accept_errors = int(np.sum(~y))
    always_accept_total = always_accept_compute + always_accept_errors * hallucination_penalty_gbp

    # 2. Single-Stage Full-Compute Selective Router (no early exit)
    single_stage_compute = n * (stage1_cost_gbp + stage2_cost_gbp)
    abstained = ~accepted
    escaped_errors = int(np.sum(accepted & (~y)))
    single_stage_total = (
        single_stage_compute
        + int(np.sum(abstained)) * human_review_cost_gbp
        + escaped_errors * hallucination_penalty_gbp
    )

    # 3. 3S-CCR Two-Stage Conformal Cascade (early-exits high/low Stage-1 queries)
    cascade_compute = n * stage1_cost_gbp + int(np.sum(s2)) * stage2_cost_gbp
    cascade_total = (
        cascade_compute
        + int(np.sum(abstained)) * human_review_cost_gbp
        + escaped_errors * hallucination_penalty_gbp
    )

    scale = 1000.0 / n
    compute_savings_pct = (
        100.0 * (single_stage_compute - cascade_compute) / max(single_stage_compute, 1e-9)
    )
    total_savings_pct = (
        100.0 * (always_accept_total - cascade_total) / max(always_accept_total, 1e-9)
    )

    return {
        "always_accept_gbp_per_1k": round(always_accept_total * scale, 2),
        "single_stage_gbp_per_1k": round(single_stage_total * scale, 2),
        "cascade_3sccr_gbp_per_1k": round(cascade_total * scale, 2),
        "compute_savings_vs_full_pct": round(compute_savings_pct, 2),
        "total_risk_cost_savings_pct": round(total_savings_pct, 2),
    }