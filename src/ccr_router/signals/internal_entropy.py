"""Signal 1 (S_logit): Internal token-level entropy & log-probability margin confidence."""

from __future__ import annotations

import numpy as np


def compute_s_logit(
    token_logprobs: list[float] | np.ndarray,
    top2_margins: list[float] | np.ndarray | None = None,
    vocab_size: int = 32000,
) -> float:
    """
    Compute Stage-1 internal confidence S_logit in [0, 1].

    Combines length-normalized sequence predictive probability (geometric mean token prob)
    with top-1 vs top-2 logit margin when available.
    """
    lp = np.asarray(token_logprobs, dtype=np.float64)
    if lp.size == 0:
        return 0.5

    # Clip extreme negative log-probs for numerical stability
    lp_clipped = np.clip(lp, -20.0, 0.0)
    seq_prob = float(np.exp(np.mean(lp_clipped)))

    # Approximate normalized Shannon entropy bound from sequence probability
    max_ent = np.log(max(vocab_size, 2))
    approx_ent = -float(np.mean(lp_clipped))
    norm_certainty = float(np.clip(1.0 - (approx_ent / max_ent), 0.0, 1.0))

    if top2_margins is not None and len(top2_margins) > 0:
        margins = np.asarray(top2_margins, dtype=np.float64)
        margin_conf = float(np.mean(1.0 / (1.0 + np.exp(-margins))))
        score = 0.55 * seq_prob + 0.30 * margin_conf + 0.15 * norm_certainty
    else:
        score = 0.75 * seq_prob + 0.25 * norm_certainty

    return float(np.clip(score, 0.001, 0.999))