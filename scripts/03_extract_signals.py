"""Stage 3 CLI: Demonstrate orthogonal signal extraction (S_logit, S_sem, S_ret)."""

from __future__ import annotations

from ccr_router.generation.llm_engine import LocalLLMEngine
from ccr_router.signals.internal_entropy import compute_s_logit
from ccr_router.signals.semantic_entropy import SemanticEntropyEstimator


def main() -> None:
    engine = LocalLLMEngine()
    sem = SemanticEntropyEstimator(entailment_fn=None, jaccard_threshold=0.65)
    bundle = engine.generate_bundle(
        question="What are the UK FCA Consumer Duty obligations for retail banks?",
        domain="uk_finance",
        num_samples=5,
    )
    s_logit = compute_s_logit(bundle.token_logprobs, top2_margins=bundle.top2_margins)
    s_sem, _ = sem.score(bundle.sampled_answers)
    print(
        f"[03_extract_signals] backend={bundle.backend_used} "
        f"S_logit={s_logit:.4f} S_sem={s_sem:.4f}"
    )


if __name__ == "__main__":
    main()
