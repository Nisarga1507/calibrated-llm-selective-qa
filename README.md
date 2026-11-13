cat << 'EOF' > README.md
# 3S-CCR (v2.0): Three-Signal Conformal Cascading Router for Cost-Optimal Selective Question Answering

**Principal Investigator & Author:** Nisarga  
**Repository:** [github.com/Nisarga1507/calibrated-llm-selective-qa](https://github.com/Nisarga1507/calibrated-llm-selective-qa)  
**Version:** `2.0.0` (Conformal LTT Cascade & UK Enterprise Evaluation Suite)  
**License:** MIT  

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Calibration: Conformal LTT](https://img.shields.io/badge/Calibration-Conformal%20LTT%20(Clopper--Pearson)-1b9e77.svg)](#3-methodology--mathematical-formulation)
[![Evaluation: UK Enterprise](https://img.shields.io/badge/Domains-UK%20Healthcare%20%7C%20Law%20%7C%20Finance-08519c.svg)](#4-datasets--circularity-free-evaluation-protocol)

---

## Abstract

Deploying Large Language Models (LLMs) in safety-critical and regulated enterprise environments—such as clinical decision support, statutory legal advisory, and financial compliance—is fundamentally constrained by uncalibrated hallucinations. Standard token-level confidence heuristics fail silently when models generate fluent, high-probability falsehoods, while invoking multi-sample semantic verification and retrieval-augmented entailment on every query incurs prohibitive latency and compute overhead.

This research introduces **3S-CCR (Three-Signal Conformal Cascading Router)**, a two-stage selective prediction and abstention framework that unifies three orthogonal uncertainty signals—**internal token logit confidence ($S_{\text{logit}}$)**, **black-box semantic clustering entropy ($S_{\text{sem}}$)**, and **retrieval-augmented Natural Language Inference grounding ($S_{\text{ret}}$)**—under finite-sample **Learn Then Test (LTT)** conformal risk control. Evaluated across **5,988 queries** spanning general knowledge (`TruthfulQA`, `FEVER`) and UK enterprise domains (`UK Healthcare`, `UK Law`, `UK Finance`), 3S-CCR guarantees empirical selective risk strictly below the target tolerance ($\alpha = 0.05$) on 100% of held-out benchmarks while reducing total expected operational risk and human-escalation costs by **44.5% to 91.3%**.

---

## 1. Completed Research & Engineering Milestones (v1.0 $\rightarrow$ v2.0)

The following core components, experimental protocols, and statistical validations have been designed, implemented, and verified in **v2.0**:

1. **Leakage-Free Multi-Domain Corpus & Partitioning (`N = 5,988`):**
   * Constructed stratified, circularity-free splits (`60% train / 20% dev_calibration / 20% test`) across five benchmarks (`TruthfulQA`, `FEVER`, `UK Healthcare`, `UK Law`, and `UK Finance`) to guarantee zero data leakage between meta-learner training, conformal threshold calibration, and final held-out testing.
2. **Hybrid Reciprocal Rank Fusion (RRF) Retrieval Engine (`src/ccr_router/retrieval/`):**
   * Implemented lexical BM25 and dense semantic indexing fused via Reciprocal Rank Fusion ($k = 60$) to retrieve domain-grounded statutory, clinical, and encyclopedic evidence passages.
3. **Three Orthogonal Uncertainty Signal Extractors (`src/ccr_router/signals/`):**
   * **$S_{\text{logit}}$ (Stage 1):** Length-normalized token log-probabilities combined with top-2 token logit margin separation.
   * **$S_{\text{sem}}$ (Stage 2):** Semantic equivalence clustering across $M = 5$ stochastic generations, computing normalized cluster entropy $S_{\text{sem}} = 1 - H_{\text{sem}} / \log M$.
   * **$S_{\text{ret}}$ (Stage 2):** Claim-evidence NLI entailment scoring over top-$k$ retrieved passages to catch overconfident parametric hallucinations.
4. **Two-Stage Cost-Aware Cascade & Conformal LTT Calibration (`src/ccr_router/calibration/`):**
   * Built regularized logistic meta-learners ($f_1, f_2$) paired with a two-stage routing gate ($\tau_{\text{fast\_lo}} = 0.20, \tau_{\text{fast\_hi}} = 0.85$) and exact Clopper-Pearson upper confidence bound ($\delta = 0.05$) LTT threshold selection across $\alpha \in \{0.01, 0.03, 0.05, 0.10, 0.15, 0.20\}$.
5. **UK Enterprise Operational Economics Simulator (`src/ccr_router/evaluation/economics_sim.py`):**
   * Formulated an end-to-end GBP (£) deployment cost model incorporating Stage-1 compute (`£0.002/query`), Stage-2 verification compute (`£0.015/query`), specialist human review for abstained queries (`£18.50/query`), and unmitigated hallucination liability (`£500.00/error`).
6. **Rigorous Statistical Hypothesis Testing & Publication Figures (`scripts/05_*.py`, `scripts/06_*.py`):**
   * Executed 4-way signal ablations (`S_logit Only`, `S_sem Only`, `2S-CCR`, `3S-CCR`), paired bootstrap 95% confidence intervals ($B = 2,000$ resamples), exact McNemar's tests, and generated 300 DPI diagnostic visualizations in `results/figures/`.

---

## 2. System Architecture & Mathematical Formulation

```text
                        ┌───────────────────────────────────────┐
                        │           Incoming Query (x)          │
                        └───────────────────┬───────────────────┘
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │  STAGE 1: Fast Internal Screening     │
                        │  Extract S_logit (Token LP & Margin)  │
                        │  Compute Stage-1 Score: p₁ = f₁(x)    │
                        └───────────────────┬───────────────────┘
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              │ p₁ ≤ τ_fast_lo              │ τ_fast_lo < p₁ < τ_fast_hi  │ p₁ ≥ τ_fast_hi
              ▼                             ▼                             ▼
   ┌─────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────┐
   │  Immediate Abstain  │     │ STAGE 2: Deep Cascade   │     │   Fast-Path Score   │
   │  g(x) = p₁          │     │ • M=5 Samples → S_sem   │     │   g(x) = p₁         │
   └──────────┬──────────┘     │ • RRF + NLI   → S_ret   │     └──────────┬──────────┘
              │                │ • Unified: g(x)=f₂(x)   │                │
              │                └────────────┬────────────┘                │
              └─────────────────────────────┼─────────────────────────────┘
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │  CONFORMAL LTT DECISION GATE          │
                        │  Accept if g(x) ≥ τ̂_accept(α, δ)      │
                        │  Else Abstain → Human Review (£18.50) │
                        └───────────────────────────────────────┘
                        