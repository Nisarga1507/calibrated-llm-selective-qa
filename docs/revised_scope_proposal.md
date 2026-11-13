# Revised MSc Dissertation Scope Proposal
**Title:** Calibrated Multi-Signal Hallucination Guardrails & Selective Routers for Large Language Models

## 1. Research Objective & Shift in Framing
Rather than training or fine-tuning a generator to eliminate hallucinations, this dissertation investigates **post-hoc calibrated uncertainty estimation and selective prediction** for frozen instruction-tuned LLMs (`Llama-3.1-8B-Instruct`). Specifically, we test whether combining orthogonal epistemic, retrieval-grounding, and semantic-consistency signals yields well-calibrated correctness probabilities ($\hat{p} \in [0, 1]$) that improve risk-coverage trade-offs over single-signal baselines.

## 2. Core Hypotheses
* **H1 (Signal Complementarity):** Internal token probabilities ($S_{\text{int}}$), external retrieval NLI entailment ($S_{\text{ret}}$), and multi-sample semantic consistency ($S_{\text{con}}$) exhibit low rank correlation and capture distinct failure modes (confabulations vs. systematic misconceptions).
* **H2 (Calibration & Discrimination):** Learned low-dimensional meta-combiners (L2 Logistic Regression / regularised MLP) with post-hoc calibration (Platt Scaling / Isotonic Regression) significantly reduce Expected Calibration Error (ECE) and Area Under the Risk-Coverage Curve (AURC) compared to individual signals and Kadavath's $P(\text{True})$ baseline.
* **H3 (3-Way Selective Routing):** Conditioning an adaptive **Accept / Refine & Verify / Abstain** router on calibrated confidence $\hat{p}$ achieves lower expected risk at fixed coverage than binary abstention alone.

## 3. Experimental Protocol & Anti-Leakage Controls
* **Primary Benchmark:** TruthfulQA ($N = 817$), stratified into locked `20% Train / 30% Dev / 50% Test` splits (`seed = 42`).
* **Out-of-Domain Generalisation:** Held-out FEVER claim-verification subset ($N = 1,500$) evaluated using frozen Dev-optimised combiners, calibrators, and routing thresholds $(\tau_{\text{abstain}}, \tau_{\text{accept}})$.
* **Judge Validation:** Automated LLM-as-a-Judge correctness labels validated against a 50-question blind human audit (reporting Cohen's $\kappa$).
* **Statistical Rigour:** All headline test metrics reported with 1,000-resample 95% paired bootstrap confidence intervals and McNemar's significance tests.