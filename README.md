# Calibrated Multi-Signal Hallucination Guardrail & Selective Router for LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **MSc Data Science Dissertation & Applied AI Reliability Framework**  
> Shifting LLM evaluation from raw answer generation to **calibrated risk control**—unifying internal token probabilities, external retrieval NLI entailment, and semantic self-consistency into a three-branch selective router (**Accept / Refine / Abstain**).

---

## 1. Problem & Motivation
Large Language Models (LLMs) frequently generate fluent, plausible falsehoods across two distinct failure modes:
1. **Confabulations** — arbitrary, random-seed-sensitive guesses when the model lacks knowledge (*Farquhar et al., Nature 2024*).
2. **Systematic Misconceptions** — confident, consistent repetition of common human myths learned during pretraining (*Lin et al., TruthfulQA 2022*).

Single-signal uncertainty estimators fail to catch both: internal token probabilities and sample consistency remain high on systematic misconceptions, while uncalibrated heuristic combinations produce overconfident scores (*Guo et al., 2017; Kadavath et al., 2022*).

This repository implements an end-to-end **post-hoc calibrated confidence estimator** and **selective prediction router** (*Geifman & El-Yaniv, 2017*) that maps multi-signal reliability features into true correctness probabilities $\hat{p} \in [0, 1]$ to govern adaptive routing without fine-tuning the underlying 8B LLM.

---

## 2. System Architecture

```text
                       ┌──────────────────────────┐
                       │    Input Query (x)       │
                       └────────────┬─────────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
      ┌────────────────────┐ ┌─────────────┐ ┌───────────────────┐
      │ Greedy Generation  │ │ Dense FAISS │ │ M=5 Stochastic    │
      │ + Token Log-Probs  │ │ Retrieval   │ │ Samples (T=0.5)   │
      └─────────┬──────────┘ └──────┬──────┘ └─────────┬─────────┘
                │                   │                  │
                ▼                   ▼                  ▼
      Signal 1: S_int       Signal 2: S_ret    Signal 3: S_con
     (Length-Norm Prob)    (DeBERTa NLI Max)  (SelfCheckGPT-NLI &
                                               Semantic Entropy)
                │                   │                  │
                └───────────────────┼──────────────────┘
                                    ▼
                 ┌─────────────────────────────────────┐
                 │  Meta-Combiner (LogReg / MLP)       │
                 │  + Post-Hoc Calibration (Platt/Iso) │
                 └──────────────────┬──────────────────┘
                                    │  Calibrated Confidence p̂ ∈ [0, 1]
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
          p̂ >= τ_accept    τ_abstain <= p̂ < τ_acc   p̂ < τ_abstain
          ┌───────────┐    ┌─────────────────────┐  ┌───────────┐
          │  ACCEPT   │    │  REFINE & VERIFY    │  │  ABSTAIN  │
          │ (Return y)│    │ (RAG Re-Ask + Gate) │  │ (Refuse)  │
          └───────────┘    └─────────────────────┘  └───────────┘
```

### Formal Signal Definitions
1. **Internal Uncertainty ($S_{\text{int}}$):** Length-normalised sequence probability $\exp\left(\frac{1}{T}\sum_{t=1}^T \log P_\theta(y_t \mid x, y_{<t})\right)$ preventing length bias on multi-sentence responses.
2. **External Retrieval Grounding ($S_{\text{ret}}$):** Top-$k$ ($k=3$) Wikipedia passages retrieved via `BAAI/bge-base-en-v1.5` (FAISS) and scored via `DeBERTa-v3-large` NLI max entailment probability.
3. **Verification Consistency ($S_{\text{con}}$):** Question-conditioned **SelfCheckGPT-NLI** 2-class entailment consistency (*Manakul et al., 2023*) and non-defeating bidirectional **Discrete Semantic Entropy** (*Farquhar et al., 2024*) across $M=5$ stochastic samples ($T=0.5$).

---

## 3. Repository Layout

```text
calibrated-llm-selective-qa/
├── app.py                       # Interactive Streamlit reliability & routing dashboard
├── scripts/run_pipeline.py      # CLI entry point for automated end-to-end execution
├── tests/                       # Pytest suite verifying ECE, NLI scoring & router boundaries
├── docs/                        # Supervisor scope proposal & formal mathematical architecture
├── configs/                     # YAML hyperparameters & frozen Dev-optimised thresholds
├── data/splits/                 # Locked 20% Train / 30% Dev / 50% Test splits (zero leakage)
├── src/                         # Modular Python package (signals, combiners, calibration, stats)
├── notebooks/                   # Sequential reproducible notebooks (01 to 05)
└── results/                     # Publication-ready Reliability Diagrams, AURC curves & LaTeX tables
```

---

## 4. Quickstart & Reproducibility

### Installation
```bash
git clone git clone https://github.com/Nisarga1507/calibrated-llm-selective-qa.git
cd calibrated-llm-selective-qa
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

### Experimental Execution Order
1. **Splits & Baseline Inference (`notebooks/01_data_splits_and_baseline.ipynb`):** Locks `train / dev / test` splits (`seed=42`), runs `Llama-3.1-8B-Instruct` greedy decoding, logs token log-probs, and validates LLM-as-a-Judge labels against a 50-sample human audit (Cohen's $\kappa$).
2. **Calibration Plumbing Check (`notebooks/02_calibration_plumbing_check.ipynb`):** Verifies 10-bin ECE, Brier Score, AUROC, and two-panel Reliability Diagrams on `dev` using $S_{\text{int}}$ alone.
3. **Multi-Signal Extraction & Ablations (`notebooks/03_signal_extraction_and_ablation.ipynb`):** Extracts and caches $S_{\text{ret}}$ and $S_{\text{con}}$; evaluates signal orthogonality (Spearman correlation) and class separability (Mann-Whitney $U$ test).
4. **Meta-Combination, Calibration & Thresholding (`notebooks/04_combiner_and_calibration_dev.ipynb`):** Trains Linear, L2-Logistic Regression, and MLP combiners on `train`; applies Platt Scaling and Isotonic Regression on `dev`; optimises $(\tau_{\text{abstain}}, \tau_{\text{accept}})$ via Risk-Coverage utility and freezes them in `configs/thresholds.yaml`.
5. **Final Held-Out Evaluation (`notebooks/05_final_test_and_fever_eval.ipynb`):** Runs a single locked evaluation on `truthfulqa_test.jsonl` ($N=409$) and out-of-domain `fever_test.jsonl` ($N=1,500$) with 95% paired bootstrap confidence intervals and McNemar's significance tests.

### Launch Interactive Demo
```bash
streamlit run app.py
```

---

## 5. Experimental Results Tracker (TruthfulQA Test Split, $N=409$)

| Estimator / Policy | Post-Hoc Calibrator | ECE ($\downarrow$) | Brier ($\downarrow$) | AUROC ($\uparrow$) | AURC ($\downarrow$) | Selective Acc @ 75% Cov ($\uparrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline ($S_{\text{int}}$ only) | None | — | — | — | — | — |
| Retrieval Only ($S_{\text{ret}}$) | None | — | — | — | — | — |
| Consistency Only ($S_{\text{con}}$) | None | — | — | — | — | — |
| Kadavath $P(\text{True})$ Baseline | None | — | — | — | — | — |
| Convex Linear Sum | Platt Scaling | — | — | — | — | — |
| L2 Logistic Regression | Platt Scaling | — | — | — | — | — |
| Regularised MLP (`[3->16->1]`) | Isotonic Reg. | — | — | — | — | — |
| **3-Way Router (`Accept/Refine/Abstain`)** | **Best Calibrated** | — | — | — | — | — |

---

## 6. Key References
* **Guo et al. (2017).** *On Calibration of Modern Neural Networks.* ICML.
* **Geifman & El-Yaniv (2017).** *Selective Classification for Deep Neural Networks.* NeurIPS.
* **Kadavath et al. (2022).** *Language Models (Mostly) Know What They Don't Know.* arXiv:2207.05221.
* **Lin et al. (2022).** *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* ACL.
* **Kuhn et al. (2023).** *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG.* ICLR.
* **Manakul et al. (2023).** *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative LLMs.* EMNLP.
* **Farquhar et al. (2024).** *Detecting hallucinations in large language models using semantic entropy.* Nature, 630, 625–630.