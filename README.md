# 3S-CCR: Three-Signal Conformal Cascading Router for Selective LLM Question Answering

[![CI](https://github.com/Nisarga1507/calibrated-llm-selective-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Nisarga1507/calibrated-llm-selective-qa/actions)
[![Release](https://img.shields.io/badge/release-v2.3.0-blue.svg)](https://github.com/Nisarga1507/calibrated-llm-selective-qa/releases)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**3S-CCR** (*Three-Signal Conformal Cascading Router*) is a finite-sample calibrated selective question-answering (QA) architecture and production FastAPI microservice built for regulated UK domains (**UK Healthcare**, **UK Statutory Law**, and **FCA-Regulated UK Finance**) alongside general factuality benchmarks (**TruthfulQA** and **FEVER**).

Instead of emitting unverified hallucinations or paying multi-sample retrieval latency on every query, 3S-CCR gates routine queries in **Stage 1** using fast token log-probability confidence ($S_{\\text{logit}}$) and escalates only ambiguous boundary queries to **Stage 2** for stochastic semantic clustering ($S_{\\text{sem}}$) and hybrid Reciprocal Rank Fusion + NLI evidence grounding ($S_{\\text{ret}}$). Routing thresholds are calibrated via **Learn-Then-Test (LTT)** with exact **Clopper-Pearson upper confidence bounds** ($\\alpha = 0.05, \\delta = 0.05$), guaranteeing empirical selective risk $\\le \\alpha$ while reducing expected UK enterprise operational risk costs by **44.5% to 91.3%**.

---

## 1. System Architecture & How It Works

* **Stage 1 (Fast Gate — $S_{\\text{logit}}$):** Computes length-normalized token log-probabilities and top-2 logit margins (£0.002/query, `2.45 ms` median latency). Resolves ~80% of routine queries immediately without multi-sample or retrieval overhead.
* **Stage 2 (Boundary Escalation — $S_{\\text{sem}} + S_{\\text{ret}}$):** Escalates ambiguous queries ($0.35 \\le S_{\\text{logit}} < 0.85$) to $M=5$ stochastic semantic clustering ($S_{\\text{sem}}$) and Hybrid BM25 + Dense RRF + NLI grounding ($S_{\\text{ret}}$), fused as $0.45 S_{\\text{logit}} + 0.25 S_{\\text{sem}} + 0.30 S_{\\text{ret}}$ (£0.018/query).
* **LTT Conformal Risk Control:** Selects $\\hat{\\tau}_{\\text{accept}}$ on `dev_calibration.jsonl` via exact Clopper-Pearson bounds ($\\alpha = 0.05, \\delta = 0.05$, sample complexity floor $n_{\\min} = 59$). Queries below $\\hat{\\tau}_{\\text{accept}}$ trigger `ABSTAIN_ESCALATE_HUMAN` (£18.50 specialist review vs. £500.00 hallucination liability).
* **Production Observability (`GET /v2/metrics`):** Tracks $p_{50}/p_{95}$ latency, Stage-1 fast-exit rate, cumulative GBP cost, and rolling $S_{\\text{logit}}$ drift alerts ($|\\Delta \\bar{S}_{\\text{logit}}| > 0.15$) in `results/telemetry/inference_log.jsonl`.
