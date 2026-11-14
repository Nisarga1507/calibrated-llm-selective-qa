"""FastAPI production service for the 3S-CCR Conformal Cascading Router (v2.0)."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ccr_router.calibration.conformal_router import ConformalCascadingRouter
from ccr_router.retrieval.hybrid_retriever import HybridRRFRetriever
from ccr_router.signals.internal_entropy import compute_s_logit
from ccr_router.signals.nli_grounding import NLIGroundingScorer
from ccr_router.signals.semantic_entropy import SemanticEntropyEstimator

DOMAINS = ("truthful_qa", "fever", "uk_healthcare", "uk_law", "uk_finance")


class RouteRequest(BaseModel):
    domain: Literal["truthful_qa", "fever", "uk_healthcare", "uk_law", "uk_finance"] = Field(
        default="uk_healthcare", description="Target enterprise or benchmark domain."
    )
    question: str = Field(..., min_length=3, description="User query to evaluate.")
    candidate_answer: str = Field(..., min_length=1, description="Stage-1 primary LLM answer.")
    token_logprobs: list[float] = Field(
        default_factory=lambda: [-0.18, -0.15, -0.22, -0.14],
        description="Token-level log probabilities from Stage-1 generation.",
    )
    top2_margins: list[float] | None = Field(
        default=None, description="Optional top-1 vs top-2 logit margins."
    )
    sampled_answers: list[str] | None = Field(
        default=None,
        description="Optional M stochastic Stage-2 responses for semantic entropy.",
    )
    target_alpha: float = Field(
        default=0.05, ge=0.01, le=0.20, description="Target conformal selective risk bound."
    )


class RouteResponse(BaseModel):
    domain: str
    decision: Literal["ACCEPT", "ABSTAIN_ESCALATE_HUMAN"]
    stage_executed: Literal[1, 2]
    s_logit: float
    s_sem: float | None
    s_ret: float | None
    unified_score: float
    tau_accept: float
    target_alpha: float
    estimated_query_cost_gbp: float
    retrieved_evidence_top1: str | None


STATE: dict = {}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = yaml.safe_load(Path("configs/v2_conformal_config.yaml").read_text(encoding="utf-8"))
    report_path = Path("results/tables/v2_full_conformal_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    retrievers: dict[str, HybridRRFRetriever] = {}
    for ds in DOMAINS:
        rows = (
            _load_jsonl(Path("data/splits") / ds / "train.jsonl")
            + _load_jsonl(Path("data/splits") / ds / "dev_calibration.jsonl")
            + _load_jsonl(Path("data/splits") / ds / "test.jsonl")
        )
        doc_ids = [r["id"] for r in rows if str(r.get("evidence", "")).strip()]
        docs = [str(r["evidence"]) for r in rows if str(r.get("evidence", "")).strip()]
        ret = HybridRRFRetriever(use_dense=False)
        if docs:
            ret.index(doc_ids, docs)
        retrievers[ds] = ret

    STATE["cfg"] = cfg
    STATE["report"] = report
    STATE["retrievers"] = retrievers
    STATE["sem_estimator"] = SemanticEntropyEstimator(entailment_fn=None, jaccard_threshold=0.65)
    STATE["nli_scorer"] = NLIGroundingScorer(use_cross_encoder=False)
    yield
    STATE.clear()


app = FastAPI(
    title="3S-CCR Conformal Cascading Router API",
    version="2.0.0",
    description="Finite-sample calibrated selective QA router for UK regulated domains.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": "2.0.0",
        "domains_indexed": list(STATE.get("retrievers", {}).keys()),
    }


@app.post("/v2/route", response_model=RouteResponse)
def route_query(req: RouteRequest) -> RouteResponse:
    cfg = STATE["cfg"]
    econ = cfg["uk_enterprise_economics"]
    tau_lo = float(cfg["cascade"]["tau_fast_lo"])
    tau_hi = float(cfg["cascade"]["tau_fast_hi"])

    s_logit = compute_s_logit(req.token_logprobs, top2_margins=req.top2_margins)

    # Resolve calibrated tau_accept from v2 conformal report
    ds_sweep = STATE["report"].get(req.domain, {}).get("alpha_sweep", {})
    alpha_key = str(round(req.target_alpha, 2))
    tau_accept = float(ds_sweep.get(alpha_key, {}).get("tau_accept", 0.85))

    ConformalCascadingRouter(
        tau_fast_lo=tau_lo,
        tau_fast_hi=tau_hi,
        delta=float(cfg["conformal_ltt"]["delta"]),
    )

    # Stage 1 fast gate check
    if s_logit < tau_lo or s_logit >= tau_hi:
        stage_executed: Literal[1, 2] = 1
        s_sem = None
        s_ret = None
        top_ev = None
        unified_score = float(s_logit)
        compute_cost = float(econ["stage1_cost_gbp"])
    else:
        # Escalate boundary query to Stage 2
        stage_executed = 2
        samples = req.sampled_answers or [req.candidate_answer]
        s_sem, _ = STATE["sem_estimator"].score(samples)
        passages = STATE["retrievers"][req.domain].retrieve(req.question, top_k=3)
        ev_texts = [p.text for p in passages]
        top_ev = ev_texts[0] if ev_texts else None
        s_ret = STATE["nli_scorer"].score(req.question, req.candidate_answer, ev_texts)
        # Calibrated multi-signal fusion
        unified_score = float(0.45 * s_logit + 0.25 * s_sem + 0.30 * s_ret)
        compute_cost = float(econ["stage1_cost_gbp"]) + float(econ["stage2_cost_gbp"])

    accepted = unified_score >= tau_accept
    decision: Literal["ACCEPT", "ABSTAIN_ESCALATE_HUMAN"] = (
        "ACCEPT" if accepted else "ABSTAIN_ESCALATE_HUMAN"
    )
    total_cost = compute_cost + (0.0 if accepted else float(econ["human_review_cost_gbp"]))

    return RouteResponse(
        domain=req.domain,
        decision=decision,
        stage_executed=stage_executed,
        s_logit=round(float(s_logit), 4),
        s_sem=round(float(s_sem), 4) if s_sem is not None else None,
        s_ret=round(float(s_ret), 4) if s_ret is not None else None,
        unified_score=round(unified_score, 4),
        tau_accept=round(tau_accept, 4),
        target_alpha=req.target_alpha,
        estimated_query_cost_gbp=round(total_cost, 4),
        retrieved_evidence_top1=top_ev,
    )
