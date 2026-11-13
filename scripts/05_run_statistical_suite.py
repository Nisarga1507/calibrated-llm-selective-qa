"""Run 4-way signal ablation & paired statistical significance tests across all 5 benchmarks."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml
from run_pipeline import DATASETS, build_split_signals, load_jsonl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ccr_router.calibration.conformal_router import ConformalCascadingRouter
from ccr_router.calibration.meta_learners import ThreeSignalMetaLearner
from ccr_router.evaluation.selective_metrics import evaluate_selective_operating_point
from ccr_router.evaluation.stats_tests import (
    mcnemar_selective_test,
    paired_bootstrap_aurc_auroc,
)
from ccr_router.retrieval.hybrid_retriever import HybridRRFRetriever
from ccr_router.signals.nli_grounding import NLIGroundingScorer
from ccr_router.signals.semantic_entropy import SemanticEntropyEstimator


def _fit_and_score_subset(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    test_x: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    sc = StandardScaler().fit(train_x)
    clf = LogisticRegression(C=1.0, random_state=seed, max_iter=1000)
    clf.fit(sc.transform(train_x), train_y.astype(int))
    dev_p = np.clip(clf.predict_proba(sc.transform(dev_x))[:, 1], 0.001, 0.999)
    test_p = np.clip(clf.predict_proba(sc.transform(test_x))[:, 1], 0.001, 0.999)
    return dev_p, test_p


def main() -> None:
    cfg = yaml.safe_load(Path("configs/v2_conformal_config.yaml").read_text())
    seed = int(cfg["project"]["seed"])
    m_samples = int(cfg["models"]["semantic_samples_m"])
    tau_lo = float(cfg["cascade"]["tau_fast_lo"])
    tau_hi = float(cfg["cascade"]["tau_fast_hi"])
    delta = float(cfg["conformal_ltt"]["delta"])
    alpha_grid = [float(a) for a in cfg["conformal_ltt"]["alpha_grid"]]
    alpha = float(cfg["conformal_ltt"]["default_alpha"])

    sem_estimator = SemanticEntropyEstimator(entailment_fn=None, jaccard_threshold=0.65)
    nli_scorer = NLIGroundingScorer(use_cross_encoder=False)

    ablation_rows: list[dict] = []
    stats_rows: list[dict] = []

    for idx, ds_name in enumerate(DATASETS):
        split_dir = Path("data/splits") / ds_name
        train_rows = load_jsonl(split_dir / "train.jsonl")
        dev_rows = load_jsonl(split_dir / "dev_calibration.jsonl")
        test_rows = load_jsonl(split_dir / "test.jsonl")

        all_rows = train_rows + dev_rows + test_rows
        doc_ids = [r["id"] for r in all_rows if str(r.get("evidence", "")).strip()]
        docs = [str(r["evidence"]) for r in all_rows if str(r.get("evidence", "")).strip()]
        retriever = HybridRRFRetriever(use_dense=False)
        if docs:
            retriever.index(doc_ids, docs)

        tr = build_split_signals(
            train_rows, retriever, sem_estimator, nli_scorer, seed + idx * 10 + 1, m_samples
        )
        dv = build_split_signals(
            dev_rows, retriever, sem_estimator, nli_scorer, seed + idx * 10 + 2, m_samples
        )
        te = build_split_signals(
            test_rows, retriever, sem_estimator, nli_scorer, seed + idx * 10 + 3, m_samples
        )

        # Variant A: S_logit only (Stage-1 baseline)
        dv_s1, te_s1 = _fit_and_score_subset(
            tr["s_logit"][:, None],
            tr["is_correct"],
            dv["s_logit"][:, None],
            te["s_logit"][:, None],
            seed=seed,
        )

        # Variant B: S_sem only (Semantic entropy baseline)
        dv_s2, te_s2 = _fit_and_score_subset(
            tr["s_sem"][:, None],
            tr["is_correct"],
            dv["s_sem"][:, None],
            te["s_sem"][:, None],
            seed=seed,
        )

        # Variant C: 2S-CCR (S_logit + S_sem, no retrieval grounding)
        tr_2s = np.column_stack([tr["s_logit"], tr["s_sem"], tr["s_logit"] * tr["s_sem"]])
        dv_2s = np.column_stack([dv["s_logit"], dv["s_sem"], dv["s_logit"] * dv["s_sem"]])
        te_2s = np.column_stack([te["s_logit"], te["s_sem"], te["s_logit"] * te["s_sem"]])
        dv_p2s, te_p2s = _fit_and_score_subset(tr_2s, tr["is_correct"], dv_2s, te_2s, seed=seed)

        # Variant D: Full 3S-CCR Conformal Cascade (S_logit + S_sem + S_ret)
        meta = ThreeSignalMetaLearner(c_reg=1.0, seed=seed)
        meta.fit(tr["s_logit"], tr["s_sem"], tr["s_ret"], tr["is_correct"])
        dv_p1 = meta.predict_stage1(dv["s_logit"])
        dv_pfull = meta.predict_stage2(dv["s_logit"], dv["s_sem"], dv["s_ret"])
        te_p1 = meta.predict_stage1(te["s_logit"])
        te_pfull = meta.predict_stage2(te["s_logit"], te["s_sem"], te["s_ret"])

        router = ConformalCascadingRouter(
            tau_fast_lo=tau_lo, tau_fast_hi=tau_hi, delta=delta, max_refinements=1
        )
        cal_3s = router.calibrate(
            dv_p1, dv_pfull, dv["is_correct"], alpha_grid=alpha_grid, min_accepted=10
        )
        te_g3s, _ = router.compute_unified_scores(te_p1, te_pfull)
        tau_3s = cal_3s[alpha].tau_accept

        # Calibrate single-stage baselines at same alpha
        cal_base_router = ConformalCascadingRouter(
            tau_fast_lo=0.0, tau_fast_hi=1.0, delta=delta, max_refinements=1
        )
        tau_s1 = cal_base_router.calibrate(
            dv_s1, dv_s1, dv["is_correct"], alpha_grid=alpha_grid, min_accepted=10
        )[alpha].tau_accept
        tau_s2 = cal_base_router.calibrate(
            dv_s2, dv_s2, dv["is_correct"], alpha_grid=alpha_grid, min_accepted=10
        )[alpha].tau_accept
        tau_2s = router.calibrate(
            dv_p1, dv_p2s, dv["is_correct"], alpha_grid=alpha_grid, min_accepted=10
        )[alpha].tau_accept
        te_g2s, _ = router.compute_unified_scores(te_p1, te_p2s)

        variants = [
            ("1_S_logit_only", te_s1, tau_s1),
            ("2_S_sem_only", te_s2, tau_s2),
            ("3_2S_CCR_no_ret", te_g2s, tau_2s),
            ("4_3S_CCR_full", te_g3s, tau_3s),
        ]
        for label, scores, tau in variants:
            m = evaluate_selective_operating_point(scores, te["is_correct"], tau)
            ablation_rows.append({"dataset": ds_name, "variant": label, **m})

        # Paired statistical tests: 3S-CCR vs S_logit_only
        boot_vs_s1 = paired_bootstrap_aurc_auroc(te_s1, te_g3s, te["is_correct"], seed=seed)
        mcn_vs_s1 = mcnemar_selective_test(
            te_s1 >= tau_s1, te_g3s >= tau_3s, te["is_correct"]
        )
        stats_rows.append(
            {
                "dataset": ds_name,
                "comparison": "3S-CCR vs S_logit",
                **boot_vs_s1,
                **mcn_vs_s1,
            }
        )

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    ab_csv = out_dir / "v2_ablation_comparison.csv"
    st_csv = out_dir / "v2_statistical_significance.csv"

    with ab_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        w.writeheader()
        w.writerows(ablation_rows)

    with st_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)

    print("\n=== Paired Statistical Significance: 3S-CCR (Full) vs S_logit (Baseline) ===")
    hdr = (
        f"{'Dataset':<15} {'dAURC':>7} {'95% CI':>17} {'p(AURC)':>8} "
        f"{'dAUROC':>7} {'McNemar p':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in stats_rows:
        ci = f"[{r['delta_aurc_ci_lo']:+.4f}, {r['delta_aurc_ci_hi']:+.4f}]"
        print(
            f"{r['dataset']:<15} {r['delta_aurc']:>+7.4f} {ci:>17} "
            f"{r['p_value_aurc']:>8.4f} {r['delta_auroc']:>+7.4f} {r['mcnemar_p_value']:>10.4f}"
        )
    print(f"\nSaved Ablation CSV -> {ab_csv}")
    print(f"Saved Stats CSV    -> {st_csv}")


if __name__ == "__main__":
    main()