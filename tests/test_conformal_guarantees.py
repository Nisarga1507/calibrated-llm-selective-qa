"""Unit tests verifying LTT Selective Risk Control and cascade state machine."""

import numpy as np

from ccr_router.calibration.conformal_router import ConformalCascadingRouter, RouteAction


def test_ltt_selective_risk_guarantee_on_held_out_data() -> None:
    rng = np.random.default_rng(42)
    n_cal, n_test = 3000, 3000

    true_prob_cal = np.clip(rng.beta(3.5, 1.2, size=n_cal), 0.02, 0.995)
    y_cal = rng.binomial(1, true_prob_cal).astype(bool)
    p1_cal = np.clip(true_prob_cal + rng.normal(0, 0.03, n_cal), 0.001, 0.999)
    pfull_cal = np.clip(true_prob_cal + rng.normal(0, 0.015, n_cal), 0.001, 0.999)

    true_prob_test = np.clip(rng.beta(3.5, 1.2, size=n_test), 0.02, 0.995)
    y_test = rng.binomial(1, true_prob_test).astype(bool)
    p1_test = np.clip(true_prob_test + rng.normal(0, 0.03, n_test), 0.001, 0.999)
    pfull_test = np.clip(true_prob_test + rng.normal(0, 0.015, n_test), 0.001, 0.999)

    router = ConformalCascadingRouter(tau_fast_lo=0.20, tau_fast_hi=0.85, delta=0.05)
    results = router.calibrate(p1_cal, pfull_cal, y_cal, alpha_grid=[0.05, 0.10])

    assert 0.15 < results[0.05].stage2_trigger_rate < 0.85

    g_test, _ = router.compute_unified_scores(p1_test, pfull_test)
    for alpha in [0.05, 0.10]:
        assert results[alpha].is_feasible is True
        tau = results[alpha].tau_accept
        accepted = g_test >= tau
        assert np.sum(accepted) > 50
        empirical_test_error = np.mean(~y_test[accepted])
        assert empirical_test_error <= alpha + 0.01


def test_infeasible_alpha_triggers_safe_full_abstention() -> None:
    rng = np.random.default_rng(7)
    p1 = rng.uniform(0.1, 0.9, size=500)
    pfull = rng.uniform(0.1, 0.9, size=500)
    y_true = rng.binomial(1, 0.65, size=500).astype(bool)

    router = ConformalCascadingRouter()
    res = router.calibrate(p1, pfull, y_true, alpha_grid=[0.01])
    assert res[0.01].is_feasible is False
    assert res[0.01].tau_accept > 1.0


def test_cascade_fast_exit_and_one_hop_refinement_cap() -> None:
    router = ConformalCascadingRouter(tau_fast_lo=0.20, tau_fast_hi=0.85)
    router.calibrate(
        stage1_probs=np.linspace(0.05, 0.99, 500),
        full_probs=np.linspace(0.05, 0.99, 500),
        is_correct=(np.linspace(0.05, 0.99, 500) > 0.45),
        alpha_grid=[0.10],
    )

    fast_accept = router.route_single(p_stage1=0.96, p_full=None, alpha=0.10)
    assert fast_accept.stage2_triggered is False
    assert fast_accept.action == RouteAction.ACCEPT

    cal = router.calibrated_thresholds[0.10]
    mid_score = (cal.tau_abstain + cal.tau_accept) / 2.0
    first_pass = router.route_single(p_stage1=0.50, p_full=mid_score, alpha=0.10)
    assert first_pass.stage2_triggered is True
    assert first_pass.action == RouteAction.REFINE

    second_pass_fail = router.route_single(
        p_stage1=0.50, p_full=mid_score, alpha=0.10, refined_score=mid_score
    )
    assert second_pass_fail.refinement_used is True
    assert second_pass_fail.action == RouteAction.ABSTAIN