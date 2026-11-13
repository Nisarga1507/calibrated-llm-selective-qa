"""Unit tests verifying LTT Selective Risk Control and cascade state machine."""

import numpy as np

from ccr_router.calibration.conformal_router import ConformalCascadingRouter, RouteAction


def test_ltt_selective_risk_guarantee_on_held_out_data() -> None:
    rng = np.random.default_rng(42)
    n_cal, n_test = 3000, 3000

    # High-signal regime where confident predictions have >98% accuracy
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

    # Verify Stage-2 cascade skips a substantial portion of queries
    assert 0.15 < results[0.05].stage2_trigger_rate < 0.85

    # Verify out-of-sample accepted hallucination rate respects alpha bound
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
    # Noisy model with 35% baseline error everywhere -> alpha=0.01 is impossible
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
    assert

hi


"""3S-CCR: Unified Conformal Cascading Router with LTT Selective Risk Control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import beta


class RouteAction(str, Enum):
    ACCEPT = "ACCEPT"
    REFINE = "REFINE"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class ConformalCalibrationResult:
    alpha: float
    delta: float
    tau_accept: float
    tau_abstain: float
    empirical_risk: float
    ucb_risk: float
    empirical_coverage: float
    stage2_trigger_rate: float
    is_feasible: bool


@dataclass(frozen=True)
class RoutingDecision:
    action: RouteAction
    unified_score: float
    stage2_triggered: bool
    refinement_used: bool
    tau_accept: float
    tau_abstain: float
    alpha_bound: float


class ConformalCascadingRouter:
    """Two-stage cascading router with finite-sample LTT selective risk guarantees."""

    def __init__(
        self,
        tau_fast_lo: float = 0.20,
        tau_fast_hi: float = 0.85,
        delta: float = 0.05,
        max_refinements: int = 1,
        abstain_margin: float = 0.25,
    ) -> None:
        if not (0.0 <= tau_fast_lo < tau_fast_hi <= 1.0):
            raise ValueError("Require 0.0 <= tau_fast_lo < tau_fast_hi <= 1.0")
        if max_refinements != 1:
            raise ValueError("v2.0 enforces strict max_refinements = 1 for deterministic latency.")

        self.tau_fast_lo = tau_fast_lo
        self.tau_fast_hi = tau_fast_hi
        self.delta = delta
        self.max_refinements = max_refinements
        self.abstain_margin = abstain_margin
        self.calibrated_thresholds: dict[float, ConformalCalibrationResult] = {}

    def compute_unified_scores(
        self,
        stage1_probs: np.ndarray,
        full_probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute unified functional g(x) across Stage-1 early exits and Stage-2 queries."""
        p1 = np.asarray(stage1_probs, dtype=np.float64)
        p_full = np.asarray(full_probs, dtype=np.float64)
        if p1.shape != p_full.shape:
            raise ValueError("stage1_probs and full_probs must share identical shape.")

        trigger_stage2 = (p1 > self.tau_fast_lo) & (p1 < self.tau_fast_hi)
        unified_g = np.where(trigger_stage2, p_full, p1)
        return unified_g, trigger_stage2

    @staticmethod
    def clopper_pearson_ucb(k_errors: int, n_accepted: int, delta: float) -> float:
        """Exact finite-sample upper confidence bound on binomial error proportion."""
        if n_accepted <= 0:
            return 1.0
        if k_errors >= n_accepted:
            return 1.0
        return float(beta.ppf(1.0 - delta, k_errors + 1, n_accepted - k_errors))

    def calibrate(
        self,
        stage1_probs: np.ndarray,
        full_probs: np.ndarray,
        is_correct: np.ndarray,
        alpha_grid: list[float] | None = None,
        min_accepted: int = 10,
    ) -> dict[float, ConformalCalibrationResult]:
        """Calibrate tau_accept(alpha) on dev split via Learn-Then-Test UCB scan."""
        if alpha_grid is None:
            alpha_grid = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]

        y_true = np.asarray(is_correct, dtype=bool)
        unified_g, trigger_stage2 = self.compute_unified_scores(stage1_probs, full_probs)
        n_total = len(y_true)
        stage2_rate = float(np.mean(trigger_stage2))

        candidate_taus = np.linspace(0.01, 0.995, 986)

        for alpha in alpha_grid:
            chosen_tau = 1.0001
            chosen_emp_risk = 0.0
            chosen_ucb = 1.0
            chosen_cov = 0.0
            feasible = False

            for tau in candidate_taus:
                accepted_mask = unified_g >= tau
                n_acc = int(np.sum(accepted_mask))
                if n_acc < min_accepted:
                    continue

                k_err = int(np.sum(accepted_mask & (~y_true)))
                ucb = self.clopper_pearson_ucb(k_err, n_acc, self.delta)

                if ucb <= alpha:
                    chosen_tau = float(tau)
                    chosen_emp_risk = k_err / n_acc
                    chosen_ucb = ucb
                    chosen_cov = n_acc / n_total
                    feasible = True
                    break

            tau_abstain = max(0.05, round(min(chosen_tau, 1.0) - self.abstain_margin, 4))
            self.calibrated_thresholds[float(alpha)] = ConformalCalibrationResult(
                alpha=float(alpha),
                delta=self.delta,
                tau_accept=chosen_tau,
                tau_abstain=tau_abstain,
                empirical_risk=chosen_emp_risk,
                ucb_risk=chosen_ucb,
                empirical_coverage=chosen_cov,
                stage2_trigger_rate=stage2_rate,
                is_feasible=feasible,
            )

        return self.calibrated_thresholds

    def route_single(
        self,
        p_stage1: float,
        p_full: float | None = None,
        alpha: float = 0.05,
        refined_score: float | None = None,
    ) -> RoutingDecision:
        """Route a single query with deterministic 1-hop refinement cap."""
        if alpha not in self.calibrated_thresholds:
            raise KeyError(f"Alpha {alpha} has not been calibrated yet.")

        cal = self.calibrated_thresholds[alpha]
        trigger_s2 = (p_stage1 > self.tau_fast_lo) and (p_stage1 < self.tau_fast_hi)

        if trigger_s2:
            if p_full is None:
                raise ValueError("p_full must be provided when Stage-2 cascade is triggered.")
            g_score = float(p_full)
        else:
            g_score = float(p_stage1)

        if g_score >= cal.tau_accept:
            return RoutingDecision(
                action=RouteAction.ACCEPT,
                unified_score=g_score,
                stage2_triggered=trigger_s2,
                refinement_used=False,
                tau_accept=cal.tau_accept,
                tau_abstain=cal.tau_abstain,
                alpha_bound=alpha,
            )

        if g_score >= cal.tau_abstain:
            if refined_score is None:
                return RoutingDecision(
                    action=RouteAction.REFINE,
                    unified_score=g_score,
                    stage2_triggered=trigger_s2,
                    refinement_used=False,
                    tau_accept=cal.tau_accept,
                    tau_abstain=cal.tau_abstain,
                    alpha_bound=alpha,
                )
            final_action = (
                RouteAction.ACCEPT if refined_score >= cal.tau_accept else RouteAction.ABSTAIN
            )
            return RoutingDecision(
                action=final_action,
                unified_score=float(refined_score),
                stage2_triggered=trigger_s2,
                refinement_used=True,
                tau_accept=cal.tau_accept,
                tau_abstain=cal.tau_abstain,
                alpha_bound=alpha,
            )

        return RoutingDecision(
            action=RouteAction.ABSTAIN,
            unified_score=g_score,
            stage2_triggered=trigger_s2,
            refinement_used=False,
            tau_accept=cal.tau_accept,
            tau_abstain=cal.tau_abstain,
            alpha_bound=alpha,
        )
"""3S-CCR: Unified Conformal Cascading Router with LTT Selective Risk Control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import beta


class RouteAction(str, Enum):
    ACCEPT = "ACCEPT"
    REFINE = "REFINE"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class ConformalCalibrationResult:
    alpha: float
    delta: float
    tau_accept: float
    tau_abstain: float
    empirical_risk: float
    ucb_risk: float
    empirical_coverage: float
    stage2_trigger_rate: float
    is_feasible: bool


@dataclass(frozen=True)
class RoutingDecision:
    action: RouteAction
    unified_score: float
    stage2_triggered: bool
    refinement_used: bool
    tau_accept: float
    tau_abstain: float
    alpha_bound: float


class ConformalCascadingRouter:
    """Two-stage cascading router with finite-sample LTT selective risk guarantees."""

    def __init__(
        self,
        tau_fast_lo: float = 0.20,
        tau_fast_hi: float = 0.85,
        delta: float = 0.05,
        max_refinements: int = 1,
        abstain_margin: float = 0.25,
    ) -> None:
        if not (0.0 <= tau_fast_lo < tau_fast_hi <= 1.0):
            raise ValueError("Require 0.0 <= tau_fast_lo < tau_fast_hi <= 1.0")
        if max_refinements != 1:
            raise ValueError("v2.0 enforces strict max_refinements = 1 for deterministic latency.")

        self.tau_fast_lo = tau_fast_lo
        self.tau_fast_hi = tau_fast_hi
        self.delta = delta
        self.max_refinements = max_refinements
        self.abstain_margin = abstain_margin
        self.calibrated_thresholds: dict[float, ConformalCalibrationResult] = {}

    def compute_unified_scores(
        self,
        stage1_probs: np.ndarray,
        full_probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute unified functional g(x) across Stage-1 early exits and Stage-2 queries."""
        p1 = np.asarray(stage1_probs, dtype=np.float64)
        p_full = np.asarray(full_probs, dtype=np.float64)
        if p1.shape != p_full.shape:
            raise ValueError("stage1_probs and full_probs must share identical shape.")

        trigger_stage2 = (p1 > self.tau_fast_lo) & (p1 < self.tau_fast_hi)
        unified_g = np.where(trigger_stage2, p_full, p1)
        return unified_g, trigger_stage2

    @staticmethod
    def clopper_pearson_ucb(k_errors: int, n_accepted: int, delta: float) -> float:
        """Exact finite-sample upper confidence bound on binomial error proportion."""
        if n_accepted <= 0:
            return 1.0
        if k_errors >= n_accepted:
            return 1.0
        return float(beta.ppf(1.0 - delta, k_errors + 1, n_accepted - k_errors))

    def calibrate(
        self,
        stage1_probs: np.ndarray,
        full_probs: np.ndarray,
        is_correct: np.ndarray,
        alpha_grid: list[float] | None = None,
        min_accepted: int = 10,
    ) -> dict[float, ConformalCalibrationResult]:
        """Calibrate tau_accept(alpha) on dev split via Learn-Then-Test UCB scan."""
        if alpha_grid is None:
            alpha_grid = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]

        y_true = np.asarray(is_correct, dtype=bool)
        unified_g, trigger_stage2 = self.compute_unified_scores(stage1_probs, full_probs)
        n_total = len(y_true)
        stage2_rate = float(np.mean(trigger_stage2))

        candidate_taus = np.linspace(0.01, 0.995, 986)

        for alpha in alpha_grid:
            chosen_tau = 1.0001
            chosen_emp_risk = 0.0
            chosen_ucb = 1.0
            chosen_cov = 0.0
            feasible = False

            for tau in candidate_taus:
                accepted_mask = unified_g >= tau
                n_acc = int(np.sum(accepted_mask))
                if n_acc < min_accepted:
                    continue

                k_err = int(np.sum(accepted_mask & (~y_true)))
                ucb = self.clopper_pearson_ucb(k_err, n_acc, self.delta)

                if ucb <= alpha:
                    chosen_tau = float(tau)
                    chosen_emp_risk = k_err / n_acc
                    chosen_ucb = ucb
                    chosen_cov = n_acc / n_total
                    feasible = True
                    break

            tau_abstain = max(0.05, round(min(chosen_tau, 1.0) - self.abstain_margin, 4))
            self.calibrated_thresholds[float(alpha)] = ConformalCalibrationResult(
                alpha=float(alpha),
                delta=self.delta,
                tau_accept=chosen_tau,
                tau_abstain=tau_abstain,
                empirical_risk=chosen_emp_risk,
                ucb_risk=chosen_ucb,
                empirical_coverage=chosen_cov,
                stage2_trigger_rate=stage2_rate,
                is_feasible=feasible,
            )

        return self.calibrated_thresholds

    def route_single(
        self,
        p_stage1: float,
        p_full: float | None = None,
        alpha: float = 0.05,
        refined_score: float | None = None,
    ) -> RoutingDecision:
        """Route a single query with deterministic 1-hop refinement cap."""
        if alpha not in self.calibrated_thresholds:
            raise KeyError(f"Alpha {alpha} has not been calibrated yet.")

        cal = self.calibrated_thresholds[alpha]
        trigger_s2 = (p_stage1 > self.tau_fast_lo) and (p_stage1 < self.tau_fast_hi)

        if trigger_s2:
            if p_full is None:
                raise ValueError("p_full must be provided when Stage-2 cascade is triggered.")
            g_score = float(p_full)
        else:
            g_score = float(p_stage1)

        if g_score >= cal.tau_accept:
            return RoutingDecision(
                action=RouteAction.ACCEPT,
                unified_score=g_score,
                stage2_triggered=trigger_s2,
                refinement_used=False,
                tau_accept=cal.tau_accept,
                tau_abstain=cal.tau_abstain,
                alpha_bound=alpha,
            )

        if g_score >= cal.tau_abstain:
            if refined_score is None:
                return RoutingDecision(
                    action=RouteAction.REFINE,
                    unified_score=g_score,
                    stage2_triggered=trigger_s2,
                    refinement_used=False,
                    tau_accept=cal.tau_accept,
                    tau_abstain=cal.tau_abstain,
                    alpha_bound=alpha,
                )
            final_action = (
                RouteAction.ACCEPT if refined_score >= cal.tau_accept else RouteAction.ABSTAIN
            )
            return RoutingDecision(
                action=final_action,
                unified_score=float(refined_score),
                stage2_triggered=trigger_s2,
                refinement_used=True,
                tau_accept=cal.tau_accept,
                tau_abstain=cal.tau_abstain,
                alpha_bound=alpha,
            )

        return RoutingDecision(
            action=RouteAction.ABSTAIN,
            unified_score=g_score,
            stage2_triggered=trigger_s2,
            refinement_used=False,
            tau_accept=cal.tau_accept,
            tau_abstain=cal.tau_abstain,
            alpha_bound=alpha,
        )
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
ruff check src tests
pytest tests/test_conformal_guarantees.py -v
