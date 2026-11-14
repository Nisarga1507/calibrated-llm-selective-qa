"""Production observability, latency profiling, and distribution drift monitor."""

from __future__ import annotations

import json
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TelemetryTracker:
    """Tracks live inference latency, cascade exit rates, and confidence drift."""

    log_path: Path = field(default_factory=lambda: Path("results/telemetry/inference_log.jsonl"))
    baseline_s_logit_mean: float = 0.82
    drift_threshold: float = 0.15
    window_size: int = 500
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    s_logit_history: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    total_requests: int = 0
    stage1_exits: int = 0
    stage2_escalations: int = 0
    accepted_count: int = 0
    abstained_count: int = 0
    cumulative_cost_gbp: float = 0.0

    def record(
        self,
        *,
        domain: str,
        decision: str,
        stage_executed: int,
        backend_used: str,
        s_logit: float,
        unified_score: float,
        tau_accept: float,
        cost_gbp: float,
        latency_ms: float,
    ) -> None:
        self.total_requests += 1
        if stage_executed == 1:
            self.stage1_exits += 1
        else:
            self.stage2_escalations += 1

        if decision == "ACCEPT":
            self.accepted_count += 1
        else:
            self.abstained_count += 1

        self.cumulative_cost_gbp += float(cost_gbp)
        self.latencies_ms.append(float(latency_ms))
        self.s_logit_history.append(float(s_logit))

        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "domain": domain,
            "decision": decision,
            "stage_executed": stage_executed,
            "backend_used": backend_used,
            "s_logit": round(float(s_logit), 4),
            "unified_score": round(float(unified_score), 4),
            "tau_accept": round(float(tau_accept), 4),
            "cost_gbp": round(float(cost_gbp), 4),
            "latency_ms": round(float(latency_ms), 3),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def summary(self) -> dict:
        lats = sorted(self.latencies_ms)
        n = len(lats)
        p50 = lats[n // 2] if n else 0.0
        p95 = lats[min(n - 1, int(0.95 * n))] if n else 0.0
        mean_lat = statistics.mean(lats) if n else 0.0

        live_s_logit_mean = (
            statistics.mean(self.s_logit_history)
            if self.s_logit_history
            else self.baseline_s_logit_mean
        )
        drift_delta = abs(live_s_logit_mean - self.baseline_s_logit_mean)
        drift_alert = bool(n >= 5 and drift_delta > self.drift_threshold)

        return {
            "total_requests": self.total_requests,
            "stage1_fast_exit_rate": round(
                self.stage1_exits / max(1, self.total_requests), 4
            ),
            "stage2_escalation_rate": round(
                self.stage2_escalations / max(1, self.total_requests), 4
            ),
            "empirical_coverage_rate": round(
                self.accepted_count / max(1, self.total_requests), 4
            ),
            "empirical_abstention_rate": round(
                self.abstained_count / max(1, self.total_requests), 4
            ),
            "cumulative_operational_cost_gbp": round(self.cumulative_cost_gbp, 4),
            "latency_ms": {
                "mean": round(mean_lat, 3),
                "p50": round(p50, 3),
                "p95": round(p95, 3),
            },
            "drift_monitor": {
                "baseline_s_logit_mean": round(self.baseline_s_logit_mean, 4),
                "live_s_logit_mean": round(live_s_logit_mean, 4),
                "abs_drift_delta": round(drift_delta, 4),
                "recalibration_alert": drift_alert,
            },
        }
