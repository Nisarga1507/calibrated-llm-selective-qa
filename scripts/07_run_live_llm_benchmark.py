"""Stage 7 CLI: Run live open-weight LLM (Ollama) or calibrated fallback benchmark + telemetry."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api import app

SAMPLE_QUERIES = [
    ("uk_finance", "What are the retail customer obligations under UK FCA Consumer Duty?"),
    ("uk_healthcare", "What is the first-line NICE treatment for uncomplicated hypertension?"),
    ("uk_healthcare", "What is the experimental off-label dosing protocol for refractory cases?"),
    ("uk_law", "Under UK statutory employment law, when does unfair dismissal protection apply?"),
    ("fever", "Was the Eiffel Tower completed in 1889 for the Exposition Universelle?"),
]


def main() -> None:
    results: list[dict] = []
    with TestClient(app) as client:
        for domain, question in SAMPLE_QUERIES:
            resp = client.post(
                "/v2/generate-and-route",
                json={
                    "domain": domain,
                    "question": question,
                    "model_name": "llama3.1:8b",
                    "num_samples": 5,
                    "target_alpha": 0.05,
                },
            )
            resp.raise_for_status()
            results.append(resp.json())

        telemetry = client.get("/v2/metrics").json()

    out_path = Path("results/tables/live_llm_telemetry.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"queries": results, "telemetry_summary": telemetry}, indent=2),
        encoding="utf-8",
    )
    print(f"[07_run_live_llm_benchmark] Saved {len(results)} routed queries to {out_path}")
    print(json.dumps(telemetry, indent=2))


if __name__ == "__main__":
    main()
