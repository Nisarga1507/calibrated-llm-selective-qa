"""Integration tests for the 3S-CCR FastAPI routing and generation service."""

from fastapi.testclient import TestClient

from api import app


def test_health_and_routing_cascade() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert "uk_healthcare" in body["domains_indexed"]

        # 1. Stage-1 high-confidence fast accept on UK Finance
        fast_res = client.post(
            "/v2/route",
            json={
                "domain": "uk_finance",
                "question": "What is the UK Financial Conduct Authority Consumer Duty?",
                "candidate_answer": "Requires firms to deliver good outcomes for retail customers.",
                "token_logprobs": [-0.02, -0.01, -0.02, -0.01],
                "top2_margins": [4.5, 4.8, 4.6, 4.9],
                "target_alpha": 0.05,
            },
        )
        assert fast_res.status_code == 200
        fast_data = fast_res.json()
        assert fast_data["stage_executed"] == 1
        assert fast_data["decision"] == "ACCEPT"
        assert fast_data["s_sem"] is None

        # 2. Boundary query escalated to Stage-2 semantic + RRF verification
        esc_res = client.post(
            "/v2/route",
            json={
                "domain": "uk_healthcare",
                "question": "What is the first-line NICE treatment for uncomplicated hypertension?",
                "candidate_answer": "An ACE inhibitor or angiotensin receptor blocker.",
                "token_logprobs": [-0.45, -0.50, -0.42, -0.48],
                "top2_margins": [0.9, 1.1, 0.8, 1.0],
                "sampled_answers": [
                    "An ACE inhibitor or angiotensin receptor blocker.",
                    "ACE inhibitor or ARB therapy.",
                    "Calcium channel blocker.",
                ],
                "target_alpha": 0.05,
            },
        )
        assert esc_res.status_code == 200
        esc_data = esc_res.json()
        assert esc_data["stage_executed"] == 2
        assert esc_data["s_sem"] is not None
        assert esc_data["s_ret"] is not None
        assert esc_data["decision"] == "ABSTAIN_ESCALATE_HUMAN"


def test_end_to_end_generate_and_route() -> None:
    with TestClient(app) as client:
        res = client.post(
            "/v2/generate-and-route",
            json={
                "domain": "uk_law",
                "question": "What is the experimental unverified statutory threshold?",
                "num_samples": 5,
                "target_alpha": 0.05,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["stage_executed"] == 2
        assert data["decision"] == "ABSTAIN_ESCALATE_HUMAN"
        assert data["candidate_answer"]
