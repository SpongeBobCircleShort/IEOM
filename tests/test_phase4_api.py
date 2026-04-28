from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from hesitation.api.main import app
from hesitation.ml.pipeline import train_classical
from tests.phase4_support import frames_to_payload, sample_session_frames, write_synth_dataset


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "rules" in payload["supported_backends"]


def test_policy_recommend_endpoint() -> None:
    client = TestClient(app)
    response = client.post(
        "/policy/recommend",
        json={
            "inferred_current_state": "mild_hesitation",
            "current_hesitation_probability": 0.6,
            "future_hesitation_probability": 0.7,
            "future_correction_probability": 0.2,
            "workspace_distance": 0.2,
        },
    )
    assert response.status_code == 200
    assert response.json()["recommended_robot_mode"] == "assistive_slow"


def test_infer_current_state_rules_endpoint() -> None:
    client = TestClient(app)
    frames = sample_session_frames()
    response = client.post(
        "/infer/current-state",
        json={
            "frames": frames_to_payload(frames),
            "artifact": {
                "backend": "rules",
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "rules"
    assert payload["predicted_state"]
    assert "state_probabilities" in payload


def test_infer_full_classical_endpoint(tmp_path: Path) -> None:
    client = TestClient(app)
    data = tmp_path / "synth.jsonl"
    artifacts = tmp_path / "classical"
    write_synth_dataset(data)
    train_classical(str(data), str(artifacts), window_size=15, pause_speed_threshold=0.03, horizon_frames=10)

    frames = sample_session_frames(n_frames=20)
    response = client.post(
        "/infer/full",
        json={
            "frames": frames_to_payload(frames),
            "artifact": {
                "backend": "classical",
                "model_path": str(artifacts / "classical_model.json"),
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["current_state"]["backend"] == "classical"
    assert "predicted_state" in payload["current_state"]
    assert "future_hesitation_probability" in payload["future_risk"]
    assert "recommended_robot_mode" in payload["policy"]


def test_reports_compare_endpoint(tmp_path: Path) -> None:
    client = TestClient(app)
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    (left_dir / "metrics.json").write_text(json.dumps({"accuracy": 0.5, "nested": {"f1": 0.4}}), encoding="utf-8")
    (right_dir / "metrics.json").write_text(json.dumps({"accuracy": 0.7, "nested": {"f1": 0.6}}), encoding="utf-8")

    response = client.post(
        "/reports/compare",
        json={
            "left_path": str(left_dir),
            "right_path": str(right_dir),
            "left_label": "baseline",
            "right_label": "candidate",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["shared_metric_count"] >= 2
    assert any(row["metric"] == "accuracy" for row in payload["comparison_rows"])
