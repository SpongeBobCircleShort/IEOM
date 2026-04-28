from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.deep

pytest.importorskip("fastapi")
pytest.importorskip("torch")
from fastapi.testclient import TestClient

from hesitation.api.main import app
from hesitation.ml.deep import train_deep, tune_thresholds
from tests.phase4_support import frames_to_payload, sample_session_frames, write_synth_dataset


def test_infer_full_deep_endpoint(tmp_path: Path) -> None:
    client = TestClient(app)
    data = tmp_path / "synth.jsonl"
    artifacts = tmp_path / "deep"
    thresholds_path = artifacts / "thresholds.json"
    write_synth_dataset(data)
    train_deep(str(data), str(artifacts), window_size=15, horizon_frames=10, epochs=3, batch_size=16, hidden_size=12)
    tune_thresholds(str(data), str(artifacts / "deep_model.pt"), output_path=str(thresholds_path))

    response = client.post(
        "/infer/full",
        json={
            "frames": frames_to_payload(sample_session_frames(n_frames=20)),
            "artifact": {
                "backend": "deep",
                "model_path": str(artifacts / "deep_model.pt"),
                "threshold_path": str(thresholds_path),
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["current_state"]["backend"] == "deep"
    assert payload["future_risk"]["thresholds"]["future_hesitation"] >= 0.0
