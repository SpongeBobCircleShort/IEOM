from __future__ import annotations

from pathlib import Path

import pytest

from hesitation.serving.runtime import ArtifactSpec, infer_from_frames
from hesitation.ml.pipeline import train_classical
from tests.phase4_support import sample_session_frames, write_synth_dataset


def test_classical_model_loading_smoke(tmp_path: Path) -> None:
    data = tmp_path / "synth.jsonl"
    artifacts = tmp_path / "classical"
    write_synth_dataset(data)
    train_classical(str(data), str(artifacts), window_size=15, pause_speed_threshold=0.03, horizon_frames=10)

    result = infer_from_frames(
        sample_session_frames(n_frames=20),
        ArtifactSpec(backend="classical", model_path=str(artifacts / "classical_model.json")),
    )
    assert result.backend == "classical"
    assert result.model_source.endswith("classical_model.json")
    assert result.predicted_state


@pytest.mark.deep
def test_deep_model_loading_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    from hesitation.ml.deep import train_deep

    data = tmp_path / "synth.jsonl"
    artifacts = tmp_path / "deep"
    write_synth_dataset(data)
    train_deep(str(data), str(artifacts), window_size=15, horizon_frames=10, epochs=3, batch_size=16, hidden_size=12)

    result = infer_from_frames(
        sample_session_frames(n_frames=20),
        ArtifactSpec(backend="deep", model_path=str(artifacts / "deep_model.pt")),
    )
    assert result.backend == "deep"
    assert result.model_source.endswith("deep_model.pt")
    assert result.predicted_state
