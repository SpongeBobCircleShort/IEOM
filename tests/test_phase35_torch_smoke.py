import importlib.util
import json
from pathlib import Path

import pytest

from hesitation.deep.pipeline import train_deep
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


def _write_synth(path: Path, n_sessions: int = 4) -> None:
    scenario = ScenarioConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(n_sessions):
            traj, latent = generate_session(f"s{i}", scenario, frame_rate_hz=10, seed=1200 + i)
            for frame, state in zip(traj.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_true_torch_training_path(tmp_path: Path) -> None:
    data = tmp_path / "synth.jsonl"
    artifacts = tmp_path / "deep"
    _write_synth(data)
    metrics = train_deep(str(data), str(artifacts), window_size=12, horizon_frames=8, epochs=1, hidden_dim=8, learning_rate=0.01, batch_size=16)
    assert metrics["backend"] == "torch"
    assert (artifacts / "deep_model.pt").exists()
