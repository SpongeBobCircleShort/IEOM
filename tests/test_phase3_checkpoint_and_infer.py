import json
from pathlib import Path

from hesitation.deep.pipeline import infer_sequence_deep, train_deep
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


def _write_synth(path: Path, n_sessions: int = 5) -> None:
    scenario = ScenarioConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(n_sessions):
            traj, latent = generate_session(f"s{i}", scenario, frame_rate_hz=10, seed=700 + i)
            for frame, state in zip(traj.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


def test_checkpoint_load_and_prediction_range(tmp_path: Path) -> None:
    data = tmp_path / "synth.jsonl"
    artifacts = tmp_path / "deep"
    _write_synth(data)
    train_deep(str(data), str(artifacts), window_size=12, horizon_frames=8, epochs=2, hidden_dim=16, learning_rate=0.005)

    model_path = artifacts / ("deep_model.pt" if (artifacts / "deep_model.pt").exists() else "deep_model.json")
    rows = infer_sequence_deep(str(data), str(model_path))
    assert len(rows) > 0
    first = rows[0]
    assert 0.99 <= sum(first["state_probabilities"].values()) <= 1.01
    assert 0.0 <= first["future_hesitation_within_horizon"] <= 1.0
    assert 0.0 <= first["future_correction_within_horizon"] <= 1.0
