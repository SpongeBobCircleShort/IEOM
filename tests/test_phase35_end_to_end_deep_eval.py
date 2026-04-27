import json
from pathlib import Path

from hesitation.deep.pipeline import evaluate_deep_calibrated, train_deep, tune_thresholds
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


def _write_synth(path: Path, n_sessions: int = 6) -> None:
    scenario = ScenarioConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(n_sessions):
            traj, latent = generate_session(f"s{i}", scenario, frame_rate_hz=10, seed=1300 + i)
            for frame, state in zip(traj.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


def test_end_to_end_calibrated_eval(tmp_path: Path) -> None:
    data = tmp_path / "synth.jsonl"
    artifacts = tmp_path / "deep"
    thresholds = tmp_path / "thresholds.json"
    _write_synth(data)

    train_deep(str(data), str(artifacts), window_size=12, horizon_frames=8, epochs=2, hidden_dim=16, learning_rate=0.005)
    model_path = artifacts / ("deep_model.pt" if (artifacts / "deep_model.pt").exists() else "deep_model.json")
    tune_thresholds(str(data), str(model_path), str(thresholds))
    metrics = evaluate_deep_calibrated(str(data), str(model_path), str(thresholds))

    assert "thresholds" in metrics
    assert "future_hesitation" in metrics
    assert "future_correction" in metrics
