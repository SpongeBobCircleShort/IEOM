import json
from pathlib import Path

from hesitation.deep.pipeline import evaluate_deep, train_deep
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


def _write_synth(path: Path, n_sessions: int = 6) -> None:
    scenario = ScenarioConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(n_sessions):
            traj, latent = generate_session(f"s{i}", scenario, frame_rate_hz=10, seed=600 + i)
            for frame, state in zip(traj.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


def test_train_and_evaluate_deep(tmp_path: Path) -> None:
    data = tmp_path / "synth.jsonl"
    artifacts = tmp_path / "deep"
    _write_synth(data)

    metrics = train_deep(
        input_path=str(data),
        output_dir=str(artifacts),
        window_size=15,
        horizon_frames=10,
        epochs=2,
        hidden_dim=16,
        learning_rate=0.005,
        seed=42,
    )
    assert "current_state_deep" in metrics
    assert (artifacts / "deep_metrics.json").exists()

    model_path = artifacts / ("deep_model.pt" if (artifacts / "deep_model.pt").exists() else "deep_model.json")
    eval_out = evaluate_deep(str(data), str(model_path))
    assert eval_out["windows"] > 0
