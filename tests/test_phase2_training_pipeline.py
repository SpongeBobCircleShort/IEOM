import json
from pathlib import Path

from hesitation.ml.pipeline import evaluate_classical, train_classical
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


def _write_synth(path: Path, n_sessions: int = 6) -> None:
    scenario = ScenarioConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(n_sessions):
            traj, latent = generate_session(f"s{i}", scenario, frame_rate_hz=10, seed=100 + i)
            for frame, state in zip(traj.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


def test_train_and_evaluate_pipeline(tmp_path: Path) -> None:
    data = tmp_path / "synth.jsonl"
    out_dir = tmp_path / "artifacts"
    _write_synth(data)

    metrics = train_classical(str(data), str(out_dir), window_size=15, pause_speed_threshold=0.03, horizon_frames=10)
    assert "current_state_classical" in metrics
    assert (out_dir / "classical_model.json").exists()

    eval_metrics = evaluate_classical(str(data), str(out_dir / "classical_model.json"))
    assert "future_hesitation" in eval_metrics
    assert eval_metrics["windows"] > 0
