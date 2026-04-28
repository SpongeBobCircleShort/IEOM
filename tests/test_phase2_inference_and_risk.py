import json
from pathlib import Path

from hesitation.ml.pipeline import infer_sequence, predict_future_risk, train_classical
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


def _write_synth(path: Path, n_sessions: int = 5) -> None:
    scenario = ScenarioConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(n_sessions):
            traj, latent = generate_session(f"s{i}", scenario, frame_rate_hz=10, seed=200 + i)
            for frame, state in zip(traj.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


def test_model_load_and_infer_probabilities(tmp_path: Path) -> None:
    data = tmp_path / "synth.jsonl"
    out_dir = tmp_path / "artifacts"
    _write_synth(data)
    train_classical(str(data), str(out_dir), window_size=15, pause_speed_threshold=0.03, horizon_frames=10)

    state_rows = infer_sequence(str(data), str(out_dir / "classical_model.json"))
    assert len(state_rows) > 0
    sample_probs = state_rows[0]["state_probabilities"]
    assert 0.99 <= sum(sample_probs.values()) <= 1.01

    risk_rows = predict_future_risk(str(data), str(out_dir / "classical_model.json"))
    assert len(risk_rows) > 0
    assert 0.0 <= risk_rows[0]["future_hesitation_within_horizon"] <= 1.0
    assert 0.0 <= risk_rows[0]["future_correction_within_horizon"] <= 1.0
