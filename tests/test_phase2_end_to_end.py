import json
from pathlib import Path

from hesitation.ml.pipeline import infer_sequence, predict_future_risk, train_classical
from hesitation.policy.recommender import PolicyInput, recommend_policy
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


def _write_synth(path: Path) -> None:
    scenario = ScenarioConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(6):
            traj, latent = generate_session(f"s{i}", scenario, frame_rate_hz=10, seed=400 + i)
            for frame, state in zip(traj.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


def test_end_to_end_synth_to_policy(tmp_path: Path) -> None:
    data = tmp_path / "synth.jsonl"
    artifacts = tmp_path / "artifacts"
    _write_synth(data)

    train_classical(str(data), str(artifacts), window_size=15, pause_speed_threshold=0.03, horizon_frames=10)
    states = infer_sequence(str(data), str(artifacts / "classical_model.json"))
    risks = predict_future_risk(str(data), str(artifacts / "classical_model.json"))

    row_s = states[0]
    row_r = risks[0]
    current_prob = max(row_s["state_probabilities"].values())

    rec = recommend_policy(
        PolicyInput(
            inferred_current_state=row_s["predicted_state"],
            current_hesitation_probability=current_prob,
            future_hesitation_probability=float(row_r["future_hesitation_within_horizon"]),
            future_correction_probability=float(row_r["future_correction_within_horizon"]),
            workspace_distance=0.3,
        )
    )
    assert 0.0 <= rec.recommended_speed_scale <= 1.0
    assert rec.recommended_wait_time_ms >= 0
