import json
from pathlib import Path

from hesitation.deep.pipeline import compare_models, train_deep
from hesitation.ml.pipeline import train_classical
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


def _write_synth(path: Path, n_sessions: int = 8) -> None:
    scenario = ScenarioConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(n_sessions):
            traj, latent = generate_session(f"s{i}", scenario, frame_rate_hz=10, seed=800 + i)
            for frame, state in zip(traj.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


def test_compare_pipeline_smoke(tmp_path: Path) -> None:
    data = tmp_path / "synth.jsonl"
    classical = tmp_path / "classical"
    deep = tmp_path / "deep"
    reports = tmp_path / "reports"
    _write_synth(data)

    train_classical(str(data), str(classical), window_size=12, pause_speed_threshold=0.03, horizon_frames=8)
    train_deep(str(data), str(deep), window_size=12, horizon_frames=8, epochs=2, hidden_dim=16, learning_rate=0.005)

    deep_model = deep / ("deep_model.pt" if (deep / "deep_model.pt").exists() else "deep_model.json")
    result = compare_models(
        input_path=str(data),
        classical_model_path=str(classical / "classical_model.json"),
        deep_model_path=str(deep_model),
        output_dir=str(reports),
    )
    assert "summary" in result
    assert (reports / "comparison.json").exists()
    assert (reports / "comparison_report.md").exists()
    assert (reports / "comparison_table.csv").exists()
