import json
from pathlib import Path

from hesitation.deep.pipeline import compare_models_multiseed, train_deep_multiseed
from hesitation.ml.pipeline import train_classical
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


def _write_synth(path: Path, n_sessions: int = 8) -> None:
    scenario = ScenarioConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(n_sessions):
            traj, latent = generate_session(f"s{i}", scenario, frame_rate_hz=10, seed=1100 + i)
            for frame, state in zip(traj.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


def test_multiseed_aggregation(tmp_path: Path) -> None:
    data = tmp_path / "synth.jsonl"
    classical = tmp_path / "classical"
    deep_root = tmp_path / "deep_ms"
    compare_dir = tmp_path / "compare_ms"
    _write_synth(data)

    train_classical(str(data), str(classical), window_size=12, pause_speed_threshold=0.03, horizon_frames=8)
    agg = train_deep_multiseed(
        input_path=str(data),
        output_dir=str(deep_root),
        seeds=[1, 2],
        window_size=12,
        horizon_frames=8,
        epochs=2,
        hidden_dim=16,
        learning_rate=0.005,
        batch_size=32,
    )
    assert agg["n_seeds"] == 2

    comp = compare_models_multiseed(
        input_path=str(data),
        classical_model_path=str(classical / "classical_model.json"),
        deep_root_dir=str(deep_root),
        seeds=[1, 2],
        output_dir=str(compare_dir),
    )
    assert comp["n_seeds"] == 2
    assert (compare_dir / "comparison_multiseed.json").exists()
