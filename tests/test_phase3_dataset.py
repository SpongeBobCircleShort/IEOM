import json
from pathlib import Path

from hesitation.deep.dataset import build_sequence_windows, load_rows
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


def _write_synth(path: Path, n_sessions: int = 4) -> None:
    scenario = ScenarioConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(n_sessions):
            traj, latent = generate_session(f"s{i}", scenario, frame_rate_hz=10, seed=500 + i)
            for frame, state in zip(traj.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


def test_sequence_windows_shape(tmp_path: Path) -> None:
    data = tmp_path / "synth.jsonl"
    _write_synth(data)
    rows = load_rows(data)
    windows = build_sequence_windows(rows, window_size=12, horizon_frames=8)
    assert len(windows) > 0
    assert len(windows[0]["sequence"]) == 12
    assert len(windows[0]["sequence"][0]) == 8
