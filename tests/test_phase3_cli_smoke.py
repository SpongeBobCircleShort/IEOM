import json
import subprocess
import sys
from pathlib import Path

from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


def _write_synth(path: Path, n_sessions: int = 6) -> None:
    scenario = ScenarioConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(n_sessions):
            traj, latent = generate_session(f"s{i}", scenario, frame_rate_hz=10, seed=900 + i)
            for frame, state in zip(traj.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


def test_phase3_cli_commands(tmp_path: Path) -> None:
    data = tmp_path / "synth.jsonl"
    classical = tmp_path / "classical"
    deep = tmp_path / "deep"
    infer_out = tmp_path / "deep_infer.jsonl"
    compare_out = tmp_path / "compare"
    _write_synth(data)

    subprocess.run(
        [
            sys.executable,
            "scripts/phase2_cli.py",
            "train-classical",
            "--input",
            str(data),
            "--output-dir",
            str(classical),
            "--window-size",
            "12",
            "--pause-speed-threshold",
            "0.03",
            "--horizon-frames",
            "8",
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/phase2_cli.py",
            "train-deep",
            "--input",
            str(data),
            "--output-dir",
            str(deep),
            "--window-size",
            "12",
            "--horizon-frames",
            "8",
            "--epochs",
            "2",
            "--hidden-dim",
            "16",
            "--learning-rate",
            "0.005",
        ],
        check=True,
    )

    model_path = deep / ("deep_model.pt" if (deep / "deep_model.pt").exists() else "deep_model.json")

    subprocess.run(
        [
            sys.executable,
            "scripts/phase2_cli.py",
            "infer-sequence-deep",
            "--input",
            str(data),
            "--model-path",
            str(model_path),
            "--output",
            str(infer_out),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/phase2_cli.py",
            "compare-models",
            "--input",
            str(data),
            "--classical-model-path",
            str(classical / "classical_model.json"),
            "--deep-model-path",
            str(model_path),
            "--output-dir",
            str(compare_out),
        ],
        check=True,
    )

    assert infer_out.exists()
    assert (compare_out / "comparison.json").exists()
