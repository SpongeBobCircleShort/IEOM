import json
import subprocess
import sys
from pathlib import Path

from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


def _write_synth(path: Path, n_sessions: int = 4) -> None:
    scenario = ScenarioConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(n_sessions):
            traj, latent = generate_session(f"s{i}", scenario, frame_rate_hz=10, seed=300 + i)
            for frame, state in zip(traj.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


def test_cli_train_eval_and_policy(tmp_path: Path) -> None:
    data = tmp_path / "synth.jsonl"
    out_dir = tmp_path / "artifacts"
    infer_out = tmp_path / "infer.jsonl"
    risk_out = tmp_path / "risk.jsonl"
    _write_synth(data)

    subprocess.run(
        [
            sys.executable,
            "scripts/phase2_cli.py",
            "train-classical",
            "--input",
            str(data),
            "--output-dir",
            str(out_dir),
            "--window-size",
            "15",
            "--pause-speed-threshold",
            "0.03",
            "--horizon-frames",
            "10",
        ],
        check=True,
    )
    assert (out_dir / "classical_model.json").exists()

    subprocess.run(
        [
            sys.executable,
            "scripts/phase2_cli.py",
            "infer-sequence",
            "--input",
            str(data),
            "--model-path",
            str(out_dir / "classical_model.json"),
            "--output",
            str(infer_out),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/phase2_cli.py",
            "predict-risk",
            "--input",
            str(data),
            "--model-path",
            str(out_dir / "classical_model.json"),
            "--output",
            str(risk_out),
        ],
        check=True,
    )
    out = subprocess.run(
        [
            sys.executable,
            "scripts/phase2_cli.py",
            "recommend-policy",
            "--current-state",
            "mild_hesitation",
            "--current-hesitation-prob",
            "0.5",
            "--future-hesitation-prob",
            "0.6",
            "--future-correction-prob",
            "0.3",
            "--workspace-distance",
            "0.3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "recommended_robot_mode" in out.stdout
