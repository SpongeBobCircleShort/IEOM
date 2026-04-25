import json
import subprocess
import sys
from pathlib import Path

import pytest

from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


pytestmark = pytest.mark.deep


def _write_synth(path: Path, n_sessions: int = 6) -> None:
    scenario = ScenarioConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(n_sessions):
            traj, latent = generate_session(f"s{i}", scenario, frame_rate_hz=10, seed=500 + i)
            for frame, state in zip(traj.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


def test_phase35_torch_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    data = tmp_path / "synth.jsonl"
    artifacts = tmp_path / "artifacts"
    thresholds = artifacts / "deep_thresholds.json"
    eval_metrics = artifacts / "deep_eval.json"
    calibrated_metrics = artifacts / "deep_eval_calibrated.json"
    _write_synth(data)

    subprocess.run(
        [
            sys.executable,
            "scripts/phase2_cli.py",
            "train-deep",
            "--input",
            str(data),
            "--output-dir",
            str(artifacts),
            "--window-size",
            "15",
            "--horizon-frames",
            "10",
            "--seed",
            "13",
            "--epochs",
            "6",
            "--batch-size",
            "16",
        ],
        check=True,
    )
    assert (artifacts / "deep_model.pt").exists()

    subprocess.run(
        [
            sys.executable,
            "scripts/phase2_cli.py",
            "evaluate-deep",
            "--input",
            str(data),
            "--model-path",
            str(artifacts / "deep_model.pt"),
            "--output",
            str(eval_metrics),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/phase2_cli.py",
            "tune-thresholds",
            "--input",
            str(data),
            "--model-path",
            str(artifacts / "deep_model.pt"),
            "--output",
            str(thresholds),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/phase2_cli.py",
            "evaluate-deep-calibrated",
            "--input",
            str(data),
            "--model-path",
            str(artifacts / "deep_model.pt"),
            "--threshold-path",
            str(thresholds),
            "--output",
            str(calibrated_metrics),
        ],
        check=True,
    )

    eval_payload = json.loads(eval_metrics.read_text(encoding="utf-8"))
    thresholds_payload = json.loads(thresholds.read_text(encoding="utf-8"))
    calibrated_payload = json.loads(calibrated_metrics.read_text(encoding="utf-8"))

    assert "current_state_deep" in eval_payload
    assert set(thresholds_payload) == {"future_hesitation", "future_correction"}
    assert calibrated_payload["thresholds"] == thresholds_payload
