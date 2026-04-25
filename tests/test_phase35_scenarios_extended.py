import json
import subprocess
import sys
from pathlib import Path


def test_generate_scenarios_extended(tmp_path: Path) -> None:
    out = tmp_path / "extended.jsonl"
    subprocess.run(
        [
            sys.executable,
            "scripts/phase2_cli.py",
            "generate-scenarios-extended",
            "--output",
            str(out),
            "--configs",
            "configs/simulation/default_scene.yaml,configs/simulation/ambiguous_scene.yaml",
            "--sessions-per-scenario",
            "2",
            "--frame-rate",
            "10",
            "--seed",
            "55",
        ],
        check=True,
    )
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0
    sample = json.loads(lines[0])
    assert "scenario_name" in sample
