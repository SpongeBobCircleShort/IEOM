import json
import subprocess
import sys
from pathlib import Path


from hesitation.io.config import load_config


def _build_fixture_scale_config(path: Path, input_root: Path) -> None:
    config = load_config("configs/benchmark/paper_final_locked.yaml")
    config["benchmark"]["deep"]["epochs"] = 1
    config["benchmark"]["deep"]["batch_size"] = 8
    config["datasets"]["chico"]["splits"] = {
        "train": ["chico_s0", "chico_s1"],
        "test": ["chico_s2"],
    }
    config["datasets"]["ha_vid"]["splits"] = {
        "train": ["havid_s0", "havid_s1"],
        "test": ["havid_s2", "havid_s3"],
    }
    config["datasets"]["chico"]["input_path"] = str(input_root / "chico_model_input.jsonl")
    config["datasets"]["ha_vid"]["input_path"] = str(input_root / "havid_model_input.jsonl")
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def _run_pipeline(output_root: Path, config_path: Path, seed: int) -> dict:
    cmd = [
        sys.executable,
        "scripts/run_full_paper_pipeline.py",
        "--output-root",
        str(output_root),
        "--config",
        str(config_path),
        "--seed",
        str(seed),
        "--strict",
    ]
    subprocess.run(cmd, check=True)
    return json.loads((output_root / "reproducibility_manifest.json").read_text(encoding="utf-8"))


def test_run_full_paper_pipeline_smoke_stable_checksums(tmp_path: Path) -> None:
    config_path = tmp_path / "fixture_locked.yaml"
    input_root = tmp_path / "placeholder_inputs"
    _build_fixture_scale_config(config_path, input_root)

    first_manifest = _run_pipeline(tmp_path / "run_a", config_path, seed=321)
    second_manifest = _run_pipeline(tmp_path / "run_b", config_path, seed=321)

    selected_keys = [
        "suite/paper/final_benchmark_table.csv",
        "suite/paper/ablation_table.csv",
        "suite/paper/transfer_gap_notes.md",
    ]
    for key in selected_keys:
        assert key in first_manifest["checksums"]
        assert key in second_manifest["checksums"]
        assert first_manifest["checksums"][key] == second_manifest["checksums"][key]

    assert first_manifest["seed_map"]["pipeline_seed"] == 321
    assert second_manifest["seed_map"]["pipeline_seed"] == 321
