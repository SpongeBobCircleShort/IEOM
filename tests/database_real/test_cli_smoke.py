import json
import subprocess
import sys


def test_cli_normalize_smoke(tmp_path) -> None:
    cmd = [
        sys.executable,
        "scripts/real_dataset_onboarding.py",
        "normalize-chico",
        "--raw",
        "tests/fixtures/chico/raw/chico_realistic_sample.jsonl",
        "--mapping",
        "merged_database/configs/chico_mapping_rules.yaml",
        "--output",
        str(tmp_path / "norm.jsonl"),
        "--report",
        str(tmp_path / "map.json"),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["result"][0].endswith("norm.jsonl")
