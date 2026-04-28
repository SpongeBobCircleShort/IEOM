import json
import subprocess
import sys


def test_cli_normalize_chico_smoke(tmp_path) -> None:
    cmd = [
        sys.executable,
        "scripts/real_dataset_onboarding.py",
        "normalize-chico",
        "--raw",
        "tests/fixtures/chico/raw/chico_realistic_sample.jsonl",
        "--mapping",
        "merged_database/configs/chico_mapping_rules.yaml",
        "--output",
        str(tmp_path / "chico_norm.jsonl"),
        "--report",
        str(tmp_path / "chico_map.json"),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["result"][0].endswith("chico_norm.jsonl")


def test_cli_normalize_havid_smoke(tmp_path) -> None:
    cmd = [
        "python",
        "scripts/real_dataset_onboarding.py",
        "normalize-havid",
        "--raw",
        "tests/fixtures/havid/raw/havid_realistic_sample.jsonl",
        "--mapping",
        "merged_database/configs/havid_mapping_rules.yaml",
        "--output",
        str(tmp_path / "havid_norm.jsonl"),
        "--report",
        str(tmp_path / "havid_map.json"),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["result"][0].endswith("havid_norm.jsonl")
