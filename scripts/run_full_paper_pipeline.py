#!/usr/bin/env python3
"""Orchestrate the full reproducible paper benchmark pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hesitation.evaluation.suite import run_benchmark_suite
from hesitation.io.config import load_config


KEY_OUTPUT_RELATIVE_PATHS = [
    "input_manifest.json",
    "suite/suite_summary.json",
    "suite/manifest_snapshot.json",
    "suite/paper/final_benchmark_table.csv",
    "suite/paper/ablation_table.csv",
    "suite/paper/transfer_gap_notes.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="artifacts/paper")
    parser.add_argument("--config", default="configs/benchmark/paper_final_locked.yaml")
    parser.add_argument("--seed", type=int, default=140)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _stage_inputs(output_root: Path, seed: int) -> dict[str, Any]:
    inputs_dir = output_root / "inputs"
    cmd = [
        sys.executable,
        "scripts/generate_paper_benchmark_inputs.py",
        "--output-dir",
        str(inputs_dir),
        "--seed",
        str(seed),
    ]
    subprocess.run(cmd, check=True)
    return json.loads((inputs_dir / "input_manifest.json").read_text(encoding="utf-8"))


def _materialize_runtime_config(config_path: str, staged_inputs: dict[str, Any], output_root: Path, seed: int) -> tuple[Path, dict[str, Any]]:
    config = load_config(config_path)
    datasets = config.get("datasets", {})
    if "chico" in datasets:
        datasets["chico"]["input_path"] = staged_inputs["chico_path"]
    if "ha_vid" in datasets:
        datasets["ha_vid"]["input_path"] = staged_inputs["havid"]["path"]

    benchmark = config.setdefault("benchmark", {})
    deep = benchmark.setdefault("deep", {})
    deep["seed"] = int(seed)

    runtime_path = output_root / "runtime_config.yaml"
    runtime_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return runtime_path, config


def _run_locked_suite(runtime_config_path: Path, output_root: Path) -> dict[str, Any]:
    suite_dir = output_root / "suite"
    return run_benchmark_suite(str(runtime_config_path), str(suite_dir))


def _strict_integrity_check(output_root: Path, summary: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    required = [
        output_root / "input_manifest.json",
        output_root / "suite" / "suite_summary.json",
        output_root / "suite" / "benchmarks" / "chico_within" / "run_summary.json",
        output_root / "suite" / "benchmarks" / "havid_within" / "run_summary.json",
        output_root / "suite" / "benchmarks" / "merged_train_eval" / "run_summary.json",
        output_root / "suite" / "paper" / "final_benchmark_table.csv",
        output_root / "suite" / "paper" / "ablation_table.csv",
        output_root / "suite" / "paper" / "figures" / "pipeline_overview.svg",
    ]
    for path in required:
        if not path.exists():
            issues.append(f"missing:{path}")

    benchmark_runs = summary.get("benchmark_runs", [])
    ablation_runs = summary.get("ablation_runs", [])
    if len(benchmark_runs) < 3:
        issues.append("benchmark_runs_lt_3")
    if len(ablation_runs) < 1:
        issues.append("ablation_runs_lt_1")
    return issues


def _package_artifacts(output_root: Path) -> Path:
    package_dir = output_root / "package"
    package_dir.mkdir(parents=True, exist_ok=True)
    archive_base = package_dir / "paper_artifacts"
    archive_path = shutil.make_archive(str(archive_base), "gztar", root_dir=output_root, base_dir="suite")
    return Path(archive_path)


def _package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scikit-learn", "pyyaml", "torch"]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _write_reproducibility_manifest(
    output_root: Path,
    config_path: str,
    resolved_config: dict[str, Any],
    seed: int,
    archive_path: Path,
) -> Path:
    checksums: dict[str, str] = {}
    for rel in KEY_OUTPUT_RELATIVE_PATHS:
        path = output_root / rel
        if path.exists():
            checksums[rel] = _sha256(path)
    checksums[str(archive_path.relative_to(output_root))] = _sha256(archive_path)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_commit_hash(),
        "config_path": config_path,
        "config_snapshot": resolved_config,
        "seed_map": {
            "pipeline_seed": seed,
            "havid_input_seed": seed,
            "suite_deep_seed": seed,
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "env_vars": {
                "PYTHONHASHSEED": os.getenv("PYTHONHASHSEED", ""),
            },
            "packages": _package_versions(),
        },
        "checksums": checksums,
    }
    manifest_path = output_root / "reproducibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    staged_manifest = _stage_inputs(output_root, args.seed)
    shutil.copyfile(
        output_root / "inputs" / "input_manifest.json",
        output_root / "input_manifest.json",
    )
    runtime_config_path, resolved_config = _materialize_runtime_config(
        args.config,
        staged_manifest,
        output_root,
        args.seed,
    )
    suite_summary = _run_locked_suite(runtime_config_path, output_root)

    integrity_issues = _strict_integrity_check(output_root, suite_summary)
    integrity = {
        "strict_mode": bool(args.strict),
        "ok": not integrity_issues,
        "issues": integrity_issues,
    }
    (output_root / "integrity_report.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    if args.strict and integrity_issues:
        print(json.dumps(integrity, indent=2))
        return 2

    archive_path = _package_artifacts(output_root)
    manifest_path = _write_reproducibility_manifest(
        output_root,
        args.config,
        resolved_config,
        args.seed,
        archive_path,
    )

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "suite_summary": str(output_root / "suite" / "suite_summary.json"),
                "integrity_report": str(output_root / "integrity_report.json"),
                "artifact_archive": str(archive_path),
                "reproducibility_manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
