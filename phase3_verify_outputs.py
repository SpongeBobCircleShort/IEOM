#!/usr/bin/env python3
"""Phase 3 output verification for three fixed scenarios.

Runs the inference CLI on deterministic feature vectors and validates:
- output schema fields exist
- confidence is in [0, 1]
- probabilities sum approximately to 1.0
- repeated invocation is deterministic for each scenario
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Scenario:
    name: str
    features: dict[str, float | int]


SCENARIOS: list[Scenario] = [
    Scenario(
        name="normal_progress_like",
        features={
            "mean_hand_speed": 0.55,
            "pause_ratio": 0.08,
            "progress_delta": 0.82,
            "reversal_count": 0,
            "retry_count": 0,
            "task_step_id": 2,
            "human_robot_distance": 0.75,
        },
    ),
    Scenario(
        name="hesitation_like",
        features={
            "mean_hand_speed": 0.07,
            "pause_ratio": 0.72,
            "progress_delta": 0.18,
            "reversal_count": 2,
            "retry_count": 1,
            "task_step_id": 4,
            "human_robot_distance": 0.33,
        },
    ),
    Scenario(
        name="overlap_risk_like",
        features={
            "mean_hand_speed": 0.30,
            "pause_ratio": 0.20,
            "progress_delta": 0.48,
            "reversal_count": 1,
            "retry_count": 0,
            "task_step_id": 5,
            "human_robot_distance": 0.12,
        },
    ),
]


def _python_executable() -> str:
    preferred = Path("/opt/homebrew/bin/python3.11")
    if preferred.exists():
        return str(preferred)
    return sys.executable


def _run_cli_predict(features: dict[str, float | int]) -> dict[str, Any]:
    model_root = Path(__file__).parent
    cmd = [
        _python_executable(),
        "-m",
        "hesitation.inference.cli",
        "--format",
        "json",
        "predict",
        "--mean-hand-speed",
        str(features["mean_hand_speed"]),
        "--pause-ratio",
        str(features["pause_ratio"]),
        "--progress-delta",
        str(features["progress_delta"]),
        "--reversal-count",
        str(features["reversal_count"]),
        "--retry-count",
        str(features["retry_count"]),
        "--task-step-id",
        str(features["task_step_id"]),
        "--human-robot-distance",
        str(features["human_robot_distance"]),
    ]
    env = os.environ.copy()
    src_path = str(model_root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}:{existing}" if existing else src_path
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=model_root, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"CLI failed: {result.stderr or result.stdout}")
    return json.loads(result.stdout)


def _validate_output(output: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    required = [
        "state",
        "state_probabilities",
        "future_hesitation_prob",
        "future_correction_prob",
        "confidence",
    ]
    for key in required:
        if key not in output:
            issues.append(f"missing_field:{key}")

    confidence = float(output.get("confidence", -1))
    if not (0.0 <= confidence <= 1.0):
        issues.append("confidence_out_of_range")

    probs = output.get("state_probabilities", {})
    if isinstance(probs, dict) and probs:
        total = sum(float(v) for v in probs.values())
        if abs(total - 1.0) > 1e-6:
            issues.append(f"probability_sum_not_1:{total}")
    else:
        issues.append("invalid_probabilities")
    return issues


def main() -> int:
    report_rows: list[dict[str, Any]] = []
    all_ok = True

    for scenario in SCENARIOS:
        out1 = _run_cli_predict(scenario.features)
        out2 = _run_cli_predict(scenario.features)
        issues = _validate_output(out1)
        deterministic = out1 == out2
        if not deterministic:
            issues.append("non_deterministic_output")
        if issues:
            all_ok = False

        report_rows.append(
            {
                "scenario": scenario.name,
                "features": scenario.features,
                "state": out1.get("state"),
                "confidence": out1.get("confidence"),
                "future_hesitation_prob": out1.get("future_hesitation_prob"),
                "future_correction_prob": out1.get("future_correction_prob"),
                "deterministic": deterministic,
                "issues": issues,
            }
        )

    summary = {
        "all_passed": all_ok,
        "scenario_count": len(report_rows),
        "rows": report_rows,
    }
    out_dir = Path(__file__).parent / "reports" / "phase3_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "output_verification_report.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out_path), "all_passed": all_ok}, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
