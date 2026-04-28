#!/usr/bin/env python3
"""Stable Stage 3 prediction bridge CLI for MATLAB integration."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hesitation.inference.stage3_bridge import STATE_ORDER, predict


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _validate_output(payload: dict[str, Any]) -> None:
    required = [
        "predicted_state",
        "state_probabilities",
        "future_hesitation_probability",
        "future_correction_probability",
    ]
    for key in required:
        if key not in payload:
            raise ValueError(f"Missing output key: {key}")

    if payload["predicted_state"] not in STATE_ORDER:
        raise ValueError(f"Invalid predicted_state: {payload['predicted_state']}")

    probs = payload["state_probabilities"]
    if not isinstance(probs, dict):
        raise ValueError("state_probabilities must be dict")

    total = 0.0
    for key in STATE_ORDER:
        value = _to_float(probs.get(key), -1.0)
        if value < 0.0 or value > 1.0 or not math.isfinite(value):
            raise ValueError(f"Invalid probability {key}={value}")
        total += value
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"state_probabilities sum != 1.0 ({total})")

    for risk_key in ["future_hesitation_probability", "future_correction_probability"]:
        value = _to_float(payload[risk_key], -1.0)
        if value < 0.0 or value > 1.0 or not math.isfinite(value):
            raise ValueError(f"Invalid risk value {risk_key}={value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-json", required=True)
    args = parser.parse_args()
    try:
        features = json.loads(args.features_json)
        output = predict(features)
        _validate_output(output)
        print(json.dumps(output))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
