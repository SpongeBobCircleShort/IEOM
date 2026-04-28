"""Stable Stage 3 prediction API used by MATLAB bridge."""

from __future__ import annotations

from typing import Any

from hesitation.inference.predictor import HesitationPredictor

STATE_ORDER = [
    "normal_progress",
    "mild_hesitation",
    "strong_hesitation",
    "correction_rework",
    "ready_for_robot_action",
    "overlap_risk",
]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_features(feature_window: dict[str, Any]) -> dict[str, float | int]:
    return {
        "mean_hand_speed": _to_float(feature_window.get("mean_speed")),
        "pause_ratio": _to_float(feature_window.get("pause_ratio")),
        "progress_delta": _to_float(feature_window.get("progress_delta")),
        "reversal_count": _to_int(feature_window.get("reversal_count")),
        "retry_count": _to_int(feature_window.get("retry_count")),
        "task_step_id": _to_int(feature_window.get("task_step"), default=1),
        "human_robot_distance": _to_float(feature_window.get("human_robot_distance")),
    }


def _normalize_probabilities(raw: dict[str, Any]) -> dict[str, float]:
    probs = {state: max(0.0, _to_float(raw.get(state), 0.0)) for state in STATE_ORDER}
    total = sum(probs.values())
    if total <= 0.0:
        return {state: 1.0 / len(STATE_ORDER) for state in STATE_ORDER}
    return {state: value / total for state, value in probs.items()}


def predict(features_dict: dict[str, Any]) -> dict[str, Any]:
    predictor = HesitationPredictor.load_default()
    normalized = _normalize_features(features_dict)
    pred = predictor.predict_single(normalized)

    probabilities = _normalize_probabilities(pred.state_probabilities)
    state = str(pred.state)
    if state not in STATE_ORDER:
        state = "normal_progress"

    return {
        "predicted_state": state,
        "state_probabilities": probabilities,
        "future_hesitation_probability": float(pred.future_hesitation_prob),
        "future_correction_probability": float(pred.future_correction_prob),
    }
