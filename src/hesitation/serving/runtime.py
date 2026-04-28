from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from hesitation.baselines.rules_engine import classify_window
from hesitation.features.pipeline import window_to_features
from hesitation.io.config import load_config
from hesitation.ml.deep import (
    SEQUENCE_FEATURE_ORDER,
    build_sequence_windows,
    load_deep_runtime,
    predict_deep_window,
    torch,
)
from hesitation.ml.dataset import FEATURE_ORDER
from hesitation.ml.pipeline import load_classical_runtime, predict_classical_window
from hesitation.policy.recommender import PolicyInput, PolicyRecommendation, recommend_policy
from hesitation.schemas.events import FrameObservation
from hesitation.schemas.features import FeatureWindow
from hesitation.schemas.labels import HesitationState

BackendKind = Literal["rules", "classical", "deep"]
ADVISORY_NOTICE = "Predictions and recommendations are advisory only and are not safety-certified robot control."


@dataclass(slots=True)
class ArtifactSpec:
    """Describe which backend to run and where to load saved artifacts."""

    backend: BackendKind
    model_path: str | None = None
    threshold_path: str | None = None
    rules_config_path: str = "configs/baseline/rules_v1.yaml"
    pause_speed_threshold: float = 0.03


@dataclass(slots=True)
class InferenceResult:
    """Unified output contract used by the API and demo."""

    backend: BackendKind
    predicted_state: str
    state_probabilities: dict[str, float]
    current_hesitation_probability: float
    future_hesitation_probability: float
    future_correction_probability: float
    thresholds: dict[str, float]
    feature_window: dict[str, float | int | str]
    model_source: str
    window_size_used: int
    overlap_risk: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert the dataclass into a JSON-serializable payload."""
        return asdict(self)


def supported_backends() -> list[str]:
    """Return the serving backends available in the current environment."""
    backends = ["rules", "classical"]
    if torch is not None:
        backends.append("deep")
    return backends


def _require_model_path(spec: ArtifactSpec) -> str:
    if not spec.model_path:
        raise ValueError(f"`model_path` is required for backend `{spec.backend}`.")
    return spec.model_path


def _validate_frames(frames: list[FrameObservation], window_size: int) -> list[FrameObservation]:
    if len(frames) < window_size:
        raise ValueError(f"Need at least {window_size} frames, received {len(frames)}.")
    ordered = sorted(frames, key=lambda frame: frame.frame_idx)
    return ordered[-window_size:]


def _hesitation_probability_from_state_probabilities(state_probabilities: dict[str, float]) -> float:
    risky_states = {
        HesitationState.MILD_HESITATION.value,
        HesitationState.STRONG_HESITATION.value,
        HesitationState.CORRECTION_REWORK.value,
        HesitationState.OVERLAP_RISK.value,
    }
    return min(1.0, sum(float(state_probabilities.get(state_name, 0.0)) for state_name in risky_states))


def _feature_window_to_dict(feature_window: FeatureWindow) -> dict[str, float | int | str]:
    return {
        "session_id": feature_window.session_id,
        "end_frame_idx": feature_window.end_frame_idx,
        "mean_speed": feature_window.mean_speed,
        "speed_variance": feature_window.speed_variance,
        "pause_ratio": feature_window.pause_ratio,
        "direction_changes": feature_window.direction_changes,
        "progress_delta": feature_window.progress_delta,
        "backtrack_ratio": feature_window.backtrack_ratio,
        "mean_workspace_distance": feature_window.mean_workspace_distance,
    }


def _compute_feature_window(frames: list[FrameObservation], pause_speed_threshold: float) -> FeatureWindow:
    return window_to_features(frames, pause_speed_threshold=pause_speed_threshold)


def _feature_vector(feature_window: FeatureWindow) -> list[float]:
    values = _feature_window_to_dict(feature_window)
    return [float(values[name]) for name in FEATURE_ORDER]


def _predict_rules(frames: list[FrameObservation], spec: ArtifactSpec) -> InferenceResult:
    config = load_config(spec.rules_config_path)
    window_frames = _validate_frames(frames, window_size=2)
    features = _compute_feature_window(window_frames, pause_speed_threshold=spec.pause_speed_threshold)
    label = classify_window(features, thresholds=config["thresholds"], risk_cfg=config["risk"])
    state_probabilities = {state.value: 0.0 for state in HesitationState}
    state_probabilities[label.current_state.value] = 1.0
    return InferenceResult(
        backend="rules",
        predicted_state=label.current_state.value,
        state_probabilities=state_probabilities,
        current_hesitation_probability=float(label.hesitation_risk),
        future_hesitation_probability=float(label.hesitation_risk),
        future_correction_probability=float(label.correction_rework_risk),
        thresholds={
            "future_hesitation": float(config["risk"]["high_pause_ratio"]),
            "future_correction": 0.5,
        },
        feature_window=_feature_window_to_dict(features),
        model_source=spec.rules_config_path,
        window_size_used=len(window_frames),
        overlap_risk=float(label.overlap_risk),
    )


def _predict_classical(frames: list[FrameObservation], spec: ArtifactSpec) -> InferenceResult:
    model_path = _require_model_path(spec)
    runtime = load_classical_runtime(model_path)
    payload = runtime["payload"]
    window_frames = _validate_frames(frames, window_size=int(payload["window_size"]))
    features = _compute_feature_window(window_frames, pause_speed_threshold=float(payload["pause_speed_threshold"]))
    prediction = predict_classical_window(runtime, _feature_vector(features))
    state_probabilities = {name: float(value) for name, value in prediction["state_probabilities"].items()}
    return InferenceResult(
        backend="classical",
        predicted_state=str(prediction["predicted_state"]),
        state_probabilities=state_probabilities,
        current_hesitation_probability=_hesitation_probability_from_state_probabilities(state_probabilities),
        future_hesitation_probability=float(prediction["future_hesitation_probability"]),
        future_correction_probability=float(prediction["future_correction_probability"]),
        thresholds={key: float(value) for key, value in prediction["thresholds"].items()},
        feature_window=_feature_window_to_dict(features),
        model_source=model_path,
        window_size_used=len(window_frames),
    )


def _predict_deep(frames: list[FrameObservation], spec: ArtifactSpec) -> InferenceResult:
    model_path = _require_model_path(spec)
    runtime = load_deep_runtime(model_path, threshold_path=spec.threshold_path)
    checkpoint = runtime["checkpoint"]
    window_size = int(checkpoint["window_size"])
    window_frames = _validate_frames(frames, window_size=window_size)
    sequence_rows = [
        {
            "session_id": frame.session_id,
            "frame_idx": frame.frame_idx,
            "timestamp_ms": frame.timestamp_ms,
            "task_step_id": frame.task_step_id,
            "hand_x": frame.hand_x,
            "hand_y": frame.hand_y,
            "hand_speed": frame.hand_speed,
            "hand_accel": frame.hand_accel,
            "distance_to_robot_workspace": frame.distance_to_robot_workspace,
            "progress": frame.progress,
            "confidence": frame.confidence,
            "is_dropout": frame.is_dropout,
        }
        for frame in window_frames
    ]
    windows = build_sequence_windows(sequence_rows, window_size=window_size, horizon_frames=1)
    if not windows:
        sequence_features = [
            [
                float(getattr(frame, field_name))
                if field_name != "is_dropout"
                else float(int(frame.is_dropout))
                for field_name in SEQUENCE_FEATURE_ORDER
            ]
            for frame in window_frames
        ]
    else:
        sequence_features = windows[-1]["sequence_features"]  # type: ignore[index]
    prediction = predict_deep_window(runtime, sequence_features)
    features = _compute_feature_window(window_frames, pause_speed_threshold=spec.pause_speed_threshold)
    state_probabilities = {name: float(value) for name, value in prediction["state_probabilities"].items()}
    return InferenceResult(
        backend="deep",
        predicted_state=str(prediction["predicted_state"]),
        state_probabilities=state_probabilities,
        current_hesitation_probability=_hesitation_probability_from_state_probabilities(state_probabilities),
        future_hesitation_probability=float(prediction["future_hesitation_probability"]),
        future_correction_probability=float(prediction["future_correction_probability"]),
        thresholds={key: float(value) for key, value in prediction["thresholds"].items()},
        feature_window=_feature_window_to_dict(features),
        model_source=model_path,
        window_size_used=len(window_frames),
    )


def infer_from_frames(frames: list[FrameObservation], spec: ArtifactSpec) -> InferenceResult:
    """Run one inference pass over an ordered frame sequence."""
    if spec.backend == "rules":
        return _predict_rules(frames, spec)
    if spec.backend == "classical":
        return _predict_classical(frames, spec)
    if spec.backend == "deep":
        return _predict_deep(frames, spec)
    raise ValueError(f"Unsupported backend `{spec.backend}`.")


def recommend_from_inference(
    inference: InferenceResult,
    workspace_distance: float | None = None,
) -> PolicyRecommendation:
    """Generate an advisory policy recommendation from a prediction result."""
    if workspace_distance is None:
        workspace_distance = float(inference.feature_window["mean_workspace_distance"])
    return recommend_policy(
        PolicyInput(
            inferred_current_state=inference.predicted_state,
            current_hesitation_probability=inference.current_hesitation_probability,
            future_hesitation_probability=inference.future_hesitation_probability,
            future_correction_probability=inference.future_correction_probability,
            workspace_distance=float(workspace_distance),
        )
    )
