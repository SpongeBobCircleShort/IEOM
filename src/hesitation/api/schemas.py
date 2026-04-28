from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FrameObservationPayload(BaseModel):
    """JSON request schema for one observed frame."""

    session_id: str = Field(..., description="Session identifier for the sequence.")
    frame_idx: int = Field(..., ge=0, description="Monotonic frame index.")
    timestamp_ms: int = Field(..., ge=0, description="Capture time in milliseconds.")
    task_step_id: int = Field(..., ge=0, description="Task step index for the frame.")
    hand_x: float = Field(..., description="Hand x-position.")
    hand_y: float = Field(..., description="Hand y-position.")
    hand_speed: float = Field(..., ge=0.0, description="Hand speed magnitude.")
    hand_accel: float = Field(..., description="Hand acceleration.")
    distance_to_robot_workspace: float = Field(..., ge=0.0, description="Distance to the robot workspace.")
    progress: float = Field(..., ge=0.0, le=1.0, description="Task progress in [0, 1].")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Observation confidence in [0, 1].")
    is_dropout: bool = Field(False, description="Whether the frame came from a dropout/imputed interval.")


class ArtifactSpecPayload(BaseModel):
    """Describe which backend and artifacts to use for inference."""

    backend: Literal["rules", "classical", "deep"] = Field(..., description="Inference backend to run.")
    model_path: str | None = Field(None, description="Saved classical JSON or deep `.pt` artifact path.")
    threshold_path: str | None = Field(None, description="Optional calibrated threshold JSON for deep inference.")
    rules_config_path: str = Field("configs/baseline/rules_v1.yaml", description="Rules config path for the rules backend.")
    pause_speed_threshold: float = Field(0.03, ge=0.0, description="Fallback pause threshold for inline feature extraction.")


class InferenceRequest(BaseModel):
    """Run inference over one frame sequence."""

    frames: list[FrameObservationPayload] = Field(..., min_length=1, description="Ordered or orderable frame sequence.")
    artifact: ArtifactSpecPayload = Field(..., description="Backend selection and artifact paths.")
    workspace_distance_override: float | None = Field(
        None,
        ge=0.0,
        description="Optional override used only for policy recommendation in `/infer/full`.",
    )


class PolicyRequest(BaseModel):
    """Direct advisory policy input payload."""

    inferred_current_state: str = Field(..., description="Current predicted or inferred state.")
    current_hesitation_probability: float = Field(..., ge=0.0, le=1.0)
    future_hesitation_probability: float = Field(..., ge=0.0, le=1.0)
    future_correction_probability: float = Field(..., ge=0.0, le=1.0)
    workspace_distance: float = Field(..., ge=0.0)


class ReportCompareRequest(BaseModel):
    """Compare metrics from two report artifacts."""

    left_path: str = Field(..., description="Path to a JSON report file or artifact directory.")
    right_path: str = Field(..., description="Path to a JSON report file or artifact directory.")
    left_label: str = Field("left", description="Display label for the left artifact.")
    right_label: str = Field("right", description="Display label for the right artifact.")


class HealthResponse(BaseModel):
    """Service health payload."""

    status: str
    torch_available: bool
    supported_backends: list[str]
    advisory_notice: str


class CurrentStateResponse(BaseModel):
    """Current-state inference response payload."""

    backend: str
    predicted_state: str
    state_probabilities: dict[str, float]
    current_hesitation_probability: float
    feature_window: dict[str, Any]
    model_source: str
    window_size_used: int
    overlap_risk: float | None = None
    advisory_notice: str


class FutureRiskResponse(BaseModel):
    """Future-risk inference response payload."""

    backend: str
    future_hesitation_probability: float
    future_correction_probability: float
    thresholds: dict[str, float]
    feature_window: dict[str, Any]
    model_source: str
    window_size_used: int
    overlap_risk: float | None = None
    advisory_notice: str


class PolicyRecommendationResponse(BaseModel):
    """Advisory policy recommendation response."""

    recommended_robot_mode: str
    recommended_speed_scale: float
    recommended_clearance_scale: float
    recommended_wait_time_ms: int
    recommend_confirmation_gate: bool
    rationale: str
    advisory_notice: str


class FullInferenceResponse(BaseModel):
    """Combined inference and policy response."""

    current_state: CurrentStateResponse
    future_risk: FutureRiskResponse
    policy: PolicyRecommendationResponse


class ReportCompareResponse(BaseModel):
    """Report comparison payload."""

    left_label: str
    right_label: str
    left: dict[str, Any]
    right: dict[str, Any]
    shared_metric_count: int
    comparison_rows: list[dict[str, float | str | int | None]]
