from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException

from hesitation.api.schemas import (
    ArtifactSpecPayload,
    CurrentStateResponse,
    FullInferenceResponse,
    FutureRiskResponse,
    HealthResponse,
    InferenceRequest,
    PolicyRecommendationResponse,
    PolicyRequest,
    ReportCompareRequest,
    ReportCompareResponse,
)
from hesitation.policy.recommender import PolicyInput, recommend_policy
from hesitation.schemas.events import FrameObservation
from hesitation.serving import (
    ADVISORY_NOTICE,
    ArtifactSpec,
    compare_report_sources,
    infer_from_frames,
    recommend_from_inference,
    supported_backends,
)
from hesitation.serving.runtime import InferenceResult

app = FastAPI(
    title="Hesitation Phase 4 API",
    version="0.4.0",
    description="Inference and advisory serving layer for the hesitation prototype. Advisory only; not safety-certified control.",
)


def _build_frames(request: InferenceRequest) -> list[FrameObservation]:
    """Convert API payload frames into validated runtime observations."""
    return [FrameObservation(**frame.model_dump()) for frame in request.frames]


def _build_artifact_spec(payload: ArtifactSpecPayload) -> ArtifactSpec:
    """Convert the API artifact payload into the serving-layer spec."""
    return ArtifactSpec(
        backend=payload.backend,
        model_path=payload.model_path,
        threshold_path=payload.threshold_path,
        rules_config_path=payload.rules_config_path,
        pause_speed_threshold=payload.pause_speed_threshold,
    )


def _current_state_response(result: InferenceResult) -> CurrentStateResponse:
    """Project the unified inference result into the current-state contract."""
    return CurrentStateResponse(
        backend=result.backend,
        predicted_state=result.predicted_state,
        state_probabilities=result.state_probabilities,
        current_hesitation_probability=result.current_hesitation_probability,
        feature_window=result.feature_window,
        model_source=result.model_source,
        window_size_used=result.window_size_used,
        overlap_risk=result.overlap_risk,
        advisory_notice=ADVISORY_NOTICE,
    )


def _future_risk_response(result: InferenceResult) -> FutureRiskResponse:
    """Project the unified inference result into the future-risk contract."""
    return FutureRiskResponse(
        backend=result.backend,
        future_hesitation_probability=result.future_hesitation_probability,
        future_correction_probability=result.future_correction_probability,
        thresholds=result.thresholds,
        feature_window=result.feature_window,
        model_source=result.model_source,
        window_size_used=result.window_size_used,
        overlap_risk=result.overlap_risk,
        advisory_notice=ADVISORY_NOTICE,
    )


def _policy_response_from_recommendation(recommendation: PolicyInput | object) -> PolicyRecommendationResponse:
    """Normalize a policy dataclass into the API response model."""
    return PolicyRecommendationResponse(
        recommended_robot_mode=getattr(recommendation, "recommended_robot_mode"),
        recommended_speed_scale=float(getattr(recommendation, "recommended_speed_scale")),
        recommended_clearance_scale=float(getattr(recommendation, "recommended_clearance_scale")),
        recommended_wait_time_ms=int(getattr(recommendation, "recommended_wait_time_ms")),
        recommend_confirmation_gate=bool(getattr(recommendation, "recommend_confirmation_gate")),
        rationale=str(getattr(recommendation, "rationale")),
        advisory_notice=ADVISORY_NOTICE,
    )


def _handle_inference(request: InferenceRequest) -> InferenceResult:
    """Run one inference request and map local errors into HTTP 400 responses."""
    try:
        return infer_from_frames(_build_frames(request), _build_artifact_spec(request.artifact))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return basic service health and backend availability."""
    return HealthResponse(
        status="ok",
        torch_available="deep" in supported_backends(),
        supported_backends=supported_backends(),
        advisory_notice=ADVISORY_NOTICE,
    )


@app.post("/infer/current-state", response_model=CurrentStateResponse)
def infer_current_state(request: InferenceRequest) -> CurrentStateResponse:
    """Run only the current-state portion of the inference flow."""
    return _current_state_response(_handle_inference(request))


@app.post("/infer/future-risk", response_model=FutureRiskResponse)
def infer_future_risk(request: InferenceRequest) -> FutureRiskResponse:
    """Run only the future-risk portion of the inference flow."""
    return _future_risk_response(_handle_inference(request))


@app.post("/infer/full", response_model=FullInferenceResponse)
def infer_full(request: InferenceRequest) -> FullInferenceResponse:
    """Run inference and advisory policy recommendation in one request."""
    result = _handle_inference(request)
    policy = recommend_from_inference(result, workspace_distance=request.workspace_distance_override)
    return FullInferenceResponse(
        current_state=_current_state_response(result),
        future_risk=_future_risk_response(result),
        policy=_policy_response_from_recommendation(policy),
    )


@app.post("/policy/recommend", response_model=PolicyRecommendationResponse)
def policy_recommend(request: PolicyRequest) -> PolicyRecommendationResponse:
    """Produce an advisory robot recommendation from probability inputs."""
    recommendation = recommend_policy(
        PolicyInput(
            inferred_current_state=request.inferred_current_state,
            current_hesitation_probability=request.current_hesitation_probability,
            future_hesitation_probability=request.future_hesitation_probability,
            future_correction_probability=request.future_correction_probability,
            workspace_distance=request.workspace_distance,
        )
    )
    return _policy_response_from_recommendation(recommendation)


@app.post("/reports/compare", response_model=ReportCompareResponse)
def reports_compare(request: ReportCompareRequest) -> ReportCompareResponse:
    """Compare two report artifacts by flattening shared numeric metrics."""
    try:
        payload = compare_report_sources(
            request.left_path,
            request.right_path,
            left_label=request.left_label,
            right_label=request.right_label,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:  # type: ignore[name-defined]
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ReportCompareResponse(**payload)
