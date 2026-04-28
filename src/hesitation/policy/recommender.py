from dataclasses import dataclass

from hesitation.schemas.labels import HesitationState


@dataclass(slots=True)
class PolicyInput:
    inferred_current_state: str
    current_hesitation_probability: float
    future_hesitation_probability: float
    future_correction_probability: float
    workspace_distance: float


@dataclass(slots=True)
class PolicyRecommendation:
    recommended_robot_mode: str
    recommended_speed_scale: float
    recommended_clearance_scale: float
    recommended_wait_time_ms: int
    recommend_confirmation_gate: bool
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "recommended_robot_mode": self.recommended_robot_mode,
            "recommended_speed_scale": self.recommended_speed_scale,
            "recommended_clearance_scale": self.recommended_clearance_scale,
            "recommended_wait_time_ms": self.recommended_wait_time_ms,
            "recommend_confirmation_gate": self.recommend_confirmation_gate,
            "rationale": self.rationale,
        }


def recommend_policy(inp: PolicyInput) -> PolicyRecommendation:
    risk = max(
        inp.current_hesitation_probability,
        inp.future_hesitation_probability,
        inp.future_correction_probability
    )
    overlap_sensitive = inp.workspace_distance < 0.25 or inp.inferred_current_state == HesitationState.OVERLAP_RISK.value

    if inp.inferred_current_state in {HesitationState.CORRECTION_REWORK.value, HesitationState.STRONG_HESITATION.value}:
        return PolicyRecommendation(
            recommended_robot_mode="hold",
            recommended_speed_scale=0.0,
            recommended_clearance_scale=1.4,
            recommended_wait_time_ms=1800,
            recommend_confirmation_gate=True,
            rationale="high uncertainty/correction: hold and request confirmation",
        )

    if risk >= 0.7 or overlap_sensitive:
        return PolicyRecommendation(
            recommended_robot_mode="assistive_slow",
            recommended_speed_scale=0.35,
            recommended_clearance_scale=1.25,
            recommended_wait_time_ms=900,
            recommend_confirmation_gate=True,
            rationale="elevated near-term risk or overlap sensitivity",
        )

    if inp.inferred_current_state == HesitationState.READY_FOR_ROBOT_ACTION.value and risk < 0.35:
        return PolicyRecommendation(
            recommended_robot_mode="proceed",
            recommended_speed_scale=0.9,
            recommended_clearance_scale=1.0,
            recommended_wait_time_ms=50,
            recommend_confirmation_gate=False,
            rationale="ready state with low predicted risk",
        )

    return PolicyRecommendation(
        recommended_robot_mode="normal_monitor",
        recommended_speed_scale=0.65,
        recommended_clearance_scale=1.1,
        recommended_wait_time_ms=300,
        recommend_confirmation_gate=False,
        rationale="default conservative advisory mode",
    )
