from dataclasses import dataclass, field
from enum import Enum


class HesitationState(str, Enum):
    NORMAL_PROGRESS = "normal_progress"
    MILD_HESITATION = "mild_hesitation"
    STRONG_HESITATION = "strong_hesitation"
    CORRECTION_REWORK = "correction_rework"
    READY_FOR_ROBOT_ACTION = "ready_for_robot_action"
    OVERLAP_RISK = "overlap_risk"


@dataclass
class LabelOutput:
    current_state: HesitationState
    hesitation_risk: float
    correction_rework_risk: float
    overlap_risk: float
    triggered_rules: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for value in (self.hesitation_risk, self.correction_rework_risk, self.overlap_risk):
            if value < 0.0 or value > 1.0:
                raise ValueError("Risk values must be in [0, 1]")

    def model_dump(self) -> dict[str, object]:
        return {
            "current_state": self.current_state.value,
            "hesitation_risk": self.hesitation_risk,
            "correction_rework_risk": self.correction_rework_risk,
            "overlap_risk": self.overlap_risk,
            "triggered_rules": self.triggered_rules,
        }
