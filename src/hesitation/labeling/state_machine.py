from hesitation.schemas.labels import HesitationState

_ALLOWED: dict[HesitationState, set[HesitationState]] = {
    HesitationState.NORMAL_PROGRESS: {
        HesitationState.NORMAL_PROGRESS,
        HesitationState.MILD_HESITATION,
        HesitationState.READY_FOR_ROBOT_ACTION,
        HesitationState.OVERLAP_RISK,
    },
    HesitationState.MILD_HESITATION: {
        HesitationState.NORMAL_PROGRESS,
        HesitationState.MILD_HESITATION,
        HesitationState.STRONG_HESITATION,
        HesitationState.CORRECTION_REWORK,
    },
    HesitationState.STRONG_HESITATION: {
        HesitationState.MILD_HESITATION,
        HesitationState.STRONG_HESITATION,
        HesitationState.CORRECTION_REWORK,
    },
    HesitationState.CORRECTION_REWORK: {
        HesitationState.NORMAL_PROGRESS,
        HesitationState.MILD_HESITATION,
        HesitationState.CORRECTION_REWORK,
    },
    HesitationState.READY_FOR_ROBOT_ACTION: {
        HesitationState.NORMAL_PROGRESS,
        HesitationState.READY_FOR_ROBOT_ACTION,
        HesitationState.OVERLAP_RISK,
    },
    HesitationState.OVERLAP_RISK: {
        HesitationState.NORMAL_PROGRESS,
        HesitationState.MILD_HESITATION,
        HesitationState.OVERLAP_RISK,
    },
}


def allowed_transition(current: HesitationState, target: HesitationState) -> bool:
    return target in _ALLOWED[current]
