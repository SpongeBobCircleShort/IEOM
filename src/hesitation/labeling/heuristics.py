from hesitation.schemas.features import FeatureWindow
from hesitation.schemas.labels import HesitationState


def infer_state_from_features(
    f: FeatureWindow,
    pause_ratio_mild: float,
    pause_ratio_strong: float,
    correction_backtrack: float,
    ready_progress: float,
    overlap_distance: float,
) -> tuple[HesitationState, list[str]]:
    rules: list[str] = []

    if f.backtrack_ratio >= correction_backtrack:
        rules.append("correction_backtrack")
        return HesitationState.CORRECTION_REWORK, rules

    if f.mean_workspace_distance <= overlap_distance and f.pause_ratio > pause_ratio_mild:
        rules.append("overlap_distance_pause")
        return HesitationState.OVERLAP_RISK, rules

    if f.progress_delta >= ready_progress and f.pause_ratio < pause_ratio_mild:
        rules.append("ready_progress")
        return HesitationState.READY_FOR_ROBOT_ACTION, rules

    if f.pause_ratio >= pause_ratio_strong:
        rules.append("strong_pause")
        return HesitationState.STRONG_HESITATION, rules

    if f.pause_ratio >= pause_ratio_mild:
        rules.append("mild_pause")
        return HesitationState.MILD_HESITATION, rules

    rules.append("normal_progress")
    return HesitationState.NORMAL_PROGRESS, rules
