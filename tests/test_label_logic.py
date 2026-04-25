from hesitation.labeling.heuristics import infer_state_from_features
from hesitation.labeling.state_machine import allowed_transition
from hesitation.schemas.features import FeatureWindow
from hesitation.schemas.labels import HesitationState


def _feature(**kwargs: float) -> FeatureWindow:
    defaults = dict(
        session_id="s",
        end_frame_idx=10,
        mean_speed=0.05,
        speed_variance=0.01,
        pause_ratio=0.0,
        direction_changes=0,
        progress_delta=0.02,
        backtrack_ratio=0.0,
        mean_workspace_distance=0.5,
    )
    defaults.update(kwargs)
    return FeatureWindow(**defaults)


def test_transition_table() -> None:
    assert allowed_transition(HesitationState.NORMAL_PROGRESS, HesitationState.MILD_HESITATION)
    assert not allowed_transition(HesitationState.STRONG_HESITATION, HesitationState.READY_FOR_ROBOT_ACTION)


def test_heuristics_detect_strong() -> None:
    state, _ = infer_state_from_features(_feature(pause_ratio=0.5), 0.2, 0.4, 0.2, 0.005, 0.3)
    assert state == HesitationState.STRONG_HESITATION


def test_heuristics_detect_correction() -> None:
    state, _ = infer_state_from_features(_feature(backtrack_ratio=0.25), 0.2, 0.4, 0.2, 0.005, 0.3)
    assert state == HesitationState.CORRECTION_REWORK
