from hesitation.baselines.rules_engine import classify_window
from hesitation.schemas.features import FeatureWindow
from hesitation.schemas.labels import HesitationState


def test_rules_baseline_outputs_label_and_risks() -> None:
    feature = FeatureWindow(
        session_id="s",
        end_frame_idx=20,
        mean_speed=0.01,
        speed_variance=0.02,
        pause_ratio=0.45,
        direction_changes=1,
        progress_delta=0.01,
        backtrack_ratio=0.05,
        mean_workspace_distance=0.4,
    )
    out = classify_window(
        feature,
        thresholds={
            "pause_ratio_mild": 0.2,
            "pause_ratio_strong": 0.4,
            "correction_backtrack": 0.2,
            "ready_progress": 0.005,
            "overlap_distance": 0.3,
        },
        risk_cfg={"high_pause_ratio": 0.3},
    )
    assert out.current_state == HesitationState.STRONG_HESITATION
    assert 0.0 <= out.hesitation_risk <= 1.0
