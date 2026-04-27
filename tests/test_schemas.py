import pytest

from hesitation.schemas.events import FrameObservation
from hesitation.schemas.labels import HesitationState


def test_frame_schema_accepts_valid_payload() -> None:
    frame = FrameObservation(
        session_id="s1",
        frame_idx=0,
        timestamp_ms=0,
        task_step_id=0,
        hand_x=0.0,
        hand_y=0.0,
        hand_speed=0.1,
        hand_accel=0.0,
        distance_to_robot_workspace=0.2,
        progress=0.1,
        confidence=1.0,
    )
    assert frame.session_id == "s1"


def test_frame_schema_rejects_invalid_progress() -> None:
    with pytest.raises(ValueError):
        FrameObservation(
            session_id="s1",
            frame_idx=0,
            timestamp_ms=0,
            task_step_id=0,
            hand_x=0.0,
            hand_y=0.0,
            hand_speed=0.1,
            hand_accel=0.0,
            distance_to_robot_workspace=0.2,
            progress=2.0,
            confidence=1.0,
        )


def test_state_enum_has_expected_values() -> None:
    assert HesitationState.NORMAL_PROGRESS.value == "normal_progress"
    assert HesitationState.OVERLAP_RISK.value == "overlap_risk"
