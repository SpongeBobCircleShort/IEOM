from hesitation.features.pipeline import window_to_features
from hesitation.schemas.events import FrameObservation


def _mk_frame(idx: int, speed: float, progress: float, x: float) -> FrameObservation:
    return FrameObservation(
        session_id="s",
        frame_idx=idx,
        timestamp_ms=idx * 100,
        task_step_id=0,
        hand_x=x,
        hand_y=0.0,
        hand_speed=speed,
        hand_accel=0.0,
        distance_to_robot_workspace=abs(0.5 - x),
        progress=progress,
        confidence=1.0,
    )


def test_window_features_basic() -> None:
    frames = [
        _mk_frame(0, 0.0, 0.1, 0.1),
        _mk_frame(1, 0.2, 0.15, 0.2),
        _mk_frame(2, 0.1, 0.2, 0.3),
    ]
    f = window_to_features(frames, pause_speed_threshold=0.05)
    assert f.pause_ratio == 1 / 3
    assert f.progress_delta > 0
    assert f.mean_speed > 0
