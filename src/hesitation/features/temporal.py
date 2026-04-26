from hesitation.schemas.events import FrameObservation


def compute_temporal_features(frames: list[FrameObservation], pause_speed_threshold: float) -> dict[str, float]:
    if not frames:
        return {"pause_ratio": 0.0, "progress_delta": 0.0, "backtrack_ratio": 0.0}
    pause_count = sum(1 for f in frames if f.hand_speed <= pause_speed_threshold)
    progress_delta = frames[-1].progress - frames[0].progress
    backtrack_count = sum(
        1
        for i in range(1, len(frames))
        if frames[i].progress < frames[i - 1].progress
    )
    return {
        "pause_ratio": pause_count / len(frames),
        "progress_delta": progress_delta,
        "backtrack_ratio": backtrack_count / max(1, len(frames) - 1),
    }
