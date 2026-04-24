import statistics

from hesitation.schemas.events import FrameObservation


def compute_kinematic_features(frames: list[FrameObservation]) -> dict[str, float]:
    speeds = [f.hand_speed for f in frames]
    if not speeds:
        return {"mean_speed": 0.0, "speed_variance": 0.0}
    mean_speed = statistics.fmean(speeds)
    variance = statistics.pvariance(speeds) if len(speeds) > 1 else 0.0
    return {"mean_speed": mean_speed, "speed_variance": variance}
