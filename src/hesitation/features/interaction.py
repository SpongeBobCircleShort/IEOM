import statistics

from hesitation.schemas.events import FrameObservation


def compute_interaction_features(frames: list[FrameObservation]) -> dict[str, float]:
    distances = [f.distance_to_robot_workspace for f in frames]
    mean_distance = statistics.fmean(distances) if distances else 0.0
    direction_changes = 0
    for i in range(2, len(frames)):
        prev_delta = frames[i - 1].hand_x - frames[i - 2].hand_x
        curr_delta = frames[i].hand_x - frames[i - 1].hand_x
        if prev_delta * curr_delta < 0:
            direction_changes += 1
    return {
        "mean_workspace_distance": mean_distance,
        "direction_changes": float(direction_changes),
    }
