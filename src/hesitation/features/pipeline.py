from hesitation.features.interaction import compute_interaction_features
from hesitation.features.kinematics import compute_kinematic_features
from hesitation.features.temporal import compute_temporal_features
from hesitation.schemas.events import FrameObservation
from hesitation.schemas.features import FeatureWindow


def window_to_features(
    frames: list[FrameObservation],
    pause_speed_threshold: float,
) -> FeatureWindow:
    k = compute_kinematic_features(frames)
    t = compute_temporal_features(frames, pause_speed_threshold=pause_speed_threshold)
    i = compute_interaction_features(frames)
    return FeatureWindow(
        session_id=frames[-1].session_id,
        end_frame_idx=frames[-1].frame_idx,
        mean_speed=k["mean_speed"],
        speed_variance=k["speed_variance"],
        pause_ratio=t["pause_ratio"],
        direction_changes=int(i["direction_changes"]),
        progress_delta=t["progress_delta"],
        backtrack_ratio=t["backtrack_ratio"],
        mean_workspace_distance=i["mean_workspace_distance"],
    )
