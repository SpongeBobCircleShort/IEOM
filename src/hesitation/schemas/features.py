from dataclasses import dataclass


@dataclass
class FeatureWindow:
    session_id: str
    end_frame_idx: int
    mean_speed: float
    speed_variance: float
    pause_ratio: float
    direction_changes: int
    progress_delta: float
    backtrack_ratio: float
    mean_workspace_distance: float

    def __post_init__(self) -> None:
        if self.end_frame_idx < 0 or self.direction_changes < 0:
            raise ValueError("Indices must be non-negative")
        if self.mean_speed < 0.0 or self.speed_variance < 0.0 or self.mean_workspace_distance < 0.0:
            raise ValueError("Non-negative features violated")
        if not (0.0 <= self.pause_ratio <= 1.0):
            raise ValueError("pause_ratio must be in [0, 1]")
        if not (0.0 <= self.backtrack_ratio <= 1.0):
            raise ValueError("backtrack_ratio must be in [0, 1]")
