from dataclasses import dataclass


@dataclass
class FrameObservation:
    session_id: str
    frame_idx: int
    timestamp_ms: int
    task_step_id: int
    hand_x: float
    hand_y: float
    hand_speed: float
    hand_accel: float
    distance_to_robot_workspace: float
    progress: float
    confidence: float
    is_dropout: bool = False

    def __post_init__(self) -> None:
        if self.frame_idx < 0 or self.timestamp_ms < 0 or self.task_step_id < 0:
            raise ValueError("Indices and timestamp must be non-negative")
        if self.hand_speed < 0.0 or self.distance_to_robot_workspace < 0.0:
            raise ValueError("Speed and workspace distance must be non-negative")
        if not (0.0 <= self.progress <= 1.0):
            raise ValueError("progress must be in [0, 1]")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")

    def model_dump(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "frame_idx": self.frame_idx,
            "timestamp_ms": self.timestamp_ms,
            "task_step_id": self.task_step_id,
            "hand_x": self.hand_x,
            "hand_y": self.hand_y,
            "hand_speed": self.hand_speed,
            "hand_accel": self.hand_accel,
            "distance_to_robot_workspace": self.distance_to_robot_workspace,
            "progress": self.progress,
            "confidence": self.confidence,
            "is_dropout": self.is_dropout,
        }

    @classmethod
    def model_validate(cls, payload: dict[str, object]) -> "FrameObservation":
        allowed = {
            "session_id", "frame_idx", "timestamp_ms", "task_step_id", "hand_x", "hand_y",
            "hand_speed", "hand_accel", "distance_to_robot_workspace", "progress", "confidence", "is_dropout"
        }
        filtered = {k: v for k, v in payload.items() if k in allowed}
        return cls(**filtered)
