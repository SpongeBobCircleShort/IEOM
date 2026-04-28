from dataclasses import dataclass

from hesitation.schemas.events import FrameObservation


@dataclass
class SessionTrajectory:
    session_id: str
    frame_rate_hz: int
    frames: list[FrameObservation]

    def __post_init__(self) -> None:
        if self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be > 0")
