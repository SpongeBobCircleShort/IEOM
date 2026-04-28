from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hesitation.schemas.events import FrameObservation


@dataclass(slots=True)
class IngestionStubResult:
    """Describe the outcome of the placeholder video/pose ingestion flow."""

    status: str
    message: str
    frames: list[FrameObservation]
    source_type: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the stub result into a JSON-serializable payload."""
        payload = asdict(self)
        payload["frames"] = [frame.model_dump() for frame in self.frames]
        return payload


def ingest_video_or_pose_stub(path: str | Path) -> IngestionStubResult:
    """Load precomputed frame observations when possible, otherwise return a clear placeholder result."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Ingestion source does not exist: {source}")

    suffix = source.suffix.lower()
    if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
        return IngestionStubResult(
            status="stub_only",
            message="Full video inference is not implemented in Phase 4. Upload precomputed frame-observation JSONL instead.",
            frames=[],
            source_type="video",
        )

    if suffix == ".jsonl":
        frames: list[FrameObservation] = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    frames.append(FrameObservation.model_validate(payload))
        return IngestionStubResult(
            status="loaded_frames",
            message="Loaded precomputed frame observations from JSONL.",
            frames=frames,
            source_type="jsonl",
        )

    return IngestionStubResult(
        status="stub_only",
        message="Unsupported pose/video stub input. Use `.jsonl` frame observations or a video file placeholder.",
        frames=[],
        source_type=suffix.lstrip(".") or "unknown",
    )
