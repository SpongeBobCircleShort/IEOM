import json
from pathlib import Path

from hesitation.schemas.events import FrameObservation


def load_jsonl_frames(path: str | Path) -> list[FrameObservation]:
    records: list[FrameObservation] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(FrameObservation.model_validate(json.loads(line)))
    return records
