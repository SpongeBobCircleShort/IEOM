import json
from pathlib import Path
from typing import Any

from hesitation.schemas.events import FrameObservation


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_jsonl_frames(path: str | Path) -> list[FrameObservation]:
    records = load_jsonl_records(path)
    return [FrameObservation.model_validate(r) for r in records]
