from __future__ import annotations

import json
from pathlib import Path

from hesitation.database.schemas import CanonicalRecord


def load_canonical(path: str) -> list[CanonicalRecord]:
    rows: list[CanonicalRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(CanonicalRecord(**json.loads(line)))
    return rows
