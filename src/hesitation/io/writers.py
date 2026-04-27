import json
from pathlib import Path
from typing import Any


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
