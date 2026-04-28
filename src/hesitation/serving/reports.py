from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ArtifactInspection:
    """Describe the report artifacts found at a file or directory path."""

    root_path: str
    json_files: list[str]
    csv_files: list[str]
    markdown_files: list[str]
    chosen_report_path: str | None
    chosen_report_payload: dict[str, Any] | None
    preview_rows: list[dict[str, str]]
    markdown_preview: str | None

    def to_dict(self) -> dict[str, object]:
        """Convert the inspection summary into a JSON-serializable payload."""
        return asdict(self)


def _flatten_numeric_metrics(payload: Any, prefix: str = "") -> dict[str, float]:
    results: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            results.update(_flatten_numeric_metrics(value, next_prefix))
        return results
    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            results.update(_flatten_numeric_metrics(value, f"{prefix}[{idx}]"))
        return results
    if isinstance(payload, bool):
        return results
    if isinstance(payload, (int, float)):
        results[prefix] = float(payload)
    return results


def _read_csv_preview(path: Path, limit: int = 10) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for index, row in enumerate(reader):
            if index >= limit:
                break
            rows.append({str(key): str(value) for key, value in row.items()})
        return rows


def _choose_json_report(json_files: list[Path]) -> Path | None:
    if not json_files:
        return None
    preferred = {
        "metrics.json",
        "deep_metrics.json",
        "deep_eval.json",
        "deep_eval_calibrated.json",
        "report.json",
    }
    for candidate in json_files:
        if candidate.name in preferred:
            return candidate
    return json_files[0]


def inspect_artifact_path(path: str | Path) -> ArtifactInspection:
    """Inspect a report artifact path and summarize displayable content."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {root}")

    if root.is_dir():
        json_files = sorted(root.glob("*.json"))
        csv_files = sorted(root.glob("*.csv"))
        markdown_files = sorted(root.glob("*.md"))
    else:
        json_files = [root] if root.suffix.lower() == ".json" else []
        csv_files = [root] if root.suffix.lower() == ".csv" else []
        markdown_files = [root] if root.suffix.lower() == ".md" else []

    chosen = _choose_json_report(json_files)
    payload = None
    if chosen is not None:
        payload = json.loads(chosen.read_text(encoding="utf-8"))

    preview_rows: list[dict[str, str]] = []
    if csv_files:
        preview_rows = _read_csv_preview(csv_files[0])

    markdown_preview = None
    if markdown_files:
        markdown_preview = markdown_files[0].read_text(encoding="utf-8")[:4000]

    return ArtifactInspection(
        root_path=str(root),
        json_files=[str(file_path) for file_path in json_files],
        csv_files=[str(file_path) for file_path in csv_files],
        markdown_files=[str(file_path) for file_path in markdown_files],
        chosen_report_path=str(chosen) if chosen is not None else None,
        chosen_report_payload=payload,
        preview_rows=preview_rows,
        markdown_preview=markdown_preview,
    )


def compare_report_sources(
    left_path: str | Path,
    right_path: str | Path,
    left_label: str = "left",
    right_label: str = "right",
) -> dict[str, object]:
    """Compare numeric metrics across two artifact/report sources."""
    left = inspect_artifact_path(left_path)
    right = inspect_artifact_path(right_path)

    left_metrics = _flatten_numeric_metrics(left.chosen_report_payload or {})
    right_metrics = _flatten_numeric_metrics(right.chosen_report_payload or {})
    shared_keys = sorted(set(left_metrics) & set(right_metrics))

    comparison_rows = [
        {
            "metric": key,
            left_label: left_metrics[key],
            right_label: right_metrics[key],
            "delta": right_metrics[key] - left_metrics[key],
        }
        for key in shared_keys
    ]

    return {
        "left_label": left_label,
        "right_label": right_label,
        "left": left.to_dict(),
        "right": right.to_dict(),
        "shared_metric_count": len(shared_keys),
        "comparison_rows": comparison_rows,
    }
