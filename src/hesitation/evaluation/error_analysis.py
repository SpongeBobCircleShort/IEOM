"""Error-analysis artifact generation for benchmark runs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hesitation.evaluation.metrics import multiclass_metrics


def _safe_probability(record: dict[str, Any], key: str) -> float:
    value = record.get(key, 0.0)
    return float(value) if value is not None else 0.0


def hardest_confusion_pairs(
    confusion_matrix: dict[str, dict[str, int]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return the largest off-diagonal confusion pairs."""
    pairs: list[dict[str, Any]] = []
    for true_label, predicted_counts in confusion_matrix.items():
        for predicted_label, count in predicted_counts.items():
            if true_label == predicted_label or count <= 0:
                continue
            pairs.append(
                {
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": int(count),
                }
            )
    pairs.sort(key=lambda item: (-item["count"], item["true_label"], item["predicted_label"]))
    return pairs[:top_k]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _state_error_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in predictions:
        if record["true_state"] == record["predicted_state"]:
            continue
        rows.append(
            {
                "dataset_name": record["dataset_name"],
                "session_id": record["session_id"],
                "end_frame_idx": record["end_frame_idx"],
                "true_state": record["true_state"],
                "predicted_state": record["predicted_state"],
            }
        )
    return rows


def _binary_error_rows(
    predictions: list[dict[str, Any]],
    label_key: str,
    probability_key: str,
    threshold: float,
    error_kind: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in predictions:
        label = int(record[label_key])
        probability = _safe_probability(record, probability_key)
        predicted = 1 if probability >= threshold else 0
        if error_kind == "fp" and not (label == 0 and predicted == 1):
            continue
        if error_kind == "fn" and not (label == 1 and predicted == 0):
            continue
        rows.append(
            {
                "dataset_name": record["dataset_name"],
                "session_id": record["session_id"],
                "end_frame_idx": record["end_frame_idx"],
                "label": label,
                "probability": probability,
                "threshold": threshold,
            }
        )
    return rows


def _trigger_audit(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    trigger_counts: Counter[str] = Counter()
    trigger_pairs: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in predictions:
        for trigger in record.get("triggered_rules", []):
            trigger_counts[trigger] += 1
            trigger_pairs[trigger][str(record["true_state"])] += 1
    return {
        "trigger_counts": dict(trigger_counts),
        "trigger_state_breakdown": {
            trigger: dict(counter)
            for trigger, counter in sorted(trigger_pairs.items())
        },
    }


def write_model_error_report(
    predictions: list[dict[str, Any]],
    output_dir: str | Path,
    future_hesitation_threshold: float = 0.5,
    future_correction_threshold: float = 0.5,
) -> dict[str, Any]:
    """Persist detailed error-analysis artifacts for one model family."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    true_state = [str(record["true_state"]) for record in predictions]
    predicted_state = [str(record["predicted_state"]) for record in predictions]
    classes = sorted(set(true_state) | set(predicted_state))
    state_metrics = multiclass_metrics(true_state, predicted_state, classes)
    hardest_pairs = hardest_confusion_pairs(
        state_metrics["confusion_matrix"],  # type: ignore[arg-type]
    )

    confusion_rows: list[dict[str, Any]] = []
    confusion_matrix = state_metrics["confusion_matrix"]
    for true_label, predicted_counts in confusion_matrix.items():  # type: ignore[union-attr]
        row: dict[str, Any] = {"true_label": true_label}
        row.update(predicted_counts)
        confusion_rows.append(row)
    _write_csv(
        out / "confusion_matrix.csv",
        ["true_label", *classes],
        confusion_rows,
    )

    per_class_rows = [
        {
            "class_name": class_name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
        for class_name, metrics in state_metrics["per_class"].items()  # type: ignore[union-attr]
    ]
    _write_csv(
        out / "per_class_metrics.csv",
        ["class_name", "precision", "recall", "f1"],
        per_class_rows,
    )

    state_error_rows = _state_error_rows(predictions)
    _write_csv(
        out / "current_state_errors.csv",
        ["dataset_name", "session_id", "end_frame_idx", "true_state", "predicted_state"],
        state_error_rows,
    )

    fh_fp = _binary_error_rows(
        predictions,
        label_key="true_future_hesitation",
        probability_key="future_hesitation_probability",
        threshold=future_hesitation_threshold,
        error_kind="fp",
    )
    fh_fn = _binary_error_rows(
        predictions,
        label_key="true_future_hesitation",
        probability_key="future_hesitation_probability",
        threshold=future_hesitation_threshold,
        error_kind="fn",
    )
    fc_fp = _binary_error_rows(
        predictions,
        label_key="true_future_correction",
        probability_key="future_correction_probability",
        threshold=future_correction_threshold,
        error_kind="fp",
    )
    fc_fn = _binary_error_rows(
        predictions,
        label_key="true_future_correction",
        probability_key="future_correction_probability",
        threshold=future_correction_threshold,
        error_kind="fn",
    )
    binary_fields = ["dataset_name", "session_id", "end_frame_idx", "label", "probability", "threshold"]
    _write_csv(out / "future_hesitation_false_positives.csv", binary_fields, fh_fp)
    _write_csv(out / "future_hesitation_false_negatives.csv", binary_fields, fh_fn)
    _write_csv(out / "future_correction_false_positives.csv", binary_fields, fc_fp)
    _write_csv(out / "future_correction_false_negatives.csv", binary_fields, fc_fn)

    trigger_audit = _trigger_audit(predictions)
    summary = {
        "state_metrics": state_metrics,
        "hardest_confusion_pairs": hardest_pairs,
        "future_hesitation_threshold": future_hesitation_threshold,
        "future_correction_threshold": future_correction_threshold,
        "future_hesitation_fp": len(fh_fp),
        "future_hesitation_fn": len(fh_fn),
        "future_correction_fp": len(fc_fp),
        "future_correction_fn": len(fc_fn),
        "trigger_audit": trigger_audit,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    hardest_pairs_text = ", ".join(
        f"{item['true_label']} -> {item['predicted_label']} ({item['count']})"
        for item in hardest_pairs
    ) or "none"
    lines = [
        "# Error Analysis",
        "",
        f"- Current-state accuracy: {float(state_metrics['accuracy']):.4f}",
        f"- Current-state macro F1: {float(state_metrics['macro_f1']):.4f}",
        f"- Hardest confusion pairs: {hardest_pairs_text}",
        f"- Future hesitation FP/FN: {len(fh_fp)}/{len(fh_fn)}",
        f"- Future correction FP/FN: {len(fc_fp)}/{len(fc_fn)}",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
