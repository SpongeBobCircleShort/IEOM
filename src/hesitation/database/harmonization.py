from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path

from hesitation.database.label_audit import audit_labels
from hesitation.database.io_utils import load_canonical


def build_harmonization_report(
    chico_labeled_path: str,
    havid_labeled_path: str,
    output_json: str,
    output_md: str,
) -> tuple[str, str]:
    chico = load_canonical(chico_labeled_path)
    havid = load_canonical(havid_labeled_path)

    coverage_fields = [
        "timestamp_ms",
        "hand_left",
        "hand_right",
        "pose_confidence",
        "task_step",
        "action_label_raw",
        "canonical_action_label",
        "human_robot_distance",
        "tool",
        "manipulated_object",
    ]

    def coverage(records, field):
        total = max(1, len(records))
        present = sum(1 for r in records if getattr(r, field) is not None)
        return present / total

    field_coverage = {
        field: {
            "chico": coverage(chico, field),
            "havid": coverage(havid, field),
            "gap": abs(coverage(chico, field) - coverage(havid, field)),
        }
        for field in coverage_fields
    }

    label_dist = {
        "chico": dict(Counter(r.hesitation_state or "unknown" for r in chico)),
        "havid": dict(Counter(r.hesitation_state or "unknown" for r in havid)),
    }

    trigger_dist = {
        "chico": audit_labels(chico).trigger_counts,
        "havid": audit_labels(havid).trigger_counts,
    }

    transfer_issues: list[str] = []
    for field, info in field_coverage.items():
        if info["gap"] > 0.35:
            transfer_issues.append(f"coverage_gap:{field}:{info['gap']:.2f}")

    all_labels = sorted(set(label_dist["chico"]) | set(label_dist["havid"]))
    for label in all_labels:
        c = label_dist["chico"].get(label, 0)
        h = label_dist["havid"].get(label, 0)
        if c == 0 or h == 0:
            transfer_issues.append(f"label_missing_in_one_dataset:{label}")

    payload = {
        "datasets": ["chico", "havid"],
        "field_coverage": field_coverage,
        "label_distribution": label_dist,
        "trigger_distribution": trigger_dist,
        "transfer_issues": transfer_issues,
    }

    Path(output_json).write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# CHICO vs HA-ViD harmonization report",
        "",
        "## Field coverage gaps",
        "| Field | CHICO | HA-ViD | Gap |",
        "|---|---:|---:|---:|",
    ]
    for field, info in field_coverage.items():
        md_lines.append(f"| {field} | {info['chico']:.2f} | {info['havid']:.2f} | {info['gap']:.2f} |")

    md_lines += ["", "## Transfer issues", ""]
    if transfer_issues:
        md_lines.extend([f"- {issue}" for issue in transfer_issues])
    else:
        md_lines.append("- none")

    Path(output_md).write_text("\n".join(md_lines), encoding="utf-8")
    return output_json, output_md
