"""Reporting helpers for Phase 3 model comparison outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hesitation.deep.serialize import save_json


def write_comparison_report(output_dir: str, report: dict[str, Any]) -> None:
    """Persist machine-readable and markdown summary artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "comparison.json", report)

    summary = report["summary"]
    lines = [
        "# Phase 3 Model Comparison",
        "",
        "| Metric | Rules | Classical | Deep |",
        "|---|---:|---:|---:|",
        f"| Current-state accuracy | {summary['current_state_accuracy']['rules']:.4f} | {summary['current_state_accuracy']['classical']:.4f} | {summary['current_state_accuracy']['deep']:.4f} |",
        f"| Current-state macro F1 | {summary['current_state_macro_f1']['rules']:.4f} | {summary['current_state_macro_f1']['classical']:.4f} | {summary['current_state_macro_f1']['deep']:.4f} |",
        f"| Future hesitation AUPRC | n/a | {summary['future_hesitation_auprc']['classical']:.4f} | {summary['future_hesitation_auprc']['deep']:.4f} |",
        f"| Future correction AUPRC | n/a | {summary['future_correction_auprc']['classical']:.4f} | {summary['future_correction_auprc']['deep']:.4f} |",
    ]
    (out / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")

    csv_lines = [
        "metric,rules,classical,deep",
        f"current_state_accuracy,{summary['current_state_accuracy']['rules']},{summary['current_state_accuracy']['classical']},{summary['current_state_accuracy']['deep']}",
        f"current_state_macro_f1,{summary['current_state_macro_f1']['rules']},{summary['current_state_macro_f1']['classical']},{summary['current_state_macro_f1']['deep']}",
        f"future_hesitation_auprc,,{summary['future_hesitation_auprc']['classical']},{summary['future_hesitation_auprc']['deep']}",
        f"future_correction_auprc,,{summary['future_correction_auprc']['classical']},{summary['future_correction_auprc']['deep']}",
    ]
    (out / "comparison_table.csv").write_text("\n".join(csv_lines), encoding="utf-8")
