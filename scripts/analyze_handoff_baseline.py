#!/usr/bin/env python3
"""Analyze baseline handoff outputs and generate an actionable next-steps report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SummaryRow = dict[str, float | str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        default="data/baseline_scraped/handoff_baseline_summary.csv",
        help="Path to handoff_baseline_summary.csv",
    )
    parser.add_argument(
        "--report-path",
        default="data/baseline_scraped/handoff_next_steps.md",
        help="Output markdown report path.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> list[SummaryRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[SummaryRow] = []
        for row in reader:
            rows.append(
                {
                    "scenario": row["scenario"],
                    "robot_speed_mps": float(row["robot_speed_mps"]),
                    "completion_time_s": float(row["completion_time_s"]),
                    "min_separation_before_first_arrival_m": float(
                        row["min_separation_before_first_arrival_m"]
                    ),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in summary CSV: {path}")
    return rows


def build_report(rows: list[SummaryRow]) -> str:
    sorted_by_speed = sorted(rows, key=lambda r: float(r["robot_speed_mps"]))
    slow = sorted_by_speed[0]
    fastest_completion = min(rows, key=lambda r: float(r["completion_time_s"]))
    safest = max(rows, key=lambda r: float(r["min_separation_before_first_arrival_m"]))

    lines: list[str] = [
        "# Baseline Handoff: What to Do Next",
        "",
        "## 1) Verify baseline tradeoff",
        "",
        "| Scenario | Robot speed (m/s) | Completion time (s) | Min separation before first arrival (m) |",
        "|---|---:|---:|---:|",
    ]

    for row in sorted_by_speed:
        lines.append(
            "| {scenario} | {robot_speed_mps:.2f} | {completion_time_s:.3f} | "
            "{min_separation_before_first_arrival_m:.3f} |".format(
                scenario=row["scenario"],
                robot_speed_mps=float(row["robot_speed_mps"]),
                completion_time_s=float(row["completion_time_s"]),
                min_separation_before_first_arrival_m=float(
                    row["min_separation_before_first_arrival_m"]
                ),
            )
        )

    lines.extend(
        [
            "",
            "## 2) Key findings",
            "",
            (
                "- Fastest completion: **{scenario}** at **{time:.3f}s**."
            ).format(
                scenario=fastest_completion["scenario"],
                time=float(fastest_completion["completion_time_s"]),
            ),
            (
                "- Largest safety margin: **{scenario}** at "
                "**{sep:.3f}m** minimum separation."
            ).format(
                scenario=safest["scenario"],
                sep=float(safest["min_separation_before_first_arrival_m"]),
            ),
            "",
            "## 3) Introduce hesitation (next experiment)",
            "",
            "Use this baseline to create a controlled hesitation test:",
            "",
            "1. Keep the same three robot speeds.",
            "2. Add human hesitation windows (e.g., 0.3s, 0.6s, 1.0s pauses).",
            "3. Recompute completion time and minimum separation.",
            "4. Compare each hesitant run to this baseline using deltas:",
            "   - `delta_time = hesitant_completion_time - baseline_completion_time`",
            "   - `delta_sep = hesitant_min_separation - baseline_min_separation`",
            "",
            "## 4) Adaptive-control success criteria",
            "",
            "Evaluate your future adaptive policy against two goals:",
            "",
            "- Safety: keep minimum separation at or above the slow baseline.",
            "- Efficiency: stay close to moderate/aggressive completion time when no hesitation occurs.",
        ]
    )

    if float(slowest_time := slow["completion_time_s"]) > 0:
        for row in sorted_by_speed[1:]:
            efficiency_gain_pct = (
                (float(slowest_time) - float(row["completion_time_s"]))
                / float(slowest_time)
                * 100.0
            )
            safety_change = float(row["min_separation_before_first_arrival_m"]) - float(
                slow["min_separation_before_first_arrival_m"]
            )
            lines.append(
                (
                    "- Relative to slow, **{scenario}** improves completion by "
                    "{gain:.1f}% and changes safety margin by {sep_change:+.3f}m."
                ).format(
                    scenario=row["scenario"],
                    gain=efficiency_gain_pct,
                    sep_change=safety_change,
                )
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_csv)
    report_path = Path(args.report_path)

    rows = load_summary(summary_path)
    report_text = build_report(rows)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Wrote report: {report_path}")
    print(report_text)


if __name__ == "__main__":
    main()
