#!/usr/bin/env python3
"""Render canonical benchmark figures from master_aggregated_results.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "artifacts" / "canonical_results" / "tables" / "master_aggregated_results.csv"
FIG_DIR = ROOT / "artifacts" / "canonical_results" / "figures"


def read_rows() -> list[dict[str, float | str]]:
    with TABLE_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "N_runs",
        "PolicyA_mean_overlaps",
        "PolicyA_std_overlaps",
        "PolicyB_mean_overlaps",
        "PolicyB_std_overlaps",
        "Overlap_Reduction_Pct",
        "PolicyA_mean_time_sec",
        "PolicyB_mean_time_sec",
        "Time_Cost_Pct",
        "paired_ttest_p",
    }
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def main() -> None:
    rows = read_rows()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    labels = [str(row["environment"]).replace("_", " ") for row in rows]
    x = list(range(len(rows)))
    width = 0.36

    a_ov = [float(row["PolicyA_mean_overlaps"]) for row in rows]
    b_ov = [float(row["PolicyB_mean_overlaps"]) for row in rows]
    a_sd = [float(row["PolicyA_std_overlaps"]) for row in rows]
    b_sd = [float(row["PolicyB_std_overlaps"]) for row in rows]
    reduction = [float(row["Overlap_Reduction_Pct"]) for row in rows]
    a_time = [float(row["PolicyA_mean_time_sec"]) for row in rows]
    b_time = [float(row["PolicyB_mean_time_sec"]) for row in rows]
    time_cost = [float(row["Time_Cost_Pct"]) for row in rows]
    pvals = [float(row["paired_ttest_p"]) for row in rows]

    plt.figure(figsize=(10, 4.8))
    plt.bar([i - width / 2 for i in x], a_ov, width, yerr=a_sd, label="Policy A Baseline", color="#337fbd", capsize=3)
    plt.bar([i + width / 2 for i in x], b_ov, width, yerr=b_sd, label="Policy B Hesitation-Aware", color="#df4a3f", capsize=3)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Mean Overlap Events (N=50 runs)")
    plt.title("Safety: Overlap Risk Events")
    plt.legend(loc="upper left")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_overlap_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4.8))
    colors = ["#2ca02c" if value > 50 else "#d6a900" if value > 0 else "#c43c35" for value in reduction]
    plt.bar(x, reduction, color=colors)
    for idx, pvalue in enumerate(pvals):
        if pvalue < 0.001:
            plt.text(idx, reduction[idx] + 1, "***", ha="center", fontweight="bold")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Overlap Reduction % (Policy B vs A)")
    plt.title("Overlap Reduction by Environment")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_reduction_pct.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5.2))
    plt.scatter(time_cost, reduction, s=110, color="#4c78a8", edgecolor="black")
    for idx, label in enumerate(labels):
        plt.text(time_cost[idx] + 0.05, reduction[idx], label, fontsize=8)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Time Cost % (+ = Policy B slower)")
    plt.ylabel("Overlap Reduction %")
    plt.title("Safety-Efficiency Tradeoff")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_cost_benefit.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4.8))
    plt.bar([i - width / 2 for i in x], a_time, width, label="Policy A", color="#337fbd")
    plt.bar([i + width / 2 for i in x], b_time, width, label="Policy B", color="#df4a3f")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Mean Task Completion Time (s), N=50 runs")
    plt.title("Efficiency: Task Completion Time")
    plt.legend(loc="upper left")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_completion_time.png", dpi=180)
    plt.close()

    print(f"[SAVED] 4 Python-rendered figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
