#!/usr/bin/env python3
"""Generate baseline heavy-machinery handoff data for fixed-speed scenarios.

The script models an idealized robot-human handoff in 1D:
- Human and robot move toward a fixed handoff point at constant speeds.
- No hesitation, adaptation, or stochasticity is included.
- Robot speed is varied across scenarios while human speed is fixed.

Outputs:
1) Frame-level CSV with positions and separation over time.
2) Scenario-level summary CSV with safety/efficiency indicators.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    name: str
    robot_speed_mps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/baseline_scraped",
        help="Directory for frame and summary CSV outputs.",
    )
    parser.add_argument("--dt", type=float, default=0.05, help="Time step in seconds.")
    parser.add_argument(
        "--human-speed",
        type=float,
        default=0.80,
        help="Human speed in m/s (constant across scenarios).",
    )
    parser.add_argument(
        "--robot-speeds",
        type=float,
        nargs=3,
        default=[0.40, 0.60, 0.75],
        metavar=("SLOW", "MODERATE", "AGGRESSIVE"),
        help="Robot speed values in m/s for slow/moderate/aggressive scenarios.",
    )
    parser.add_argument(
        "--human-start",
        type=float,
        default=-1.0,
        help="Human start position in meters.",
    )
    parser.add_argument(
        "--robot-start",
        type=float,
        default=1.0,
        help="Robot start position in meters.",
    )
    parser.add_argument(
        "--handoff-point",
        type=float,
        default=0.0,
        help="Shared target handoff position in meters.",
    )
    parser.add_argument(
        "--sqlite-path",
        default="data/baseline_scraped/handoff_baseline.db",
        help=(
            "SQLite output path. Use an existing .db path to refresh baseline tables "
            "for VS Code database workflows."
        ),
    )
    return parser.parse_args()


def simulate_scenario(
    scenario: Scenario,
    human_speed_mps: float,
    dt_s: float,
    human_start_m: float,
    robot_start_m: float,
    handoff_point_m: float,
) -> tuple[list[dict[str, float | str]], dict[str, float | str]]:
    t_h = (handoff_point_m - human_start_m) / human_speed_mps
    t_r = (robot_start_m - handoff_point_m) / scenario.robot_speed_mps
    completion_time_s = max(t_h, t_r)

    rows: list[dict[str, float | str]] = []
    min_separation_before_first_arrival_m: float | None = None
    first_arrival_time_s = min(t_h, t_r)

    n_steps = int(completion_time_s / dt_s) + 1
    for step in range(n_steps + 1):
        t = min(step * dt_s, completion_time_s)
        human_x = min(handoff_point_m, human_start_m + human_speed_mps * t)
        robot_x = max(handoff_point_m, robot_start_m - scenario.robot_speed_mps * t)
        separation = abs(robot_x - human_x)

        if t <= first_arrival_time_s:
            if (
                min_separation_before_first_arrival_m is None
                or separation < min_separation_before_first_arrival_m
            ):
                min_separation_before_first_arrival_m = separation

        rows.append(
            {
                "scenario": scenario.name,
                "time_s": round(t, 4),
                "human_x_m": round(human_x, 6),
                "robot_x_m": round(robot_x, 6),
                "separation_m": round(separation, 6),
                "human_speed_mps": human_speed_mps,
                "robot_speed_mps": scenario.robot_speed_mps,
            }
        )

    if min_separation_before_first_arrival_m is None:
        min_separation_before_first_arrival_m = 0.0

    summary = {
        "scenario": scenario.name,
        "human_speed_mps": human_speed_mps,
        "robot_speed_mps": scenario.robot_speed_mps,
        "human_arrival_time_s": round(t_h, 4),
        "robot_arrival_time_s": round(t_r, 4),
        "completion_time_s": round(completion_time_s, 4),
        "min_separation_before_first_arrival_m": round(
            min_separation_before_first_arrival_m, 6
        ),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_sqlite(
    path: Path,
    frame_rows: list[dict[str, float | str]],
    summary_rows: list[dict[str, float | str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS handoff_baseline_frames (
                scenario TEXT NOT NULL,
                time_s REAL NOT NULL,
                human_x_m REAL NOT NULL,
                robot_x_m REAL NOT NULL,
                separation_m REAL NOT NULL,
                human_speed_mps REAL NOT NULL,
                robot_speed_mps REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS handoff_baseline_summary (
                scenario TEXT PRIMARY KEY,
                human_speed_mps REAL NOT NULL,
                robot_speed_mps REAL NOT NULL,
                human_arrival_time_s REAL NOT NULL,
                robot_arrival_time_s REAL NOT NULL,
                completion_time_s REAL NOT NULL,
                min_separation_before_first_arrival_m REAL NOT NULL
            )
            """
        )

        scenarios = [row["scenario"] for row in summary_rows]
        placeholders = ",".join("?" for _ in scenarios)
        conn.execute(
            f"DELETE FROM handoff_baseline_frames WHERE scenario IN ({placeholders})",
            scenarios,
        )
        conn.execute(
            f"DELETE FROM handoff_baseline_summary WHERE scenario IN ({placeholders})",
            scenarios,
        )

        conn.executemany(
            """
            INSERT INTO handoff_baseline_frames (
                scenario, time_s, human_x_m, robot_x_m, separation_m, human_speed_mps, robot_speed_mps
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["scenario"],
                    row["time_s"],
                    row["human_x_m"],
                    row["robot_x_m"],
                    row["separation_m"],
                    row["human_speed_mps"],
                    row["robot_speed_mps"],
                )
                for row in frame_rows
            ],
        )
        conn.executemany(
            """
            INSERT INTO handoff_baseline_summary (
                scenario, human_speed_mps, robot_speed_mps, human_arrival_time_s,
                robot_arrival_time_s, completion_time_s, min_separation_before_first_arrival_m
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["scenario"],
                    row["human_speed_mps"],
                    row["robot_speed_mps"],
                    row["human_arrival_time_s"],
                    row["robot_arrival_time_s"],
                    row["completion_time_s"],
                    row["min_separation_before_first_arrival_m"],
                )
                for row in summary_rows
            ],
        )
        conn.commit()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    scenarios = [
        Scenario(name="slow", robot_speed_mps=args.robot_speeds[0]),
        Scenario(name="moderate", robot_speed_mps=args.robot_speeds[1]),
        Scenario(name="aggressive", robot_speed_mps=args.robot_speeds[2]),
    ]

    frame_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []

    for scenario in scenarios:
        rows, summary = simulate_scenario(
            scenario=scenario,
            human_speed_mps=args.human_speed,
            dt_s=args.dt,
            human_start_m=args.human_start,
            robot_start_m=args.robot_start,
            handoff_point_m=args.handoff_point,
        )
        frame_rows.extend(rows)
        summary_rows.append(summary)

    write_csv(output_dir / "handoff_baseline_frames.csv", frame_rows)
    write_csv(output_dir / "handoff_baseline_summary.csv", summary_rows)
    write_sqlite(Path(args.sqlite_path), frame_rows, summary_rows)


if __name__ == "__main__":
    main()
