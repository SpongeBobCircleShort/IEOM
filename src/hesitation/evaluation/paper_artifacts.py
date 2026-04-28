"""Paper- and slide-friendly artifact writers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


STATE_COLORS = {
    "normal_progress": "#5B8E7D",
    "mild_hesitation": "#E9C46A",
    "strong_hesitation": "#F4A261",
    "correction_rework": "#E76F51",
    "ready_for_robot_action": "#4C78A8",
    "overlap_risk": "#9C755F",
}


def write_table_bundle(
    output_stem: str | Path,
    headers: list[str],
    rows: list[list[Any]],
) -> None:
    """Write the same table as CSV and Markdown."""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    with stem.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)

    markdown_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        markdown_lines.append("| " + " | ".join(str(value) for value in row) + " |")
    stem.with_suffix(".md").write_text("\n".join(markdown_lines), encoding="utf-8")


def render_pipeline_figure(path: str | Path) -> None:
    """Render a static pipeline overview figure as SVG."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="980" height="280" viewBox="0 0 980 280">
<rect width="980" height="280" fill="#F7F3EA"/>
<text x="40" y="48" font-family="Georgia, serif" font-size="28" fill="#222222">Paper-Ready Benchmark Pipeline</text>
<rect x="40" y="90" width="170" height="110" rx="18" fill="#DCE8D8" stroke="#2F5D50" stroke-width="2"/>
<text x="64" y="126" font-family="Helvetica, Arial, sans-serif" font-size="18" fill="#1F2A2A">Dataset Inputs</text>
<text x="64" y="154" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#1F2A2A">CHICO</text>
<text x="64" y="176" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#1F2A2A">HA-ViD</text>
<rect x="255" y="90" width="190" height="110" rx="18" fill="#F0E2B6" stroke="#8B6F2F" stroke-width="2"/>
<text x="281" y="126" font-family="Helvetica, Arial, sans-serif" font-size="18" fill="#3B2F14">Harmonize + Split</text>
<text x="281" y="154" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#3B2F14">train/test manifests</text>
<text x="281" y="176" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#3B2F14">coverage-gap tracking</text>
<rect x="490" y="90" width="190" height="110" rx="18" fill="#E7D8E8" stroke="#6A4C72" stroke-width="2"/>
<text x="524" y="126" font-family="Helvetica, Arial, sans-serif" font-size="18" fill="#34203A">Baselines</text>
<text x="524" y="154" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#34203A">rules / classical / deep</text>
<text x="524" y="176" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#34203A">within / cross / merged</text>
<rect x="725" y="90" width="215" height="110" rx="18" fill="#D8E5F2" stroke="#365C7A" stroke-width="2"/>
<text x="756" y="126" font-family="Helvetica, Arial, sans-serif" font-size="18" fill="#1D3144">Reports + Figures</text>
<text x="756" y="154" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#1D3144">benchmark matrix</text>
<text x="756" y="176" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#1D3144">ablations / error analysis</text>
<path d="M210 145 L255 145" stroke="#444444" stroke-width="3" fill="none"/>
<path d="M445 145 L490 145" stroke="#444444" stroke-width="3" fill="none"/>
<path d="M680 145 L725 145" stroke="#444444" stroke-width="3" fill="none"/>
</svg>
"""
    target.write_text(svg, encoding="utf-8")


def render_qualitative_panel(
    path: str | Path,
    title: str,
    tracks: list[tuple[str, list[str]]],
) -> None:
    """Render a strip-chart qualitative panel for one sequence."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    width = 1080
    row_height = 42
    left_pad = 180
    top_pad = 70
    n_steps = max((len(values) for _, values in tracks), default=1)
    usable_width = width - left_pad - 40
    cell_width = max(10, usable_width // max(1, n_steps))
    height = top_pad + row_height * len(tracks) + 80

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FCF9F2"/>',
        f'<text x="36" y="42" font-family="Georgia, serif" font-size="26" fill="#222222">{title}</text>',
    ]

    for row_index, (label, values) in enumerate(tracks):
        y = top_pad + row_index * row_height
        parts.append(
            f'<text x="36" y="{y + 24}" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#303030">{label}</text>'
        )
        for index, state in enumerate(values):
            x = left_pad + index * cell_width
            color = STATE_COLORS.get(state, "#BBBBBB")
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 1}" height="28" fill="{color}" opacity="0.95"/>'
            )

    legend_y = top_pad + row_height * len(tracks) + 24
    legend_x = 36
    for state, color in STATE_COLORS.items():
        parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="18" height="18" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 26}" y="{legend_y + 14}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#303030">{state}</text>'
        )
        legend_x += 150

    parts.append("</svg>")
    target.write_text("".join(parts), encoding="utf-8")
