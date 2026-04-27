from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hesitation.deep.pipeline import train_deep
from hesitation.ml.dataset import build_windows, load_rows, split_train_val
from hesitation.ml.pipeline import train_classical


def _dataset_stats(model_input_path: str, window_size: int, horizon_frames: int) -> dict[str, Any]:
    rows = load_rows(model_input_path)
    sessions = sorted({str(row["session_id"]) for row in rows})
    labels: dict[str, int] = {}
    for row in rows:
        state = str(row.get("latent_state", "unknown"))
        labels[state] = labels.get(state, 0) + 1

    windows = build_windows(rows, window_size=window_size, pause_speed_threshold=0.03, horizon_frames=horizon_frames)
    train, val = split_train_val(windows)
    return {
        "records": len(rows),
        "sessions": len(sessions),
        "label_distribution": labels,
        "split_sizes": {"train_windows": len(train), "val_windows": len(val)},
    }


def run_first_benchmark(
    model_input_path: str,
    output_dir: str,
    window_size: int = 10,
    horizon_frames: int = 5,
) -> dict[str, Any]:
    """Run first benchmark (rules + classical + deep) on onboarded dataset export."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_stats = _dataset_stats(model_input_path, window_size, horizon_frames)

    classical_metrics = train_classical(
        input_path=model_input_path,
        output_dir=str(out / "classical"),
        window_size=window_size,
        pause_speed_threshold=0.03,
        horizon_frames=horizon_frames,
    )
    deep_metrics = train_deep(
        input_path=model_input_path,
        output_dir=str(out / "deep"),
        window_size=window_size,
        horizon_frames=horizon_frames,
        epochs=3,
        hidden_dim=16,
        learning_rate=0.005,
        seed=13,
        batch_size=16,
    )

    summary = {
        "dataset_size": data_stats,
        "qc_summary": {
            "note": "qc report generated in onboarding flow and should be attached alongside benchmark",
        },
        "rules_baseline": classical_metrics.get("current_state_rules", {}),
        "classical_baseline": classical_metrics.get("current_state_classical", {}),
        "deep_baseline": deep_metrics,
        "future_hesitation": classical_metrics.get("future_hesitation", {}),
        "future_correction": classical_metrics.get("future_correction", {}),
        "known_caveats": [
            "fixture-based real-data slice is class-imbalanced",
            "future-risk metrics may degenerate when positives are sparse",
            "deep backend may use fallback if torch is unavailable",
        ],
    }
    (out / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
