"""Paper-ready benchmark suite orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hesitation.deep.dataset import SequenceWindow, build_sequence_windows
from hesitation.deep.pipeline import (
    DeepTrainConfig,
    ThresholdConfig,
    evaluate_deep_windows,
    infer_sequence_deep_windows,
    train_deep_on_windows,
)
from hesitation.evaluation.error_analysis import write_model_error_report
from hesitation.evaluation.metrics import binary_metrics
from hesitation.evaluation.paper_artifacts import (
    render_pipeline_figure,
    render_qualitative_panel,
    write_table_bundle,
)
from hesitation.io.config import load_config
from hesitation.io.writers import write_jsonl
from hesitation.ml.dataset import FEATURE_ORDER, build_windows, load_rows
from hesitation.ml.pipeline import (
    evaluate_classical_windows,
    load_classical_runtime,
    predict_classical_window,
    predict_rules_windows,
    train_classical_on_windows,
)


@dataclass(slots=True)
class SuiteDeepConfig:
    """Deep model configuration for the benchmark suite."""

    epochs: int = 3
    hidden_dim: int = 16
    learning_rate: float = 0.005
    seed: int = 13
    batch_size: int = 16


@dataclass(slots=True)
class SuiteBenchmarkConfig:
    """Shared training/evaluation window configuration."""

    window_size: int = 10
    pause_speed_threshold: float = 0.03
    horizon_frames: int = 5
    deep: SuiteDeepConfig = field(default_factory=SuiteDeepConfig)


@dataclass(slots=True)
class DatasetSpec:
    """Dataset manifest used by the paper-ready benchmark suite."""

    name: str
    display_name: str
    input_path: str
    splits: dict[str, list[str]]
    harmonization_fields: dict[str, bool]
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RunSpec:
    """One benchmark matrix run definition."""

    name: str
    display_name: str
    train_datasets: list[str]
    eval_datasets: list[str]
    description: str = ""


@dataclass(slots=True)
class AblationSpec:
    """One benchmark ablation definition."""

    name: str
    display_name: str
    base_run: str
    feature_indices: list[int] | None = None
    frame_feature_indices: list[int] | None = None
    horizon_frames: int | None = None
    row_overrides: dict[str, float | int | bool | str | None] | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkSuiteConfig:
    """Top-level benchmark suite configuration."""

    benchmark: SuiteBenchmarkConfig
    datasets: dict[str, DatasetSpec]
    runs: list[RunSpec]
    ablations: list[AblationSpec]


def load_benchmark_suite_config(path: str | Path) -> BenchmarkSuiteConfig:
    """Load the benchmark suite manifest."""
    payload = load_config(path)
    benchmark_payload = payload.get("benchmark", {})
    deep_payload = benchmark_payload.get("deep", {})
    benchmark = SuiteBenchmarkConfig(
        window_size=int(benchmark_payload.get("window_size", 10)),
        pause_speed_threshold=float(benchmark_payload.get("pause_speed_threshold", 0.03)),
        horizon_frames=int(benchmark_payload.get("horizon_frames", 5)),
        deep=SuiteDeepConfig(
            epochs=int(deep_payload.get("epochs", 3)),
            hidden_dim=int(deep_payload.get("hidden_dim", 16)),
            learning_rate=float(deep_payload.get("learning_rate", 0.005)),
            seed=int(deep_payload.get("seed", 13)),
            batch_size=int(deep_payload.get("batch_size", 16)),
        ),
    )
    datasets = {
        name: DatasetSpec(
            name=name,
            display_name=str(spec.get("display_name", name)),
            input_path=str(spec["input_path"]),
            splits={key: [str(value) for value in values] for key, values in spec.get("splits", {}).items()},
            harmonization_fields={key: bool(value) for key, value in spec.get("harmonization_fields", {}).items()},
            notes=[str(note) for note in spec.get("notes", [])],
        )
        for name, spec in payload.get("datasets", {}).items()
    }
    runs = [
        RunSpec(
            name=str(spec["name"]),
            display_name=str(spec.get("display_name", spec["name"])),
            train_datasets=[str(value) for value in spec.get("train_datasets", [])],
            eval_datasets=[str(value) for value in spec.get("eval_datasets", [])],
            description=str(spec.get("description", "")),
        )
        for spec in payload.get("runs", [])
    ]
    ablations = [
        AblationSpec(
            name=str(spec["name"]),
            display_name=str(spec.get("display_name", spec["name"])),
            base_run=str(spec["base_run"]),
            feature_indices=[int(value) for value in spec["feature_indices"]] if spec.get("feature_indices") is not None else None,
            frame_feature_indices=[int(value) for value in spec["frame_feature_indices"]] if spec.get("frame_feature_indices") is not None else None,
            horizon_frames=int(spec["horizon_frames"]) if spec.get("horizon_frames") is not None else None,
            row_overrides=dict(spec["row_overrides"]) if spec.get("row_overrides") is not None else None,
            notes=[str(note) for note in spec.get("notes", [])],
        )
        for spec in payload.get("ablations", [])
    ]
    return BenchmarkSuiteConfig(
        benchmark=benchmark,
        datasets=datasets,
        runs=runs,
        ablations=ablations,
    )


def _deep_train_config(config: SuiteBenchmarkConfig, horizon_frames: int) -> DeepTrainConfig:
    return DeepTrainConfig(
        window_size=config.window_size,
        horizon_frames=horizon_frames,
        epochs=config.deep.epochs,
        hidden_dim=config.deep.hidden_dim,
        learning_rate=config.deep.learning_rate,
        seed=config.deep.seed,
        batch_size=config.deep.batch_size,
    )


def _load_dataset_rows(config: BenchmarkSuiteConfig) -> dict[str, list[dict[str, object]]]:
    return {
        name: load_rows(spec.input_path)
        for name, spec in config.datasets.items()
    }


def _filter_rows_by_sessions(
    rows: list[dict[str, object]],
    sessions: list[str],
) -> list[dict[str, object]]:
    wanted = set(sessions)
    return [dict(row) for row in rows if str(row["session_id"]) in wanted]


def _apply_row_overrides(
    rows: list[dict[str, object]],
    overrides: dict[str, float | int | bool | str | None] | None,
) -> list[dict[str, object]]:
    if not overrides:
        return [dict(row) for row in rows]
    adjusted: list[dict[str, object]] = []
    for row in rows:
        copied = dict(row)
        copied.update(overrides)
        adjusted.append(copied)
    return adjusted


def _rows_for_datasets(
    dataset_rows: dict[str, list[dict[str, object]]],
    datasets: dict[str, DatasetSpec],
    dataset_names: list[str],
    split_name: str,
    overrides: dict[str, float | int | bool | str | None] | None,
) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for name in dataset_names:
        spec = datasets[name]
        split_rows = _filter_rows_by_sessions(dataset_rows[name], spec.splits[split_name])
        merged.extend(_apply_row_overrides(split_rows, overrides))
    return merged


def _build_window_splits(
    train_rows: list[dict[str, object]],
    eval_rows: list[dict[str, object]],
    config: SuiteBenchmarkConfig,
    horizon_frames: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[SequenceWindow], list[SequenceWindow]]:
    classical_train = build_windows(
        train_rows,
        window_size=config.window_size,
        pause_speed_threshold=config.pause_speed_threshold,
        horizon_frames=horizon_frames,
    )
    classical_eval = build_windows(
        eval_rows,
        window_size=config.window_size,
        pause_speed_threshold=config.pause_speed_threshold,
        horizon_frames=horizon_frames,
    )
    deep_train = build_sequence_windows(
        train_rows,
        window_size=config.window_size,
        horizon_frames=horizon_frames,
    )
    deep_eval = build_sequence_windows(
        eval_rows,
        window_size=config.window_size,
        horizon_frames=horizon_frames,
    )
    return classical_train, classical_eval, deep_train, deep_eval


def _standardize_rules_predictions(
    eval_windows: list[dict[str, object]],
) -> list[dict[str, Any]]:
    lookup = {
        (record["session_id"], record["end_frame_idx"]): record
        for record in predict_rules_windows(eval_windows)
    }
    standardized: list[dict[str, Any]] = []
    for window in eval_windows:
        key = (str(window["session_id"]), int(window["end_frame_idx"]))
        prediction = lookup[key]
        standardized.append(
            {
                "dataset_name": str(window.get("dataset_name", "unknown")),
                "session_id": key[0],
                "end_frame_idx": key[1],
                "true_state": str(window["current_state"]),
                "predicted_state": str(prediction["predicted_state"]),
                "true_future_hesitation": int(window["future_hesitation"]),
                "true_future_correction": int(window["future_correction"]),
                "future_hesitation_probability": float(prediction["hesitation_risk"]),
                "future_correction_probability": float(prediction["correction_rework_risk"]),
                "triggered_rules": list(prediction["triggered_rules"]),
            }
        )
    return standardized


def _standardize_classical_predictions(
    eval_windows: list[dict[str, object]],
    model_path: Path,
) -> list[dict[str, Any]]:
    runtime = load_classical_runtime(model_path)
    standardized: list[dict[str, Any]] = []
    for window in eval_windows:
        prediction = predict_classical_window(runtime, [float(value) for value in window["features"]])
        standardized.append(
            {
                "dataset_name": str(window.get("dataset_name", "unknown")),
                "session_id": str(window["session_id"]),
                "end_frame_idx": int(window["end_frame_idx"]),
                "true_state": str(window["current_state"]),
                "predicted_state": str(prediction["predicted_state"]),
                "true_future_hesitation": int(window["future_hesitation"]),
                "true_future_correction": int(window["future_correction"]),
                "future_hesitation_probability": float(prediction["future_hesitation_probability"]),
                "future_correction_probability": float(prediction["future_correction_probability"]),
                "state_probabilities": prediction["state_probabilities"],
            }
        )
    return standardized


def _standardize_deep_predictions(
    eval_windows: list[SequenceWindow],
    model_path: Path,
) -> list[dict[str, Any]]:
    lookup = {
        (record["session_id"], record["end_frame_idx"]): record
        for record in infer_sequence_deep_windows(eval_windows, str(model_path))
    }
    standardized: list[dict[str, Any]] = []
    for window in eval_windows:
        key = (window["session_id"], window["end_frame_idx"])
        prediction = lookup[key]
        standardized.append(
            {
                "dataset_name": window["dataset_name"],
                "session_id": window["session_id"],
                "end_frame_idx": int(window["end_frame_idx"]),
                "true_state": str(window["current_state"]),
                "predicted_state": str(prediction["predicted_state"]),
                "true_future_hesitation": int(window["future_hesitation"]),
                "true_future_correction": int(window["future_correction"]),
                "future_hesitation_probability": float(prediction["future_hesitation_within_horizon"]),
                "future_correction_probability": float(prediction["future_correction_within_horizon"]),
                "state_probabilities": prediction["state_probabilities"],
            }
        )
    return standardized


def _write_run_summary(
    output_dir: Path,
    display_name: str,
    metrics: dict[str, Any],
) -> None:
    lines = [
        f"# {display_name}",
        "",
        "| Model | State Accuracy | State Macro F1 | Future Hesitation AUPRC | Future Correction AUPRC |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Rules | {metrics['rules']['current_state']['accuracy']:.4f} | {metrics['rules']['current_state']['macro_f1']:.4f} | {metrics['rules']['future_hesitation']['auprc']:.4f} | {metrics['rules']['future_correction']['auprc']:.4f} |",
        f"| Classical | {metrics['classical']['current_state']['accuracy']:.4f} | {metrics['classical']['current_state']['macro_f1']:.4f} | {metrics['classical']['future_hesitation']['auprc']:.4f} | {metrics['classical']['future_correction']['auprc']:.4f} |",
        f"| Deep | {metrics['deep']['current_state']['accuracy']:.4f} | {metrics['deep']['current_state']['macro_f1']:.4f} | {metrics['deep']['future_hesitation']['auprc']:.4f} | {metrics['deep']['future_correction']['auprc']:.4f} |",
    ]
    (output_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _binary_metrics_from_predictions(
    predictions: list[dict[str, Any]],
    label_key: str,
    probability_key: str,
) -> dict[str, Any]:
    labels = [int(record[label_key]) for record in predictions]
    probabilities = [float(record[probability_key]) for record in predictions]
    return binary_metrics(labels, probabilities, threshold=0.5)


def _run_one_benchmark(
    output_dir: Path,
    run_spec: RunSpec,
    benchmark_config: SuiteBenchmarkConfig,
    dataset_specs: dict[str, DatasetSpec],
    dataset_rows: dict[str, list[dict[str, object]]],
    run_name: str | None = None,
    display_name: str | None = None,
    feature_indices: list[int] | None = None,
    frame_feature_indices: list[int] | None = None,
    horizon_frames: int | None = None,
    row_overrides: dict[str, float | int | bool | str | None] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    effective_horizon = horizon_frames or benchmark_config.horizon_frames
    effective_run_name = run_name or run_spec.name
    effective_display_name = display_name or run_spec.display_name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _rows_for_datasets(
        dataset_rows,
        dataset_specs,
        run_spec.train_datasets,
        "train",
        row_overrides,
    )
    eval_rows = _rows_for_datasets(
        dataset_rows,
        dataset_specs,
        run_spec.eval_datasets,
        "test",
        row_overrides,
    )
    classical_train, classical_eval, deep_train, deep_eval = _build_window_splits(
        train_rows,
        eval_rows,
        benchmark_config,
        effective_horizon,
    )
    if not classical_train or not classical_eval or not deep_train or not deep_eval:
        raise ValueError(f"Run {run_spec.name} did not produce enough windows")

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run": effective_run_name,
                "display_name": effective_display_name,
                "train_datasets": run_spec.train_datasets,
                "eval_datasets": run_spec.eval_datasets,
                "feature_indices": feature_indices,
                "frame_feature_indices": frame_feature_indices,
                "horizon_frames": effective_horizon,
                "row_overrides": row_overrides,
                "notes": notes or [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    classical_metrics = train_classical_on_windows(
        train_windows=classical_train,
        eval_windows=classical_eval,
        output_dir=str(output_dir / "classical"),
        window_size=benchmark_config.window_size,
        pause_speed_threshold=benchmark_config.pause_speed_threshold,
        horizon_frames=effective_horizon,
        feature_indices=feature_indices,
    )
    deep_metrics = train_deep_on_windows(
        train_windows=deep_train,
        eval_windows=deep_eval,
        output_dir=str(output_dir / "deep"),
        cfg=_deep_train_config(benchmark_config, effective_horizon),
        frame_feature_indices=frame_feature_indices,
    )
    deep_model_path = output_dir / "deep" / ("deep_model.pt" if (output_dir / "deep" / "deep_model.pt").exists() else "deep_model.json")

    rules_predictions = _standardize_rules_predictions(classical_eval)
    classical_predictions = _standardize_classical_predictions(classical_eval, output_dir / "classical" / "classical_model.json")
    deep_predictions = _standardize_deep_predictions(deep_eval, deep_model_path)

    rules_metrics = write_model_error_report(rules_predictions, output_dir / "error_analysis" / "rules")
    classical_error = write_model_error_report(classical_predictions, output_dir / "error_analysis" / "classical")
    deep_error = write_model_error_report(deep_predictions, output_dir / "error_analysis" / "deep")

    write_jsonl(output_dir / "rules" / "predictions.jsonl", rules_predictions)
    write_jsonl(output_dir / "classical" / "predictions.jsonl", classical_predictions)
    write_jsonl(output_dir / "deep" / "predictions.jsonl", deep_predictions)

    rules_future_hesitation = _binary_metrics_from_predictions(
        rules_predictions,
        label_key="true_future_hesitation",
        probability_key="future_hesitation_probability",
    )
    rules_future_correction = _binary_metrics_from_predictions(
        rules_predictions,
        label_key="true_future_correction",
        probability_key="future_correction_probability",
    )
    classical_eval_metrics = evaluate_classical_windows(load_classical_runtime(output_dir / "classical" / "classical_model.json"), classical_eval)
    deep_eval_metrics = evaluate_deep_windows(deep_eval, str(deep_model_path), ThresholdConfig())
    summary_metrics = {
        "rules": {
            "current_state": rules_metrics["state_metrics"],
            "future_hesitation": rules_future_hesitation,
            "future_correction": rules_future_correction,
        },
        "classical": {
            "current_state": classical_eval_metrics["current_state_classical"],
            "future_hesitation": classical_eval_metrics["future_hesitation"],
            "future_correction": classical_eval_metrics["future_correction"],
        },
        "deep": {
            "current_state": deep_eval_metrics["current_state_deep"],
            "future_hesitation": deep_eval_metrics["future_hesitation"],
            "future_correction": deep_eval_metrics["future_correction"],
        },
        "counts": {
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_classical_windows": len(classical_train),
            "eval_classical_windows": len(classical_eval),
            "train_deep_windows": len(deep_train),
            "eval_deep_windows": len(deep_eval),
        },
        "notes": notes or [],
        "rules_trigger_audit": rules_metrics["trigger_audit"],
        "classical_error_summary": classical_error,
        "deep_error_summary": deep_error,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary_metrics, indent=2), encoding="utf-8")
    _write_run_summary(output_dir, effective_display_name, summary_metrics)
    return {
        "run_name": effective_run_name,
        "display_name": effective_display_name,
        "train_datasets": list(run_spec.train_datasets),
        "eval_datasets": list(run_spec.eval_datasets),
        "summary": summary_metrics,
        "predictions": {
            "rules": rules_predictions,
            "classical": classical_predictions,
            "deep": deep_predictions,
        },
    }


def _benchmark_table_rows(
    runs: list[dict[str, Any]],
    family_key: str,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for run in runs:
        metrics = run["summary"][family_key]
        rows.append(
            [
                run["display_name"],
                family_key,
                ",".join(run["train_datasets"]),
                ",".join(run["eval_datasets"]),
                f"{float(metrics['current_state']['accuracy']):.4f}",
                f"{float(metrics['current_state']['macro_f1']):.4f}",
                f"{float(metrics['future_hesitation']['auprc']):.4f}",
                f"{float(metrics['future_correction']['auprc']):.4f}",
            ]
        )
    return rows


def _write_label_distribution_table(
    output_dir: Path,
    dataset_specs: dict[str, DatasetSpec],
    dataset_rows: dict[str, list[dict[str, object]]],
) -> None:
    counts: list[list[Any]] = []
    for dataset_name, spec in dataset_specs.items():
        for split_name, sessions in spec.splits.items():
            subset = _filter_rows_by_sessions(dataset_rows[dataset_name], sessions)
            label_counts: dict[str, int] = {}
            for row in subset:
                label = str(row.get("latent_state", "unknown"))
                label_counts[label] = label_counts.get(label, 0) + 1
            for label, count in sorted(label_counts.items()):
                counts.append([spec.display_name, split_name, label, count, len(sessions)])
    write_table_bundle(
        output_dir / "label_distribution_table",
        ["Dataset", "Split", "Label", "Frames", "Sessions"],
        counts,
    )


def _write_harmonization_table(
    output_dir: Path,
    dataset_specs: dict[str, DatasetSpec],
) -> None:
    all_fields = sorted({field for spec in dataset_specs.values() for field in spec.harmonization_fields})
    dataset_names = list(dataset_specs.keys())
    rows: list[list[Any]] = []
    for field_name in all_fields:
        values = ["yes" if dataset_specs[name].harmonization_fields.get(field_name, False) else "no" for name in dataset_names]
        coverage_gap = "gap" if len(set(values)) > 1 else "aligned"
        rows.append([field_name, *values, coverage_gap])
    headers = ["Field", *[dataset_specs[name].display_name for name in dataset_names], "Gap"]
    write_table_bundle(output_dir / "harmonization_coverage_gap_table", headers, rows)


def _write_transfer_gap_notes(
    output_dir: Path,
    benchmark_runs: list[dict[str, Any]],
    dataset_specs: dict[str, DatasetSpec],
) -> None:
    run_lookup = {run["run_name"]: run for run in benchmark_runs}
    lines = ["# Dataset-Specific Transfer Gap Notes", ""]

    if "chico_to_havid" in run_lookup and "havid_within" in run_lookup:
        cross = run_lookup["chico_to_havid"]["summary"]["deep"]["current_state"]["macro_f1"]
        within = run_lookup["havid_within"]["summary"]["deep"]["current_state"]["macro_f1"]
        delta = float(cross) - float(within)
        lines.append(
            f"- CHICO -> HA-ViD deep macro F1: {float(cross):.4f} vs HA-ViD within-dataset {float(within):.4f} (delta {delta:.4f})."
        )
        lines.append(
            f"- HA-ViD harmonization gaps: {', '.join(field for field, covered in dataset_specs['ha_vid'].harmonization_fields.items() if not covered) or 'none'}."
        )
    if "havid_to_chico" in run_lookup and "chico_within" in run_lookup:
        cross = run_lookup["havid_to_chico"]["summary"]["deep"]["current_state"]["macro_f1"]
        within = run_lookup["chico_within"]["summary"]["deep"]["current_state"]["macro_f1"]
        delta = float(cross) - float(within)
        lines.append(
            f"- HA-ViD -> CHICO deep macro F1: {float(cross):.4f} vs CHICO within-dataset {float(within):.4f} (delta {delta:.4f})."
        )
        lines.append(
            f"- CHICO harmonization gaps: {', '.join(field for field, covered in dataset_specs['chico'].harmonization_fields.items() if not covered) or 'none'}."
        )

    (output_dir / "transfer_gap_notes.md").write_text("\n".join(lines), encoding="utf-8")


def _render_qualitative_panels(
    output_dir: Path,
    benchmark_runs: list[dict[str, Any]],
) -> None:
    run_lookup = {run["run_name"]: run for run in benchmark_runs}
    targets = [
        ("chico_to_havid", "qualitative_cross_dataset"),
        ("merged_train_eval", "qualitative_merged"),
    ]
    for run_name, filename in targets:
        if run_name not in run_lookup:
            continue
        run = run_lookup[run_name]
        deep_predictions = run["predictions"]["deep"]
        counts: dict[str, int] = {}
        for record in deep_predictions:
            session_id = str(record["session_id"])
            counts[session_id] = counts.get(session_id, 0) + int(record["true_state"] != record["predicted_state"])
        selected_session = max(counts.items(), key=lambda item: (item[1], item[0]))[0] if counts else str(deep_predictions[0]["session_id"])

        def track(records: list[dict[str, Any]], key: str) -> list[str]:
            values = [record for record in records if str(record["session_id"]) == selected_session]
            values.sort(key=lambda item: int(item["end_frame_idx"]))
            return [str(record[key]) for record in values]

        render_qualitative_panel(
            output_dir / "figures" / f"{filename}.svg",
            f"{run['display_name']} | {selected_session}",
            [
                ("Ground Truth", track(run["predictions"]["deep"], "true_state")),
                ("Rules", track(run["predictions"]["rules"], "predicted_state")),
                ("Classical", track(run["predictions"]["classical"], "predicted_state")),
                ("Deep", track(run["predictions"]["deep"], "predicted_state")),
            ],
        )


def run_benchmark_suite(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the full paper-ready benchmark suite and write artifacts."""
    config = load_benchmark_suite_config(config_path)
    dataset_rows = _load_dataset_rows(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest_snapshot.json").write_text(
        json.dumps(load_config(config_path), indent=2),
        encoding="utf-8",
    )

    benchmark_runs: list[dict[str, Any]] = []
    for run_spec in config.runs:
        benchmark_runs.append(
            _run_one_benchmark(
                output_dir=out / "benchmarks" / run_spec.name,
                run_spec=run_spec,
                benchmark_config=config.benchmark,
                dataset_specs=config.datasets,
                dataset_rows=dataset_rows,
            )
        )

    ablation_runs: list[dict[str, Any]] = []
    run_lookup = {run.name: run for run in config.runs}
    for ablation in config.ablations:
        ablation_runs.append(
            _run_one_benchmark(
                output_dir=out / "ablations" / ablation.name,
                run_spec=run_lookup[ablation.base_run],
                benchmark_config=config.benchmark,
                dataset_specs=config.datasets,
                dataset_rows=dataset_rows,
                run_name=ablation.name,
                display_name=ablation.display_name,
                feature_indices=ablation.feature_indices,
                frame_feature_indices=ablation.frame_feature_indices,
                horizon_frames=ablation.horizon_frames,
                row_overrides=ablation.row_overrides,
                notes=ablation.notes,
            )
        )

    benchmark_headers = [
        "Run",
        "Model",
        "Train Datasets",
        "Eval Datasets",
        "State Accuracy",
        "State Macro F1",
        "Future Hesitation AUPRC",
        "Future Correction AUPRC",
    ]
    benchmark_rows = (
        _benchmark_table_rows(benchmark_runs, "rules")
        + _benchmark_table_rows(benchmark_runs, "classical")
        + _benchmark_table_rows(benchmark_runs, "deep")
    )
    write_table_bundle(out / "paper" / "final_benchmark_table", benchmark_headers, benchmark_rows)

    ablation_rows = (
        _benchmark_table_rows(ablation_runs, "rules")
        + _benchmark_table_rows(ablation_runs, "classical")
        + _benchmark_table_rows(ablation_runs, "deep")
    )
    write_table_bundle(out / "paper" / "ablation_table", benchmark_headers, ablation_rows)

    _write_harmonization_table(out / "paper", config.datasets)
    _write_label_distribution_table(out / "paper", config.datasets, dataset_rows)
    _write_transfer_gap_notes(out / "paper", benchmark_runs, config.datasets)
    render_pipeline_figure(out / "paper" / "figures" / "pipeline_overview.svg")
    _render_qualitative_panels(out / "paper", benchmark_runs)

    summary = {
        "benchmark_runs": [
            {
                "run_name": run["run_name"],
                "display_name": run["display_name"],
                "summary": run["summary"],
            }
            for run in benchmark_runs
        ],
        "ablation_runs": [
            {
                "run_name": run["run_name"],
                "display_name": run["display_name"],
                "summary": run["summary"],
            }
            for run in ablation_runs
        ],
        "paper_artifacts_dir": str(out / "paper"),
    }
    (out / "suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
