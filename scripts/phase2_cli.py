#!/usr/bin/env python3
"""Unified CLI for Phase 2 and Phase 3.5 workflows."""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hesitation.deep.pipeline import (
    compare_models,
    compare_models_multiseed,
    evaluate_deep,
    evaluate_deep_calibrated,
    infer_sequence_deep,
    train_deep,
    train_deep_multiseed,
    tune_thresholds,
)
from hesitation.evaluation.suite import run_benchmark_suite
from hesitation.io.config import load_config
from hesitation.io.writers import write_jsonl
from hesitation.ml.pipeline import evaluate_classical, infer_sequence, predict_future_risk, train_classical
from hesitation.policy.recommender import PolicyInput, recommend_policy
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import NoiseConfig, ScenarioConfig


def _add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--pause-speed-threshold", type=float, default=0.03)
    parser.add_argument("--horizon-frames", type=int, default=20)


def _parse_seed_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _generate_scenarios_extended(
    output: str,
    config_paths: list[str],
    sessions_per_scenario: int,
    frame_rate: int,
    seed: int,
) -> dict[str, object]:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    scenario_counts: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as out:
        global_session_idx = 0
        for cfg_path in config_paths:
            scenario_dict = load_config(cfg_path)
            noise_dict = scenario_dict.get("noise", {})
            scenario = ScenarioConfig(**{**scenario_dict, "noise": NoiseConfig(**noise_dict)})
            scenario_name = scenario.name
            scenario_counts[scenario_name] = 0
            for local_idx in range(sessions_per_scenario):
                sid = f"{scenario_name}_session_{global_session_idx}"
                traj, latent = generate_session(
                    session_id=sid,
                    scenario=scenario,
                    frame_rate_hz=frame_rate,
                    seed=seed + global_session_idx + local_idx,
                )
                for frame, state in zip(traj.frames, latent):
                    row = frame.model_dump()
                    row["latent_state"] = state.value
                    row["scenario_name"] = scenario_name
                    out.write(json.dumps(row) + "\n")
                    total_rows += 1
                scenario_counts[scenario_name] += 1
                global_session_idx += 1

    return {
        "output": output,
        "total_rows": total_rows,
        "scenario_sessions": scenario_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2/3.5 ML workflows")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train-classical")
    _add_common_dataset_args(train)
    train.add_argument("--output-dir", required=True)

    evaluate = sub.add_parser("evaluate-classical")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--model-path", required=True)
    evaluate.add_argument("--output", required=False)

    infer = sub.add_parser("infer-sequence")
    infer.add_argument("--input", required=True)
    infer.add_argument("--model-path", required=True)
    infer.add_argument("--output", required=True)

    risk = sub.add_parser("predict-risk")
    risk.add_argument("--input", required=True)
    risk.add_argument("--model-path", required=True)
    risk.add_argument("--output", required=True)

    train_deep_p = sub.add_parser("train-deep")
    train_deep_p.add_argument("--input", required=True)
    train_deep_p.add_argument("--output-dir", required=True)
    train_deep_p.add_argument("--window-size", type=int, default=20)
    train_deep_p.add_argument("--horizon-frames", type=int, default=20)
    train_deep_p.add_argument("--epochs", type=int, default=20)
    train_deep_p.add_argument("--hidden-dim", type=int, default=64)
    train_deep_p.add_argument("--learning-rate", type=float, default=0.001)
    train_deep_p.add_argument("--seed", type=int, default=42)
    train_deep_p.add_argument("--batch-size", type=int, default=64)

    train_deep_ms = sub.add_parser("train-deep-multiseed")
    train_deep_ms.add_argument("--input", required=True)
    train_deep_ms.add_argument("--output-dir", required=True)
    train_deep_ms.add_argument("--seeds", default="11,22,33")
    train_deep_ms.add_argument("--window-size", type=int, default=20)
    train_deep_ms.add_argument("--horizon-frames", type=int, default=20)
    train_deep_ms.add_argument("--epochs", type=int, default=20)
    train_deep_ms.add_argument("--hidden-dim", type=int, default=64)
    train_deep_ms.add_argument("--learning-rate", type=float, default=0.001)
    train_deep_ms.add_argument("--batch-size", type=int, default=64)

    eval_deep_p = sub.add_parser("evaluate-deep")
    eval_deep_p.add_argument("--input", required=True)
    eval_deep_p.add_argument("--model-path", required=True)
    eval_deep_p.add_argument("--output", required=False)

    eval_cal_p = sub.add_parser("evaluate-deep-calibrated")
    eval_cal_p.add_argument("--input", required=True)
    eval_cal_p.add_argument("--model-path", required=True)
    eval_cal_p.add_argument("--threshold-path", required=True)
    eval_cal_p.add_argument("--output", required=False)

    tune_p = sub.add_parser("tune-thresholds")
    tune_p.add_argument("--input", required=True)
    tune_p.add_argument("--model-path", required=True)
    tune_p.add_argument("--output", required=True)

    infer_deep_p = sub.add_parser("infer-sequence-deep")
    infer_deep_p.add_argument("--input", required=True)
    infer_deep_p.add_argument("--model-path", required=True)
    infer_deep_p.add_argument("--output", required=True)

    compare_p = sub.add_parser("compare-models")
    compare_p.add_argument("--input", required=True)
    compare_p.add_argument("--classical-model-path", required=True)
    compare_p.add_argument("--deep-model-path", required=True)
    compare_p.add_argument("--output-dir", required=True)

    compare_ms = sub.add_parser("compare-models-multiseed")
    compare_ms.add_argument("--input", required=True)
    compare_ms.add_argument("--classical-model-path", required=True)
    compare_ms.add_argument("--deep-root-dir", required=True)
    compare_ms.add_argument("--seeds", default="11,22,33")
    compare_ms.add_argument("--output-dir", required=True)

    bench_suite = sub.add_parser("run-benchmark-suite")
    bench_suite.add_argument("--config", required=True)
    bench_suite.add_argument("--output-dir", required=True)

    gen_ext = sub.add_parser("generate-scenarios-extended")
    gen_ext.add_argument("--output", required=True)
    gen_ext.add_argument(
        "--configs",
        default="configs/simulation/default_scene.yaml,configs/simulation/stress_scene.yaml,configs/simulation/ambiguous_scene.yaml,configs/simulation/domain_gap_scene.yaml",
    )
    gen_ext.add_argument("--sessions-per-scenario", type=int, default=3)
    gen_ext.add_argument("--frame-rate", type=int, default=10)
    gen_ext.add_argument("--seed", type=int, default=101)

    policy = sub.add_parser("recommend-policy")
    policy.add_argument("--current-state", required=True)
    policy.add_argument("--current-hesitation-prob", type=float, required=True)
    policy.add_argument("--future-hesitation-prob", type=float, required=True)
    policy.add_argument("--future-correction-prob", type=float, required=True)
    policy.add_argument("--workspace-distance", type=float, default=0.5)

    args = parser.parse_args()

    if args.cmd == "train-classical":
        metrics = train_classical(
            input_path=args.input,
            output_dir=args.output_dir,
            window_size=args.window_size,
            pause_speed_threshold=args.pause_speed_threshold,
            horizon_frames=args.horizon_frames,
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.cmd == "evaluate-classical":
        metrics = evaluate_classical(args.input, args.model_path)
        if args.output:
            Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return

    if args.cmd == "infer-sequence":
        records = infer_sequence(args.input, args.model_path)
        write_jsonl(args.output, records)
        print(f"wrote {len(records)} inference rows -> {args.output}")
        return

    if args.cmd == "predict-risk":
        records = predict_future_risk(args.input, args.model_path)
        write_jsonl(args.output, records)
        print(f"wrote {len(records)} risk rows -> {args.output}")
        return

    if args.cmd == "train-deep":
        metrics = train_deep(
            input_path=args.input,
            output_dir=args.output_dir,
            window_size=args.window_size,
            horizon_frames=args.horizon_frames,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.cmd == "train-deep-multiseed":
        metrics = train_deep_multiseed(
            input_path=args.input,
            output_dir=args.output_dir,
            seeds=_parse_seed_list(args.seeds),
            window_size=args.window_size,
            horizon_frames=args.horizon_frames,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.cmd == "evaluate-deep":
        metrics = evaluate_deep(args.input, args.model_path)
        if args.output:
            Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return

    if args.cmd == "evaluate-deep-calibrated":
        metrics = evaluate_deep_calibrated(args.input, args.model_path, args.threshold_path)
        if args.output:
            Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return

    if args.cmd == "tune-thresholds":
        tuned = tune_thresholds(args.input, args.model_path, args.output)
        print(json.dumps(tuned, indent=2))
        return

    if args.cmd == "infer-sequence-deep":
        records = infer_sequence_deep(args.input, args.model_path)
        write_jsonl(args.output, records)
        print(f"wrote {len(records)} deep inference rows -> {args.output}")
        return

    if args.cmd == "compare-models":
        metrics = compare_models(
            input_path=args.input,
            classical_model_path=args.classical_model_path,
            deep_model_path=args.deep_model_path,
            output_dir=args.output_dir,
        )
        print(json.dumps(metrics["summary"], indent=2))
        return

    if args.cmd == "compare-models-multiseed":
        metrics = compare_models_multiseed(
            input_path=args.input,
            classical_model_path=args.classical_model_path,
            deep_root_dir=args.deep_root_dir,
            seeds=_parse_seed_list(args.seeds),
            output_dir=args.output_dir,
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.cmd == "run-benchmark-suite":
        metrics = run_benchmark_suite(args.config, args.output_dir)
        print(json.dumps(metrics, indent=2))
        return

    if args.cmd == "generate-scenarios-extended":
        report = _generate_scenarios_extended(
            output=args.output,
            config_paths=[item.strip() for item in args.configs.split(",") if item.strip()],
            sessions_per_scenario=args.sessions_per_scenario,
            frame_rate=args.frame_rate,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2))
        return

    if args.cmd == "recommend-policy":
        rec = recommend_policy(
            PolicyInput(
                inferred_current_state=args.current_state,
                current_hesitation_probability=args.current_hesitation_prob,
                future_hesitation_probability=args.future_hesitation_prob,
                future_correction_probability=args.future_correction_prob,
                workspace_distance=args.workspace_distance,
            )
        )
        print(json.dumps(rec.to_dict(), indent=2))
        return

    raise RuntimeError("Unhandled command")


if __name__ == "__main__":
    main()
