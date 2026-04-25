#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hesitation.io.writers import write_jsonl
from hesitation.ml.pipeline import evaluate_classical, infer_sequence, predict_future_risk, train_classical
from hesitation.policy.recommender import PolicyInput, recommend_policy


def _add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--pause-speed-threshold", type=float, default=0.03)
    parser.add_argument("--horizon-frames", type=int, default=20)


def _add_common_deep_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--horizon-frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 classical + Phase 3 deep ML workflows")
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

    policy = sub.add_parser("recommend-policy")
    policy.add_argument("--current-state", required=True)
    policy.add_argument("--current-hesitation-prob", type=float, required=True)
    policy.add_argument("--future-hesitation-prob", type=float, required=True)
    policy.add_argument("--future-correction-prob", type=float, required=True)
    policy.add_argument("--workspace-distance", type=float, default=0.5)

    train_deep = sub.add_parser("train-deep")
    _add_common_deep_args(train_deep)
    train_deep.add_argument("--output-dir", required=True)
    train_deep.add_argument("--hidden-size", type=int, default=24)
    train_deep.add_argument("--epochs", type=int, default=12)
    train_deep.add_argument("--batch-size", type=int, default=32)
    train_deep.add_argument("--learning-rate", type=float, default=1e-2)

    evaluate_deep = sub.add_parser("evaluate-deep")
    evaluate_deep.add_argument("--input", required=True)
    evaluate_deep.add_argument("--model-path", required=True)
    evaluate_deep.add_argument("--output", required=False)

    tune = sub.add_parser("tune-thresholds")
    tune.add_argument("--input", required=True)
    tune.add_argument("--model-path", required=True)
    tune.add_argument("--output", required=True)

    evaluate_calibrated = sub.add_parser("evaluate-deep-calibrated")
    evaluate_calibrated.add_argument("--input", required=True)
    evaluate_calibrated.add_argument("--model-path", required=True)
    evaluate_calibrated.add_argument("--threshold-path", required=True)
    evaluate_calibrated.add_argument("--output", required=False)

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

    if args.cmd == "train-deep":
        from hesitation.ml.deep import train_deep

        metrics = train_deep(
            input_path=args.input,
            output_dir=args.output_dir,
            window_size=args.window_size,
            horizon_frames=args.horizon_frames,
            seed=args.seed,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.cmd == "evaluate-deep":
        from hesitation.ml.deep import evaluate_deep

        metrics = evaluate_deep(args.input, args.model_path, output_path=args.output)
        print(json.dumps(metrics, indent=2))
        return

    if args.cmd == "tune-thresholds":
        from hesitation.ml.deep import tune_thresholds

        thresholds = tune_thresholds(args.input, args.model_path, output_path=args.output)
        print(json.dumps(thresholds, indent=2))
        return

    if args.cmd == "evaluate-deep-calibrated":
        from hesitation.ml.deep import evaluate_deep_calibrated

        metrics = evaluate_deep_calibrated(
            args.input,
            args.model_path,
            args.threshold_path,
            output_path=args.output,
        )
        print(json.dumps(metrics, indent=2))
        return

    raise RuntimeError("Unhandled command")


if __name__ == "__main__":
    main()
