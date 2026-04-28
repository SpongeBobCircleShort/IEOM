"""Command-line interface for hesitation model inference.

Enables MATLAB to call the model via subprocess/system calls.

Usage:
    python -m hesitation.inference.cli --format json predict \\
        --mean-hand-speed 0.5 \\
        --pause-ratio 0.1 \\
        --progress-delta 0.8 \\
        --reversal-count 0 \\
        --retry-count 1 \\
        --task-step-id 2 \\
        --human-robot-distance 0.3
"""

import argparse
import json
import sys

from hesitation.inference import HesitationPredictor


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Hesitation model inference CLI")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (.pt or .json)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "text"],
        default="json",
        help="Output format",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make a single prediction")
    predict_parser.add_argument("--mean-hand-speed", type=float, required=True)
    predict_parser.add_argument("--pause-ratio", type=float, required=True)
    predict_parser.add_argument("--progress-delta", type=float, required=True)
    predict_parser.add_argument("--reversal-count", type=int, required=True)
    predict_parser.add_argument("--retry-count", type=int, required=True)
    predict_parser.add_argument("--task-step-id", type=int, required=True)
    predict_parser.add_argument("--human-robot-distance", type=float, required=True)

    # Health check command
    subparsers.add_parser("health", help="Check if model is loaded")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Load model
    try:
        predictor = HesitationPredictor.load_default()
        if args.model_path:
            predictor = HesitationPredictor(torch_model_path=args.model_path)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {e}"}))
        return 1

    if args.command == "health":
        status = {"model_loaded": predictor.use_torch or predictor.fallback_model is not None}
        print(json.dumps(status))
        return 0

    if args.command == "predict":
        features = {
            "mean_hand_speed": args.mean_hand_speed,
            "pause_ratio": args.pause_ratio,
            "progress_delta": args.progress_delta,
            "reversal_count": args.reversal_count,
            "retry_count": args.retry_count,
            "task_step_id": args.task_step_id,
            "human_robot_distance": args.human_robot_distance,
        }

        try:
            prediction = predictor.predict_single(features)

            if args.format == "json":
                print(prediction.to_json())
            elif args.format == "csv":
                # CSV: state,confidence,future_hesitation,future_correction
                row = (  # noqa: E501
                    f"{prediction.state},{prediction.confidence:.3f},"
                    f"{prediction.future_hesitation_prob:.3f},"
                    f"{prediction.future_correction_prob:.3f}"
                )
                print(row)
            elif args.format == "text":
                print(f"State: {prediction.state}")
                print(f"Confidence: {prediction.confidence:.2%}")
                print(f"Future Hesitation: {prediction.future_hesitation_prob:.2%}")
                print(f"Future Correction: {prediction.future_correction_prob:.2%}")
                print("\nState Probabilities:")
                for state, prob in sorted(prediction.state_probabilities.items()):
                    print(f"  {state}: {prob:.2%}")

            return 0

        except Exception as e:
            print(json.dumps({"error": f"Prediction failed: {e}"}))
            return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
