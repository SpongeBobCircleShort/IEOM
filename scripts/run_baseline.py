#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hesitation.baselines.rules_engine import classify_window
from hesitation.features.pipeline import window_to_features
from hesitation.io.config import load_config
from hesitation.io.loaders import load_jsonl_frames
from hesitation.io.writers import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rules-config", default="configs/baseline/rules_v1.yaml")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--pause-speed-threshold", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.rules_config))
    frames = load_jsonl_frames(args.input)
    records = []
    for end in range(args.window_size, len(frames) + 1):
        window = frames[end - args.window_size : end]
        feat = window_to_features(window, pause_speed_threshold=args.pause_speed_threshold)
        out = classify_window(feat, thresholds=cfg["thresholds"], risk_cfg=cfg["risk"])
        records.append(
            {
                "session_id": feat.session_id,
                "end_frame_idx": feat.end_frame_idx,
                **out.model_dump(),
            }
        )
    write_jsonl(args.output, records)


if __name__ == "__main__":
    main()
