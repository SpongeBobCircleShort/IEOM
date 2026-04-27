#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hesitation.io.config import load_config
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import NoiseConfig, ScenarioConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/simulation/default_scene.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-sessions", type=int, default=5)
    parser.add_argument("--frame-rate", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_dict = load_config(args.config)
    noise_dict = scenario_dict.get("noise", {})
    scenario = ScenarioConfig(**{**scenario_dict, "noise": NoiseConfig(**noise_dict)})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for i in range(args.n_sessions):
            traj, latent = generate_session(
                session_id=f"session_{i}",
                scenario=scenario,
                frame_rate_hz=args.frame_rate,
                seed=args.seed + i,
            )
            for frame, state in zip(traj.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
