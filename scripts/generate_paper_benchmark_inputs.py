#!/usr/bin/env python3
"""Generate deterministic local inputs for the paper-ready benchmark suite."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hesitation.io.config import load_config
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import NoiseConfig, ScenarioConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--havid-config", default="configs/simulation/domain_gap_scene.yaml")
    parser.add_argument("--havid-sessions", type=int, default=6)
    parser.add_argument("--frame-rate", type=int, default=10)
    parser.add_argument("--seed", type=int, default=140)
    return parser.parse_args()


def _write_havid_fixture(
    output_path: Path,
    config_path: str,
    sessions: int,
    frame_rate: int,
    seed: int,
) -> dict[str, object]:
    scenario_dict = load_config(config_path)
    noise_dict = scenario_dict.get("noise", {})
    scenario = ScenarioConfig(**{**scenario_dict, "noise": NoiseConfig(**noise_dict)})

    total_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for session_index in range(sessions):
            session_id = f"havid_s{session_index}"
            trajectory, latent_states = generate_session(
                session_id=session_id,
                scenario=scenario,
                frame_rate_hz=frame_rate,
                seed=seed + session_index,
            )
            for frame, state in zip(trajectory.frames, latent_states, strict=False):
                row = frame.model_dump()
                row["latent_state"] = state.value
                row["dataset_name"] = "ha_vid"
                handle.write(json.dumps(row) + "\n")
                total_rows += 1
    return {
        "path": str(output_path),
        "sessions": sessions,
        "rows": total_rows,
        "config": config_path,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chico_source = Path("merged_database/sample_outputs/chico_model_input_fixture.jsonl")
    chico_target = out_dir / "chico_model_input.jsonl"
    shutil.copyfile(chico_source, chico_target)

    havid_report = _write_havid_fixture(
        output_path=out_dir / "havid_model_input.jsonl",
        config_path=args.havid_config,
        sessions=args.havid_sessions,
        frame_rate=args.frame_rate,
        seed=args.seed,
    )
    report = {
        "chico_path": str(chico_target),
        "havid": havid_report,
    }
    (out_dir / "input_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
