from __future__ import annotations

import json
from pathlib import Path

from hesitation.schemas.events import FrameObservation
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


def write_synth_dataset(path: Path, n_sessions: int = 6, seed_base: int = 600) -> None:
    """Write a small synthetic dataset to JSONL for Phase 4 tests."""
    scenario = ScenarioConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for index in range(n_sessions):
            trajectory, latent = generate_session(f"s{index}", scenario, frame_rate_hz=10, seed=seed_base + index)
            for frame, state in zip(trajectory.frames, latent):
                row = frame.model_dump()
                row["latent_state"] = state.value
                out.write(json.dumps(row) + "\n")


def sample_session_frames(n_frames: int = 24, seed: int = 700) -> list[FrameObservation]:
    """Generate one synthetic session and return the first `n_frames` observations."""
    scenario = ScenarioConfig()
    trajectory, _ = generate_session("phase4_session", scenario, frame_rate_hz=10, seed=seed)
    return trajectory.frames[:n_frames]


def frames_to_payload(frames: list[FrameObservation]) -> list[dict[str, object]]:
    """Serialize observations for API request bodies."""
    return [frame.model_dump() for frame in frames]
