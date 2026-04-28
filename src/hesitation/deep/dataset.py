"""Sequence dataset construction for deep temporal models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from hesitation.schemas.labels import HesitationState

FRAME_FEATURE_ORDER = [
    "hand_x",
    "hand_y",
    "hand_speed",
    "hand_accel",
    "distance_to_robot_workspace",
    "progress",
    "confidence",
    "is_dropout",
]


class SequenceWindow(TypedDict):
    session_id: str
    dataset_name: str
    end_frame_idx: int
    sequence: list[list[float]]
    current_state: str
    future_hesitation: int
    future_correction: int


def load_rows(path: str | Path) -> list[dict[str, object]]:
    """Load raw jsonl rows."""
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_sequence_windows(
    rows: list[dict[str, object]],
    window_size: int,
    horizon_frames: int,
) -> list[SequenceWindow]:
    """Build per-window sequence examples with current-state and future-risk targets."""
    sessions: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        sid = str(row["session_id"])
        sessions.setdefault(sid, []).append(row)

    windows: list[SequenceWindow] = []
    for sid, session_rows in sessions.items():
        session_rows.sort(key=lambda r: int(r["frame_idx"]))
        states = [str(r.get("latent_state", HesitationState.NORMAL_PROGRESS.value)) for r in session_rows]
        for end in range(window_size, len(session_rows) - horizon_frames):
            seq_rows = session_rows[end - window_size : end]
            sequence = [
                [
                    float(r["hand_x"]),
                    float(r["hand_y"]),
                    float(r["hand_speed"]),
                    float(r["hand_accel"]),
                    float(r["distance_to_robot_workspace"]),
                    float(r["progress"]),
                    float(r["confidence"]),
                    1.0 if bool(r.get("is_dropout", False)) else 0.0,
                ]
                for r in seq_rows
            ]
            future_slice = states[end : end + horizon_frames]
            future_hesitation = int(
                any(s in {HesitationState.MILD_HESITATION.value, HesitationState.STRONG_HESITATION.value} for s in future_slice)
            )
            future_correction = int(any(s == HesitationState.CORRECTION_REWORK.value for s in future_slice))

            windows.append(
                SequenceWindow(
                    session_id=sid,
                    dataset_name=str(session_rows[end - 1].get("dataset_name", "unknown")),
                    end_frame_idx=int(session_rows[end - 1]["frame_idx"]),
                    sequence=sequence,
                    current_state=states[end - 1],
                    future_hesitation=future_hesitation,
                    future_correction=future_correction,
                )
            )
    return windows


def split_train_val(
    windows: list[SequenceWindow],
    val_fraction: float = 0.2,
) -> tuple[list[SequenceWindow], list[SequenceWindow]]:
    """Session-based split for reproducible comparison with Phase 2."""
    session_ids = sorted({w["session_id"] for w in windows})
    val_count = max(1, int(len(session_ids) * val_fraction))
    val_sessions = set(session_ids[-val_count:])
    train = [w for w in windows if w["session_id"] not in val_sessions]
    val = [w for w in windows if w["session_id"] in val_sessions]
    return train, val
