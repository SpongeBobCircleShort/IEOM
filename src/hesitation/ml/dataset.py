from __future__ import annotations

import json
from pathlib import Path

from hesitation.features.pipeline import window_to_features
from hesitation.schemas.events import FrameObservation
from hesitation.schemas.labels import HesitationState

FEATURE_ORDER = [
    "mean_speed",
    "speed_variance",
    "pause_ratio",
    "direction_changes",
    "progress_delta",
    "backtrack_ratio",
    "mean_workspace_distance",
]


class DatasetRow(dict):
    pass


def load_rows(path: str | Path) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(DatasetRow(json.loads(line)))
    return rows


def build_windows(
    rows: list[DatasetRow],
    window_size: int,
    pause_speed_threshold: float,
    horizon_frames: int,
) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    sessions: dict[str, list[DatasetRow]] = {}
    for row in rows:
        sessions.setdefault(str(row["session_id"]), []).append(row)

    for session_id, session_rows in sessions.items():
        session_rows.sort(key=lambda r: int(r["frame_idx"]))
        observations = [FrameObservation.model_validate(r) for r in session_rows]
        states = [str(r.get("latent_state", HesitationState.NORMAL_PROGRESS.value)) for r in session_rows]
        for end in range(window_size, len(observations) - horizon_frames):
            window_frames = observations[end - window_size : end]
            feature = window_to_features(window_frames, pause_speed_threshold=pause_speed_threshold)
            current_state = states[end - 1]
            future_slice = states[end : end + horizon_frames]
            future_hesitation = int(
                any(s in {HesitationState.MILD_HESITATION.value, HesitationState.STRONG_HESITATION.value} for s in future_slice)
            )
            future_correction = int(any(s == HesitationState.CORRECTION_REWORK.value for s in future_slice))

            windows.append(
                {
                    "session_id": session_id,
                    "dataset_name": str(session_rows[end - 1].get("dataset_name", "unknown")),
                    "end_frame_idx": observations[end - 1].frame_idx,
                    "features": [float(getattr(feature, name)) for name in FEATURE_ORDER],
                    "current_state": current_state,
                    "future_hesitation": future_hesitation,
                    "future_correction": future_correction,
                }
            )
    return windows


def split_train_val(
    windows: list[dict[str, object]],
    val_fraction: float = 0.2,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    session_ids = sorted({str(w["session_id"]) for w in windows})
    val_count = max(1, int(len(session_ids) * val_fraction))
    val_sessions = set(session_ids[-val_count:])
    train = [w for w in windows if str(w["session_id"]) not in val_sessions]
    val = [w for w in windows if str(w["session_id"]) in val_sessions]
    return train, val
