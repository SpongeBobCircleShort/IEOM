from __future__ import annotations

from hesitation.database.schemas import CanonicalRecord
from hesitation.schemas.events import FrameObservation


def to_model_rows(records: list[CanonicalRecord]) -> list[dict[str, object]]:
    """Convert canonical records to existing model input row format."""
    rows: list[dict[str, object]] = []
    by_session: dict[str, list[CanonicalRecord]] = {}
    for record in records:
        by_session.setdefault(record.session_id, []).append(record)

    for session_rows in by_session.values():
        session_rows.sort(key=lambda r: r.frame_index)
        prev_x = None
        prev_y = None
        prev_speed = 0.0
        first_idx = session_rows[0].frame_index
        last_idx = session_rows[-1].frame_index
        for record in session_rows:
            x = (record.hand_left or [0.0, 0.0])[0]
            y = (record.hand_left or [0.0, 0.0])[1]
            if prev_x is None or prev_y is None:
                speed = 0.0
            else:
                speed = ((x - prev_x) ** 2 + (y - prev_y) ** 2) ** 0.5
            accel = speed - prev_speed
            prev_speed = speed
            prev_x, prev_y = x, y

            progress = 0.0 if last_idx == first_idx else (record.frame_index - first_idx) / (last_idx - first_idx)
            workspace_dist = record.human_robot_distance if record.human_robot_distance is not None else 1.0
            confidence = record.pose_confidence if record.pose_confidence is not None else 0.5
            task_step_id = 0
            if record.task_step and record.task_step.replace("_", "").isdigit():
                task_step_id = int(record.task_step.replace("_", ""))

            frame = FrameObservation(
                session_id=record.session_id,
                frame_idx=record.frame_index,
                timestamp_ms=record.timestamp_ms or 0,
                task_step_id=task_step_id,
                hand_x=float(x),
                hand_y=float(y),
                hand_speed=max(float(speed), 0.0),
                hand_accel=float(accel),
                distance_to_robot_workspace=max(float(workspace_dist), 0.0),
                progress=max(0.0, min(1.0, float(progress))),
                confidence=max(0.0, min(1.0, float(confidence))),
                is_dropout=False,
            )
            payload = frame.model_dump()
            payload["latent_state"] = record.hesitation_state or "normal_progress"
            payload["dataset_name"] = record.dataset_name
            rows.append(payload)
    return rows
