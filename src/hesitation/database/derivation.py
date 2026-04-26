from __future__ import annotations

from collections import defaultdict

from hesitation.database.schemas import CanonicalRecord


def derive_hesitation_labels(records: list[CanonicalRecord], horizon_frames: int = 15) -> list[CanonicalRecord]:
    """Derive transparent hesitation labels from motion proxies and context fields."""
    by_session: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for record in records:
        by_session[record.session_id].append(record)

    for session_rows in by_session.values():
        session_rows.sort(key=lambda r: r.frame_index)
        prev_speed = 0.0
        for idx, record in enumerate(session_rows):
            speed = _estimate_speed(record, session_rows[idx - 1] if idx > 0 else None)
            if speed < 0.01:
                record.pause_duration = (record.pause_duration or 0.0) + 1.0
                record.micro_stop_count = (record.micro_stop_count or 0) + 1
            if idx > 0:
                prev = session_rows[idx - 1]
                prev_dx = (prev.hand_left or [0.0, 0.0])[0] - (session_rows[idx - 2].hand_left or [0.0, 0.0])[0] if idx > 1 else 0.0
                dx = (record.hand_left or [0.0, 0.0])[0] - (prev.hand_left or [0.0, 0.0])[0]
                if prev_dx * dx < 0:
                    record.motion_reversal_count = (record.motion_reversal_count or 0) + 1
                record.jerk_proxy = abs(speed - prev_speed)
            prev_speed = speed

            state, triggers, confidence = _derive_state(record)
            record.hesitation_state = state
            record.hesitation_binary = state in {"mild_hesitation", "strong_hesitation", "correction_rework", "overlap_risk"}
            record.correction_rework = state == "correction_rework"
            record.overlap_risk = state == "overlap_risk"
            record.label_rule_triggers = triggers
            record.label_confidence = confidence
            record.derived_label_flag = True

        states = [r.hesitation_state or "normal_progress" for r in session_rows]
        for idx, record in enumerate(session_rows):
            future = states[idx + 1 : idx + 1 + horizon_frames]
            record.future_hesitation_within_horizon = any(s in {"mild_hesitation", "strong_hesitation"} for s in future)
            record.future_correction_within_horizon = any(s == "correction_rework" for s in future)
    return records


def _estimate_speed(record: CanonicalRecord, previous: CanonicalRecord | None) -> float:
    if previous is None or record.hand_left is None or previous.hand_left is None:
        return 0.0
    dx = record.hand_left[0] - previous.hand_left[0]
    dy = record.hand_left[1] - previous.hand_left[1]
    return (dx * dx + dy * dy) ** 0.5


def _derive_state(record: CanonicalRecord) -> tuple[str, list[str], float]:
    triggers: list[str] = []
    if record.rework_native_flag:
        triggers.append("native_rework")
        return "correction_rework", triggers, 0.95
    if record.shared_workspace_flag and (record.pause_duration or 0) >= 2.0:
        triggers.append("shared_zone_pause")
        return "overlap_risk", triggers, 0.8
    if (record.motion_reversal_count or 0) >= 2 or (record.pause_duration or 0.0) >= 3.0:
        triggers.append("reversal_or_long_pause")
        return "strong_hesitation", triggers, 0.75
    if (record.micro_stop_count or 0) >= 2 or (record.pause_duration or 0.0) >= 1.0:
        triggers.append("micro_stop_or_pause")
        return "mild_hesitation", triggers, 0.6
    if (record.completion_state or "") == "ready":
        triggers.append("completion_ready")
        return "ready_for_robot_action", triggers, 0.9
    triggers.append("default_normal")
    return "normal_progress", triggers, 0.95
