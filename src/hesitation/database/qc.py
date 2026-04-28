from __future__ import annotations

from collections import defaultdict

from hesitation.database.schemas import CanonicalRecord, QCSummary


def compute_qc(records: list[CanonicalRecord], dataset_name: str) -> QCSummary:
    by_session: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for record in records:
        by_session[record.session_id].append(record)

    missing_ts = sum(1 for record in records if record.timestamp_ms is None)
    duplicate = 0
    non_monotonic = 0
    impossible = 0
    low_conf = 0
    label_conflicts = 0

    for session_rows in by_session.values():
        seen: set[int] = set()
        previous = -1
        for record in sorted(session_rows, key=lambda r: r.frame_index):
            if record.frame_index in seen:
                duplicate += 1
            seen.add(record.frame_index)
            if record.frame_index < previous:
                non_monotonic += 1
            previous = record.frame_index

            if record.hand_left and any(abs(v) > 1e4 for v in record.hand_left):
                impossible += 1
            if record.pose_confidence is not None and record.pose_confidence < 0.1:
                low_conf += 1
            if record.hesitation_state == "normal_progress" and bool(record.hesitation_binary):
                label_conflicts += 1

    def missing_ratio(field: str) -> float:
        total = max(1, len(records))
        return sum(1 for record in records if getattr(record, field) is None) / total

    missingness = {
        "timestamp_ms": missing_ratio("timestamp_ms"),
        "hand_left": missing_ratio("hand_left"),
        "pose_confidence": missing_ratio("pose_confidence"),
        "task_step": missing_ratio("task_step"),
        "human_robot_distance": missing_ratio("human_robot_distance"),
    }

    return QCSummary(
        dataset_name=dataset_name,
        missing_timestamps=missing_ts,
        duplicate_frames=duplicate,
        non_monotonic_frames=non_monotonic,
        impossible_coordinates=impossible,
        low_confidence_pose=low_conf,
        label_conflicts=label_conflicts,
        missingness=missingness,
    )
