from __future__ import annotations

from collections import Counter, defaultdict

from hesitation.database.schemas import CanonicalRecord, LabelAuditSummary


def audit_labels(records: list[CanonicalRecord]) -> LabelAuditSummary:
    label_counts: Counter[str] = Counter()
    confidence_buckets: Counter[str] = Counter()
    trigger_counts: Counter[str] = Counter()
    by_session: dict[str, list[CanonicalRecord]] = defaultdict(list)

    for record in records:
        by_session[record.session_id].append(record)
        label_counts[record.hesitation_state or "unknown"] += 1
        conf = record.label_confidence if record.label_confidence is not None else 0.0
        if conf < 0.5:
            confidence_buckets["low"] += 1
        elif conf < 0.8:
            confidence_buckets["medium"] += 1
        else:
            confidence_buckets["high"] += 1
        for trigger in record.label_rule_triggers:
            trigger_counts[trigger] += 1

    suspicious: list[str] = []
    for session_id, session_rows in by_session.items():
        pause_heavy = sum(1 for row in session_rows if (row.pause_duration or 0) >= 2)
        strong_states = sum(1 for row in session_rows if row.hesitation_state in {"strong_hesitation", "correction_rework"})
        if pause_heavy >= 3 and strong_states == 0:
            suspicious.append(session_id)

    return LabelAuditSummary(
        label_counts=dict(label_counts),
        confidence_buckets=dict(confidence_buckets),
        trigger_counts=dict(trigger_counts),
        suspicious_sessions=sorted(suspicious),
    )
