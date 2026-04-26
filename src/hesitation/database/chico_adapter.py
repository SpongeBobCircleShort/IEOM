from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from hesitation.database.mapping import CHICOMappingPack, read_first_available
from hesitation.database.schemas import CanonicalRecord, MappingReport
from hesitation.io.loaders import load_jsonl_records


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(float(value.strip()))
    return None


class CHICOAdapter:
    """CHICO-first adapter with explicit field-variant handling and assumptions."""

    def __init__(self, mapping_pack: CHICOMappingPack) -> None:
        self.mapping_pack = mapping_pack

    def normalize(self, raw_path: str | Path) -> tuple[list[CanonicalRecord], MappingReport]:
        path = Path(raw_path)
        files: list[Path] = [path] if path.is_file() else sorted(path.glob("*.jsonl"))

        out: list[CanonicalRecord] = []
        unsupported_counter: Counter[str] = Counter()
        dropped = 0

        for file in files:
            for row in load_jsonl_records(file):
                record = self._normalize_row(row, str(file), unsupported_counter)
                if record is None:
                    dropped += 1
                    continue
                out.append(record)

        report = MappingReport(
            dataset_name=self.mapping_pack.dataset_name,
            records_total=len(out) + dropped,
            mapped_records=len(out),
            dropped_records=dropped,
            unsupported_fields=dict(unsupported_counter),
            assumptions=[
                "timestamp converted to milliseconds when source provides seconds",
                "action labels normalized through action_map",
                "missing hand_right/torso fields preserved as null",
            ],
        )
        return out, report

    def _normalize_row(
        self,
        row: dict[str, Any],
        source_file: str,
        unsupported_counter: Counter[str],
    ) -> CanonicalRecord | None:
        def get(field: str) -> Any:
            rule = self.mapping_pack.fields.get(field)
            if rule is None:
                return None
            return read_first_available(row, rule.source)

        session_id = get("session_id")
        frame_idx = _as_int(get("frame_index"))
        if session_id is None or frame_idx is None:
            return None

        ts = get("timestamp")
        ts_ms = _as_int(ts)
        if ts_ms is not None and ts_ms < 100000:
            # assume seconds for small values
            ts_ms *= 1000

        hand_left = [_as_float(get("left_x")) or 0.0, _as_float(get("left_y")) or 0.0]
        hand_right_x = _as_float(get("right_x"))
        hand_right_y = _as_float(get("right_y"))
        hand_right = [hand_right_x, hand_right_y] if hand_right_x is not None and hand_right_y is not None else None

        action_raw = get("action_label")
        action_raw_s = str(action_raw) if action_raw is not None else None
        action_norm = self.mapping_pack.action_map.get(action_raw_s or "", action_raw_s)

        known_inputs: set[str] = set()
        for rule in self.mapping_pack.fields.values():
            known_inputs.update(rule.source)
        unknown = sorted([k for k in row.keys() if k not in known_inputs])
        for name in unknown:
            unsupported_counter[name] += 1

        pose_conf = _as_float(get("pose_confidence"))

        return CanonicalRecord(
            dataset_name=self.mapping_pack.dataset_name,
            dataset_version=self.mapping_pack.version,
            source_file=source_file,
            source_record_id=str(get("record_id") or f"{session_id}:{frame_idx}"),
            session_id=str(session_id),
            sequence_id=str(get("sequence_id")) if get("sequence_id") is not None else None,
            subject_id=str(get("subject_id")) if get("subject_id") is not None else None,
            frame_index=frame_idx,
            timestamp_ms=ts_ms,
            original_annotation_reference=str(get("annotation_reference")) if get("annotation_reference") is not None else None,
            mapping_rule_id="chico_mapping_v1",
            derived_label_flag=False,
            confidence_in_mapping=pose_conf,
            keypoints_2d=None,
            keypoints_3d=None,
            hand_left=hand_left,
            hand_right=hand_right,
            torso_center=None,
            pose_confidence=pose_conf,
            keypoint_visibility=None,
            robot_present=bool(get("robot_present")) if get("robot_present") is not None else None,
            robot_state=str(get("robot_state")) if get("robot_state") is not None else None,
            robot_pose=None,
            human_robot_distance=_as_float(get("human_robot_distance")),
            shared_workspace_flag=bool(get("shared_workspace_flag")) if get("shared_workspace_flag") is not None else None,
            workspace_region=str(get("workspace_region")) if get("workspace_region") is not None else None,
            overlap_event_native=bool(get("overlap_event_native")) if get("overlap_event_native") is not None else None,
            collision_event_native=bool(get("collision_event_native")) if get("collision_event_native") is not None else None,
            task_name=str(get("task_name")) if get("task_name") is not None else None,
            task_step=str(get("task_step")) if get("task_step") is not None else None,
            action_label_raw=action_raw_s,
            canonical_action_label=action_norm,
            manipulated_object=str(get("manipulated_object")) if get("manipulated_object") is not None else None,
            target_object=str(get("target_object")) if get("target_object") is not None else None,
            tool=str(get("tool")) if get("tool") is not None else None,
            completion_state=str(get("completion_state")) if get("completion_state") is not None else None,
            rework_native_flag=bool(get("rework_native_flag")) if get("rework_native_flag") is not None else None,
            unsupported_source_fields=unknown,
        )
