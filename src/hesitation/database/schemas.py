from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Provenance:
    dataset_name: str
    dataset_version: str
    source_file: str
    source_record_id: str
    original_annotation_reference: str | None = None
    mapping_rule_id: str | None = None
    derived_label_flag: bool = False
    confidence_in_mapping: float | None = None
    unsupported_fields: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CanonicalRecord:
    # identity / provenance
    dataset_name: str
    dataset_version: str
    source_file: str
    source_record_id: str
    session_id: str
    sequence_id: str | None
    subject_id: str | None
    frame_index: int
    timestamp_ms: int | None
    original_annotation_reference: str | None
    mapping_rule_id: str | None
    derived_label_flag: bool
    confidence_in_mapping: float | None

    # pose / motion
    keypoints_2d: list[list[float]] | None = None
    keypoints_3d: list[list[float]] | None = None
    hand_left: list[float] | None = None
    hand_right: list[float] | None = None
    torso_center: list[float] | None = None
    pose_confidence: float | None = None
    keypoint_visibility: list[float] | None = None

    # workspace / robot context
    robot_present: bool | None = None
    robot_state: str | None = None
    robot_pose: list[float] | None = None
    human_robot_distance: float | None = None
    shared_workspace_flag: bool | None = None
    workspace_region: str | None = None
    overlap_event_native: bool | None = None
    collision_event_native: bool | None = None

    # task context
    task_name: str | None = None
    task_step: str | None = None
    action_label_raw: str | None = None
    canonical_action_label: str | None = None
    manipulated_object: str | None = None
    target_object: str | None = None
    tool: str | None = None
    completion_state: str | None = None
    rework_native_flag: bool | None = None

    # derived behavior
    pause_duration: float | None = None
    stop_count: int | None = None
    micro_stop_count: int | None = None
    motion_reversal_count: int | None = None
    target_distance: float | None = None
    step_duration_deviation: float | None = None
    unfinished_attempt_count: int | None = None
    object_reposition_count: int | None = None
    jerk_proxy: float | None = None
    overlap_risk_proxy: float | None = None

    # hesitation labels
    hesitation_binary: bool | None = None
    hesitation_state: str | None = None
    future_hesitation_within_horizon: bool | None = None
    future_correction_within_horizon: bool | None = None
    correction_rework: bool | None = None
    overlap_risk: bool | None = None

    # audit metadata
    label_rule_triggers: list[str] = field(default_factory=list)
    label_confidence: float | None = None
    unsupported_source_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MappingReport:
    dataset_name: str
    records_total: int
    mapped_records: int
    dropped_records: int
    unsupported_fields: dict[str, int]
    assumptions: list[str]


@dataclass(slots=True)
class QCSummary:
    dataset_name: str
    missing_timestamps: int
    duplicate_frames: int
    non_monotonic_frames: int
    impossible_coordinates: int
    low_confidence_pose: int
    label_conflicts: int
    missingness: dict[str, float]


@dataclass(slots=True)
class LabelAuditSummary:
    label_counts: dict[str, int]
    confidence_buckets: dict[str, int]
    trigger_counts: dict[str, int]
    suspicious_sessions: list[str]
