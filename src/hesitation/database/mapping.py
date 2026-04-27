from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hesitation.io.config import load_config


@dataclass(slots=True)
class FieldRule:
    source: list[str]
    required: bool
    unit: str | None
    notes: str


@dataclass(slots=True)
class DatasetMappingPack:
    dataset_name: str
    version: str
    fps_default: int
    fields: dict[str, FieldRule]
    action_map: dict[str, str]


def load_dataset_mapping_pack(path: str | Path) -> DatasetMappingPack:
    payload = load_config(path)
    fields = {
        key: FieldRule(
            source=list(value.get("source", [])),
            required=bool(value.get("required", False)),
            unit=value.get("unit"),
            notes=str(value.get("notes", "")),
        )
        for key, value in payload.get("fields", {}).items()
    }
    return DatasetMappingPack(
        dataset_name=str(payload.get("dataset_name", "dataset")),
        version=str(payload.get("version", "unknown")),
        fps_default=int(payload.get("fps_default", 30)),
        fields=fields,
        action_map={str(k): str(v) for k, v in payload.get("action_map", {}).items()},
    )


def load_chico_mapping_pack(path: str | Path) -> DatasetMappingPack:
    return load_dataset_mapping_pack(path)


def load_havid_mapping_pack(path: str | Path) -> DatasetMappingPack:
    return load_dataset_mapping_pack(path)


def read_first_available(payload: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None
