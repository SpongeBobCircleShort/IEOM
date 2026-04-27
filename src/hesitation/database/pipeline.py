from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hesitation.database.benchmark import run_first_benchmark
from hesitation.database.chico_adapter import CHICOAdapter
from hesitation.database.derivation import derive_hesitation_labels
from hesitation.database.export import to_model_rows
from hesitation.database.label_audit import audit_labels
from hesitation.database.mapping import load_chico_mapping_pack
from hesitation.database.qc import compute_qc
from hesitation.database.schemas import CanonicalRecord
from hesitation.io.writers import write_jsonl


def normalize_chico(
    raw_path: str,
    mapping_config: str,
    output_path: str,
    report_path: str
) -> tuple[str, str]:
    pack = load_chico_mapping_pack(mapping_config)
    adapter = CHICOAdapter(pack)
    records, report = adapter.normalize(raw_path)

    write_jsonl(output_path, [record.to_dict() for record in records])
    Path(report_path).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return output_path, report_path


def load_canonical(path: str) -> list[CanonicalRecord]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(CanonicalRecord(**json.loads(line)))
    return rows


def derive_labels_and_audit(
    normalized_path: str,
    labeled_path: str,
    audit_path: str,
    horizon_frames: int = 15,
) -> tuple[str, str]:
    records = load_canonical(normalized_path)
    derived = derive_hesitation_labels(records, horizon_frames=horizon_frames)
    write_jsonl(labeled_path, [row.to_dict() for row in derived])

    audit = audit_labels(derived)
    Path(audit_path).write_text(json.dumps(asdict(audit), indent=2), encoding="utf-8")
    return labeled_path, audit_path


def run_qc_report(labeled_path: str, output_path: str, dataset_name: str) -> str:
    records = load_canonical(labeled_path)
    qc = compute_qc(records, dataset_name=dataset_name)
    Path(output_path).write_text(json.dumps(asdict(qc), indent=2), encoding="utf-8")
    return output_path


def build_splits(labeled_path: str, output_path: str) -> str:
    records = load_canonical(labeled_path)
    sessions = sorted({record.session_id for record in records})
    n = len(sessions)
    train = sessions[: max(1, int(0.6 * n))]
    val = sessions[max(1, int(0.6 * n)) : max(2, int(0.8 * n))]
    test = sessions[max(2, int(0.8 * n)) :]
    payload: dict[str, Any] = {
        "within_dataset": {"train": train, "val": val, "test": test},
        "cross_dataset": {"source_dataset": "chico", "target_dataset": "tbd"},
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def export_for_models(labeled_path: str, output_path: str) -> str:
    records = load_canonical(labeled_path)
    rows = to_model_rows(records)
    write_jsonl(output_path, rows)
    return output_path


def run_benchmark_export(model_input_path: str, output_dir: str) -> str:
    summary = run_first_benchmark(model_input_path=model_input_path, output_dir=output_dir)
    report_path = Path(output_dir) / "benchmark_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(report_path)
