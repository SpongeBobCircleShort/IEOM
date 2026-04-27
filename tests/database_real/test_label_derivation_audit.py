from hesitation.database.derivation import derive_hesitation_labels
from hesitation.database.label_audit import audit_labels
from hesitation.database.io_utils import load_canonical
from hesitation.database.pipeline import normalize_chico


def test_label_derivation_and_audit(tmp_path) -> None:
    norm = tmp_path / "norm.jsonl"
    report = tmp_path / "map.json"
    normalize_chico(
        raw_path="tests/fixtures/chico/raw/chico_realistic_sample.jsonl",
        mapping_config="merged_database/configs/chico_mapping_rules.yaml",
        output_path=str(norm),
        report_path=str(report),
    )
    records = load_canonical(str(norm))
    records = derive_hesitation_labels(records, horizon_frames=8)
    audit = audit_labels(records)
    assert "normal_progress" in audit.label_counts
    assert len(audit.trigger_counts) > 0
