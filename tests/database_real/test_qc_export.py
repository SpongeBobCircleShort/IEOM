from hesitation.database.export import to_model_rows
from hesitation.database.io_utils import load_canonical
from hesitation.database.pipeline import normalize_chico
from hesitation.database.qc import compute_qc


def test_qc_and_export_compatibility(tmp_path) -> None:
    norm = tmp_path / "norm.jsonl"
    report = tmp_path / "map.json"
    normalize_chico(
        raw_path="tests/fixtures/chico/raw/chico_realistic_sample.jsonl",
        mapping_config="merged_database/configs/chico_mapping_rules.yaml",
        output_path=str(norm),
        report_path=str(report),
    )
    records = load_canonical(str(norm))
    qc = compute_qc(records, dataset_name="chico")
    assert qc.duplicate_frames == 0

    rows = to_model_rows(records)
    first = rows[0]
    assert "frame_idx" in first and "hand_speed" in first and "latent_state" in first
