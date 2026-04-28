from hesitation.database.derivation import derive_hesitation_labels
from hesitation.database.export import to_model_rows
from hesitation.database.io_utils import load_canonical
from hesitation.database.pipeline import normalize_havid


def test_havid_label_derivation_and_export(tmp_path) -> None:
    norm = tmp_path / "havid_norm.jsonl"
    report = tmp_path / "havid_map.json"
    normalize_havid(
        raw_path="tests/fixtures/havid/raw/havid_realistic_sample.jsonl",
        mapping_config="merged_database/configs/havid_mapping_rules.yaml",
        output_path=str(norm),
        report_path=str(report),
    )

    records = load_canonical(str(norm))
    derived = derive_hesitation_labels(records, horizon_frames=8)
    assert any(r.hesitation_state is not None for r in derived)
    rows = to_model_rows(derived)
    assert "frame_idx" in rows[0] and "latent_state" in rows[0]
