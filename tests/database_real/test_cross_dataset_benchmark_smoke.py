from pathlib import Path

from hesitation.database.pipeline import (
    derive_labels_and_audit,
    export_for_models,
    normalize_chico,
    normalize_havid,
    run_cross_benchmark,
    run_harmonization,
)


def test_cross_dataset_benchmark_and_harmonization_smoke(tmp_path) -> None:
    chico_norm = tmp_path / "chico_norm.jsonl"
    chico_map = tmp_path / "chico_map.json"
    normalize_chico(
        raw_path="tests/fixtures/chico/raw/chico_realistic_sample.jsonl",
        mapping_config="merged_database/configs/chico_mapping_rules.yaml",
        output_path=str(chico_norm),
        report_path=str(chico_map),
    )

    havid_norm = tmp_path / "havid_norm.jsonl"
    havid_map = tmp_path / "havid_map.json"
    normalize_havid(
        raw_path="tests/fixtures/havid/raw/havid_realistic_sample.jsonl",
        mapping_config="merged_database/configs/havid_mapping_rules.yaml",
        output_path=str(havid_norm),
        report_path=str(havid_map),
    )

    chico_labeled = tmp_path / "chico_labeled.jsonl"
    chico_audit = tmp_path / "chico_audit.json"
    derive_labels_and_audit(str(chico_norm), str(chico_labeled), str(chico_audit), 8)

    havid_labeled = tmp_path / "havid_labeled.jsonl"
    havid_audit = tmp_path / "havid_audit.json"
    derive_labels_and_audit(str(havid_norm), str(havid_labeled), str(havid_audit), 8)

    chico_model = tmp_path / "chico_model.jsonl"
    havid_model = tmp_path / "havid_model.jsonl"
    export_for_models(str(chico_labeled), str(chico_model))
    export_for_models(str(havid_labeled), str(havid_model))

    out = tmp_path / "cross"
    run_cross_benchmark(str(chico_model), str(havid_model), str(out))
    assert Path(out / "cross_dataset_benchmark_summary.json").exists()

    h_json = tmp_path / "harmonization.json"
    h_md = tmp_path / "harmonization.md"
    run_harmonization(str(chico_labeled), str(havid_labeled), str(h_json), str(h_md))
    assert h_json.exists() and h_md.exists()
