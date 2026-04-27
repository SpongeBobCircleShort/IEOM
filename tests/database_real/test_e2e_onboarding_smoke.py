from pathlib import Path

from hesitation.database.pipeline import (
    build_splits,
    derive_labels_and_audit,
    export_for_models,
    normalize_chico,
    run_benchmark_export,
    run_qc_report,
)


def test_end_to_end_real_onboarding_smoke(tmp_path) -> None:
    norm = tmp_path / "norm.jsonl"
    map_report = tmp_path / "mapping.json"
    normalize_chico(
        raw_path="tests/fixtures/chico/raw/chico_realistic_sample.jsonl",
        mapping_config="merged_database/configs/chico_mapping_rules.yaml",
        output_path=str(norm),
        report_path=str(map_report),
    )

    labeled = tmp_path / "labeled.jsonl"
    audit = tmp_path / "audit.json"
    derive_labels_and_audit(str(norm), str(labeled), str(audit), horizon_frames=8)

    qc = tmp_path / "qc.json"
    run_qc_report(str(labeled), str(qc), "chico")

    splits = tmp_path / "splits.json"
    build_splits(str(labeled), str(splits), "chico")

    model_input = tmp_path / "model_input.jsonl"
    export_for_models(str(labeled), str(model_input))

    bench_dir = tmp_path / "bench"
    run_benchmark_export(str(model_input), str(bench_dir))

    assert Path(bench_dir / "benchmark_summary.json").exists()
