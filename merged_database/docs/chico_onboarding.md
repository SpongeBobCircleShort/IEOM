# CHICO-first real dataset onboarding

This flow supports either mounted CHICO raw JSONL or the included realistic fixture slice.

## Mapping pack
- Config: `merged_database/configs/chico_mapping_rules.yaml`
- Explicit field variants, units, and assumptions are defined there.

## End-to-end commands

```bash
PYTHONPATH=src python -m hesitation.database.cli normalize-chico \
  --raw tests/fixtures/chico/raw/chico_realistic_sample.jsonl \
  --mapping merged_database/configs/chico_mapping_rules.yaml \
  --output /tmp/chico_normalized.jsonl \
  --report /tmp/chico_mapping_report.json

PYTHONPATH=src python -m hesitation.database.cli derive-labels \
  --input /tmp/chico_normalized.jsonl \
  --output /tmp/chico_labeled.jsonl \
  --audit /tmp/chico_label_audit.json \
  --horizon-frames 8

PYTHONPATH=src python -m hesitation.database.cli run-qc \
  --input /tmp/chico_labeled.jsonl \
  --output /tmp/chico_qc.json \
  --dataset-name chico

PYTHONPATH=src python -m hesitation.database.cli build-splits \
  --input /tmp/chico_labeled.jsonl \
  --output /tmp/chico_splits.json

PYTHONPATH=src python -m hesitation.database.cli export-model-input \
  --input /tmp/chico_labeled.jsonl \
  --output /tmp/chico_model_input.jsonl

PYTHONPATH=src python -m hesitation.database.cli run-benchmark \
  --input /tmp/chico_model_input.jsonl \
  --output-dir /tmp/chico_benchmark
```

## Notes
- If CHICO raw data is mounted, point `--raw` to that path.
- Unsupported raw fields are surfaced in mapping reports.
- Deep benchmark runs through the existing deep pipeline (torch path if available, fallback otherwise).


## Sample fixture artifacts in repo
- `merged_database/reports/chico_mapping_report_fixture.json`
- `merged_database/reports/chico_label_audit_fixture.json`
- `merged_database/reports/chico_qc_fixture.json`
- `merged_database/reports/chico_splits_fixture.json`
- `merged_database/reports/chico_benchmark_summary_fixture.json`
- `merged_database/sample_outputs/chico_model_input_fixture.jsonl`
