# Hesitation Simulation + Real Dataset Onboarding Prototype

This repository includes:
- simulation-first hesitation modeling pipelines (rules/classical/deep)
- CHICO and HA-ViD onboarding flows with mapping, QC, label audit, split/export, and benchmark
- CHICO↔HA-ViD cross-dataset benchmark + harmonization reporting

## Real dataset onboarding docs
- CHICO: `merged_database/docs/chico_onboarding.md`
- HA-ViD + cross-dataset: `merged_database/docs/havid_onboarding_and_cross_benchmark.md`

## Quick flow (fixture-backed)

```bash
# CHICO normalize -> labeled -> export
PYTHONPATH=src python -m hesitation.database.cli normalize-chico --raw tests/fixtures/chico/raw/chico_realistic_sample.jsonl --mapping merged_database/configs/chico_mapping_rules.yaml --output /tmp/chico_normalized.jsonl --report /tmp/chico_mapping_report.json
PYTHONPATH=src python -m hesitation.database.cli derive-labels --input /tmp/chico_normalized.jsonl --output /tmp/chico_labeled.jsonl --audit /tmp/chico_label_audit.json --horizon-frames 8
PYTHONPATH=src python -m hesitation.database.cli export-model-input --input /tmp/chico_labeled.jsonl --output /tmp/chico_model_input.jsonl

# HA-ViD normalize -> labeled -> export
PYTHONPATH=src python -m hesitation.database.cli normalize-havid --raw tests/fixtures/havid/raw/havid_realistic_sample.jsonl --mapping merged_database/configs/havid_mapping_rules.yaml --output /tmp/havid_normalized.jsonl --report /tmp/havid_mapping_report.json
PYTHONPATH=src python -m hesitation.database.cli derive-labels --input /tmp/havid_normalized.jsonl --output /tmp/havid_labeled.jsonl --audit /tmp/havid_label_audit.json --horizon-frames 8
PYTHONPATH=src python -m hesitation.database.cli export-model-input --input /tmp/havid_labeled.jsonl --output /tmp/havid_model_input.jsonl

# Cross-dataset benchmark + harmonization report
PYTHONPATH=src python -m hesitation.database.cli run-cross-benchmark --chico-input /tmp/chico_model_input.jsonl --havid-input /tmp/havid_model_input.jsonl --output-dir /tmp/chico_havid_cross_benchmark
PYTHONPATH=src python -m hesitation.database.cli harmonization-report --chico-labeled /tmp/chico_labeled.jsonl --havid-labeled /tmp/havid_labeled.jsonl --output-json /tmp/chico_havid_harmonization.json --output-md /tmp/chico_havid_harmonization.md
```

## Development

```bash
pytest -q
```
