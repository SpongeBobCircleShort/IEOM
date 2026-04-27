# Hesitation Simulation + Real Dataset Onboarding Prototype

This repository now includes:
- simulation-first hesitation modeling pipelines (rules/classical/deep)
- CHICO-first real dataset onboarding flow with mapping, QC, label audit, split/export, and benchmark

## Real dataset onboarding (CHICO-first)

See: `merged_database/docs/chico_onboarding.md`

Quick run (fixture-based realistic sample):

```bash
PYTHONPATH=src python -m hesitation.database.cli normalize-chico --raw tests/fixtures/chico/raw/chico_realistic_sample.jsonl --mapping merged_database/configs/chico_mapping_rules.yaml --output /tmp/chico_normalized.jsonl --report /tmp/chico_mapping_report.json
PYTHONPATH=src python -m hesitation.database.cli derive-labels --input /tmp/chico_normalized.jsonl --output /tmp/chico_labeled.jsonl --audit /tmp/chico_label_audit.json --horizon-frames 8
PYTHONPATH=src python -m hesitation.database.cli run-qc --input /tmp/chico_labeled.jsonl --output /tmp/chico_qc.json --dataset-name chico
PYTHONPATH=src python -m hesitation.database.cli build-splits --input /tmp/chico_labeled.jsonl --output /tmp/chico_splits.json
PYTHONPATH=src python -m hesitation.database.cli export-model-input --input /tmp/chico_labeled.jsonl --output /tmp/chico_model_input.jsonl
PYTHONPATH=src python -m hesitation.database.cli run-benchmark --input /tmp/chico_model_input.jsonl --output-dir /tmp/chico_benchmark
```

## Development

```bash
pytest -q
```
