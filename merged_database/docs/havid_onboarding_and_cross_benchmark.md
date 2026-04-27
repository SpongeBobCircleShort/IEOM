# HA-ViD onboarding + CHICO↔HA-ViD cross-dataset benchmark

## HA-ViD onboarding commands

```bash
PYTHONPATH=src python -m hesitation.database.cli normalize-havid \
  --raw tests/fixtures/havid/raw/havid_realistic_sample.jsonl \
  --mapping merged_database/configs/havid_mapping_rules.yaml \
  --output /tmp/havid_normalized.jsonl \
  --report /tmp/havid_mapping_report.json

PYTHONPATH=src python -m hesitation.database.cli derive-labels \
  --input /tmp/havid_normalized.jsonl \
  --output /tmp/havid_labeled.jsonl \
  --audit /tmp/havid_label_audit.json \
  --horizon-frames 8

PYTHONPATH=src python -m hesitation.database.cli run-qc \
  --input /tmp/havid_labeled.jsonl \
  --output /tmp/havid_qc.json \
  --dataset-name havid

PYTHONPATH=src python -m hesitation.database.cli build-splits \
  --input /tmp/havid_labeled.jsonl \
  --output /tmp/havid_splits.json \
  --source-dataset havid

PYTHONPATH=src python -m hesitation.database.cli export-model-input \
  --input /tmp/havid_labeled.jsonl \
  --output /tmp/havid_model_input.jsonl
```

## CHICO + HA-ViD cross-dataset benchmark

```bash
PYTHONPATH=src python -m hesitation.database.cli run-cross-benchmark \
  --chico-input /tmp/chico_model_input.jsonl \
  --havid-input /tmp/havid_model_input.jsonl \
  --output-dir /tmp/chico_havid_cross_benchmark

PYTHONPATH=src python -m hesitation.database.cli harmonization-report \
  --chico-labeled /tmp/chico_labeled.jsonl \
  --havid-labeled /tmp/havid_labeled.jsonl \
  --output-json /tmp/chico_havid_harmonization.json \
  --output-md /tmp/chico_havid_harmonization.md
```

## Notes
- Replace fixture paths with mounted raw dataset paths when available.
- Reports include mapping coverage, missingness, label/trigger distributions, and transfer-gap flags.
