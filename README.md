# Hesitation Simulation Prototype (Phase 1)

This repository contains a simulation-first prototype for human hesitation identification and near-future correction/overlap risk prediction in collaborative robot workspaces.

## Scope
- ✅ Synthetic data generator
- ✅ Label/state definitions and logic
- ✅ Feature extraction
- ✅ Rules-based baseline (no ML model)
- ✅ Tests
- ❌ No deep learning model yet
- ❌ No API/UI yet

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/generate_synthetic_dataset.py --output data/synth.jsonl --n-sessions 10
python scripts/run_baseline.py --input data/synth.jsonl --output data/predictions.jsonl
```
