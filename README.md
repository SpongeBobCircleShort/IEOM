# Hesitation Simulation Prototype (Phase 1 + Phase 2)

This repository contains a simulation-first prototype for human hesitation identification, near-future risk prediction, and advisory policy recommendation in collaborative robot workspaces.

## Scope
- ✅ Synthetic data generator
- ✅ Label/state definitions and logic
- ✅ Feature extraction
- ✅ Rules-based baseline
- ✅ Classical ML baselines (Phase 2)
- ✅ Future-risk prediction (Phase 2)
- ✅ Advisory policy recommendation layer (Phase 2)
- ✅ Tests
- ❌ No deep learning model yet
- ❌ No API/UI yet

## Quickstart
```bash
pytest

python scripts/generate_synthetic_dataset.py \
  --output data/synth.jsonl \
  --n-sessions 12

python scripts/phase2_cli.py train-classical \
  --input data/synth.jsonl \
  --output-dir artifacts/phase2 \
  --window-size 20 \
  --pause-speed-threshold 0.03 \
  --horizon-frames 20

python scripts/phase2_cli.py evaluate-classical \
  --input data/synth.jsonl \
  --model-path artifacts/phase2/classical_model.json

python scripts/phase2_cli.py infer-sequence \
  --input data/synth.jsonl \
  --model-path artifacts/phase2/classical_model.json \
  --output data/state_predictions.jsonl

python scripts/phase2_cli.py predict-risk \
  --input data/synth.jsonl \
  --model-path artifacts/phase2/classical_model.json \
  --output data/future_risk_predictions.jsonl

python scripts/phase2_cli.py recommend-policy \
  --current-state mild_hesitation \
  --current-hesitation-prob 0.62 \
  --future-hesitation-prob 0.71 \
  --future-correction-prob 0.44 \
  --workspace-distance 0.21
```

## Notes / assumptions
- Classical models are implemented as lightweight logistic regression (one-vs-rest + binary heads) trained on Phase 1 window features.
- Config files are `.yaml` names with JSON-compatible content.
- Policy recommendation is advisory only and **not** a safety-certified controller.


## Phase 3 (deep temporal baseline)
```bash
python scripts/phase2_cli.py train-deep \
  --input data/synth.jsonl \
  --output-dir artifacts/phase3_deep \
  --window-size 20 \
  --horizon-frames 20 \
  --epochs 20 \
  --hidden-dim 64 \
  --learning-rate 0.001

python scripts/phase2_cli.py evaluate-deep \
  --input data/synth.jsonl \
  --model-path artifacts/phase3_deep/deep_model.json

python scripts/phase2_cli.py infer-sequence-deep \
  --input data/synth.jsonl \
  --model-path artifacts/phase3_deep/deep_model.json \
  --output data/deep_predictions.jsonl

python scripts/phase2_cli.py compare-models \
  --input data/synth.jsonl \
  --classical-model-path artifacts/phase2/classical_model.json \
  --deep-model-path artifacts/phase3_deep/deep_model.json \
  --output-dir artifacts/phase3_compare
```

### Phase 3 assumptions
- Default deep architecture is a GRU multi-head model when PyTorch is available.
- In restricted environments without PyTorch, the CLI falls back to a sequence-aware logistic multi-head backend for runnable smoke validation.
- Comparison reports are emitted as JSON + markdown table under the configured output directory.
