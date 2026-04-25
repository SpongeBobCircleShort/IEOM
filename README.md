# Hesitation Simulation Prototype (Phase 1 + Phase 2 + Phase 3.5)

This repository contains a simulation-first prototype for human hesitation identification, near-future risk prediction, and advisory policy recommendation in collaborative robot workspaces.

## Scope
- ✅ Synthetic data generator
- ✅ Label/state definitions and logic
- ✅ Feature extraction
- ✅ Rules-based baseline
- ✅ Classical ML baselines (Phase 2)
- ✅ Future-risk prediction (Phase 2)
- ✅ Advisory policy recommendation layer (Phase 2)
- ✅ Deep temporal baseline + multi-head inference (Phase 3)
- ✅ Multi-seed deep experiments, threshold tuning, calibration metrics, and comparison reports (Phase 3.5)
- ✅ Tests
- ❌ No API/UI/video pipeline yet

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
```

## Phase 3.5 commands

### 1) Generate extended scenario mixture
```bash
python scripts/phase2_cli.py generate-scenarios-extended \
  --output data/synth_extended.jsonl \
  --configs configs/simulation/default_scene.yaml,configs/simulation/stress_scene.yaml,configs/simulation/ambiguous_scene.yaml,configs/simulation/domain_gap_scene.yaml,configs/simulation/correction_heavy_scene.yaml \
  --sessions-per-scenario 3 \
  --frame-rate 10 \
  --seed 101
```

### 2) Train deep model (single seed)
```bash
python scripts/phase2_cli.py train-deep \
  --input data/synth_extended.jsonl \
  --output-dir artifacts/phase35_deep \
  --window-size 20 \
  --horizon-frames 20 \
  --epochs 20 \
  --hidden-dim 64 \
  --learning-rate 0.001 \
  --batch-size 64 \
  --seed 42
```

### 3) Tune thresholds + calibrated evaluation
```bash
python scripts/phase2_cli.py tune-thresholds \
  --input data/synth_extended.jsonl \
  --model-path artifacts/phase35_deep/deep_model.json \
  --output artifacts/phase35_deep/tuned_thresholds.json

python scripts/phase2_cli.py evaluate-deep-calibrated \
  --input data/synth_extended.jsonl \
  --model-path artifacts/phase35_deep/deep_model.json \
  --threshold-path artifacts/phase35_deep/tuned_thresholds.json \
  --output artifacts/phase35_deep/eval_calibrated.json
```

### 4) Multi-seed deep training/evaluation
```bash
python scripts/phase2_cli.py train-deep-multiseed \
  --input data/synth_extended.jsonl \
  --output-dir artifacts/phase35_multiseed \
  --seeds 11,22,33 \
  --window-size 20 \
  --horizon-frames 20 \
  --epochs 20 \
  --hidden-dim 64 \
  --learning-rate 0.001 \
  --batch-size 64
```

### 5) Model comparison (rules vs classical vs deep) and multiseed reports
```bash
python scripts/phase2_cli.py compare-models \
  --input data/synth_extended.jsonl \
  --classical-model-path artifacts/phase2/classical_model.json \
  --deep-model-path artifacts/phase35_deep/deep_model.json \
  --output-dir artifacts/phase35_compare

python scripts/phase2_cli.py compare-models-multiseed \
  --input data/synth_extended.jsonl \
  --classical-model-path artifacts/phase2/classical_model.json \
  --deep-root-dir artifacts/phase35_multiseed \
  --seeds 11,22,33 \
  --output-dir artifacts/phase35_compare_multiseed
```

## Notes / assumptions
- **Primary deep backend:** PyTorch GRU multi-head.
- **Fallback backend:** only used when torch is unavailable in the execution environment.
- Calibration support includes threshold tuning, Brier score, and ECE metrics for future-risk heads.
- Comparison reports are reproducible and emitted as JSON + Markdown + CSV (+ PNG if matplotlib is available).
