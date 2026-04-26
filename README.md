# Hesitation Simulation Prototype (Phase 1 + Phase 2 + Phase 3)

This repository contains a simulation-first prototype for human hesitation identification, near-future risk prediction, and advisory policy recommendation in collaborative robot workspaces.

## Scope
- ✅ Synthetic data generator
- ✅ Label/state definitions and logic
- ✅ Feature extraction
- ✅ Rules-based baseline
- ✅ Classical ML baselines (Phase 2)
- ✅ Future-risk prediction (Phase 2)
- ✅ Advisory policy recommendation layer (Phase 2)
- ✅ Torch-backed GRU smoke pipeline (Phase 3)
- ✅ Tests
- ❌ No API/UI yet

## Quickstart
```bash
pip install -e ".[dev]"
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

## Deep Profile
```bash
pip install -e ".[dev,deep]"
python -c "import torch; print(torch.__version__)"
pytest -q -k "phase35_torch_smoke or phase3_model_forward"

python scripts/phase2_cli.py train-deep \
  --input data/synth.jsonl \
  --output-dir artifacts/phase3 \
  --window-size 15 \
  --horizon-frames 10 \
  --seed 17

python scripts/phase2_cli.py evaluate-deep \
  --input data/synth.jsonl \
  --model-path artifacts/phase3/deep_model.pt

python scripts/phase2_cli.py tune-thresholds \
  --input data/synth.jsonl \
  --model-path artifacts/phase3/deep_model.pt \
  --output artifacts/phase3/deep_thresholds.json

python scripts/phase2_cli.py evaluate-deep-calibrated \
  --input data/synth.jsonl \
  --model-path artifacts/phase3/deep_model.pt \
  --threshold-path artifacts/phase3/deep_thresholds.json
```

Apple Silicon and CUDA installs should use the platform-appropriate PyTorch wheel from the PyTorch install matrix instead of the CPU-only CI command.

## Notes / assumptions
- Classical models are implemented as lightweight logistic regression (one-vs-rest + binary heads) trained on Phase 1 window features.
- The Phase 3 smoke path uses a small GRU over per-frame kinematic signals and is intended to prove torch training/checkpoint/inference/calibration in CI.
- Config files are `.yaml` names with JSON-compatible content.
- Policy recommendation is advisory only and **not** a safety-certified controller.

## Baseline handoff data scraping (ideal trajectories)
```bash
python scripts/scrape_handoff_baseline_data.py \
  --output-dir data/baseline_scraped \
  --human-speed 0.80 \
  --robot-speeds 0.40 0.60 0.75 \
  --sqlite-path data/baseline_scraped/handoff_baseline.db


python scripts/analyze_handoff_baseline.py \
  --summary-csv data/baseline_scraped/handoff_baseline_summary.csv \
  --report-path data/baseline_scraped/handoff_next_steps.md
```

This generates:
- `data/baseline_scraped/handoff_baseline_frames.csv` (frame-level positions + separation)
- `data/baseline_scraped/handoff_baseline_summary.csv` (safety/efficiency metrics by robot-speed scenario)
- `data/baseline_scraped/handoff_baseline.db` (SQLite database for VS Code SQL extensions)


After generating data, produce an actionable "what next" report:
- `data/baseline_scraped/handoff_next_steps.md` summarizes the speed-vs-safety tradeoff.
- It also gives a concrete hesitation experiment plan and adaptive-control success criteria.

### Use an existing database in VS Code
1. Set `--sqlite-path` to your existing `.db` file path when running the script.
2. In VS Code, open the database with a SQLite extension (for example, SQLite Viewer).
3. Query:
   - `handoff_baseline_frames`
   - `handoff_baseline_summary`
