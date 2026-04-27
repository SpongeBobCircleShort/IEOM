# Hesitation Simulation Prototype

This repository covers:
- simulation-first hesitation modeling pipelines across rules, classical, and deep baselines
- CHICO and HA-ViD real-dataset onboarding with mapping, QC, label audit, split/export, and benchmark helpers
- CHICO↔HA-ViD cross-dataset benchmark and harmonization reporting
- a paper-ready benchmark suite with within-dataset, cross-dataset, merged, ablation, and error-analysis artifacts
- a Phase 4 serving/demo layer for inspecting saved artifacts

## Scope
- Phase 1: synthetic data generation, state/label logic, feature extraction
- Phase 2: rules baseline, classical baselines, future-risk prediction, advisory policy recommendation
- Phase 3: torch-backed deep temporal baseline
- Phase 3.5: threshold tuning, multiseed/reporting support
- Phase 4: FastAPI inference service, Streamlit demo, artifact visualization, Docker/dev-run setup, optional ingestion stub
- Paper artifacts: final benchmark matrix, ablations, transfer notes, and reproducible tables/figures

## Install Profiles

```bash
pip install -e ".[dev]"
pip install -e ".[dev,serve]"
pip install -e ".[dev,serve,deep]"
```

Defaults:
- `dev`: test dependencies
- `serve`: FastAPI, Uvicorn, Streamlit
- `deep`: PyTorch-backed deep runtime

## Phase 1-3 Quickstart

```bash
python scripts/generate_synthetic_dataset.py \
  --output data/synth.jsonl \
  --n-sessions 12 \
  --seed 17

python scripts/phase2_cli.py train-classical \
  --input data/synth.jsonl \
  --output-dir artifacts/phase2 \
  --window-size 20 \
  --pause-speed-threshold 0.03 \
  --horizon-frames 20

python scripts/phase2_cli.py evaluate-classical \
  --input data/synth.jsonl \
  --model-path artifacts/phase2/classical_model.json

python scripts/phase2_cli.py train-deep \
  --input data/synth.jsonl \
  --output-dir artifacts/phase3 \
  --window-size 15 \
  --horizon-frames 10 \
  --seed 17

python scripts/phase2_cli.py evaluate-deep \
  --input data/synth.jsonl \
  --model-path artifacts/phase3/deep_model.pt
```

## Real Dataset Onboarding

CHICO onboarding guide:
- `merged_database/docs/chico_onboarding.md`

HA-ViD + cross-dataset guide:
- `merged_database/docs/havid_onboarding_and_cross_benchmark.md`

Fixture-based quick run:

```bash
# CHICO normalize -> labeled -> export
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

# HA-ViD normalize -> labeled -> export
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

PYTHONPATH=src python -m hesitation.database.cli export-model-input \
  --input /tmp/havid_labeled.jsonl \
  --output /tmp/havid_model_input.jsonl

# Cross-dataset benchmark + harmonization report
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

## Paper-Ready Benchmark

Detailed guide:
- `docs/paper_ready_benchmark.md`

Prepare deterministic local benchmark inputs:

```bash
python scripts/generate_paper_benchmark_inputs.py \
  --output-dir reports/paper_ready/inputs
```

Run the full benchmark + ablation suite:

```bash
python scripts/phase2_cli.py run-benchmark-suite \
  --config configs/benchmark/paper_ready_suite.yaml \
  --output-dir reports/paper_ready
```

Primary artifact locations:
- `reports/paper_ready/benchmarks/<run>/`: per-run models, predictions, metrics, and error analysis
- `reports/paper_ready/ablations/<ablation>/`: ablation-specific summaries and artifacts
- `reports/paper_ready/paper/final_benchmark_table.{csv,md}`
- `reports/paper_ready/paper/harmonization_coverage_gap_table.{csv,md}`
- `reports/paper_ready/paper/label_distribution_table.{csv,md}`
- `reports/paper_ready/paper/figures/*.svg`

## MATLAB A/B Validation

Detailed guide:
- `docs/matlab_ab_milestone1.md`

Run the full MATLAB A/B benchmark:

```matlab
cd('/path/to/IEOM');
matlab_run_ab_policy_benchmark();
```

Run the smoke and deterministic checks:

```matlab
cd('/path/to/IEOM');
matlab_ab_policy_smoke();
matlab_ab_expected_check();
```

Optional Python bridge smoke:

```matlab
cd('/path/to/IEOM');
matlab_python_bridge_smoke();
```

MATLAB benchmark artifacts are written under:
- `reports/matlab_validation/ab_milestone1/<run_id>/`

## Phase 4 API

Start the API:

```bash
pip install -e ".[dev,serve]"
uvicorn hesitation.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Use the deep backend only when `torch` is installed:

```bash
pip install -e ".[dev,serve,deep]"
python -c "import torch; print(torch.__version__)"
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Implemented endpoints:
- `GET /health`
- `POST /infer/current-state`
- `POST /infer/future-risk`
- `POST /infer/full`
- `POST /policy/recommend`
- `POST /reports/compare`

## Streamlit Demo

Run the demo:

```bash
pip install -e ".[dev,serve]"
streamlit run src/hesitation/demo/app.py
```

The demo supports:
- generated synthetic sessions
- uploaded frame-observation JSONL files
- switching between `rules`, `classical`, and `deep` backends when artifacts are available
- advisory policy display
- artifact/report inspection and numeric report comparison
- optional video/pose stub uploads

## Docker

Build the default API image:

```bash
docker build -t hesitation-api .
docker run --rm -p 8000:8000 hesitation-api
```

Build with deep support:

```bash
docker build --build-arg INSTALL_DEEP=1 -t hesitation-api-deep .
docker run --rm -p 8000:8000 hesitation-api-deep
```

## Testing

Run the full suite:

```bash
pip install -e ".[dev,serve]"
pytest -q
```

Run only deep tests:

```bash
pip install -e ".[dev,serve,deep]"
pytest -q -m deep
```

## Notes
- API and demo outputs are advisory only and are not safety-certified robot control.
- Current hesitation probability for classical and deep inference is derived from the sum of hesitation-like state probabilities.
- Paper-ready benchmark inputs are deterministic fixtures bundled for reproducible local evaluation; swap the input manifest paths for full private datasets when available.
