<<<<<<< HEAD
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
=======
# Hesitation Simulation Prototype

This repository implements a simulation-first prototype for hesitation identification, future-risk prediction, advisory policy recommendation, and a Phase 4 serving/demo layer for inspecting saved artifacts.

## Scope
- Phase 1: synthetic data generation, state/label logic, feature extraction
- Phase 2: rules baseline, classical baselines, future-risk prediction, advisory policy recommendation
- Phase 3: torch-backed deep temporal baseline
- Phase 3.5: threshold tuning, multiseed/reporting support
- Phase 4: FastAPI inference service, Streamlit demo, artifact visualization, Docker/dev-run setup, optional ingestion stub

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

Example full inference request:
```bash
curl -X POST http://127.0.0.1:8000/infer/full \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

Minimal `example_request.json` shape:
```json
{
  "frames": [
    {
      "session_id": "session_0",
      "frame_idx": 0,
      "timestamp_ms": 0,
      "task_step_id": 0,
      "hand_x": 0.1,
      "hand_y": 0.2,
      "hand_speed": 0.05,
      "hand_accel": 0.01,
      "distance_to_robot_workspace": 0.4,
      "progress": 0.1,
      "confidence": 0.95,
      "is_dropout": false
    }
  ],
  "artifact": {
    "backend": "rules"
  }
}
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

The base image installs `serve` dependencies only. Deep support is optional to keep the default image lighter.

## Reports And Artifacts
Phase 4 can inspect:
- JSON metrics/report files
- CSV previews
- Markdown report previews

`/reports/compare` and the demo artifact viewer flatten shared numeric metrics and compute deltas across two report sources.

## Optional Video/Pose Stub
Phase 4 includes a stub ingestion flow:
- video files return a documented placeholder response
- precomputed frame-observation `.jsonl` files load directly

Full video inference is intentionally out of scope for Phase 4.

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

CI layout:
- `unit-fast`: installs `.[dev,serve]`, runs non-deep tests
- `unit-deep`: installs `.[dev,serve]`, installs CPU PyTorch, runs deep tests and deep CLI smoke flow

## Notes
- API and demo outputs are advisory only and are not safety-certified robot control.
- Current hesitation probability for classical and deep inference is derived from the sum of hesitation-like state probabilities.
- Apple Silicon and CUDA installs should use the platform-appropriate PyTorch wheel instead of the CPU-only CI command.
>>>>>>> bb0e772 (Implement Phase 4 serving and demo layer)
