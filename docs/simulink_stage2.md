# Simulink Stage 2: Feature Interface + Replay Mode

## Purpose

Stage 2 extends the Stage 1 A/B scaffold with:
1. a strict feature window interface,
2. feature logging/export,
3. deterministic replay mode,
4. a pluggable inference stub (`predict_hesitation_state`) used by Policy B.

No real model integration is performed in Stage 2.

## Feature window definitions

Each feature log row stores a `feature_window` with:
- `timestamp`
- `mean_speed`
- `pause_ratio`
- `progress_delta`
- `reversal_count`
- `retry_count`
- `distance_to_target`
- `human_robot_distance`
- `task_step`
- `shared_zone_occupancy`

Compatibility fields aligned with the repository ML schema are also written:
- `speed_variance`
- `direction_changes`
- `backtrack_ratio`
- `mean_workspace_distance`

## Schema contract (fail-fast)

`validateFeatureSchemaContract` enforces:
- required field presence,
- type checks (numeric scalar),
- integer-like checks (`reversal_count`, `retry_count`, `task_step`, `direction_changes`),
- range checks for normalized features (e.g., `pause_ratio`, `shared_zone_occupancy` in `[0,1]`).

Any mismatch raises an error immediately.

## Inference interface stub

Policy B no longer reads scripted state directly. It calls:

- `predict_hesitation_state(feature_window, scripted_state)`

Current behavior is heuristic/stub only, with no Python or ONNX usage.

## Replay mode

Replay mode loads saved `feature_logs/*.jsonl` and replays each row deterministically.

For each replay stream:
- schema is revalidated,
- stub inference is recomputed,
- consistency metrics are recorded against logged predicted/scripted states,
- replay logs are exported to `replay_logs/*.csv`.

This supports reproducibility and future model-validation workflows.

## Artifact structure

Outputs are written under:

- `artifacts/simulink_stage2/`
  - `feature_logs/` (`<scenario>_<policy>.jsonl`)
  - `replay_logs/` (`<scenario>_<policy>_replay.csv`)
  - `metrics/`
    - `metrics_baseline.csv`
    - `metrics_hesitation_aware.csv`
    - `metrics_baseline_replay.csv`
    - `metrics_hesitation_aware_replay.csv`
  - `comparison/`
    - `comparison_summary_simulation.csv`
    - `comparison_summary_replay.csv`

## Running Stage 2

From MATLAB at repository root:

```matlab
run_ab_scenarios
```

Explicit options:

```matlab
run_ab_scenarios('stage', 'stage2', 'enable_replay', true, 'window_size', 12, 'deterministic_seed', 42)
```

Run Stage 1 explicitly:

```matlab
run_ab_scenarios('stage', 'stage1')
```

## Stage 3 integration plan (real model)

1. Replace `predict_hesitation_state` internals with real model inference.
2. Keep the same feature schema contract unchanged.
3. Add calibrated confidence and uncertainty traces.
4. Compare replay-mode predictions against locked reference outputs.
5. Add policy-safety regression thresholds before enabling deployment-facing simulations.
