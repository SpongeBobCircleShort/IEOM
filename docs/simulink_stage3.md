# Simulink Stage 3: Real Model Integration and A/B Validation

## Purpose

Stage 3 replaces the Stage 2 inference stub with a MATLAB-to-Python bridge to the existing hesitation model while preserving:
- feature schema contract,
- scenario behavior,
- policy structure,
- deterministic replay validation.

## Real model bridge

MATLAB interface function:
- `predict_hesitation_state(feature_window, scripted_state, config)`

Modes:
- `use_stub = true`: keep Stage 2 heuristic debugging mode.
- `use_stub = false`: call Python bridge (`scripts/simulink_stage3_predict.py`).

Bridge output contract:

```json
{
  "predicted_state": "normal_progress",
  "state_probabilities": {
    "normal_progress": 1.0,
    "mild_hesitation": 0.0,
    "strong_hesitation": 0.0,
    "correction_rework": 0.0,
    "ready_for_robot_action": 0.0,
    "overlap_risk": 0.0
  },
  "future_hesitation_probability": 0.0,
  "future_correction_probability": 0.0
}
```

## Python entrypoint

Script:
- `scripts/simulink_stage3_predict.py`

Stable function API:
- `predict(features_dict) -> {...}`

This script:
1. maps Stage 2/3 feature schema to model input fields,
2. applies type normalization,
3. calls `HesitationPredictor`,
4. validates output keys/ranges/probability sum.

## Input normalization

Feature mapping before inference:
- `mean_speed -> mean_hand_speed`
- `pause_ratio -> pause_ratio`
- `progress_delta -> progress_delta`
- `reversal_count -> reversal_count`
- `retry_count -> retry_count`
- `task_step -> task_step_id`
- `human_robot_distance -> human_robot_distance`

## Output validation

After model call, Stage 3 validates:
- required keys present,
- `predicted_state` in allowed enum,
- per-state probabilities in `[0,1]`,
- probabilities sum to ~1,
- future risk probabilities in `[0,1]`.

## Logging and replay

Feature logs now include:
- `predicted_state`
- `state_probabilities`
- `future_hesitation_probability`
- `future_correction_probability`

Replay mode:
- reloads logs,
- reruns prediction,
- checks deterministic match for state and probability vectors,
- writes replay validation reports.

## Artifact layout

Outputs under:
- `artifacts/simulink_stage3/`
  - `feature_logs/`
  - `replay_logs/`
  - `tables/`
  - `figures/`
  - `reports/`

## Running Stage 3

Default Stage 3 run:

```matlab
run_ab_scenarios
```

Explicit real model run:

```matlab
run_ab_scenarios('stage', 'stage3', 'use_stub', false, 'enable_replay', true)
```

Debug/stub fallback run:

```matlab
run_ab_scenarios('stage', 'stage3', 'use_stub', true, 'enable_replay', true)
```

## Stage 3 outputs

A/B tables and summaries include:
- task completion delta,
- overlap risk event delta,
- robot hold delta,
- human wait delta,
- unnecessary slowdown delta,
- statistical summary (mean improvement, percent improvement, variance),
- safety checks (excessive holds, oscillation indicators, invalid outputs).

## Known limitations

- Real model bridge currently uses subprocess `system()` fallback path for compatibility.
- If Python environment/model artifacts are unavailable, inference will fail fast.
- Safety checks are lightweight sanity guards, not formal safety certification.

## Stage 4 recommendation

- Add calibrated thresholds and model version pinning per experiment run.
- Add confidence-based policy hysteresis to reduce oscillations.
- Add paired-seed statistical significance testing and confidence intervals.
- Add regression snapshots for replay outputs as release gates.
