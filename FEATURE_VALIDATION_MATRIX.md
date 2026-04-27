# Feature Validation Matrix (MATLAB -> Model)

Use this matrix to complete `p3-validate-features` with evidence.

## Range/Type checks

| Feature | Expected Type | Expected Range | Unit | Pass/Fail | Notes |
|---|---|---|---|---|---|
| `mean_hand_speed` | float | [0.0, 2.0] | m/s |  |  |
| `pause_ratio` | float | [0.0, 1.0] | ratio |  |  |
| `progress_delta` | float | [0.0, 1.0] | ratio |  |  |
| `reversal_count` | int | [0, 10] | count |  |  |
| `retry_count` | int | [0, 5] | count |  |  |
| `task_step_id` | int | [0, 20] | index |  |  |
| `human_robot_distance` | float | [0.0, 2.0] | m |  |  |

## Spot-check samples (manual)

Capture at least 10 sampled frames:

| Trial | Frame | mean_hand_speed | pause_ratio | progress_delta | reversal_count | retry_count | task_step_id | human_robot_distance | Reviewer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |

## Consistency checks

- [ ] 7 features are present on every predictor call
- [ ] no NaN/Inf values observed
- [ ] deterministic inputs produce deterministic outputs
- [ ] units are consistent (meters, seconds, frame counts)

## Sign-off

- MATLAB reviewer:
- Model reviewer:
- Date:
- Result:
  - [ ] PASS (ready for `p3-verify-outputs`)
  - [ ] FAIL (needs fixes)
