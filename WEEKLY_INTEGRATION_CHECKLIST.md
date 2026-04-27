# Weekly Integration Checklist (Model <-> MATLAB)

Use this checklist once per week (or per integration sprint) to avoid drift between model outputs and simulator behavior.

## Metadata

- Week:
- Date:
- Attendees:
- Branch/commit tested:
- Dataset/scenario set used:

## 1) Interface Health

- [ ] CLI health check passes:
  - `python3 -m hesitation.inference.cli health`
- [ ] Single prediction call returns valid JSON
- [ ] No schema drift in required 7 input features
- [ ] Output fields unchanged (`state`, `state_probabilities`, `future_*`, `confidence`)

## 2) Feature Correctness (MATLAB -> Model)

- [ ] `mean_hand_speed` values in expected range [0, 2.0]
- [ ] `pause_ratio` values in expected range [0, 1.0]
- [ ] `progress_delta` values in expected range [0, 1.0]
- [ ] `reversal_count` values in expected range [0, 10]
- [ ] `retry_count` values in expected range [0, 5]
- [ ] `task_step_id` values in expected range [0, 20]
- [ ] `human_robot_distance` values in expected range [0, 2.0]
- [ ] Spot-check 10 rows against expected manual calculations

## 3) Prediction Sanity

- [ ] Same input gives same output (determinism check)
- [ ] Probabilities sum to ~1.0 (tolerance +/- 1e-6)
- [ ] Confidence equals max state probability
- [ ] State distribution is plausible for tested scenarios

## 4) Policy Mapping Validation

- [ ] Each of 6 states maps to expected robot action
- [ ] Safety state (`overlap_risk`) enforces protective slowdown/delay
- [ ] `correction_rework` behavior matches agreed halt/assist policy
- [ ] No unexpected oscillation between contradictory actions

## 5) Scenario Regression Set

- [ ] Scenario A: normal progress path behaves as expected
- [ ] Scenario B: hesitation episode triggers reduced speed
- [ ] Scenario C: correction/rework path triggers stop/assist
- [ ] Scenario D: overlap risk event triggers safety action

## 6) Logging and Experiment Readiness

- [ ] Predictions are logged with timestamps and scenario IDs
- [ ] Baseline and hesitation-aware policy logs are both captured
- [ ] Metric fields needed for paper are present (safety/efficiency/quality)
- [ ] Trial reproducibility info stored (seed, config, model version)

## 7) Open Issues and Actions

- Blockers:
- Risks:
- Decisions made this week:
- Action owners and due dates:

## Weekly Sign-off

- Model team sign-off:
- MATLAB team sign-off:
- Integration status:
  - [ ] Green (ready for next phase)
  - [ ] Yellow (minor issues)
  - [ ] Red (blocking issues)
