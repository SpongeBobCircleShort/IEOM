# MATLAB/Octave Execution Verification

This document records what was executed and whether outputs matched expected values.

## Runtime

- MATLAB binary: not available in this environment.
- Octave runtime: installed and used (`GNU Octave 11.1.0`).

## 1) Direct baseline script run

Command:

```bash
octave --no-gui --quiet baseline_handoff_simulation.m
```

Result:

- Fails in Octave because `readtable` is not implemented.
- Error observed: `'readtable' undefined near line 13`.

## 2) MATLAB-compatible expected-output checker

Added executable script:

- `matlab_baseline_expected_check.m`

Command run:

```bash
octave --no-gui --quiet --eval "cd('/Users/adijain/ENGINEERING/IEOM/ieom_model'); matlab_baseline_expected_check();"
```

Generated artifacts:

- `reports/phase3_verification/matlab_baseline_expected_report.mat`
- `reports/phase3_verification/matlab_baseline_expected_report.txt`

Outcome:

- `pass_all=1` (all checks passed)

Measured rows:

- `Slow|v=0.400|t=12.500|min_sep=1.009|iso=1|row_pass=1`
- `Moderate|v=0.900|t=10.000|min_sep=2.230|iso=1|row_pass=1`
- `Aggressive|v=1.800|t=10.000|min_sep=3.629|iso=0|row_pass=1`

## 3) Feature validation evidence

Generated spotcheck evidence:

- `reports/phase3_verification/feature_spotcheck_10.csv`
- `reports/phase3_verification/feature_range_audit.json`

Notes:

- Fixture dataset supports direct checks for:
  - `mean_hand_speed`
  - `progress_delta` (proxy from `progress`)
  - `task_step_id`
  - `human_robot_distance`
- Fixture does not directly expose:
  - `pause_ratio`
  - `reversal_count`
  - `retry_count`
  so those remain MATLAB extraction checks during simulator integration.

## Final status

- MATLAB-side execution verification in this environment: **pass via Octave-compatible checker**.
- Full `baseline_handoff_simulation.m` parity in Octave: **blocked by `readtable` limitation**, not by model logic.
