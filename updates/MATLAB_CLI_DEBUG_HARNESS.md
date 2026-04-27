# MATLAB <-> Python CLI Debug Harness

This harness gives a deterministic, repeatable way to debug model integration from MATLAB before running large simulations.

## Goal

Verify end-to-end behavior:

1. MATLAB feature values are passed correctly.
2. CLI call succeeds and returns parseable JSON.
3. Output schema is stable and policy mapping can be applied.

## Prerequisites

- Python: `/opt/homebrew/bin/python3.11`
- Working directory: `/Users/adijain/ENGINEERING/IEOM/ieom_model`
- Command path: `scripts/phase2_cli.py` is not used here; we call inference CLI directly.

## Step 1: Terminal smoke checks

Run from `ieom_model` directory:

```bash
/opt/homebrew/bin/python3.11 -m hesitation.inference.cli health
```

Expected:

```json
{"model_loaded": true}
```

Then run a known prediction:

```bash
/opt/homebrew/bin/python3.11 -m hesitation.inference.cli predict \
  --mean-hand-speed 0.45 \
  --pause-ratio 0.15 \
  --progress-delta 0.75 \
  --reversal-count 1 \
  --retry-count 0 \
  --task-step-id 3 \
  --human-robot-distance 0.35 \
  --format json
```

Expected:

- valid JSON
- `state` in known state list
- `confidence` in [0, 1]

## Step 2: MATLAB command template

Use this exact pattern for debugging:

```matlab
function prediction = call_hesitation_cli_debug(features)
    pyExe = '/opt/homebrew/bin/python3.11';
    cliCmd = ['cd /Users/adijain/ENGINEERING/IEOM/ieom_model && ' ...
        pyExe ' -m hesitation.inference.cli predict ' ...
        '--mean-hand-speed %.6f --pause-ratio %.6f --progress-delta %.6f ' ...
        '--reversal-count %d --retry-count %d --task-step-id %d ' ...
        '--human-robot-distance %.6f --format json'];

    cmd = sprintf(cliCmd, ...
        features.mean_hand_speed, ...
        features.pause_ratio, ...
        features.progress_delta, ...
        int32(features.reversal_count), ...
        int32(features.retry_count), ...
        int32(features.task_step_id), ...
        features.human_robot_distance);

    [status, result] = system(cmd);
    assert(status == 0, 'CLI failed: %s', result);

    prediction = jsondecode(result);

    required = {'state','state_probabilities','future_hesitation_prob','future_correction_prob','confidence'};
    for i = 1:numel(required)
        assert(isfield(prediction, required{i}), 'Missing field: %s', required{i});
    end
end
```

## Step 3: fixed test vectors (copy/paste)

Use these rows before connecting live feature extraction:

```matlab
tests = [
    struct('mean_hand_speed',0.50,'pause_ratio',0.10,'progress_delta',0.80,'reversal_count',1,'retry_count',0,'task_step_id',3,'human_robot_distance',0.35), ...
    struct('mean_hand_speed',0.05,'pause_ratio',0.70,'progress_delta',0.10,'reversal_count',3,'retry_count',2,'task_step_id',5,'human_robot_distance',0.20), ...
    struct('mean_hand_speed',0.30,'pause_ratio',0.20,'progress_delta',0.50,'reversal_count',0,'retry_count',0,'task_step_id',1,'human_robot_distance',0.80)
];
```

For each row:

- call CLI twice
- assert outputs are identical (determinism)
- log state + confidence

## Step 4: failure signatures and fixes

- `ModuleNotFoundError: hesitation`
  - run command from `.../ieom_model` directory
- `dataclass() got an unexpected keyword argument 'slots'`
  - wrong Python version; must use Python 3.10+
- JSON parse failure in MATLAB
  - inspect raw `result`; ensure no extra print output in wrapper
- Missing required field
  - CLI version mismatch; pin to same repo revision for both teams

## Step 5: debug log format

Write one CSV during simulator debug:

```csv
timestamp,trial_id,frame_idx,mean_hand_speed,pause_ratio,progress_delta,reversal_count,retry_count,task_step_id,human_robot_distance,state,confidence,future_hesitation_prob,future_correction_prob,cli_status
```

This file is the primary artifact for `p3-debug-interface`.
