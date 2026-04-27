# MATLAB Integration Guide — Hesitation-Aware Robot Policy

## Overview

The hesitation model predicts human operator state (6 classes) and future risks from video/kinematics features. This guide shows how to integrate it into your MATLAB simulator.

---

## Input: Feature Schema

The model expects a **feature dictionary** with these 7 keys:

| Feature | Type | Range | Units | Description |
|---------|------|-------|-------|-------------|
| `mean_hand_speed` | float | [0, 2.0] | m/s | Average hand velocity in last window |
| `pause_ratio` | float | [0, 1.0] | — | Fraction of frames with near-zero velocity |
| `progress_delta` | float | [0, 1.0] | — | Fractional progress toward task goal |
| `reversal_count` | int | [0, 10] | count | Number of direction reversals in window |
| `retry_count` | int | [0, 5] | count | Number of task restarts |
| `task_step_id` | int | [0, 20] | — | Current assembly step (0-indexed) |
| `human_robot_distance` | float | [0, 2.0] | m | Min hand-to-TCP distance in window |

**Window**: 20 frames at 10 Hz = 2 seconds of history

### Example Feature Dict (Python)
```python
features = {
    "mean_hand_speed": 0.45,
    "pause_ratio": 0.15,
    "progress_delta": 0.75,
    "reversal_count": 1,
    "retry_count": 0,
    "task_step_id": 3,
    "human_robot_distance": 0.35,
}
```

---

## Output: Prediction Schema

The model returns:

```python
Prediction(
    state: str,                             # One of 6 states (below)
    state_probabilities: dict[str, float],  # 6 class probabilities (sum=1.0)
    future_hesitation_prob: float,          # Risk of hesitation in next 2s [0-1]
    future_correction_prob: float,          # Risk of rework/correction [0-1]
    confidence: float,                      # Max state probability [0-1]
    window_size_frames: int = 20,           # Window used
    frame_rate_hz: int = 10,                # Frame rate
)
```

### Example Output (JSON)
```json
{
    "state": "mild_hesitation",
    "state_probabilities": {
        "normal_progress": 0.15,
        "mild_hesitation": 0.65,
        "strong_hesitation": 0.10,
        "correction_rework": 0.05,
        "ready_for_robot_action": 0.03,
        "overlap_risk": 0.02
    },
    "future_hesitation_prob": 0.42,
    "future_correction_prob": 0.08,
    "confidence": 0.65,
    "window_size_frames": 20,
    "frame_rate_hz": 10
}
```

---

## State Definitions

| State | Meaning | Robot Response (Recommended) |
|-------|---------|------|
| **normal_progress** | Steady motion, confident assembly | Full speed (1.0×) |
| **mild_hesitation** | Brief pause (<2s), continues | Reduced speed (0.8×), increase wait tolerance |
| **strong_hesitation** | Prolonged pause (>2s), uncertainty | Slow speed (0.5×), offer assistance cue |
| **correction_rework** | Deliberate backtrack + retry | Halt (0.0×), await operator decision |
| **ready_for_robot_action** | Human waiting for robot move | Nominal speed (1.0×), proceed with task |
| **overlap_risk** | Hand enters danger zone | Slow speed (0.3×), delay until clear |

---

## Integration Method 1: Python Function (Recommended)

### Setup
```bash
cd /path/to/ieom_model
pip install -e .
pip install torch  # If using GPU; CPU fallback available
```

### Python Code
```python
from hesitation.inference import HesitationPredictor

# Load once at startup
predictor = HesitationPredictor.load_default()

# Each timestep: call predict_single()
features = {
    "mean_hand_speed": 0.45,
    "pause_ratio": 0.15,
    "progress_delta": 0.75,
    "reversal_count": 1,
    "retry_count": 0,
    "task_step_id": 3,
    "human_robot_distance": 0.35,
}

prediction = predictor.predict_single(features)
print(prediction.state)  # "mild_hesitation"
print(prediction.confidence)  # 0.65
```

---

## Integration Method 2: MATLAB via Command-Line (CLI)

### Setup
```bash
cd /path/to/ieom_model
pip install -e .
```

### MATLAB Code
```matlab
% Build feature dict as struct
features = struct(...
    'mean_hand_speed', 0.45, ...
    'pause_ratio', 0.15, ...
    'progress_delta', 0.75, ...
    'reversal_count', 1, ...
    'retry_count', 0, ...
    'task_step_id', 3, ...
    'human_robot_distance', 0.35 ...
);

% Call Python CLI
cmd = sprintf(...
    'python -m hesitation.inference.cli predict --mean-hand-speed %.2f --pause-ratio %.2f --progress-delta %.2f --reversal-count %d --retry-count %d --task-step-id %d --human-robot-distance %.2f --format json', ...
    features.mean_hand_speed, ...
    features.pause_ratio, ...
    features.progress_delta, ...
    features.reversal_count, ...
    features.retry_count, ...
    features.task_step_id, ...
    features.human_robot_distance ...
);

[status, result] = system(cmd);

% Parse JSON
prediction = jsondecode(result);
disp(prediction.state)
disp(prediction.confidence)
```

---

## Integration Method 3: MATLAB via py.module (if MATLAB Python support available)

### MATLAB Code
```matlab
% Initialize Python engine
py.sys.path.insert(0, '/path/to/ieom_model/src');

% Import and load once
predictor = py.hesitation.inference.HesitationPredictor.load_default();

% Each step
features_py = py.dict(...
    pyargs(...
        'mean_hand_speed', 0.45, ...
        'pause_ratio', 0.15, ...
        'progress_delta', 0.75, ...
        'reversal_count', int32(1), ...
        'retry_count', int32(0), ...
        'task_step_id', int32(3), ...
        'human_robot_distance', 0.35 ...
    ) ...
);

prediction = predictor.predict_single(features_py);
state = char(prediction.state);
confidence = double(prediction.confidence);
```

---

## Policy Mapping: State → Robot Action

Once you have the predicted state, map it to robot behavior:

```matlab
function [speed_factor, delay_ms, action] = get_robot_action(state)
    switch state
        case 'normal_progress'
            speed_factor = 1.0;
            delay_ms = 0;
            action = 'proceed_full_speed';
            
        case 'mild_hesitation'
            speed_factor = 0.8;
            delay_ms = 100;
            action = 'slow_down_gently';
            
        case 'strong_hesitation'
            speed_factor = 0.5;
            delay_ms = 200;
            action = 'slow_down_offer_cue';
            
        case 'correction_rework'
            speed_factor = 0.0;
            delay_ms = 500;
            action = 'halt_await_operator';
            
        case 'ready_for_robot_action'
            speed_factor = 1.0;
            delay_ms = 0;
            action = 'proceed_nominal';
            
        case 'overlap_risk'
            speed_factor = 0.3;
            delay_ms = inf;  % Until hand clears danger zone
            action = 'slow_down_wait_clear';
            
        otherwise
            speed_factor = 0.5;
            delay_ms = 0;
            action = 'unknown_state';
    end
end
```

---

## Troubleshooting

### Issue: `PyTorch not available`
**Solution**: Install PyTorch
```bash
pip install torch
```

### Issue: `Model not found`
**Solution**: Train and save the model first, or verify path in `load_default()`.

### Issue: Feature values out of range
**Solution**: Clamp or normalize features to expected ranges (see Feature Schema table).

### Issue: Predictions always same state
**Solution**: Check that features are updating correctly (not stuck in same values).

---

## Testing: Quick Validation

```python
from hesitation.inference import HesitationPredictor

predictor = HesitationPredictor.load_default()

# Test case 1: Normal progress
pred1 = predictor.predict_single({
    "mean_hand_speed": 1.0,
    "pause_ratio": 0.0,
    "progress_delta": 0.9,
    "reversal_count": 0,
    "retry_count": 0,
    "task_step_id": 5,
    "human_robot_distance": 0.5,
})
print(f"Normal: {pred1.state}")  # Should be 'normal_progress' or similar

# Test case 2: Strong hesitation
pred2 = predictor.predict_single({
    "mean_hand_speed": 0.1,
    "pause_ratio": 0.8,
    "progress_delta": 0.5,
    "reversal_count": 3,
    "retry_count": 1,
    "task_step_id": 3,
    "human_robot_distance": 0.2,
})
print(f"Hesitant: {pred2.state}")  # Should be 'strong_hesitation' or similar
```

---

## Performance Constraints

- **Latency**: <50 ms per prediction (CPU)
- **Memory**: ~100 MB for model
- **Throughput**: ~500 predictions/second (single-threaded)
- **Deterministic**: Same input → same output (no randomness at inference)

---

## Contact & Support

For issues with integration:
1. Check that all 7 features are provided and in range
2. Verify model checkpoint exists and loads correctly
3. Run `python -m hesitation.inference.cli health` to verify setup
4. Compare output JSON to examples in this guide
