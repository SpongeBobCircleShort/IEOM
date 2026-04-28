# Quick Reference for MATLAB Integration

## TL;DR

Your hesitation model is **ready to integrate**. Here's what MATLAB needs to do:

### 1. Import & Load (Once at Start)

**Python**:
```python
from hesitation.inference.standalone import HesitationPredictor
predictor = HesitationPredictor.load_default()
```

**MATLAB CLI**:
```bash
python3 -m hesitation.inference.cli health
```

### 2. Extract 7 Features (Each Frame)

```
mean_hand_speed    = average hand velocity (m/s)
pause_ratio        = (# frames with v<0.1) / 20
progress_delta     = (current_step - prev_step) / total_steps
reversal_count     = # of direction sign flips in 20-frame window
retry_count        = # times current step was restarted
task_step_id       = current assembly step (0-indexed)
human_robot_distance = min(hand_position - tcp_position) in meters
```

### 3. Call Model (Each Frame)

**Python**:
```python
prediction = predictor.predict_single({
    "mean_hand_speed": 0.5,
    "pause_ratio": 0.15,
    "progress_delta": 0.75,
    "reversal_count": 1,
    "retry_count": 0,
    "task_step_id": 3,
    "human_robot_distance": 0.35,
})
```

**MATLAB**:
```matlab
cmd = sprintf('PYTHONPATH=src python3 -m hesitation.inference.cli --format json predict --mean-hand-speed %.2f --pause-ratio %.2f ...', ...);
[~, result] = system(cmd);
prediction = jsondecode(result);
```

### 4. Get Result

```
prediction.state  = "normal_progress" | "mild_hesitation" | ... (one of 6)
prediction.state_probabilities = dict of 6 class probabilities
prediction.confidence = max(probabilities)
prediction.future_hesitation_prob = risk of hesitation in next 2s
prediction.future_correction_prob = risk of rework in next 2s
```

### 5. Map to Robot Action

```matlab
switch prediction.state
    case 'normal_progress'
        speed_factor = 1.0;  % Full speed
    case 'mild_hesitation'
        speed_factor = 0.8;  % 80% speed
    case 'strong_hesitation'
        speed_factor = 0.5;  % 50% speed
    case 'correction_rework'
        speed_factor = 0.0;  % Halt
    case 'ready_for_robot_action'
        speed_factor = 1.0;  % Full speed
    case 'overlap_risk'
        speed_factor = 0.3;  % 30% speed + delay
end

robot_velocity = base_velocity * speed_factor;
```

## 3 Integration Options (Pick One)

### Option A: Python Function (RECOMMENDED)
- Fastest, cleanest, easiest debugging
- Call from MATLAB via `py.module.function()`
- Requires MATLAB Python support

### Option B: CLI (SAFEST)
- No Python interpreter needed in MATLAB
- Each call spawns subprocess (5-10ms overhead)
- Most robust for deployment

### Option C: REST API (ADVANCED)
- Spin up Flask/FastAPI server
- MATLAB calls HTTP endpoint
- Overkill for single-robot simulation

## File Locations

```
/Users/adijain/ENGINEERING/IEOM/ieom_model/
├── src/hesitation/inference/standalone.py     ← Load this
├── src/hesitation/inference/cli.py           ← Or call this via CLI
└── MATLAB_INTEGRATION.md                     ← Full docs
```

## Sanity Checks

```bash
# 1. Verify Python environment
python3 -c "import sys; print(sys.version)"

# 2. Check model loads
python3 -m hesitation.inference.cli health

# 3. Test single prediction
PYTHONPATH=src python3 -m hesitation.inference.cli --format json predict \
    --mean-hand-speed 0.5 \
    --pause-ratio 0.1 \
    --progress-delta 0.8 \
    --reversal-count 1 \
    --retry-count 0 \
    --task-step-id 3 \
    --human-robot-distance 0.35
```

Python requirement: 3.10+ (3.11 recommended).

## Common Mistakes to Avoid

1. ❌ Forget to provide all 7 features → prediction fails
2. ❌ Features out of range → model may give weird results
3. ❌ Don't normalize features → use raw values
4. ❌ Call model multiple times per frame → call once per timestep
5. ❌ Ignore `future_hesitation_prob` and `future_correction_prob` → use for A/B comparison

## Expected Performance

- Latency: <50ms per prediction (CPU)
- Memory: ~100 MB model
- Throughput: ~500 predictions/second
- Deterministic: Yes (same input always gives same output)
