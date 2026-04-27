# MATLAB Feature Extraction Specification

**Date:** 2026-04-27  
**Status:** ✅ Specification Complete (Awaiting Simulator Integration)

---

## Overview

Three dynamic features must be extracted from MATLAB simulator kinematics/task state:
- `pause_ratio` — Fraction of frames with near-zero hand velocity
- `reversal_count` — Number of direction reversals in motion window
- `retry_count` — Number of task restarts (cumulative)

The remaining 4 features (mean_hand_speed, progress_delta, task_step_id, human_robot_distance) are computed directly from raw measurements.

---

## Feature Specifications

### 1. pause_ratio

**Definition:** Fraction of frames in the 20-frame window where hand velocity is below motion threshold.

**MATLAB Implementation:**
```matlab
% In simulator at each frame t
hand_velocity = norm(hand_position(t) - hand_position(t-1)) / dt;

% Track pause frames
if hand_velocity < VELOCITY_THRESHOLD  % 0.05 m/s
    pause_frame_count = pause_frame_count + 1;
end

% Compute rolling window (every frame)
pause_ratio = pause_frame_count / 20;  % Over last 20 frames
```

| Property | Value |
|----------|-------|
| Type | float |
| Range | [0.0, 1.0] |
| Units | ratio (unitless) |
| Threshold | velocity < 0.05 m/s |
| Window | 20 frames @ 10 Hz (2 seconds) |
| Update Frequency | Every frame |

**Validation Checks:**
- ✓ Values between 0.0 and 1.0
- ✓ Increases during pauses (hesitation events)
- ✓ Decreases during active motion
- ✓ Deterministic (same trial → same values)

---

### 2. reversal_count

**Definition:** Number of direction reversals (hand motion reverses direction) in the 20-frame window.

**MATLAB Implementation:**
```matlab
% At each frame t
hand_velocity = hand_position(t) - hand_position(t-1);
hand_velocity_direction = sign(hand_velocity);

% Count reversals (only count if both samples exceed velocity threshold)
if (abs(prev_velocity) > VELOCITY_THRESHOLD) && (abs(curr_velocity) > VELOCITY_THRESHOLD)
    if sign(prev_velocity) ~= sign(curr_velocity)
        reversal_count = reversal_count + 1;  % Direction changed
    end
end

% Rolling window (every frame)
reversals_in_window = count_reversals(last_20_frames);
```

| Property | Value |
|----------|-------|
| Type | int |
| Range | [0, 10] |
| Units | count (per 2-second window) |
| Threshold | velocity > 0.05 m/s (ignore micro-motion) |
| Window | 20 frames @ 10 Hz (2 seconds) |
| Update Frequency | Every frame |

**Validation Checks:**
- ✓ Integer values 0-10
- ✓ Increases during hesitation/correction episodes
- ✓ Low during smooth motion
- ✓ Deterministic

**Scenario Examples:**
- **Normal progress:** 0-1 reversals (smooth forward motion)
- **Mild hesitation:** 1-3 reversals (brief pauses, small backtrack)
- **Strong hesitation:** 3-5 reversals (multiple corrections)
- **Correction/rework:** 5+ reversals (deliberate backtracking)

---

### 3. retry_count

**Definition:** Cumulative number of task restarts detected in the trial.

**MATLAB Implementation:**
```matlab
% Detect task restart (e.g., operator manual reset or explicit restart event)
if task_restart_event_detected()
    retry_count = retry_count + 1;  % Increment permanently
end

% Example: operator clicks "reset" button, or system detects reset
% retry_count is NOT reset; it monotonically increases throughout trial
```

| Property | Value |
|----------|-------|
| Type | int |
| Range | [0, 5] |
| Units | count (cumulative over trial) |
| Persistence | Monotonically increasing; never decreases |
| Update Frequency | Only on restart event |
| Scope | Trial-level (not windowed) |

**Validation Checks:**
- ✓ Integer values 0-5
- ✓ Monotonically increases (never decreases)
- ✓ Reflects cumulative task resets
- ✓ Deterministic given trial events

**Scenario Examples:**
- **Normal progress:** 0 restarts (completes task first try)
- **Mild hesitation:** 0-1 restarts (recovers without reset)
- **Correction/rework:** 1-3 restarts (explicit backtrack + reset)
- **Major difficulty:** 3-5 restarts (multiple task restarts)

---

## Integration Checklist

**In MATLAB Simulator:**
- [ ] Compute `hand_velocity` each frame from kinematics
- [ ] Maintain rolling window (20-frame buffer) for pause and reversal tracking
- [ ] Increment `pause_frame_count` when velocity < 0.05 m/s
- [ ] Track direction reversals and count in rolling window
- [ ] Maintain `retry_count` that increments on task restart events
- [ ] Ensure all 3 features updated before calling model at each timestep

**Validation:**
- [ ] Extract features during 4 scenario tests (RANK 6)
- [ ] Collect 40 spot-check samples (10 per scenario)
- [ ] Verify ranges: pause_ratio ∈ [0,1], reversal_count ∈ [0,10], retry_count ∈ [0,5]
- [ ] Verify determinism: same trial → same feature values
- [ ] Update FEATURE_VALIDATION_MATRIX.md with MATLAB reviewer evidence

---

## Expected Behavior

### Scenario A: Normal Progress
- `pause_ratio`: ~0.0-0.1 (minimal pauses)
- `reversal_count`: 0-1 (smooth forward motion)
- `retry_count`: 0 (no resets)

### Scenario B: Mild Hesitation
- `pause_ratio`: ~0.2-0.4 (short pauses)
- `reversal_count`: 1-2 (brief reversals)
- `retry_count`: 0 (recovered without reset)

### Scenario C: Strong Hesitation
- `pause_ratio`: ~0.5-0.8 (prolonged pauses)
- `reversal_count`: 3-5 (multiple corrections)
- `retry_count`: 0-1 (may attempt reset)

### Scenario D: Correction/Rework
- `pause_ratio`: ~0.3-0.6 (deliberate backtrack)
- `reversal_count`: 5-10 (many direction changes)
- `retry_count`: 1-3 (explicit task restarts)

---

## Thresholds & Parameters

```matlab
VELOCITY_THRESHOLD = 0.05;  % m/s - below this = pause
WINDOW_SIZE = 20;           % frames
FRAME_RATE = 10;            % Hz
WINDOW_DURATION = 2.0;      % seconds (20 frames @ 10 Hz)
MAX_REVERSAL_COUNT = 10;    % Clamp if exceeds
MAX_RETRY_COUNT = 5;        % Clamp if exceeds
```

---

## Next Steps

1. ✅ Feature extraction specification complete
2. → Integrate feature computation into MATLAB simulator
3. → Run 4 scenario tests and collect spot-check samples
4. → Validate ranges and determinism
5. → Complete FEATURE_VALIDATION_MATRIX.md with MATLAB sign-off
