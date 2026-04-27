# Scenario Regression Test Plan

**Date:** 2026-04-27  
**Status:** ✅ Test Suite Prepared (Awaiting Trained Model)

---

## Overview

Four baseline scenarios have been designed to validate model integration with MATLAB simulator. These represent real-world operator behaviors across the hesitation spectrum.

**Note:** Current dummy model returns uniform predictions. These tests will validate predictions once trained model is loaded.

---

## Test Scenarios

### Scenario A: Normal Progress ✓ (Baseline)

**Description:** Operator performs task smoothly with steady motion and confidence.

**Feature Profile:**
| Feature | Value | Rationale |
|---------|-------|-----------|
| mean_hand_speed | 1.0 m/s | Smooth, confident motion |
| pause_ratio | 0.05 | Minimal pauses |
| progress_delta | 0.9 | Near task completion |
| reversal_count | 0 | No backtracking |
| retry_count | 0 | First attempt |
| task_step_id | 10 | Mid-task progress |
| human_robot_distance | 0.5 m | Safe working distance |

**Expected Output:**
```json
{
  "state": "normal_progress",
  "confidence": >0.7,
  "future_hesitation_prob": <0.1,
  "future_correction_prob": <0.05,
  "robot_speed_factor": 1.0
}
```

**Robot Behavior:** Proceed at full speed (1.0×)

---

### Scenario B: Mild Hesitation

**Description:** Operator briefly pauses, uncertain about next step, but recovers and continues.

**Feature Profile:**
| Feature | Value | Rationale |
|---------|-------|-----------|
| mean_hand_speed | 0.3 m/s | Reduced speed |
| pause_ratio | 0.25 | 25% pause time |
| progress_delta | 0.5 | Halfway through task |
| reversal_count | 1 | Brief backtrack |
| retry_count | 0 | No formal restart |
| task_step_id | 5 | Early-to-mid progress |
| human_robot_distance | 0.4 m | Still safe |

**Expected Output:**
```json
{
  "state": "mild_hesitation",
  "confidence": >0.6,
  "future_hesitation_prob": 0.3-0.5,
  "future_correction_prob": <0.1,
  "robot_speed_factor": 0.8
}
```

**Robot Behavior:** Reduce speed to 0.8×, increase wait tolerance

---

### Scenario C: Strong Hesitation

**Description:** Operator shows prolonged uncertainty with multiple corrections. Confused about task direction.

**Feature Profile:**
| Feature | Value | Rationale |
|---------|-------|-----------|
| mean_hand_speed | 0.1 m/s | Very slow motion |
| pause_ratio | 0.7 | 70% pause time |
| progress_delta | 0.3 | Struggling to progress |
| reversal_count | 4 | Multiple corrections |
| retry_count | 0 | Not formally restarted yet |
| task_step_id | 3 | Early in task |
| human_robot_distance | 0.2 m | Close approach, possibly uncertain |

**Expected Output:**
```json
{
  "state": "strong_hesitation",
  "confidence": >0.6,
  "future_hesitation_prob": 0.6-0.8,
  "future_correction_prob": 0.2-0.3,
  "robot_speed_factor": 0.5
}
```

**Robot Behavior:** Slow to 0.5×, offer assistance cue, +200ms delay

---

### Scenario D: Correction/Rework

**Description:** Operator deliberately backtracks and restarts task. Clear indication of mistake recognition and recovery attempt.

**Feature Profile:**
| Feature | Value | Rationale |
|---------|-------|-----------|
| mean_hand_speed | 0.2 m/s | Deliberate backtrack motion |
| pause_ratio | 0.4 | Some pauses during reset |
| progress_delta | 0.1 | Regressed to early state |
| reversal_count | 7 | Many direction changes (backtrack+restart) |
| retry_count | 2 | Explicit task restarts detected |
| task_step_id | 1 | Reset to near beginning |
| human_robot_distance | 0.3 m | Careful repositioning |

**Expected Output:**
```json
{
  "state": "correction_rework",
  "confidence": >0.7,
  "future_hesitation_prob": 0.3-0.5,
  "future_correction_prob": 0.5-0.7,
  "robot_speed_factor": 0.0
}
```

**Robot Behavior:** **HALT** (0.0×), await operator decision

---

## Validation Criteria

### ✅ Acceptance Thresholds

For each scenario, validation passes if:

1. **Predicted state matches expected state** (exact match required)
2. **Confidence > threshold**:
   - Normal progress: > 0.7
   - Mild hesitation: > 0.6
   - Strong hesitation: > 0.6
   - Correction/rework: > 0.7

3. **Future probabilities in plausible range**:
   - Sum = 1.0 (within tolerance)
   - Individual values ∈ [0, 1]
   - Hesitation prob correlates with state severity

4. **Robot speed factor matches policy**:
   - Normal progress: 1.0
   - Mild hesitation: 0.8
   - Strong hesitation: 0.5
   - Correction/rework: 0.0

5. **Behavior is consistent across runs**:
   - Same scenario features → same output
   - No stochasticity or randomness

---

## Testing Procedure

### Step 1: Feature Extraction
Extract features from MATLAB simulator for each scenario:
```bash
PYTHONPATH=src python3 -m hesitation.inference.cli predict \
  --mean-hand-speed <value> \
  --pause-ratio <value> \
  ... (all 7 features)
```

### Step 2: Prediction Validation
For each scenario:
- [ ] State matches expected
- [ ] Confidence > threshold
- [ ] Future probs in range
- [ ] Speed factor correct

### Step 3: Spot-Check Samples
During actual MATLAB simulator runs, collect 10 samples per scenario:
- [ ] Extract actual kinematics-derived features
- [ ] Record prediction output
- [ ] Verify against expected ranges

### Step 4: Documentation
Update WEEKLY_INTEGRATION_CHECKLIST.md:
- [ ] Scenario A: PASS/FAIL
- [ ] Scenario B: PASS/FAIL
- [ ] Scenario C: PASS/FAIL
- [ ] Scenario D: PASS/FAIL

---

## Expected Progression

Across scenarios, we expect to see:

| Aspect | A (Normal) | B (Mild) | C (Strong) | D (Correction) |
|--------|:----------:|:--------:|:-----------:|:---------------:|
| Pause ratio | ↓ 0.05 | ↑ 0.25 | ↑ 0.70 | → 0.40 |
| Reversal count | ↓ 0 | ↑ 1 | ↑ 4 | ↑ 7 |
| Hesitation prob | ↓ <0.1 | → 0.3-0.5 | ↑ 0.6-0.8 | → 0.3-0.5 |
| Speed factor | → 1.0 | ↓ 0.8 | ↓ 0.5 | ↓ 0.0 |
| **State** | **normal** | **mild** | **strong** | **rework** |

---

## Files

- `test_matlab_cli_integration.m` - MATLAB test harness
- `MATLAB_FEATURE_EXTRACTION_SPEC.md` - Feature computation guide
- `FEATURE_VALIDATION_MATRIX.md` - Range and spot-check evidence
- `WEEKLY_INTEGRATION_CHECKLIST.md` - Final validation checklist

---

## Next Steps

1. ✅ Scenario suite prepared
2. → Load trained model into `src/hesitation/inference/models/`
3. → Run 4 scenarios with trained model
4. → Collect spot-check samples from MATLAB simulator
5. → Validate ranges and sign off
