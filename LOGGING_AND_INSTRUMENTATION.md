# Experiment Logging and Instrumentation Guide

**Date:** 2026-04-27  
**Status:** ✅ Infrastructure Designed (Awaiting Simulator Integration)

---

## Overview

Logging infrastructure for capturing:
1. **Prediction logs** — Timestamped model outputs per frame
2. **Policy logs** — Robot action selection per frame  
3. **Experiment metadata** — Trial info, config, model version
4. **Performance metrics** — Safety, efficiency, quality scores

---

## Logging Schema

### Prediction Log

```json
{
  "trial_id": "trial_001_scenario_a",
  "timestamp_sec": 1234.56,
  "frame_idx": 100,
  "session_id": "session_20260427_001",
  
  "input_features": {
    "mean_hand_speed": 0.45,
    "pause_ratio": 0.15,
    "progress_delta": 0.75,
    "reversal_count": 1,
    "retry_count": 0,
    "task_step_id": 3,
    "human_robot_distance": 0.35
  },
  
  "model_output": {
    "state": "mild_hesitation",
    "state_probabilities": {
      "normal_progress": 0.1,
      "mild_hesitation": 0.65,
      "strong_hesitation": 0.15,
      "correction_rework": 0.05,
      "ready_for_robot_action": 0.03,
      "overlap_risk": 0.02
    },
    "future_hesitation_prob": 0.42,
    "future_correction_prob": 0.08,
    "confidence": 0.65,
    "inference_time_ms": 0.5
  }
}
```

### Policy Action Log

```json
{
  "trial_id": "trial_001_scenario_a",
  "timestamp_sec": 1234.56,
  "frame_idx": 100,
  
  "policy": "hesitation_aware",  // or "baseline"
  "predicted_state": "mild_hesitation",
  
  "robot_action": {
    "speed_factor": 0.8,
    "delay_ms": 100,
    "action_name": "slow_down_gently",
    "execution_success": true
  },
  
  "baseline_comparison": {
    "baseline_speed_factor": 1.0,
    "baseline_action": "proceed_full_speed"
  },
  
  "safety_metrics": {
    "collision_risk": 0.02,  // Estimated risk
    "hand_robot_distance": 0.35,
    "safety_constraint_violated": false
  }
}
```

### Trial Metadata Log

```json
{
  "trial_id": "trial_001_scenario_a",
  "session_id": "session_20260427_001",
  "timestamp_start": "2026-04-27T14:30:00Z",
  "timestamp_end": "2026-04-27T14:31:45Z",
  
  "experiment_config": {
    "scenario": "A",
    "scenario_name": "Normal Progress",
    "operator_skill_level": "experienced",
    "task_complexity": "medium"
  },
  
  "model_config": {
    "model_version": "v1.0",
    "model_checkpoint": "models/hesitation_model_v1.0_20260415.pt",
    "window_size": 20,
    "frame_rate": 10,
    "feature_normalization": "z-score"
  },
  
  "simulation_config": {
    "simulator_version": "MATLAB R2024a",
    "physics_engine": "built-in",
    "timestep_ms": 100,
    "trial_duration_sec": 105
  },
  
  "random_seed": 12345,  // For reproducibility
  "trial_notes": "Clean execution, no anomalies"
}
```

---

## Performance Metrics

### Safety Metrics
```json
{
  "trial_id": "trial_001",
  "safety": {
    "collision_count": 0,
    "min_hand_robot_distance": 0.25,
    "proximity_warnings": 2,
    "safety_interventions": 0,
    "override_count": 0
  }
}
```

### Efficiency Metrics
```json
{
  "trial_id": "trial_001",
  "efficiency": {
    "task_completion_time_sec": 105,
    "total_pause_time_sec": 15,
    "total_reversal_time_sec": 8,
    "progress_delta_per_second": 0.0095,
    "wasted_motion_percentage": 7.6
  }
}
```

### Quality Metrics
```json
{
  "trial_id": "trial_001",
  "quality": {
    "task_success": true,
    "assembly_accuracy_percent": 99.5,
    "rework_events": 0,
    "operator_frustration_score": 0.2,  // 0=none, 1=extreme
    "robot_interference_score": 0.1     // 0=none, 1=severe
  }
}
```

---

## Logging Implementation

### Python Logging Wrapper

```python
import json
import logging
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    def __init__(self, trial_id: str, experiment_dir: str):
        self.trial_id = trial_id
        self.log_dir = Path(experiment_dir) / trial_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers for different streams
        self.prediction_log = open(self.log_dir / "predictions.jsonl", "w")
        self.policy_log = open(self.log_dir / "policy_actions.jsonl", "w")
    
    def log_prediction(self, timestamp_sec, frame_idx, features, output):
        entry = {
            "trial_id": self.trial_id,
            "timestamp_sec": timestamp_sec,
            "frame_idx": frame_idx,
            "input_features": features,
            "model_output": output
        }
        self.prediction_log.write(json.dumps(entry) + "\n")
        self.prediction_log.flush()
    
    def log_policy_action(self, timestamp_sec, frame_idx, state, action, metrics):
        entry = {
            "trial_id": self.trial_id,
            "timestamp_sec": timestamp_sec,
            "frame_idx": frame_idx,
            "predicted_state": state,
            "robot_action": action,
            "safety_metrics": metrics
        }
        self.policy_log.write(json.dumps(entry) + "\n")
        self.policy_log.flush()
    
    def close(self):
        self.prediction_log.close()
        self.policy_log.close()

# Usage
logger = ExperimentLogger(
    trial_id="trial_001_scenario_a",
    experiment_dir="/experiments"
)

# In simulation loop
logger.log_prediction(
    timestamp_sec=123.45,
    frame_idx=100,
    features={...},
    output=prediction_json
)

logger.close()
```

---

## MATLAB Integration

```matlab
% In MATLAB simulator loop
log_entry = struct(...
    'trial_id', trial_id, ...
    'timestamp_sec', toc(), ...
    'frame_idx', frame_idx, ...
    'predicted_state', prediction.state, ...
    'robot_action', struct(...
        'speed_factor', speed_factor, ...
        'delay_ms', delay_ms, ...
        'execution_success', true ...
    ) ...
);

% Append to JSONL log
fid = fopen([log_dir '/policy_actions.jsonl'], 'a');
fprintf(fid, '%s\n', jsonencode(log_entry));
fclose(fid);
```

---

## Analysis Tools

### Query Logs After Trial

```python
import json
import pandas as pd

# Load predictions
predictions = []
with open("trial_001/predictions.jsonl") as f:
    for line in f:
        predictions.append(json.loads(line))

df_pred = pd.DataFrame(predictions)

# Summary statistics
print(df_pred['model_output'].apply(lambda x: x['state']).value_counts())
print(f"Average confidence: {df_pred['model_output'].apply(lambda x: x['confidence']).mean():.3f}")
```

---

## Metrics to Capture for Paper

**Primary metrics (always log):**
- ✓ Prediction state (all 6 classes)
- ✓ Confidence scores
- ✓ Robot speed factors
- ✓ Collision events
- ✓ Task completion time
- ✓ Operator hesitation episodes detected

**Secondary metrics (if available):**
- Safety interventions count
- Rework/correction events
- Operator frustration score
- Assembly accuracy

---

## Directory Structure

```
/experiments/
├── trial_001_scenario_a/
│   ├── predictions.jsonl          # Per-frame model outputs
│   ├── policy_actions.jsonl       # Per-frame robot actions
│   ├── trial_metadata.json        # Trial config, timestamps
│   ├── safety_metrics.json        # Collision, proximity data
│   ├── efficiency_metrics.json    # Time, motion, progress
│   └── quality_metrics.json       # Task success, accuracy
├── trial_002_scenario_b/
│   ├── ...
├── analysis_report.md             # Aggregated findings
└── baseline_vs_hesitation_aware.csv # Comparison table
```

---

## Next Steps

1. ✅ Logging infrastructure designed
2. → Integrate logger into MATLAB simulator
3. → Run 4 scenario tests with logging enabled
4. → Aggregate and analyze logs
5. → Generate comparison report (baseline vs hesitation-aware policy)

