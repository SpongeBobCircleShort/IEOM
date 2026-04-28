# Simulink Stage 1: A/B Shared-Workspace Simulation Skeleton

## Purpose

Stage 1 provides a **simple MATLAB simulation scaffold** (Simulink-ready in structure) to validate A/B policy flow in a shared human-robot workspace before connecting the real hesitation model.

The stage focuses on:
- deterministic/synthetic human hesitation behavior,
- policy switching between baseline and hesitation-aware placeholder logic,
- metric collection and artifact writing.

## Scope and assumptions

This stage intentionally keeps the simulation lightweight:
- 2D workspace abstraction (no detailed robot arm dynamics),
- scripted human hesitation states (no ML inference),
- no Python calls,
- no ONNX export/inference,
- no hardware/URSim dependency.

Workspace abstraction includes:
- human zone,
- robot zone,
- shared zone,
- fixture/target area,
- human hand point,
- robot end-effector point.

## Human state definitions (scripted)

Implemented states:
- `normal_progress`
- `mild_hesitation`
- `strong_hesitation`
- `correction_rework`
- `ready_for_robot_action`
- `overlap_risk`

Each state modulates:
- motion speed scale,
- pause probability,
- retry tendency,
- probability of entering the shared zone,
- task progress rate.

## Robot policies

### Policy A: baseline
- fixed release/wait behavior,
- fixed nominal speed,
- no hesitation-state awareness.

### Policy B: hesitation-aware placeholder
- `normal_progress` -> proceed nominally,
- `mild_hesitation` -> slow,
- `strong_hesitation` -> stronger slow + caution,
- `correction_rework` -> hold,
- `overlap_risk` -> hold/avoid,
- `ready_for_robot_action` -> proceed nominally.

## Scenario runner

Entry point:
- `run_ab_scenarios.m`

Runner executes 4 scenario profiles under both policies:
- `smooth_operator`
- `hesitation_heavy_operator`
- `correction_heavy_operator`
- `overlap_risk_operator`

## Metrics produced

Per scenario per policy:
- `task_completion_time_sec`
- `robot_idle_time_sec`
- `human_idle_time_sec`
- `overlap_risk_event_count`
- `robot_hold_count`
- `unnecessary_slowdown_count`
- `correction_rework_count`
- `total_simulated_time_sec`

## Output artifacts

Saved under:
- `artifacts/simulink_stage1/`

Files:
- `metrics_baseline.csv`
- `metrics_hesitation_aware.csv`
- `comparison_summary.csv`
- `task_completion_comparison.png`
- `overlap_event_comparison.png`

## How to run

From MATLAB at repository root:

```matlab
run_ab_scenarios
```

## Current limitations

- Simplified kinematics only (point-mass hand and end-effector),
- scripted stochastic state sampling instead of learned inference,
- no joint-space, controller, or collision-geometry realism,
- no online co-simulation with external frameworks.

## Stage 2 recommendation

Stage 2 should keep this A/B scaffold and add:
1. stricter deterministic scenario scripts and seeds,
2. richer workspace/task sequencing,
3. state-transition timing reports,
4. optional plug-in interface for real hesitation inference (still decoupled),
5. stronger parity checks between MATLAB and Python evaluation outputs.
