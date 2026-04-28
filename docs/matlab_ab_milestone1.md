git push origin main# MATLAB A/B Validation Milestone 1

This milestone adds a deterministic, script-based MATLAB simulator for paired A/B policy comparison in a shared 2D workspace.

## Entry Points
- `matlab_run_ab_policy_benchmark`
- `matlab_ab_policy_smoke`
- `matlab_ab_expected_check`
- `matlab_python_bridge_smoke`

## Run
From MATLAB with the repo root on disk:

```matlab
cd('/path/to/IEOM');
matlab_run_ab_policy_benchmark();
```

Run the reduced smoke check:

```matlab
cd('/path/to/IEOM');
matlab_ab_policy_smoke();
```

Run the deterministic expected-output check:

```matlab
cd('/path/to/IEOM');
matlab_ab_expected_check();
```

Run the optional Python bridge smoke:

```matlab
cd('/path/to/IEOM');
matlab_python_bridge_smoke();
```

## Outputs
Artifacts are written under:

`reports/matlab_validation/ab_milestone1/<run_id>/`

Files produced by the benchmark runner:
- `run_config.json`
- `episode_metrics.csv`
- `pairwise_deltas.csv`
- `scenario_policy_summary.csv`
- `summary.mat`
- `summary.txt`
- `completion_time_boxplot.png`
- `safety_events_bar.png`
- `wait_time_tradeoff_bar.png`
- `response_latency_bar.png`
- `representative_timeline.png` when a representative B-run timeline is available

## Notes
- Default backend is the deterministic `heuristic_stub`.
- The optional `python_bridge` backend reuses `src/matlab/HesitationModelCLI.m` and the existing `hesitation.inference.cli`.
- The simulator is intentionally 2D, script-based, and minimal; Simulink and 3D models remain out of scope for this milestone.
