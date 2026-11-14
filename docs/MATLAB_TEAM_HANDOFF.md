# MATLAB Team Handoff

Moved from the top-level README.

## **Handoff Notice**
The repository now contains both the MATLAB-first simulation workflow and the restored Python hesitation stack. The next phase belongs to the MATLAB team: verify the simulation path end-to-end in MATLAB, confirm the Python bridge still resolves correctly, and move the project from "runs" to "trusted and reproducible."

## **What the MATLAB Team Should Own**
- Confirm that `simulations/baseline_handoff_simulation.m` is the canonical entry point for MATLAB runs.
- Verify that supporting scripts in `simulations/` still pass when executed from a clean MATLAB session.
- Confirm that the restored Python package installs from repo root and that MATLAB bridge scripts can resolve `hesitation.inference.cli`.
- Validate that the integrated simulation behavior matches the documented three-scenario expectations.
- Tighten any parts of the MATLAB code that still depend on manual inspection instead of scripted checks.
- Keep generated figures, console outputs, and any benchmark results reproducible.

## **Recommended Progression**
1. Run the baseline flow first: `baseline_handoff_simulation.m`, then `matlab_verify_three_scenarios.m`.
2. Run smoke checks next: `matlab_cli_debug_smoke.m`, `matlab_ab_policy_smoke.m`, and `matlab_python_bridge_smoke.m`.
3. Run expected-result checks: `matlab_baseline_expected_check.m` and `matlab_ab_expected_check.m`.
4. Run integration coverage last: `test_matlab_integration.m` and `test_matlab_cli_integration.m`.
5. Run Python packaging verification from repo root: `pip install -e ".[dev,serve]"` and `python -c "import hesitation"`.
6. Record any failures with exact MATLAB version, command used, stack trace, and whether the issue is logic, environment, data-path, or Python-environment related.

## **Definition of Done for This Handoff**
- Baseline simulation runs in MATLAB without manual patching.
- Three-scenario verification produces expected outputs.
- A/B policy benchmark executes cleanly and results are interpretable.
- MATLAB bridge smoke scripts can call into the restored Python hesitation entrypoints.
- MATLAB integration tests pass or have clearly documented blockers.
- Any required code changes are pushed back into the canonical scripts, not left as local workspace edits.

## **Priority Fix Order**
- First: execution failures, path issues, missing-file problems, MATLAB-version incompatibilities.
- Second: incorrect scenario outputs or broken verification assumptions.
- Third: benchmark quality, parameter tuning, and adaptive-control improvements.
- Last: extensions such as 3D scenarios, hardware integration, or ML-based policy work.

## **Notes for the Next Team**
- Treat `data/` as the source of truth for CSV inputs.
- Treat `docs/` as the reference for expected behavior, but prefer executable MATLAB checks over prose where they disagree.
- Do not start with new features until the current MATLAB verification scripts are stable and repeatable.
- If simulation outputs change intentionally, update both the validation logic and the documentation in the same pass.
