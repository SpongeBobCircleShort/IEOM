# Paper-Ready Benchmark

## Commands
Prepare local benchmark inputs:

```bash
python scripts/generate_paper_benchmark_inputs.py \
  --output-dir reports/paper_ready/inputs
```

Run the full suite:

```bash
python scripts/phase2_cli.py run-benchmark-suite \
  --config configs/benchmark/paper_ready_suite.yaml \
  --output-dir reports/paper_ready
```

## What It Produces
- `reports/paper_ready/benchmarks/chico_within/`
- `reports/paper_ready/benchmarks/havid_within/`
- `reports/paper_ready/benchmarks/chico_to_havid/`
- `reports/paper_ready/benchmarks/havid_to_chico/`
- `reports/paper_ready/benchmarks/merged_train_eval/`
- `reports/paper_ready/ablations/feature_subset_ablation/`
- `reports/paper_ready/ablations/harmonization_mask_ablation/`
- `reports/paper_ready/ablations/label_horizon_ablation/`
- `reports/paper_ready/paper/final_benchmark_table.{csv,md}`
- `reports/paper_ready/paper/ablation_table.{csv,md}`
- `reports/paper_ready/paper/harmonization_coverage_gap_table.{csv,md}`
- `reports/paper_ready/paper/label_distribution_table.{csv,md}`
- `reports/paper_ready/paper/transfer_gap_notes.md`
- `reports/paper_ready/paper/figures/pipeline_overview.svg`
- `reports/paper_ready/paper/figures/qualitative_cross_dataset.svg`
- `reports/paper_ready/paper/figures/qualitative_merged.svg`

## Reproducibility Notes
- The suite manifest is `configs/benchmark/paper_ready_suite.yaml`.
- Input staging is deterministic and recorded in `reports/paper_ready/inputs/input_manifest.json`.
- Each benchmark or ablation run writes its effective settings to `<run>/manifest.json`.
- Error-analysis outputs live under `<run>/error_analysis/<model>/`.
