import json
import shutil
from pathlib import Path

import pytest

from hesitation.evaluation.suite import run_benchmark_suite
from hesitation.io.config import load_config
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import NoiseConfig, ScenarioConfig


def _write_havid_fixture(path: Path, sessions: int = 4) -> None:
    scenario_dict = load_config("configs/simulation/domain_gap_scene.yaml")
    noise_dict = scenario_dict.get("noise", {})
    scenario = ScenarioConfig(**{**scenario_dict, "noise": NoiseConfig(**noise_dict)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for session_index in range(sessions):
            trajectory, latent_states = generate_session(
                session_id=f"havid_s{session_index}",
                scenario=scenario,
                frame_rate_hz=10,
                seed=220 + session_index,
            )
            for frame, state in zip(trajectory.frames, latent_states, strict=False):
                row = frame.model_dump()
                row["latent_state"] = state.value
                row["dataset_name"] = "ha_vid"
                handle.write(json.dumps(row) + "\n")


@pytest.fixture(scope="module")
def paper_ready_suite_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("paper_ready_suite")
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    shutil.copyfile("merged_database/sample_outputs/chico_model_input_fixture.jsonl", inputs / "chico_model_input.jsonl")
    _write_havid_fixture(inputs / "havid_model_input.jsonl")

    config = load_config("configs/benchmark/paper_ready_suite.yaml")
    config["benchmark"]["deep"]["epochs"] = 1
    config["datasets"]["chico"]["input_path"] = str(inputs / "chico_model_input.jsonl")
    config["datasets"]["ha_vid"]["input_path"] = str(inputs / "havid_model_input.jsonl")
    config["datasets"]["ha_vid"]["splits"] = {
        "train": ["havid_s0", "havid_s1"],
        "test": ["havid_s2", "havid_s3"],
    }

    config_path = root / "suite_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    output_dir = root / "outputs"
    run_benchmark_suite(str(config_path), str(output_dir))
    return output_dir


def test_benchmark_pipeline_smoke(paper_ready_suite_dir: Path) -> None:
    assert (paper_ready_suite_dir / "suite_summary.json").exists()
    assert (paper_ready_suite_dir / "benchmarks" / "chico_within" / "run_summary.json").exists()
    assert (paper_ready_suite_dir / "benchmarks" / "havid_within" / "run_summary.json").exists()
    assert (paper_ready_suite_dir / "benchmarks" / "merged_train_eval" / "run_summary.json").exists()


def test_ablation_smoke(paper_ready_suite_dir: Path) -> None:
    ablation_root = paper_ready_suite_dir / "ablations"
    assert (ablation_root / "feature_subset_ablation" / "manifest.json").exists()
    assert (ablation_root / "harmonization_mask_ablation" / "run_summary.json").exists()
    assert (ablation_root / "label_horizon_ablation" / "run_summary.md").exists()


def test_figure_and_table_generation_smoke(paper_ready_suite_dir: Path) -> None:
    paper_root = paper_ready_suite_dir / "paper"
    assert (paper_root / "final_benchmark_table.csv").exists()
    assert (paper_root / "ablation_table.md").exists()
    assert (paper_root / "harmonization_coverage_gap_table.csv").exists()
    assert (paper_root / "label_distribution_table.md").exists()
    assert (paper_root / "figures" / "pipeline_overview.svg").exists()
    assert (paper_root / "figures" / "qualitative_cross_dataset.svg").exists()


def test_report_integrity(paper_ready_suite_dir: Path) -> None:
    summary = json.loads((paper_ready_suite_dir / "suite_summary.json").read_text(encoding="utf-8"))
    assert len(summary["benchmark_runs"]) == 5
    assert len(summary["ablation_runs"]) == 3

    run_summary = json.loads(
        (paper_ready_suite_dir / "benchmarks" / "chico_to_havid" / "run_summary.json").read_text(encoding="utf-8")
    )
    assert "rules_trigger_audit" in run_summary
    assert run_summary["counts"]["eval_classical_windows"] > 0

    deep_error_summary = json.loads(
        (
            paper_ready_suite_dir
            / "benchmarks"
            / "chico_to_havid"
            / "error_analysis"
            / "deep"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    assert "hardest_confusion_pairs" in deep_error_summary
    assert (paper_ready_suite_dir / "paper" / "transfer_gap_notes.md").exists()
