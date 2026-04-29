from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MATLAB_DIR = REPO_ROOT / "src" / "matlab"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_run_episode_logs_hesitation_onset_in_frame_rows() -> None:
    content = _read(MATLAB_DIR / "runABEpisode.m")
    assert "buildFrameRow(sim, schedule, features, prediction, event_flags)" in content
    assert "row.hesitation_onset = double(event_flags.hesitation_onset);" in content
    assert "'hesitation_onset', 0" in content


def test_compute_metrics_tracks_hesitation_event_count_with_dual_sources() -> None:
    content = _read(MATLAB_DIR / "computeABEpisodeMetrics.m")
    assert "hesitation_events_from_states = countHesitationOnsets(human_states);" in content
    assert "if isfield(frame_log, 'hesitation_onset')" in content
    assert "hesitation_events_from_flags = sum([frame_log.hesitation_onset] > 0.5);" in content
    assert "'hesitation_event_count', hesitation_events" in content
    assert "'hesitation_event_count_from_states', hesitation_events_from_states" in content
    assert "'hesitation_event_count_consistent'" in content


def test_compute_metrics_hesitation_onset_definition_is_transition_based() -> None:
    content = _read(MATLAB_DIR / "computeABEpisodeMetrics.m")
    assert "function count = countHesitationOnsets(human_states)" in content
    assert "hesitation_states = {'mild_hesitation', 'strong_hesitation', 'correction_rework'};" in content
    assert "if current_hesitating && ~previous_hesitating" in content


def test_summary_pipeline_includes_hesitation_metric_aggregation_and_plot() -> None:
    content = _read(MATLAB_DIR / "writeABSummaryOutputs.m")
    assert "'hesitation_event_count'" in content
    assert "renderHesitationEventsBar(fullfile(run_dir, 'hesitation_events_bar.png'), episode_metrics);" in content
    assert "function renderHesitationEventsBar(file_path, episode_metrics)" in content
    assert "ylabel('Mean Hesitation Onset Events');" in content


def test_summary_text_warns_on_zero_safety_nonzero_hesitation_pattern() -> None:
    content = _read(MATLAB_DIR / "writeABSummaryOutputs.m")
    assert "zero_safety_nonzero_hesitation" in content
    assert "warning=found_%d_episodes_with_zero_safety_events_but_nonzero_hesitation_events" in content
