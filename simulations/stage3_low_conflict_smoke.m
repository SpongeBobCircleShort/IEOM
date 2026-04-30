function stage3_low_conflict_smoke()
% stage3_low_conflict_smoke: Guard against low-conflict Policy B false holds.

    summary = stage3_run_ab_scenarios( ...
        'deterministic_seed', 6395, ...
        'enable_replay', false, ...
        'no_figures', true);

    aware = summary.hesitation_aware_metrics;
    base = summary.baseline_metrics;
    idx = findScenarioRow(aware, 'low_conflict_open');
    bidx = findScenarioRow(base, 'low_conflict_open');

    assert(idx > 0, 'Missing low_conflict_open aware metrics.');
    assert(bidx > 0, 'Missing low_conflict_open baseline metrics.');

    hold_ratio = aware.robot_hold_count(idx) / max(aware.total_simulated_time_sec(idx) / 0.1, 1);
    slowdown = aware.unnecessary_slowdown_count(idx);
    overlap_delta = aware.overlap_risk_event_count(idx) - base.overlap_risk_event_count(bidx);

    assert(hold_ratio < 0.10, 'low_conflict_open hold ratio too high: %.4f', hold_ratio);
    assert(slowdown <= 100, 'low_conflict_open unnecessary slowdown too high: %d', slowdown);
    assert(overlap_delta <= 0, 'low_conflict_open overlap delta regressed: %.4f', overlap_delta);

    fprintf('Stage 3 low-conflict smoke passed. hold_ratio=%.4f slowdown=%d overlap_delta=%.4f\n', ...
        hold_ratio, slowdown, overlap_delta);
end

function idx = findScenarioRow(tbl, scenario_name)
    idx = 0;
    for i = 1:numel(tbl.scenario)
        if strcmp(char(tbl.scenario(i)), scenario_name)
            idx = i;
            return;
        end
    end
end
