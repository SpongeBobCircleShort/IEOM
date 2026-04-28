function matlab_ab_expected_check()
% matlab_ab_expected_check: Deterministic rerun and metric ordering checks.

    repo_root = fileparts(mfilename('fullpath'));
    addpath(fullfile(repo_root, 'src', 'matlab'));

    paused_buffers = struct('human_pos_xy', [], 'human_speed', [], 'progress', [], 'separation', [], 'task_restart_flag', []);
    smooth_buffers = paused_buffers;
    config = buildABConfig('run_id', 'expected_check_run', 'paired_seed_count', 2, ...
        'scenario_names', {'nominal_handoff', 'strong_pause_near_zone', 'robot_overlap_pressure'});

    dummy_human = struct('pos_xy', [0.0, 0.0], 'vel_xy', [0.0, 0.0], 'progress_01', 0.0, 'rework_count', 0, 'target_idx', 2);
    dummy_robot = struct('pos_xy', [0.5, 0.5]);
    dummy_interaction = struct('separation_m', 0.5);
    for idx = 1:config.window_size_frames
        dummy_human.vel_xy = [0.0, 0.0];
        dummy_human.progress_01 = idx / config.window_size_frames;
        [paused_features, paused_buffers] = extractRollingFeatures2D(paused_buffers, dummy_human, dummy_robot, dummy_interaction, false, config); %#ok<ASGLU>

        dummy_human.vel_xy = [0.1, 0.0];
        [smooth_features, smooth_buffers] = extractRollingFeatures2D(smooth_buffers, dummy_human, dummy_robot, dummy_interaction, false, config); %#ok<ASGLU>
    end
    assert(paused_features.pause_ratio > smooth_features.pause_ratio, 'Paused trace should produce higher pause ratio.');

    summary_one = matlab_run_ab_policy_benchmark( ...
        'run_id', 'expected_check_run_one', ...
        'paired_seed_count', 2, ...
        'scenario_names', config.scenario_names, ...
        'backend', 'heuristic_stub');
    summary_two = matlab_run_ab_policy_benchmark( ...
        'run_id', 'expected_check_run_two', ...
        'paired_seed_count', 2, ...
        'scenario_names', config.scenario_names, ...
        'backend', 'heuristic_stub');

    csv_one = fileread(fullfile(config.output_base_dir, 'expected_check_run_one', 'episode_metrics.csv'));
    csv_two = fileread(fullfile(config.output_base_dir, 'expected_check_run_two', 'episode_metrics.csv'));
    assert(strcmp(csv_one, csv_two), 'episode_metrics.csv must be identical across deterministic reruns.');

    pairwise = summary_one.pairwise_deltas;
    nominal = pairwise(strcmp({pairwise.scenario_name}, 'nominal_handoff'));
    for idx = 1:numel(nominal)
        assert(nominal(idx).delta_completion_time_sec <= 0.05 * summaryOneBaseline(summary_one, nominal(idx).seed, 'nominal_handoff') + 1e-9, ...
            'Nominal handoff completion for B exceeded 5%% tolerance.');
    end

    strong_subset = pairwise(strcmp({pairwise.scenario_name}, 'strong_pause_near_zone') | ...
        strcmp({pairwise.scenario_name}, 'robot_overlap_pressure'));
    for idx = 1:numel(strong_subset)
        assert(strong_subset(idx).delta_overlap_event_count <= 0.0, ...
            'Policy B should not increase overlap event count in strong-pressure scenarios.');
    end

    fprintf('MATLAB A/B expected-output check passed.\n');
end

function baseline_completion = summaryOneBaseline(summary, seed, scenario_name)
    metrics = summary.episode_metrics(strcmp({summary.episode_metrics.scenario_name}, scenario_name) & ...
        strcmp({summary.episode_metrics.policy_name}, 'A') & ...
        [summary.episode_metrics.seed] == seed);
    baseline_completion = metrics.completion_time_sec;
end
