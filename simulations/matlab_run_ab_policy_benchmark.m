function summary = matlab_run_ab_policy_benchmark(varargin)
% matlab_run_ab_policy_benchmark: Run paired A/B MATLAB benchmark and write outputs.

    repo_root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(repo_root, 'src', 'matlab'));

    config = buildABConfig(varargin{:});
    run_dir = fullfile(config.output_base_dir, config.run_id);
    if ~exist(run_dir, 'dir')
        mkdir(run_dir);
    end

    episode_metrics = struct([]);
    pairwise_deltas = struct([]);
    episode_timelines = cell(0, 1);
    representative_timeline = [];
    metric_idx = 1;
    delta_idx = 1;

    for scenario_idx = 1:numel(config.scenario_names)
        scenario_name = config.scenario_names{scenario_idx};
        for seed_offset = 0:(config.paired_seed_count - 1)
            seed = config.seed_start + seed_offset;
            schedule = generateHumanEventSchedule(scenario_name, seed, config);

            [metrics_a, frame_log_a] = runABEpisode('A', schedule, config);
            [metrics_b, frame_log_b] = runABEpisode('B', schedule, config);
            metrics_a.policy_name = 'A';
            metrics_a.backend = config.backend;
            metrics_b.policy_name = 'B';
            metrics_b.backend = config.backend;

            episode_metrics(metric_idx) = metrics_a; %#ok<AGROW>
            episode_metrics(metric_idx + 1) = metrics_b; %#ok<AGROW>
            episode_timelines{end + 1, 1} = struct( ... %#ok<AGROW>
                'scenario_name', scenario_name, ...
                'seed', seed, ...
                'policy_name', 'A', ...
                'frame_log', frame_log_a);
            episode_timelines{end + 1, 1} = struct( ... %#ok<AGROW>
                'scenario_name', scenario_name, ...
                'seed', seed, ...
                'policy_name', 'B', ...
                'frame_log', frame_log_b);
            metric_idx = metric_idx + 2;

            pairwise_deltas(delta_idx) = buildPairwiseDelta(schedule, metrics_a, metrics_b); %#ok<AGROW>
            if strcmp(schedule.scenario_name, config.metrics.timeline_scenario) && seed_offset == (config.metrics.timeline_seed - 1)
                representative_timeline = struct( ...
                    'schedule', schedule, ...
                    'frame_log', frame_log_b, ...
                    'baseline_frame_log', frame_log_a);
            end
            delta_idx = delta_idx + 1;
        end
    end

    summary = writeABSummaryOutputs(config, run_dir, episode_metrics, pairwise_deltas, representative_timeline, episode_timelines);
end

function row = buildPairwiseDelta(schedule, metrics_a, metrics_b)
    row = struct( ...
        'scenario_name', schedule.scenario_name, ...
        'seed', schedule.seed, ...
        'delta_completion_time_sec', metrics_b.completion_time_sec - metrics_a.completion_time_sec, ...
        'delta_overlap_event_count', metrics_b.overlap_event_count - metrics_a.overlap_event_count, ...
        'delta_unsafe_close_call_count', metrics_b.unsafe_close_call_count - metrics_a.unsafe_close_call_count, ...
        'delta_robot_wait_time_sec', metrics_b.robot_wait_time_sec - metrics_a.robot_wait_time_sec, ...
        'delta_human_wait_time_sec', metrics_b.human_wait_time_sec - metrics_a.human_wait_time_sec, ...
        'delta_response_latency_sec', metrics_b.response_latency_sec);
end
