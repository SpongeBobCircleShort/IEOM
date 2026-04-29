function summary = writeABSummaryOutputs(config, run_dir, episode_metrics, pairwise_deltas, representative_timeline, episode_timelines)
% writeABSummaryOutputs: Persist benchmark CSV, MAT, TXT, and PNG outputs.

    if ~exist(run_dir, 'dir')
        mkdir(run_dir);
    end

    if exist('OCTAVE_VERSION', 'builtin')
        config_json = jsonencode(config);
    else
        config_json = jsonencode(config, 'PrettyPrint', true);
    end
    writeTextFile(fullfile(run_dir, 'run_config.json'), config_json);

    writeStructCsv(fullfile(run_dir, 'episode_metrics.csv'), episode_metrics);
    writeStructCsv(fullfile(run_dir, 'pairwise_deltas.csv'), pairwise_deltas);

    scenario_policy_summary = summarizeScenarioPolicy(episode_metrics);
    writeStructCsv(fullfile(run_dir, 'scenario_policy_summary.csv'), scenario_policy_summary);

    summary = struct();
    summary.episode_metrics = episode_metrics;
    summary.pairwise_deltas = pairwise_deltas;
    summary.scenario_policy_summary = scenario_policy_summary;
    summary.episode_timelines = episode_timelines;
    summary.representative_timeline = representative_timeline;
    save(fullfile(run_dir, 'summary.mat'), 'summary');

    writeTextFile(fullfile(run_dir, 'summary.txt'), buildSummaryText(episode_metrics, pairwise_deltas));

    if config.enable_plots
        renderCompletionTimeBoxplot(fullfile(run_dir, 'completion_time_boxplot.png'), episode_metrics);
        renderSafetyEventsBar(fullfile(run_dir, 'safety_events_bar.png'), episode_metrics);
        renderWaitTimeTradeoffBar(fullfile(run_dir, 'wait_time_tradeoff_bar.png'), episode_metrics);
        renderResponseLatencyBar(fullfile(run_dir, 'response_latency_bar.png'), episode_metrics);
    end

    if config.enable_plots && ~isempty(representative_timeline)
        renderRepresentativeTimeline(fullfile(run_dir, 'representative_timeline.png'), representative_timeline);
    end
end

function writeStructCsv(file_path, rows)
    if isempty(rows)
        fid = fopen(file_path, 'w');
        fclose(fid);
        return;
    end

    field_names = fieldnames(rows);
    fid = fopen(file_path, 'w');
    fprintf(fid, '%s\n', strjoin(field_names', ','));
    for idx = 1:numel(rows)
        values = cell(1, numel(field_names));
        for field_idx = 1:numel(field_names)
            values{field_idx} = encodeCsvValue(rows(idx).(field_names{field_idx}));
        end
        fprintf(fid, '%s\n', strjoin(values, ','));
    end
    fclose(fid);
end

function value = encodeCsvValue(raw_value)
    if ischar(raw_value)
        text = raw_value;
        text = strrep(text, '"', '""');
        value = ['"', text, '"'];
    elseif exist('isstring', 'builtin') && isstring(raw_value)
        text = char(raw_value);
        text = strrep(text, '"', '""');
        value = ['"', text, '"'];
    elseif isnumeric(raw_value) || islogical(raw_value)
        if isempty(raw_value)
            value = '';
        elseif isscalar(raw_value)
            if isnan(raw_value)
                value = 'NaN';
            else
                value = num2str(raw_value, '%.12g');
            end
        else
            value = ['"', mat2str(raw_value), '"'];
        end
    else
        value = ['"', strrep(jsonencode(raw_value), '"', '""'), '"'];
    end
end

function writeTextFile(file_path, contents)
    fid = fopen(file_path, 'w');
    fprintf(fid, '%s\n', contents);
    fclose(fid);
end

function summary_rows = summarizeScenarioPolicy(episode_metrics)
    summary_rows = struct([]);
    scenario_names = unique({episode_metrics.scenario_name});
    policy_names = unique({episode_metrics.policy_name});
    row_idx = 1;
    metrics_to_summarize = { ...
        'completion_time_sec', ...
        'overlap_event_count', ...
        'unsafe_close_call_count', ...
        'robot_wait_time_sec', ...
        'human_wait_time_sec', ...
        'response_latency_sec'};

    for scenario_idx = 1:numel(scenario_names)
        for policy_idx = 1:numel(policy_names)
            subset = episode_metrics(strcmp({episode_metrics.scenario_name}, scenario_names{scenario_idx}) & ...
                strcmp({episode_metrics.policy_name}, policy_names{policy_idx}));
            if isempty(subset)
                continue;
            end

            row = struct( ...
                'scenario_name', scenario_names{scenario_idx}, ...
                'policy_name', policy_names{policy_idx}, ...
                'episode_count', numel(subset));
            for metric_idx = 1:numel(metrics_to_summarize)
                metric_name = metrics_to_summarize{metric_idx};
                values = [subset.(metric_name)];
                values = values(~isnan(values));
                if isempty(values)
                    row.([metric_name, '_mean']) = NaN;
                    row.([metric_name, '_std']) = NaN;
                else
                    row.([metric_name, '_mean']) = mean(values);
                    row.([metric_name, '_std']) = std(values, 0);
                end
            end
            summary_rows(row_idx) = row; %#ok<AGROW>
            row_idx = row_idx + 1;
        end
    end
end

function text = buildSummaryText(episode_metrics, pairwise_deltas)
    lines = {
        'MATLAB A/B hesitation-aware benchmark summary', ...
        sprintf('episodes=%d', numel(episode_metrics)), ...
        sprintf('pairs=%d', numel(pairwise_deltas))
    };

    scenario_names = unique({pairwise_deltas.scenario_name});
    for idx = 1:numel(scenario_names)
        subset = pairwise_deltas(strcmp({pairwise_deltas.scenario_name}, scenario_names{idx}));
        lines{end + 1} = sprintf( ...
            '%s | delta_completion_mean=%.4f | delta_overlap_mean=%.4f | delta_wait_mean=%.4f', ...
            scenario_names{idx}, ...
            mean([subset.delta_completion_time_sec]), ...
            mean([subset.delta_overlap_event_count]), ...
            mean([subset.delta_robot_wait_time_sec])); %#ok<AGROW>
    end

    text = strjoin(lines, newline);
end

function renderCompletionTimeBoxplot(file_path, episode_metrics)
    figure('Visible', 'off');
    labels = strcat({episode_metrics.scenario_name}', '_', {episode_metrics.policy_name}');
    boxplot([episode_metrics.completion_time_sec]', labels, 'LabelOrientation', 'inline');
    ylabel('Completion Time (s)');
    title('Completion Time by Scenario and Policy');
    xtickangle(30);
    saveas(gcf, file_path);
    close(gcf);
end

function renderSafetyEventsBar(file_path, episode_metrics)
    scenarios = unique({episode_metrics.scenario_name});
    overlap_a = zeros(numel(scenarios), 1);
    overlap_b = zeros(numel(scenarios), 1);
    unsafe_a = zeros(numel(scenarios), 1);
    unsafe_b = zeros(numel(scenarios), 1);

    for idx = 1:numel(scenarios)
        subset_a = episode_metrics(strcmp({episode_metrics.scenario_name}, scenarios{idx}) & strcmp({episode_metrics.policy_name}, 'A'));
        subset_b = episode_metrics(strcmp({episode_metrics.scenario_name}, scenarios{idx}) & strcmp({episode_metrics.policy_name}, 'B'));
        overlap_a(idx) = mean([subset_a.overlap_event_count]);
        overlap_b(idx) = mean([subset_b.overlap_event_count]);
        unsafe_a(idx) = mean([subset_a.unsafe_close_call_count]);
        unsafe_b(idx) = mean([subset_b.unsafe_close_call_count]);
    end

    figure('Visible', 'off');
    subplot(2, 1, 1);
    bar(categorical(scenarios), [overlap_a, overlap_b]);
    ylabel('Mean Overlap Events');
    legend({'A', 'B'}, 'Location', 'best');
    subplot(2, 1, 2);
    bar(categorical(scenarios), [unsafe_a, unsafe_b]);
    ylabel('Mean Unsafe Close Calls');
    legend({'A', 'B'}, 'Location', 'best');
    saveas(gcf, file_path);
    close(gcf);
end

function renderWaitTimeTradeoffBar(file_path, episode_metrics)
    scenarios = unique({episode_metrics.scenario_name});
    robot_wait = zeros(numel(scenarios), 2);
    human_wait = zeros(numel(scenarios), 2);
    policies = {'A', 'B'};

    for scenario_idx = 1:numel(scenarios)
        for policy_idx = 1:2
            subset = episode_metrics(strcmp({episode_metrics.scenario_name}, scenarios{scenario_idx}) & strcmp({episode_metrics.policy_name}, policies{policy_idx}));
            robot_wait(scenario_idx, policy_idx) = mean([subset.robot_wait_time_sec]);
            human_wait(scenario_idx, policy_idx) = mean([subset.human_wait_time_sec]);
        end
    end

    figure('Visible', 'off');
    subplot(2, 1, 1);
    bar(categorical(scenarios), robot_wait);
    ylabel('Robot Wait (s)');
    legend({'A', 'B'}, 'Location', 'best');
    subplot(2, 1, 2);
    bar(categorical(scenarios), human_wait);
    ylabel('Human Wait (s)');
    legend({'A', 'B'}, 'Location', 'best');
    saveas(gcf, file_path);
    close(gcf);
end

function renderResponseLatencyBar(file_path, episode_metrics)
    scenarios = unique({episode_metrics.scenario_name});
    latency = zeros(numel(scenarios), 1);
    for idx = 1:numel(scenarios)
        subset = episode_metrics(strcmp({episode_metrics.scenario_name}, scenarios{idx}) & strcmp({episode_metrics.policy_name}, 'B'));
        values = [subset.response_latency_sec];
        values = values(~isnan(values));
        if isempty(values)
            latency(idx) = 0.0;
        else
            latency(idx) = mean(values);
        end
    end

    figure('Visible', 'off');
    bar(categorical(scenarios), latency);
    ylabel('Mean Response Latency (s)');
    title('Policy B Response Latency');
    saveas(gcf, file_path);
    close(gcf);
end

function renderRepresentativeTimeline(file_path, representative_timeline)
    frames = representative_timeline.frame_log;
    times = [frames.time_sec];
    separation = [frames.separation_m];
    robot_speed = [frames.robot_speed_scale];
    future_hesitation = [frames.future_hesitation_prob];

    figure('Visible', 'off');
    subplot(3, 1, 1);
    plot(times, separation, 'LineWidth', 1.5);
    ylabel('Separation (m)');
    subplot(3, 1, 2);
    plot(times, robot_speed, 'LineWidth', 1.5);
    ylabel('Robot Speed');
    subplot(3, 1, 3);
    plot(times, future_hesitation, 'LineWidth', 1.5);
    ylabel('Future Hesitation');
    xlabel('Time (s)');
    saveas(gcf, file_path);
    close(gcf);
end
