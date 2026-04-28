function stage1_summary = stage1_run_ab_scenarios()
% stage1_run_ab_scenarios: Stage 1 A/B shared-workspace simulation scaffold.
% Runs scripted-human scenarios against baseline and hesitation-aware placeholder policies.

    rng(7, 'twister');
    repo_root = fileparts(fileparts(mfilename('fullpath')));
    output_dir = fullfile(repo_root, 'artifacts', 'simulink_stage1');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    config = defaultStage1Config();
    scenarios = buildScenarioProfiles();

    baseline_rows = struct([]);
    aware_rows = struct([]);

    for idx = 1:numel(scenarios)
        scenario = scenarios{idx};
        baseline_rows(idx) = runSingleScenario(scenario, 'baseline', config); %#ok<AGROW>
        aware_rows(idx) = runSingleScenario(scenario, 'hesitation_aware', config); %#ok<AGROW>
    end

    baseline_table = struct2table(baseline_rows);
    aware_table = struct2table(aware_rows);
    comparison_table = buildComparisonTable(baseline_table, aware_table);

    writetable(baseline_table, fullfile(output_dir, 'metrics_baseline.csv'));
    writetable(aware_table, fullfile(output_dir, 'metrics_hesitation_aware.csv'));
    writetable(comparison_table, fullfile(output_dir, 'comparison_summary.csv'));

    writeSimplePlots(output_dir, baseline_table, aware_table);

    stage1_summary = struct( ...
        'output_dir', output_dir, ...
        'baseline_table', baseline_table, ...
        'hesitation_aware_table', aware_table, ...
        'comparison_table', comparison_table);

    fprintf('Stage 1 A/B simulation complete. Artifacts: %s\n', output_dir);
end

function config = defaultStage1Config()
    config = struct();
    config.dt_sec = 0.1;
    config.max_time_sec = 120;
    config.robot_nominal_speed = 0.55;
    config.human_nominal_speed = 0.45;
    config.robot_release_delay_sec = 1.0;
    config.robot_fixed_wait_sec = 0.3;
    config.shared_zone_x = [0.42, 0.58];
    config.shared_zone_y = [0.35, 0.65];
    config.human_zone_x = [0.00, 0.50];
    config.robot_zone_x = [0.50, 1.00];
    config.fixture_pos = [0.50, 0.50];
    config.human_start = [0.12, 0.50];
    config.robot_start = [0.88, 0.50];
    config.human_target = [0.86, 0.50];
    config.robot_target = [0.14, 0.50];
    config.overlap_buffer = 0.08;
end

function scenarios = buildScenarioProfiles()
    scenarios = {
        struct('name', 'smooth_operator', 'state_weights', [0.55, 0.10, 0.05, 0.05, 0.20, 0.05], 'seed_offset', 1), ...
        struct('name', 'hesitation_heavy_operator', 'state_weights', [0.20, 0.30, 0.25, 0.10, 0.10, 0.05], 'seed_offset', 2), ...
        struct('name', 'correction_heavy_operator', 'state_weights', [0.20, 0.15, 0.10, 0.40, 0.10, 0.05], 'seed_offset', 3), ...
        struct('name', 'overlap_risk_operator', 'state_weights', [0.20, 0.15, 0.10, 0.05, 0.15, 0.35], 'seed_offset', 4) ...
    };
end

function metrics = runSingleScenario(scenario, policy_mode, config)
    state_names = {'normal_progress', 'mild_hesitation', 'strong_hesitation', 'correction_rework', 'ready_for_robot_action', 'overlap_risk'};

    rng(100 + scenario.seed_offset, 'twister');
    dt = config.dt_sec;
    steps = floor(config.max_time_sec / dt);

    human_pos = config.human_start;
    robot_pos = config.robot_start;
    human_progress = 0.0;
    robot_progress = 0.0;

    robot_idle_time = 0.0;
    human_idle_time = 0.0;
    overlap_event_count = 0;
    robot_hold_count = 0;
    unnecessary_slowdown_count = 0;
    correction_rework_count = 0;

    prev_overlap = false;
    prev_state = 'normal_progress';

    for step_idx = 1:steps
        t_sec = (step_idx - 1) * dt;
        scripted_state = sampleScriptedState(state_names, scenario.state_weights, t_sec, human_progress);
        state_params = stateParameters(scripted_state);

        if strcmp(scripted_state, 'correction_rework') && ~strcmp(prev_state, 'correction_rework')
            correction_rework_count = correction_rework_count + 1;
        end

        if rand() < state_params.pause_probability
            human_speed = 0.0;
        else
            human_speed = config.human_nominal_speed * state_params.speed_scale;
        end

        if strcmp(policy_mode, 'baseline')
            if t_sec < config.robot_release_delay_sec + config.robot_fixed_wait_sec
                robot_speed = 0.0;
                robot_mode = 'hold';
            else
                robot_speed = config.robot_nominal_speed;
                robot_mode = 'proceed';
            end
        else
            [robot_speed, robot_mode] = hesitationAwarePolicy(scripted_state, config, t_sec);
        end

        if strcmp(robot_mode, 'hold')
            robot_hold_count = robot_hold_count + 1;
        end
        if (strcmp(robot_mode, 'hold') || strcmp(robot_mode, 'slow')) && ...
                (strcmp(scripted_state, 'normal_progress') || strcmp(scripted_state, 'ready_for_robot_action'))
            unnecessary_slowdown_count = unnecessary_slowdown_count + 1;
        end

        human_pos = stepToward(human_pos, config.human_target, human_speed, dt);
        robot_pos = stepToward(robot_pos, config.robot_target, robot_speed, dt);

        if rand() < state_params.shared_zone_entry_prob
            human_pos(1) = min(max(human_pos(1), config.shared_zone_x(1)), config.shared_zone_x(2));
            human_pos(2) = config.fixture_pos(2) + (rand() - 0.5) * 0.12;
        end

        human_progress = min(1.0, human_progress + state_params.progress_rate * dt);
        robot_progress = min(1.0, robot_progress + max(robot_speed, 0.0) * dt / norm(config.robot_start - config.robot_target));

        if human_speed <= 1e-6
            human_idle_time = human_idle_time + dt;
        end
        if robot_speed <= 1e-6
            robot_idle_time = robot_idle_time + dt;
        end

        in_overlap = isInSharedZone(human_pos, config) && isInSharedZone(robot_pos, config) && norm(human_pos - robot_pos) < config.overlap_buffer;
        if in_overlap && ~prev_overlap
            overlap_event_count = overlap_event_count + 1;
        end
        prev_overlap = in_overlap;
        prev_state = scripted_state;

        if human_progress >= 1.0 && robot_progress >= 1.0
            break;
        end
    end

    task_completion_time = (step_idx - 1) * dt;
    metrics = struct( ...
        'scenario', string(scenario.name), ...
        'policy', string(policy_mode), ...
        'task_completion_time_sec', task_completion_time, ...
        'robot_idle_time_sec', robot_idle_time, ...
        'human_idle_time_sec', human_idle_time, ...
        'overlap_risk_event_count', overlap_event_count, ...
        'robot_hold_count', robot_hold_count, ...
        'unnecessary_slowdown_count', unnecessary_slowdown_count, ...
        'correction_rework_count', correction_rework_count, ...
        'total_simulated_time_sec', task_completion_time);
end

function state = sampleScriptedState(state_names, weights, t_sec, progress)
    adjusted = weights;
    if progress > 0.80
        adjusted(5) = adjusted(5) + 0.20;
        adjusted(3) = max(0.01, adjusted(3) - 0.10);
    end
    if t_sec < 5.0
        adjusted(1) = adjusted(1) + 0.15;
    end
    adjusted = adjusted ./ sum(adjusted);

    cdf = cumsum(adjusted);
    r = rand();
    idx = find(r <= cdf, 1, 'first');
    state = state_names{idx};
end

function params = stateParameters(state)
    switch state
        case 'normal_progress'
            params = mkParams(1.00, 0.02, 0.00, 0.10, 0.020);
        case 'mild_hesitation'
            params = mkParams(0.70, 0.20, 1.00, 0.20, 0.012);
        case 'strong_hesitation'
            params = mkParams(0.35, 0.55, 2.00, 0.35, 0.006);
        case 'correction_rework'
            params = mkParams(0.40, 0.45, 3.00, 0.25, 0.004);
        case 'ready_for_robot_action'
            params = mkParams(1.05, 0.00, 0.00, 0.18, 0.022);
        case 'overlap_risk'
            params = mkParams(0.85, 0.05, 0.00, 0.60, 0.015);
        otherwise
            params = mkParams(1.00, 0.05, 0.00, 0.10, 0.015);
    end
end

function params = mkParams(speed_scale, pause_probability, retry_count, shared_zone_entry_prob, progress_rate)
    params = struct( ...
        'speed_scale', speed_scale, ...
        'pause_probability', pause_probability, ...
        'retry_count', retry_count, ...
        'shared_zone_entry_prob', shared_zone_entry_prob, ...
        'progress_rate', progress_rate);
end

function [robot_speed, mode] = hesitationAwarePolicy(state, config, t_sec)
    nominal = config.robot_nominal_speed;
    if t_sec < config.robot_release_delay_sec
        robot_speed = 0.0;
        mode = 'hold';
        return;
    end

    switch state
        case 'normal_progress'
            robot_speed = nominal;
            mode = 'proceed';
        case 'mild_hesitation'
            robot_speed = 0.75 * nominal;
            mode = 'slow';
        case 'strong_hesitation'
            robot_speed = 0.45 * nominal;
            mode = 'slow';
        case 'correction_rework'
            robot_speed = 0.0;
            mode = 'hold';
        case 'overlap_risk'
            robot_speed = 0.0;
            mode = 'hold';
        case 'ready_for_robot_action'
            robot_speed = nominal;
            mode = 'proceed';
        otherwise
            robot_speed = nominal;
            mode = 'proceed';
    end
end

function pos = stepToward(pos, target, speed_mps, dt_sec)
    delta = target - pos;
    dist = norm(delta);
    if dist < 1e-9 || speed_mps <= 0.0
        return;
    end
    dir = delta / dist;
    step = min(dist, speed_mps * dt_sec);
    pos = pos + dir * step;
end

function flag = isInSharedZone(pos, config)
    flag = pos(1) >= config.shared_zone_x(1) && pos(1) <= config.shared_zone_x(2) && ...
           pos(2) >= config.shared_zone_y(1) && pos(2) <= config.shared_zone_y(2);
end

function comparison = buildComparisonTable(baseline_table, aware_table)
    comparison = table();
    comparison.scenario = baseline_table.scenario;
    comparison.completion_time_delta_sec = aware_table.task_completion_time_sec - baseline_table.task_completion_time_sec;
    comparison.robot_idle_delta_sec = aware_table.robot_idle_time_sec - baseline_table.robot_idle_time_sec;
    comparison.human_idle_delta_sec = aware_table.human_idle_time_sec - baseline_table.human_idle_time_sec;
    comparison.overlap_risk_event_delta = aware_table.overlap_risk_event_count - baseline_table.overlap_risk_event_count;
    comparison.robot_hold_delta = aware_table.robot_hold_count - baseline_table.robot_hold_count;
    comparison.unnecessary_slowdown_delta = aware_table.unnecessary_slowdown_count - baseline_table.unnecessary_slowdown_count;
    comparison.correction_rework_delta = aware_table.correction_rework_count - baseline_table.correction_rework_count;
end

function writeSimplePlots(output_dir, baseline_table, aware_table)
    figure('Visible', 'off');
    vals = [baseline_table.task_completion_time_sec, aware_table.task_completion_time_sec];
    bar(vals);
    grid on;
    title('Task Completion Time: Baseline vs Hesitation-Aware');
    ylabel('Seconds');
    xticklabels(cellstr(baseline_table.scenario));
    legend({'baseline', 'hesitation-aware'}, 'Location', 'northwest');
    saveas(gcf, fullfile(output_dir, 'task_completion_comparison.png'));
    close(gcf);

    figure('Visible', 'off');
    vals2 = [baseline_table.overlap_risk_event_count, aware_table.overlap_risk_event_count];
    bar(vals2);
    grid on;
    title('Overlap-Risk Event Count: Baseline vs Hesitation-Aware');
    ylabel('Events');
    xticklabels(cellstr(baseline_table.scenario));
    legend({'baseline', 'hesitation-aware'}, 'Location', 'northwest');
    saveas(gcf, fullfile(output_dir, 'overlap_event_comparison.png'));
    close(gcf);
end
