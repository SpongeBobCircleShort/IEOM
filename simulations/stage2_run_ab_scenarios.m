function summary = stage2_run_ab_scenarios(varargin)
% stage2_run_ab_scenarios: Stage 2 simulation runner with feature interface + replay mode.

    opts = parseStage2Options(varargin{:});
    root_dir = fileparts(fileparts(mfilename('fullpath')));
    output_root = fullfile(root_dir, 'artifacts', 'simulink_stage2');
    dirs = ensureStage2Dirs(output_root);

    scenarios = buildStage2Scenarios();
    config = buildStage2Config(opts);

    sim_baseline = struct([]);
    sim_aware = struct([]);
    for s = 1:numel(scenarios)
        sim_baseline(s) = runScenarioPolicy(scenarios{s}, 'baseline', config, dirs.feature_logs); %#ok<AGROW>
        sim_aware(s) = runScenarioPolicy(scenarios{s}, 'hesitation_aware', config, dirs.feature_logs); %#ok<AGROW>
    end

    sim_baseline_tbl = struct2table(sim_baseline);
    sim_aware_tbl = struct2table(sim_aware);
    sim_compare_tbl = buildComparisonTable(sim_baseline_tbl, sim_aware_tbl, "sim");

    writetable(sim_baseline_tbl, fullfile(dirs.metrics, 'metrics_baseline.csv'));
    writetable(sim_aware_tbl, fullfile(dirs.metrics, 'metrics_hesitation_aware.csv'));
    writetable(sim_compare_tbl, fullfile(dirs.comparison, 'comparison_summary_simulation.csv'));

    replay_baseline_tbl = table();
    replay_aware_tbl = table();
    replay_compare_tbl = table();
    if opts.enable_replay
        replay_baseline = struct([]);
        replay_aware = struct([]);
        for s = 1:numel(scenarios)
            replay_baseline(s) = replayScenarioPolicy(scenarios{s}, 'baseline', config, dirs.feature_logs, dirs.replay_logs); %#ok<AGROW>
            replay_aware(s) = replayScenarioPolicy(scenarios{s}, 'hesitation_aware', config, dirs.feature_logs, dirs.replay_logs); %#ok<AGROW>
        end
        replay_baseline_tbl = struct2table(replay_baseline);
        replay_aware_tbl = struct2table(replay_aware);
        replay_compare_tbl = buildComparisonTable(replay_baseline_tbl, replay_aware_tbl, "replay");

        writetable(replay_baseline_tbl, fullfile(dirs.metrics, 'metrics_baseline_replay.csv'));
        writetable(replay_aware_tbl, fullfile(dirs.metrics, 'metrics_hesitation_aware_replay.csv'));
        writetable(replay_compare_tbl, fullfile(dirs.comparison, 'comparison_summary_replay.csv'));
    end

    summary = struct( ...
        'output_root', output_root, ...
        'metrics_baseline', sim_baseline_tbl, ...
        'metrics_hesitation_aware', sim_aware_tbl, ...
        'comparison_simulation', sim_compare_tbl, ...
        'replay_baseline', replay_baseline_tbl, ...
        'replay_hesitation_aware', replay_aware_tbl, ...
        'comparison_replay', replay_compare_tbl);

    fprintf('Stage 2 complete. Output root: %s\n', output_root);
end

function opts = parseStage2Options(varargin)
    opts = struct('enable_replay', true, 'deterministic_seed', 42, 'window_size', 12, 'dt_sec', 0.1);
    if mod(numel(varargin), 2) ~= 0
        error('stage2_run_ab_scenarios expects name/value options.');
    end
    for idx = 1:2:numel(varargin)
        opts.(char(varargin{idx})) = varargin{idx + 1};
    end
end

function dirs = ensureStage2Dirs(output_root)
    dirs = struct();
    dirs.feature_logs = fullfile(output_root, 'feature_logs');
    dirs.replay_logs = fullfile(output_root, 'replay_logs');
    dirs.metrics = fullfile(output_root, 'metrics');
    dirs.comparison = fullfile(output_root, 'comparison');
    names = struct2cell(dirs);
    for i = 1:numel(names)
        if ~exist(names{i}, 'dir')
            mkdir(names{i});
        end
    end
end

function config = buildStage2Config(opts)
    config = struct();
    config.dt_sec = opts.dt_sec;
    config.max_steps = 1200;
    config.window_size = opts.window_size;
    config.seed = opts.deterministic_seed;

    config.robot_nominal_speed = 0.55;
    config.human_nominal_speed = 0.45;
    config.robot_release_delay_sec = 1.0;

    config.shared_zone_x = [0.42, 0.58];
    config.shared_zone_y = [0.35, 0.65];
    config.fixture_pos = [0.50, 0.50];
    config.human_start = [0.12, 0.50];
    config.robot_start = [0.88, 0.50];
    config.human_target = [0.86, 0.50];
    config.robot_target = [0.14, 0.50];
    config.overlap_buffer = 0.08;
end

function scenarios = buildStage2Scenarios()
    scenarios = {
        struct('name', 'smooth_operator', 'weights', [0.55, 0.10, 0.05, 0.05, 0.20, 0.05], 'task_step_count', 6), ...
        struct('name', 'hesitation_heavy_operator', 'weights', [0.20, 0.30, 0.25, 0.10, 0.10, 0.05], 'task_step_count', 6), ...
        struct('name', 'correction_heavy_operator', 'weights', [0.20, 0.15, 0.10, 0.40, 0.10, 0.05], 'task_step_count', 6), ...
        struct('name', 'overlap_risk_operator', 'weights', [0.20, 0.15, 0.10, 0.05, 0.15, 0.35], 'task_step_count', 6) ...
    };
end

function metrics = runScenarioPolicy(scenario, policy_name, config, feature_log_dir)
    state_names = {'normal_progress', 'mild_hesitation', 'strong_hesitation', 'correction_rework', 'ready_for_robot_action', 'overlap_risk'};
    seed_bump = strcmp(policy_name, 'hesitation_aware') * 1000;
    rng(config.seed + seed_bump + sum(double(char(scenario.name))), 'twister');

    human_pos = config.human_start;
    robot_pos = config.robot_start;
    human_progress = 0.0;
    robot_progress = 0.0;

    history = initializeFeatureHistory();
    feature_rows = struct([]);

    robot_idle_time = 0.0;
    human_idle_time = 0.0;
    overlap_event_count = 0;
    robot_hold_count = 0;
    unnecessary_slowdown_count = 0;
    correction_rework_count = 0;
    mismatch_count = 0;

    prev_overlap = false;
    prev_scripted_state = 'normal_progress';
    row_idx = 1;

    for step_idx = 1:config.max_steps
        t_sec = (step_idx - 1) * config.dt_sec;
        scripted_state = scriptedStateSample(state_names, scenario.weights, t_sec, human_progress);
        params = scriptedStateParameters(scripted_state);

        if strcmp(scripted_state, 'correction_rework') && ~strcmp(prev_scripted_state, 'correction_rework')
            correction_rework_count = correction_rework_count + 1;
        end

        pause_now = rand() < params.pause_probability;
        if pause_now
            human_speed = 0.0;
        else
            human_speed = config.human_nominal_speed * params.speed_scale;
        end

        task_step = min(scenario.task_step_count, max(1, 1 + floor(human_progress * scenario.task_step_count)));
        shared_zone_occ = double(isInSharedZone(human_pos, config));

        history = appendFeatureHistory(history, human_pos, human_speed, human_progress, params.retry_count, ...
            norm(config.human_target - human_pos), norm(human_pos - robot_pos), task_step, shared_zone_occ);

        if numel(history.speed) >= config.window_size
            feature_window = computeFeatureWindow(history, config.window_size, t_sec);
            validateFeatureSchemaContract(feature_window);
        else
            feature_window = emptyFeatureWindow(t_sec);
        end

        if strcmp(policy_name, 'baseline')
            predicted = struct('state', scripted_state, 'confidence', 1.0, 'source', 'script_passthrough');
            [robot_speed, robot_mode] = baselinePolicy(config, t_sec);
        else
            predicted = predict_hesitation_state(feature_window, scripted_state);
            [robot_speed, robot_mode] = policyBFromInference(predicted.state, config, t_sec);
        end

        if ~strcmp(predicted.state, scripted_state)
            mismatch_count = mismatch_count + 1;
        end

        if strcmp(robot_mode, 'hold')
            robot_hold_count = robot_hold_count + 1;
        end
        if (strcmp(robot_mode, 'hold') || strcmp(robot_mode, 'slow')) && ...
                (strcmp(scripted_state, 'normal_progress') || strcmp(scripted_state, 'ready_for_robot_action'))
            unnecessary_slowdown_count = unnecessary_slowdown_count + 1;
        end

        human_pos = stepToward(human_pos, config.human_target, human_speed, config.dt_sec);
        robot_pos = stepToward(robot_pos, config.robot_target, robot_speed, config.dt_sec);

        if rand() < params.shared_zone_entry_prob
            human_pos(1) = min(max(human_pos(1), config.shared_zone_x(1)), config.shared_zone_x(2));
            human_pos(2) = config.fixture_pos(2) + (rand() - 0.5) * 0.12;
        end

        human_progress = min(1.0, human_progress + params.progress_rate * config.dt_sec);
        robot_progress = min(1.0, robot_progress + max(robot_speed, 0.0) * config.dt_sec / norm(config.robot_start - config.robot_target));

        if human_speed <= 1e-6
            human_idle_time = human_idle_time + config.dt_sec;
        end
        if robot_speed <= 1e-6
            robot_idle_time = robot_idle_time + config.dt_sec;
        end

        overlap_now = isInSharedZone(human_pos, config) && isInSharedZone(robot_pos, config) && norm(human_pos - robot_pos) <= config.overlap_buffer;
        if overlap_now && ~prev_overlap
            overlap_event_count = overlap_event_count + 1;
        end
        prev_overlap = overlap_now;

        feature_rows(row_idx) = featureRow(t_sec, scenario.name, policy_name, scripted_state, predicted.state, ...
            predicted.confidence, feature_window, robot_mode, robot_speed); %#ok<AGROW>
        row_idx = row_idx + 1;

        prev_scripted_state = scripted_state;
        if human_progress >= 1.0 && robot_progress >= 1.0
            break;
        end
    end

    log_path = fullfile(feature_log_dir, sprintf('%s_%s.jsonl', scenario.name, policy_name));
    writeJsonl(log_path, feature_rows);

    sim_time = (step_idx - 1) * config.dt_sec;
    metrics = struct( ...
        'scenario', string(scenario.name), ...
        'policy', string(policy_name), ...
        'task_completion_time_sec', sim_time, ...
        'robot_idle_time_sec', robot_idle_time, ...
        'human_idle_time_sec', human_idle_time, ...
        'overlap_risk_event_count', overlap_event_count, ...
        'robot_hold_count', robot_hold_count, ...
        'unnecessary_slowdown_count', unnecessary_slowdown_count, ...
        'correction_rework_count', correction_rework_count, ...
        'state_prediction_mismatch_count', mismatch_count, ...
        'total_simulated_time_sec', sim_time);
end

function metrics = replayScenarioPolicy(scenario, policy_name, config, feature_log_dir, replay_log_dir)
    log_path = fullfile(feature_log_dir, sprintf('%s_%s.jsonl', scenario.name, policy_name));
    rows = readJsonl(log_path);
    if isempty(rows)
        error('Replay log missing or empty: %s', log_path);
    end

    mismatch_count = 0;
    overlap_count = 0;
    robot_hold_count = 0;
    unnecessary_slow = 0;
    correction_rework = 0;
    robot_idle_time = 0.0;
    human_idle_time = 0.0;
    prev_overlap = false;
    prev_scripted = '';

    replay_rows = struct([]);
    for i = 1:numel(rows)
        row = rows(i);
        feature = row.feature_window;
        validateFeatureSchemaContract(feature);

        predicted = predict_hesitation_state(feature, char(row.scripted_state));
        if ~strcmp(predicted.state, row.predicted_state)
            mismatch_count = mismatch_count + 1;
        end

        overlap_now = (feature.shared_zone_occupancy > 0.5) && (feature.human_robot_distance <= config.overlap_buffer);
        if overlap_now && ~prev_overlap
            overlap_count = overlap_count + 1;
        end
        prev_overlap = overlap_now;

        if strcmp(row.robot_mode, 'hold')
            robot_hold_count = robot_hold_count + 1;
        end
        if strcmp(row.robot_mode, 'hold') || strcmp(row.robot_mode, 'slow')
            if strcmp(row.scripted_state, 'normal_progress') || strcmp(row.scripted_state, 'ready_for_robot_action')
                unnecessary_slow = unnecessary_slow + 1;
            end
        end

        if strcmp(row.scripted_state, 'correction_rework') && ~strcmp(prev_scripted, 'correction_rework')
            correction_rework = correction_rework + 1;
        end
        prev_scripted = char(row.scripted_state);

        if row.robot_speed_cmd <= 1e-6
            robot_idle_time = robot_idle_time + config.dt_sec;
        end
        if feature.mean_speed <= 1e-6
            human_idle_time = human_idle_time + config.dt_sec;
        end

        replay_rows(i) = struct( ...
            'timestamp', row.timestamp, ...
            'scenario', row.scenario, ...
            'policy', row.policy, ...
            'scripted_state', row.scripted_state, ...
            'logged_predicted_state', row.predicted_state, ...
            'replay_predicted_state', string(predicted.state), ...
            'prediction_match', strcmp(predicted.state, row.predicted_state)); %#ok<AGROW>
    end

    replay_path = fullfile(replay_log_dir, sprintf('%s_%s_replay.csv', scenario.name, policy_name));
    writetable(struct2table(replay_rows), replay_path);

    total_time = rows(end).timestamp;
    metrics = struct( ...
        'scenario', string(scenario.name), ...
        'policy', string(policy_name), ...
        'task_completion_time_sec', total_time, ...
        'robot_idle_time_sec', robot_idle_time, ...
        'human_idle_time_sec', human_idle_time, ...
        'overlap_risk_event_count', overlap_count, ...
        'robot_hold_count', robot_hold_count, ...
        'unnecessary_slowdown_count', unnecessary_slow, ...
        'correction_rework_count', correction_rework, ...
        'state_prediction_mismatch_count', mismatch_count, ...
        'total_simulated_time_sec', total_time);
end

function table_out = buildComparisonTable(base_tbl, aware_tbl, mode_label)
    table_out = table();
    table_out.mode = repmat(mode_label, height(base_tbl), 1);
    table_out.scenario = base_tbl.scenario;
    table_out.task_completion_delta_sec = aware_tbl.task_completion_time_sec - base_tbl.task_completion_time_sec;
    table_out.robot_idle_delta_sec = aware_tbl.robot_idle_time_sec - base_tbl.robot_idle_time_sec;
    table_out.human_idle_delta_sec = aware_tbl.human_idle_time_sec - base_tbl.human_idle_time_sec;
    table_out.overlap_event_delta = aware_tbl.overlap_risk_event_count - base_tbl.overlap_risk_event_count;
    table_out.hold_count_delta = aware_tbl.robot_hold_count - base_tbl.robot_hold_count;
    table_out.unnecessary_slowdown_delta = aware_tbl.unnecessary_slowdown_count - base_tbl.unnecessary_slowdown_count;
    table_out.mismatch_delta = aware_tbl.state_prediction_mismatch_count - base_tbl.state_prediction_mismatch_count;
end

function [speed, mode] = baselinePolicy(config, t_sec)
    if t_sec < config.robot_release_delay_sec
        speed = 0.0;
        mode = 'hold';
    else
        speed = config.robot_nominal_speed;
        mode = 'proceed';
    end
end

function [speed, mode] = policyBFromInference(pred_state, config, t_sec)
    if t_sec < config.robot_release_delay_sec
        speed = 0.0;
        mode = 'hold';
        return;
    end
    switch pred_state
        case 'normal_progress'
            speed = config.robot_nominal_speed;
            mode = 'proceed';
        case 'mild_hesitation'
            speed = 0.75 * config.robot_nominal_speed;
            mode = 'slow';
        case 'strong_hesitation'
            speed = 0.45 * config.robot_nominal_speed;
            mode = 'slow';
        case 'correction_rework'
            speed = 0.0;
            mode = 'hold';
        case 'overlap_risk'
            speed = 0.0;
            mode = 'hold';
        case 'ready_for_robot_action'
            speed = config.robot_nominal_speed;
            mode = 'proceed';
        otherwise
            speed = config.robot_nominal_speed;
            mode = 'proceed';
    end
end

function pred = predict_hesitation_state(feature_window, scripted_state)
% predict_hesitation_state: Stage 2 pluggable interface stub.
    if isempty(fieldnames(feature_window))
        pred = struct('state', scripted_state, 'confidence', 0.50, 'source', 'cold_start_passthrough');
        return;
    end

    if feature_window.pause_ratio >= 0.55 || feature_window.shared_zone_occupancy >= 0.80
        state = 'overlap_risk';
    elseif feature_window.retry_count >= 2 || feature_window.progress_delta < 0.002
        state = 'correction_rework';
    elseif feature_window.pause_ratio >= 0.40
        state = 'strong_hesitation';
    elseif feature_window.pause_ratio >= 0.18
        state = 'mild_hesitation';
    elseif feature_window.task_step >= 5 && feature_window.distance_to_target < 0.18
        state = 'ready_for_robot_action';
    else
        state = 'normal_progress';
    end
    pred = struct('state', state, 'confidence', 0.75, 'source', 'stage2_stub');
end

function row = featureRow(ts, scenario, policy, scripted_state, predicted_state, confidence, feature_window, robot_mode, robot_speed)
    row = struct( ...
        'timestamp', ts, ...
        'scenario', string(scenario), ...
        'policy', string(policy), ...
        'scripted_state', string(scripted_state), ...
        'predicted_state', string(predicted_state), ...
        'prediction_confidence', confidence, ...
        'robot_mode', string(robot_mode), ...
        'robot_speed_cmd', robot_speed, ...
        'feature_window', feature_window);
end

function history = initializeFeatureHistory()
    history = struct('hand_pos', [], 'speed', [], 'progress', [], 'retry', [], ...
        'dist_target', [], 'human_robot_dist', [], 'task_step', [], 'shared_occ', []);
end

function history = appendFeatureHistory(history, hand_pos, speed, progress, retry_count, dist_target, human_robot_distance, task_step, shared_occ)
    history.hand_pos = [history.hand_pos; hand_pos];
    history.speed = [history.speed; speed];
    history.progress = [history.progress; progress];
    history.retry = [history.retry; retry_count];
    history.dist_target = [history.dist_target; dist_target];
    history.human_robot_dist = [history.human_robot_dist; human_robot_distance];
    history.task_step = [history.task_step; task_step];
    history.shared_occ = [history.shared_occ; shared_occ];
end

function feature = computeFeatureWindow(history, window_size, timestamp)
    start_idx = max(1, numel(history.speed) - window_size + 1);
    idx = start_idx:numel(history.speed);

    speed = history.speed(idx);
    progress = history.progress(idx);
    retry = history.retry(idx);
    pos = history.hand_pos(idx, :);

    move_delta = diff(pos, 1, 1);
    signed_dx = move_delta(:, 1);
    reversal_count = sum(abs(diff(sign(signed_dx))) > 0);

    progress_step = diff(progress);
    backtrack_ratio = 0.0;
    if ~isempty(progress_step)
        backtrack_ratio = sum(progress_step < 0) / numel(progress_step);
    end

    feature = struct( ...
        'timestamp', timestamp, ...
        'mean_speed', mean(speed), ...
        'pause_ratio', mean(speed <= 0.03), ...
        'progress_delta', progress(end) - progress(1), ...
        'reversal_count', reversal_count, ...
        'retry_count', max(retry), ...
        'distance_to_target', history.dist_target(end), ...
        'human_robot_distance', history.human_robot_dist(end), ...
        'task_step', history.task_step(end), ...
        'shared_zone_occupancy', mean(history.shared_occ(idx)), ...
        'speed_variance', var(speed), ...
        'direction_changes', reversal_count, ...
        'backtrack_ratio', backtrack_ratio, ...
        'mean_workspace_distance', mean(history.human_robot_dist(idx)));
end

function validateFeatureSchemaContract(feature)
    required = {
        'timestamp','mean_speed','pause_ratio','progress_delta','reversal_count','retry_count', ...
        'distance_to_target','human_robot_distance','task_step','shared_zone_occupancy', ...
        'speed_variance','direction_changes','backtrack_ratio','mean_workspace_distance'
    };
    for i = 1:numel(required)
        key = required{i};
        if ~isfield(feature, key)
            error('Stage2FeatureSchema:MissingField', 'Missing required feature field: %s', key);
        end
    end

    numeric_scalar = {'timestamp','mean_speed','pause_ratio','progress_delta','distance_to_target', ...
        'human_robot_distance','shared_zone_occupancy','speed_variance','backtrack_ratio','mean_workspace_distance'};
    for i = 1:numel(numeric_scalar)
        v = feature.(numeric_scalar{i});
        if ~(isnumeric(v) && isscalar(v))
            error('Stage2FeatureSchema:TypeMismatch', 'Field %s must be numeric scalar.', numeric_scalar{i});
        end
    end

    integer_scalar = {'reversal_count', 'retry_count', 'task_step', 'direction_changes'};
    for i = 1:numel(integer_scalar)
        v = feature.(integer_scalar{i});
        if ~(isnumeric(v) && isscalar(v) && abs(v - round(v)) < 1e-9)
            error('Stage2FeatureSchema:DimensionMismatch', 'Field %s must be scalar integer-like.', integer_scalar{i});
        end
    end

    if feature.pause_ratio < 0 || feature.pause_ratio > 1 || feature.shared_zone_occupancy < 0 || feature.shared_zone_occupancy > 1
        error('Stage2FeatureSchema:Range', 'Pause/shared occupancy features must be in [0,1].');
    end
end

function feature = emptyFeatureWindow(timestamp)
    feature = struct( ...
        'timestamp', timestamp, ...
        'mean_speed', 0.0, ...
        'pause_ratio', 0.0, ...
        'progress_delta', 0.0, ...
        'reversal_count', 0, ...
        'retry_count', 0, ...
        'distance_to_target', 1.0, ...
        'human_robot_distance', 1.0, ...
        'task_step', 1, ...
        'shared_zone_occupancy', 0.0, ...
        'speed_variance', 0.0, ...
        'direction_changes', 0, ...
        'backtrack_ratio', 0.0, ...
        'mean_workspace_distance', 1.0);
end

function state = scriptedStateSample(state_names, weights, t_sec, progress)
    adjusted = weights;
    if progress > 0.80
        adjusted(5) = adjusted(5) + 0.20;
        adjusted(3) = max(0.01, adjusted(3) - 0.10);
    end
    if t_sec < 5.0
        adjusted(1) = adjusted(1) + 0.15;
    end
    adjusted = adjusted / sum(adjusted);
    cdf = cumsum(adjusted);
    rv = rand();
    idx = find(rv <= cdf, 1, 'first');
    state = state_names{idx};
end

function params = scriptedStateParameters(state)
    switch state
        case 'normal_progress'
            params = mkParams(1.00, 0.02, 0, 0.10, 0.020);
        case 'mild_hesitation'
            params = mkParams(0.70, 0.20, 1, 0.20, 0.012);
        case 'strong_hesitation'
            params = mkParams(0.35, 0.55, 2, 0.35, 0.006);
        case 'correction_rework'
            params = mkParams(0.40, 0.45, 3, 0.25, 0.004);
        case 'ready_for_robot_action'
            params = mkParams(1.05, 0.00, 0, 0.18, 0.022);
        case 'overlap_risk'
            params = mkParams(0.85, 0.05, 0, 0.60, 0.015);
        otherwise
            params = mkParams(1.00, 0.05, 0, 0.10, 0.015);
    end
end

function params = mkParams(speed_scale, pause_probability, retry_count, shared_zone_entry_prob, progress_rate)
    params = struct('speed_scale', speed_scale, 'pause_probability', pause_probability, 'retry_count', retry_count, ...
        'shared_zone_entry_prob', shared_zone_entry_prob, 'progress_rate', progress_rate);
end

function pos = stepToward(pos, target, speed, dt)
    delta = target - pos;
    dist = norm(delta);
    if dist < 1e-9 || speed <= 0
        return;
    end
    dir = delta / dist;
    pos = pos + dir * min(dist, speed * dt);
end

function flag = isInSharedZone(pos, config)
    flag = pos(1) >= config.shared_zone_x(1) && pos(1) <= config.shared_zone_x(2) && ...
        pos(2) >= config.shared_zone_y(1) && pos(2) <= config.shared_zone_y(2);
end

function writeJsonl(path, rows)
    fid = fopen(path, 'w');
    if fid < 0
        error('Cannot open file for writing: %s', path);
    end
    cleaner = onCleanup(@() fclose(fid));
    for i = 1:numel(rows)
        fprintf(fid, '%s\n', jsonencode(rows(i)));
    end
    clear cleaner;
end

function rows = readJsonl(path)
    fid = fopen(path, 'r');
    if fid < 0
        error('Cannot open file for reading: %s', path);
    end
    cleaner = onCleanup(@() fclose(fid));

    rows = struct([]);
    idx = 1;
    while true
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        if isempty(strtrim(line))
            continue;
        end
        rows(idx) = jsondecode(line); %#ok<AGROW>
        idx = idx + 1;
    end
    clear cleaner;
end
